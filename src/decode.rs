//! Deterministic span decoding shared with the reference runtime:
//! overlap policies (`gliner2.inference.overlap`), typed relation pair proposal
//! (`TypedRelationPairGenerator`) and relation edge deduplication.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};

use crate::heads::RelationPair;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapPolicy {
    /// Keep every distinct span.
    Allow,
    /// Keep disjoint and nested spans, reject crossings.
    Nested,
    /// Maximum-total-score non-overlapping subset (`flat`).
    Disallow,
    /// Drop spans strictly contained in another candidate.
    Longest,
}

impl std::str::FromStr for OverlapPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_lowercase().replace('-', "_").as_str() {
            "allow" | "all" | "none" => Self::Allow,
            "nested" | "allow_nested" => Self::Nested,
            "flat" | "disallow" | "no_overlap" | "non_overlapping" => Self::Disallow,
            "longest" | "keep_longest" => Self::Longest,
            other => bail!("unknown overlap_policy {other:?}; expected allow, nested, flat/disallow, longest"),
        })
    }
}

/// A scored half-open span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredSpan {
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

type RankKey = (f64, usize, usize, usize);

fn rank_key(index: usize, item: &ScoredSpan) -> RankKey {
    (-(item.score as f64), item.start, item.end, index)
}

fn cmp_rank(a: &RankKey, b: &RankKey) -> Ordering {
    a.0.total_cmp(&b.0)
        .then(a.1.cmp(&b.1))
        .then(a.2.cmp(&b.2))
        .then(a.3.cmp(&b.3))
}

/// Resolve overlapping spans; output is ranked by descending score, then start/end.
pub fn resolve_overlaps(items: &[ScoredSpan], policy: OverlapPolicy) -> Vec<ScoredSpan> {
    let mut ranked: Vec<(usize, ScoredSpan)> = items.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| cmp_rank(&rank_key(a.0, &a.1), &rank_key(b.0, &b.1)));
    let mut seen = HashSet::new();
    let distinct: Vec<(usize, ScoredSpan)> =
        ranked.into_iter().filter(|(_, s)| seen.insert((s.start, s.end))).collect();

    match policy {
        OverlapPolicy::Allow => distinct.into_iter().map(|(_, s)| s).collect(),
        OverlapPolicy::Nested => {
            let mut kept: Vec<ScoredSpan> = Vec::new();
            for (_, cand) in distinct {
                let crossing = kept.iter().any(|ex| {
                    let overlaps = cand.start < ex.end && ex.start < cand.end;
                    let contains = (cand.start <= ex.start && ex.end <= cand.end)
                        || (ex.start <= cand.start && cand.end <= ex.end);
                    overlaps && !contains
                });
                if !crossing {
                    kept.push(cand);
                }
            }
            kept
        }
        OverlapPolicy::Longest => distinct
            .iter()
            .filter(|(_, cand)| {
                !distinct.iter().any(|(_, other)| {
                    other.start <= cand.start
                        && cand.end <= other.end
                        && (other.start < cand.start || cand.end < other.end)
                })
            })
            .map(|(_, s)| *s)
            .collect(),
        OverlapPolicy::Disallow => weighted_interval_scheduling(distinct),
    }
}

fn weighted_interval_scheduling(distinct: Vec<(usize, ScoredSpan)>) -> Vec<ScoredSpan> {
    let mut by_end = distinct;
    by_end.sort_by(|a, b| {
        a.1.end
            .cmp(&b.1.end)
            .then(a.1.start.cmp(&b.1.start))
            .then((-(a.1.score as f64)).total_cmp(&-(b.1.score as f64)))
            .then(a.0.cmp(&b.0))
    });
    let ends: Vec<usize> = by_end.iter().map(|(_, s)| s.end).collect();
    let predecessors: Vec<isize> = by_end
        .iter()
        .enumerate()
        .map(|(index, (_, item))| ends[..index].partition_point(|&e| e <= item.start) as isize - 1)
        .collect();

    let selection_key = |selection: &[usize]| -> Vec<RankKey> {
        let mut keys: Vec<RankKey> = selection.iter().map(|&i| rank_key(by_end[i].0, &by_end[i].1)).collect();
        keys.sort_by(cmp_rank);
        keys
    };
    let lexicographic_less = |a: &[RankKey], b: &[RankKey]| -> bool {
        for (x, y) in a.iter().zip(b) {
            match cmp_rank(x, y) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }
        a.len() < b.len()
    };

    let mut best: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
    for (index, (_, item)) in by_end.iter().enumerate() {
        let previous = &best[(predecessors[index] + 1) as usize];
        let mut with_sel = previous.1.clone();
        with_sel.push(index);
        let with_item = (previous.0 + item.score as f64, with_sel);
        let without_item = best[index].clone();
        let choice = if with_item.0 > without_item.0 {
            with_item
        } else if with_item.0 < without_item.0 {
            without_item
        } else if with_item.1.len() > without_item.1.len() {
            with_item
        } else if with_item.1.len() < without_item.1.len() {
            without_item
        } else if lexicographic_less(&selection_key(&with_item.1), &selection_key(&without_item.1)) {
            with_item
        } else {
            without_item
        };
        best.push(choice);
    }

    let mut selected: Vec<(usize, ScoredSpan)> = best.last().unwrap().1.iter().map(|&i| by_end[i]).collect();
    selected.sort_by(|a, b| cmp_rank(&rank_key(a.0, &a.1), &rank_key(b.0, &b.1)));
    selected.into_iter().map(|(_, s)| s).collect()
}

// ---------------------------------------------------------------------------
// Relation pair proposal
// ---------------------------------------------------------------------------

pub struct RelationProposalSettings {
    pub heads_per_relation: usize,
    pub tails_per_relation: usize,
    pub pair_cap: usize,
    pub argument_threshold: f32,
}

/// Head/tail query ids of one relation type.
pub struct RelationRoles {
    pub head_query: usize,
    pub tail_query: usize,
}

/// Typed, capped relation pairs from the shared entity candidates.
///
/// `probs` is `[Q][C]` sigmoid pair probabilities. Pairs are returned grouped by
/// relation, each group ordered by descending `head_prob * tail_prob`.
pub fn propose_relation_pairs(
    probs: &[Vec<f32>],
    spans: &[(usize, usize)],
    relations: &[RelationRoles],
    settings: &RelationProposalSettings,
) -> Vec<RelationPair> {
    // Secondary order: (start, end, flat index); primary: probability desc.
    let select = |query: usize, take: usize| -> Vec<(f32, (usize, usize))> {
        let Some(row) = probs.get(query) else { return Vec::new() };
        let mut valid: Vec<usize> = (0..spans.len()).filter(|&c| row[c] >= settings.argument_threshold).collect();
        valid.sort_by(|&a, &b| spans[a].cmp(&spans[b]).then(a.cmp(&b)));
        valid.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        valid.into_iter().take(take).map(|c| (row[c], spans[c])).collect()
    };

    let mut out = Vec::new();
    for (relation, roles) in relations.iter().enumerate() {
        let heads = select(roles.head_query, settings.heads_per_relation);
        let tails = select(roles.tail_query, settings.tails_per_relation);
        let mut pairs: Vec<(f32, (usize, usize), (usize, usize))> = Vec::new();
        for &(hp, hspan) in &heads {
            for &(tp, tspan) in &tails {
                if hspan != tspan {
                    pairs.push((hp * tp, hspan, tspan));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        out.extend(
            pairs
                .into_iter()
                .take(settings.pair_cap)
                .map(|(_, head, tail)| RelationPair { relation, head, tail }),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Relation edge deduplication
// ---------------------------------------------------------------------------

/// `(surface text, char start, char end)`.
pub type Mention = (String, usize, usize);

#[derive(Debug, Clone)]
pub struct RelationEdge {
    pub score: f32,
    pub head: Mention,
    pub tail: Mention,
}

fn semantic_text(value: &str) -> String {
    value.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Collapse overlap and repeated-mention cross-products into semantic edges.
pub fn deduplicate_relation_edges(edges: Vec<RelationEdge>) -> Vec<RelationEdge> {
    if edges.len() < 2 {
        return edges;
    }

    let canonical = |side: fn(&RelationEdge) -> &Mention| -> HashMap<(usize, usize), Mention> {
        let mut order: Vec<(usize, usize)> = Vec::new();
        let mut mentions: HashMap<(usize, usize), Mention> = HashMap::new();
        for edge in &edges {
            let m = side(edge);
            let coords = (m.1, m.2);
            if !mentions.contains_key(&coords) {
                order.push(coords);
            }
            mentions.insert(coords, m.clone());
        }
        let mut result = HashMap::new();
        for &(start, end) in &order {
            let mut best: Option<&Mention> = None;
            for coords in &order {
                let cand = &mentions[coords];
                if cand.1 <= start && cand.2 >= end {
                    let better = match best {
                        None => true,
                        Some(b) => (cand.2 - cand.1, -(cand.1 as i64)) > (b.2 - b.1, -(b.1 as i64)),
                    };
                    if better {
                        best = Some(cand);
                    }
                }
            }
            result.insert((start, end), best.unwrap().clone());
        }
        result
    };
    let head_canonical = canonical(|e| &e.head);
    let tail_canonical = canonical(|e| &e.tail);

    let mut exact_order: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut exact: HashMap<(usize, usize, usize, usize), RelationEdge> = HashMap::new();
    for edge in &edges {
        let head = head_canonical[&(edge.head.1, edge.head.2)].clone();
        let tail = tail_canonical[&(edge.tail.1, edge.tail.2)].clone();
        let key = (head.1, head.2, tail.1, tail.2);
        let normalized = RelationEdge { score: edge.score, head, tail };
        match exact.get(&key) {
            None => {
                exact_order.push(key);
                exact.insert(key, normalized);
            }
            Some(prev) if edge.score > prev.score => {
                exact.insert(key, normalized);
            }
            _ => {}
        }
    }

    let rank = |e: &RelationEdge| {
        let distance = (e.head.1 as i64 - e.tail.2 as i64).max(e.tail.1 as i64 - e.head.2 as i64).max(0);
        (distance, -(e.score as f64), e.head.1, e.tail.1)
    };
    let mut semantic_order: Vec<(String, String)> = Vec::new();
    let mut semantic: HashMap<(String, String), RelationEdge> = HashMap::new();
    for key in &exact_order {
        let edge = exact[key].clone();
        let skey = (semantic_text(&edge.head.0), semantic_text(&edge.tail.0));
        match semantic.get(&skey) {
            None => {
                semantic_order.push(skey.clone());
                semantic.insert(skey, edge);
            }
            Some(prev) => {
                let (a, b) = (rank(&edge), rank(prev));
                let less = a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3))
                    == Ordering::Less;
                if less {
                    semantic.insert(skey, edge);
                }
            }
        }
    }

    let values: Vec<RelationEdge> = semantic_order.iter().map(|k| semantic[k].clone()).collect();
    let token_set = |s: &str| -> HashSet<String> { semantic_text(s).split(' ').map(str::to_string).collect() };
    let strict_subset = |a: &HashSet<String>, b: &HashSet<String>| a.len() < b.len() && a.is_subset(b);
    let mut kept: Vec<RelationEdge> = Vec::new();
    for (i, edge) in values.iter().enumerate() {
        let (h, t) = (token_set(&edge.head.0), token_set(&edge.tail.0));
        let dominated = values.iter().enumerate().any(|(j, other)| {
            if i == j {
                return false;
            }
            let (oh, ot) = (token_set(&other.head.0), token_set(&other.tail.0));
            (strict_subset(&h, &oh) && t == ot) || (strict_subset(&t, &ot) && h == oh)
        });
        if !dominated {
            kept.push(edge.clone());
        }
    }
    kept.sort_by(|a, b| {
        a.head.1.cmp(&b.head.1).then(a.tail.1.cmp(&b.tail.1)).then(b.score.total_cmp(&a.score))
    });
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(score: f32, start: usize, end: usize) -> ScoredSpan {
        ScoredSpan { score, start, end }
    }

    #[test]
    fn flat_maximizes_total_score() {
        // [0,3) 0.9 overlaps both [0,1) 0.6 and [2,3) 0.6, whose sum wins.
        let items = [span(0.9, 0, 3), span(0.6, 0, 1), span(0.6, 2, 3)];
        let out = resolve_overlaps(&items, OverlapPolicy::Disallow);
        assert_eq!(out, vec![span(0.6, 0, 1), span(0.6, 2, 3)]);
    }

    #[test]
    fn flat_prefers_single_higher_score() {
        let items = [span(0.9, 0, 3), span(0.3, 0, 1), span(0.3, 2, 3)];
        assert_eq!(resolve_overlaps(&items, OverlapPolicy::Disallow), vec![span(0.9, 0, 3)]);
    }

    #[test]
    fn nested_rejects_crossings() {
        let items = [span(0.9, 0, 4), span(0.8, 1, 2), span(0.7, 3, 6)];
        assert_eq!(
            resolve_overlaps(&items, OverlapPolicy::Nested),
            vec![span(0.9, 0, 4), span(0.8, 1, 2)]
        );
    }

    #[test]
    fn longest_drops_contained() {
        let items = [span(0.5, 0, 4), span(0.9, 1, 2), span(0.7, 5, 6)];
        assert_eq!(
            resolve_overlaps(&items, OverlapPolicy::Longest),
            vec![span(0.7, 5, 6), span(0.5, 0, 4)]
        );
    }

    #[test]
    fn relation_dedup_prefers_complete_mentions() {
        let m = |t: &str, s: usize, e: usize| (t.to_string(), s, e);
        let edges = vec![
            RelationEdge { score: 0.9, head: m("Alice", 0, 5), tail: m("Acme", 16, 20) },
            RelationEdge { score: 0.8, head: m("Alice", 0, 5), tail: m("Acme Corp", 16, 25) },
        ];
        let out = deduplicate_relation_edges(edges);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tail.0, "Acme Corp");
    }
}
