//! Instance formation for structures (`gliner2.models.boundary.records`),
//! natural (anchor-driven) mode as used by `extract_json`.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// How many mentions a field may bind within one record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    OptionalOne,
    RequiredOne,
    ZeroOrMore,
    OneOrMore,
}

impl Cardinality {
    pub fn is_scalar(self) -> bool {
        matches!(self, Self::OptionalOne | Self::RequiredOne)
    }

    pub fn allows_absent(self) -> bool {
        matches!(self, Self::OptionalOne | Self::ZeroOrMore)
    }
}

#[derive(Debug, Clone)]
pub struct RecordField {
    /// Boundary query id of the field marker.
    pub query: usize,
    pub cardinality: Cardinality,
    pub is_anchor: bool,
    pub exclusive: bool,
}

/// Natural-mode record head weights.
pub struct RecordHead {
    inst_proj: Linear,
    field_proj: Linear,
    cand_proj: Linear,
    null_embed: Tensor,
}

/// Per-field assignment logits `[F][I][1 + C]` (column 0 = ABSENT) for natural
/// mode, where instances are the anchor candidates (`I == C`).
pub type AssignLogits = Vec<Vec<Vec<f32>>>;

impl RecordHead {
    pub fn load(vb: VarBuilder, hidden: usize, record_dim: usize) -> Result<Self> {
        Ok(Self {
            inst_proj: linear(hidden, record_dim, vb.pp("inst_proj"))?,
            field_proj: linear(hidden, record_dim, vb.pp("field_proj"))?,
            cand_proj: linear(hidden, record_dim, vb.pp("cand_proj"))?,
            null_embed: vb.get(record_dim, "null_embed")?,
        })
    }

    /// `candidate_states`: `[C, H]` (shared pool), `field_queries`: `[F, H]`.
    pub fn assign_logits(&self, candidate_states: &Tensor, field_queries: &Tensor) -> Result<AssignLogits> {
        let inst = self.inst_proj.forward(candidate_states)?; // [C, D]
        let field = self.field_proj.forward(field_queries)?; // [F, D]
        let query = inst.unsqueeze(0)?.broadcast_add(&field.unsqueeze(1)?)?; // [F, C, D]
        let null = query.broadcast_matmul(&self.null_embed.unsqueeze(1)?)?; // [F, C, 1]
        let cand = self.cand_proj.forward(candidate_states)?; // [C, D]
        let scores = query.broadcast_matmul(&cand.t()?)?; // [F, C, C]
        Tensor::cat(&[&null, &scores], 2)?.to_vec3::<f32>()
    }
}

#[derive(Debug, Clone)]
pub struct DecodedRecord {
    /// Field index -> selected `(candidate index, score)`.
    pub fields: HashMap<usize, Vec<(usize, f32)>>,
    /// Anchor candidate index.
    pub anchor: usize,
    pub score: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Stable descending argsort (torch `argsort(descending=True)` on distinct values).
fn argsort_desc(xs: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[b].total_cmp(&xs[a]));
    idx
}

/// `decode_group` for natural mode. `anchor_logits` are the anchor query's pair
/// logits over the `C` pool candidates, `spans` their token spans.
pub fn decode_natural(
    fields: &[RecordField],
    anchor_logits: &[f32],
    spans: &[(usize, usize)],
    assign: &AssignLogits,
    threshold: f32,
    temperature: f32,
) -> Vec<DecodedRecord> {
    let ni = anchor_logits.len();
    if ni == 0 {
        return Vec::new();
    }
    let obj_prob: Vec<f32> = anchor_logits.iter().map(|&x| sigmoid(x / temperature)).collect();
    let mut order: Vec<usize> = (0..ni).collect();
    order.sort_by(|&a, &b| obj_prob[b].total_cmp(&obj_prob[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order.into_iter().filter(|&i| obj_prob[i] >= threshold).collect();
    let anchor_field = fields.iter().position(|f| f.is_anchor);

    // Exclusive fields: joint scalar assignment and per-candidate list ownership.
    let mut scalar_choices: HashMap<(usize, usize), Option<(usize, f32)>> = HashMap::new();
    let mut list_owners: HashMap<(usize, usize), (usize, f32)> = HashMap::new();
    for (f_idx, field) in fields.iter().enumerate() {
        if !field.exclusive || selected.is_empty() || Some(f_idx) == anchor_field {
            continue;
        }
        let logits: Vec<Vec<f32>> = selected
            .iter()
            .map(|&inst| assign[f_idx][inst].iter().map(|x| x / temperature).collect())
            .collect();
        let candidate_count = logits[0].len().saturating_sub(1);
        if field.cardinality.is_scalar() {
            if candidate_count == 0 {
                for &inst in &selected {
                    scalar_choices.insert((inst, f_idx), None);
                }
                continue;
            }
            let rows = selected.len();
            let probs: Vec<Vec<f32>> = logits.iter().map(|row| softmax(row)).collect();
            let eps = f32::EPSILON;
            let candidate_cost: Vec<Vec<f32>> =
                probs.iter().map(|row| row[1..].iter().map(|p| -p.max(eps).ln()).collect()).collect();
            let max_candidate_cost = candidate_cost.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max);
            let diagonal: Vec<f32> = if field.cardinality.allows_absent() {
                probs.iter().map(|row| -row[0].max(eps).ln()).collect()
            } else {
                vec![max_candidate_cost + 50.0; rows]
            };
            let invalid = max_candidate_cost.max(diagonal.iter().copied().fold(f32::NEG_INFINITY, f32::max)) + 1000.0;
            let cost: Vec<Vec<f64>> = (0..rows)
                .map(|r| {
                    let mut row: Vec<f64> = candidate_cost[r].iter().map(|&c| c as f64).collect();
                    row.extend((0..rows).map(|k| if k == r { diagonal[r] as f64 } else { invalid as f64 }));
                    row
                })
                .collect();
            let assignment = linear_sum_assignment(&cost);
            for (row, &inst) in selected.iter().enumerate() {
                let col = assignment[row];
                if col >= candidate_count {
                    scalar_choices.insert((inst, f_idx), None);
                    continue;
                }
                let probability = probs[row][col + 1];
                if probability < threshold && field.cardinality.allows_absent() {
                    scalar_choices.insert((inst, f_idx), None);
                    continue;
                }
                scalar_choices.insert((inst, f_idx), Some((col, probability)));
            }
        } else {
            for cand in 0..candidate_count {
                // torch.max returns the first maximal row.
                let mut best_row = 0;
                let mut best = sigmoid(logits[0][cand + 1]);
                for (row, l) in logits.iter().enumerate().skip(1) {
                    let p = sigmoid(l[cand + 1]);
                    if p > best {
                        best = p;
                        best_row = row;
                    }
                }
                if best >= threshold {
                    list_owners.insert((f_idx, cand), (selected[best_row], best));
                }
            }
        }
    }

    let mut records = Vec::new();
    for &inst in &selected {
        let mut rec = DecodedRecord { fields: HashMap::new(), anchor: inst, score: obj_prob[inst] };
        for (f_idx, field) in fields.iter().enumerate() {
            if Some(f_idx) == anchor_field {
                rec.fields.entry(f_idx).or_default().push((inst, rec.score));
                continue;
            }
            let row: Vec<f32> = assign[f_idx][inst].iter().map(|x| x / temperature).collect();
            if field.cardinality.is_scalar() {
                if field.exclusive {
                    if let Some(Some((cand, p))) = scalar_choices.get(&(inst, f_idx)) {
                        rec.fields.entry(f_idx).or_default().push((*cand, *p));
                    }
                    continue;
                }
                let probs = softmax(&row);
                let mut chosen = None;
                for col in argsort_desc(&probs) {
                    if col == 0 {
                        if field.cardinality.allows_absent() {
                            chosen = Some(0);
                            break;
                        }
                        continue;
                    }
                    chosen = Some(col);
                    break;
                }
                let Some(col) = chosen.filter(|&c| c != 0) else { continue };
                if probs[col] < threshold && field.cardinality.allows_absent() {
                    continue;
                }
                rec.fields.entry(f_idx).or_default().push((col - 1, probs[col]));
            } else {
                let mut picked = Vec::new();
                for cand in 0..row.len().saturating_sub(1) {
                    let probability = if field.exclusive {
                        match list_owners.get(&(f_idx, cand)) {
                            Some((owner, p)) if *owner == inst => *p,
                            _ => continue,
                        }
                    } else {
                        let p = sigmoid(row[cand + 1]);
                        if p < threshold {
                            continue;
                        }
                        p
                    };
                    picked.push((cand, probability));
                }
                if !picked.is_empty() {
                    rec.fields.entry(f_idx).or_default().extend(picked);
                }
            }
        }
        if !rec.fields.is_empty() {
            records.push(rec);
        }
    }
    records.sort_by_key(|r| spans[r.anchor]);
    records
}

/// Minimum-cost rectangular assignment (rows <= columns), returning the column
/// of every row. A sub-ULP lexicographic offset breaks exact ties the way the
/// reference solver does.
pub fn linear_sum_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    assert!(n <= m, "linear_sum_assignment expects rows <= columns");
    let scale = cost.iter().flatten().fold(0f64, |a, &b| a.max(b.abs())).max(1.0);
    let epsilon = f64::EPSILON * scale;
    let a = |i: usize, j: usize| cost[i][j] + epsilon * (i * m + j) as f64;

    // Shortest augmenting path (Jonker-Volgenant / e-maxx), 1-indexed.
    let mut u = vec![0f64; n + 1];
    let mut v = vec![0f64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let cur = a(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![0usize; n];
    for j in 1..=m {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    assignment
}

fn is_python_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Case-insensitive, word-bounded occurrences of `choice` in `text` as character
/// offsets (`re.finditer(rf"(?<!\w){re.escape(choice)}(?!\w)", text, re.I)`).
pub fn find_choice_mentions(text: &str, choice: &str) -> Vec<(usize, usize)> {
    let Ok(re) = regex::RegexBuilder::new(&regex::escape(choice)).case_insensitive(true).build() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut from = 0;
    while from <= text.len() {
        let Some(m) = re.find_at(text, from) else { break };
        let before_ok = text[..m.start()].chars().next_back().is_none_or(|c| !is_python_word_char(c));
        let after_ok = text[m.end()..].chars().next().is_none_or(|c| !is_python_word_char(c));
        if before_ok && after_ok && m.end() > m.start() {
            let start = text[..m.start()].chars().count();
            out.push((start, start + m.as_str().chars().count()));
            from = m.end();
        } else {
            // Retry one character later, like a failed lookaround in Python.
            let step = text[m.start()..].chars().next().map_or(1, char::len_utf8);
            from = m.start() + step;
        }
    }
    out
}

/// `_record_local_choice_mentions`: bind literal choice mentions to the record
/// whose anchor precedes them. Returns `(any literal mention, owner -> mentions)`.
pub fn record_local_choice_mentions(
    text: &str,
    choices: &[String],
    anchors: &[Option<(usize, usize)>],
) -> (bool, HashMap<usize, Vec<(String, usize, usize)>>) {
    let mut mentions: Vec<(String, usize, usize)> = Vec::new();
    for choice in choices {
        mentions.extend(find_choice_mentions(text, choice).into_iter().map(|(s, e)| (choice.clone(), s, e)));
    }
    if mentions.is_empty() {
        return (false, HashMap::new());
    }
    let mut valid: Vec<(usize, (usize, usize))> =
        anchors.iter().enumerate().filter_map(|(i, a)| a.map(|a| (i, a))).collect();
    if valid.is_empty() {
        return (true, HashMap::new());
    }
    valid.sort_by_key(|&(i, a)| (a.0, i));

    mentions.sort_by_key(|m| m.1);
    let mut assigned: HashMap<usize, Vec<(String, usize, usize)>> = HashMap::new();
    for mention in mentions {
        let owner = valid
            .iter()
            .filter(|(_, a)| a.0 <= mention.1)
            .max_by_key(|&&(i, a)| (a.0, i))
            .unwrap_or(&valid[0])
            .0;
        assigned.entry(owner).or_default().push(mention);
    }
    for owned in assigned.values_mut() {
        let mut unique: Vec<(String, usize, usize)> = Vec::new();
        for m in owned.drain(..) {
            if !unique.iter().any(|u| u.0 == m.0) {
                unique.push(m);
            }
        }
        unique.sort_by_key(|m| m.1);
        *owned = unique;
    }
    (true, assigned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_is_optimal() {
        let cost = vec![vec![4.0, 1.0, 3.0], vec![2.0, 0.0, 5.0]];
        assert_eq!(linear_sum_assignment(&cost), vec![1, 0]);
        let cost = vec![vec![1.0, 2.0], vec![1.0, 10.0]];
        assert_eq!(linear_sum_assignment(&cost), vec![1, 0]);
    }

    #[test]
    fn choice_mentions_respect_word_boundaries() {
        let text = "Books and e-books, BOOKS; notebooks";
        assert_eq!(find_choice_mentions(text, "books"), vec![(0, 5), (12, 17), (19, 24)]);
    }

    #[test]
    fn choice_mentions_bind_to_preceding_anchor() {
        let text = "Amazon sells books. Walmart sells groceries and books.";
        let anchors = [Some((0, 6)), Some((20, 27))];
        let choices = ["books".to_string(), "groceries".to_string()];
        let (found, owners) = record_local_choice_mentions(text, &choices, &anchors);
        assert!(found);
        assert_eq!(owners[&0].iter().map(|m| m.0.as_str()).collect::<Vec<_>>(), ["books"]);
        assert_eq!(owners[&1].iter().map(|m| m.0.as_str()).collect::<Vec<_>>(), ["groceries", "books"]);
    }
}
