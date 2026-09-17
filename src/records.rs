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

/// How a structure's instances are formed (`gliner2.processing.records`
/// modes: `natural` / `latent` / `anchorless`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordMode {
    Natural,
    Latent,
    Anchorless,
}

/// Unified natural/latent/anchorless record head weights
/// (`gliner2.models.boundary.records.RecordHead`).
pub struct RecordHead {
    inst_proj: Linear,
    field_proj: Linear,
    cand_proj: Linear,
    null_embed: Tensor,
    object_head: Linear,
    latent_seed_head: Linear,
    instance_embed: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    record_dim: usize,
}

/// Per-field assignment logits `[F][I][1 + C]` (column 0 = ABSENT), where `I`
/// is the instance count (anchor/latent-seed candidates, or learned
/// anchorless queries) and `C` the field-candidate count.
pub type AssignLogits = Vec<Vec<Vec<f32>>>;

impl RecordHead {
    pub fn load(vb: VarBuilder, hidden: usize, record_dim: usize, instance_queries: usize) -> Result<Self> {
        Ok(Self {
            inst_proj: linear(hidden, record_dim, vb.pp("inst_proj"))?,
            field_proj: linear(hidden, record_dim, vb.pp("field_proj"))?,
            cand_proj: linear(hidden, record_dim, vb.pp("cand_proj"))?,
            null_embed: vb.get(record_dim, "null_embed")?,
            object_head: linear(hidden, 1, vb.pp("object_head"))?,
            latent_seed_head: linear(hidden, 1, vb.pp("latent_seed_head"))?,
            instance_embed: vb.get((instance_queries, hidden), "instance_embed")?,
            q_proj: linear(hidden, record_dim, vb.pp("q_proj"))?,
            k_proj: linear(hidden, record_dim, vb.pp("k_proj"))?,
            v_proj: linear(hidden, hidden, vb.pp("v_proj"))?,
            record_dim,
        })
    }

    /// `latent_seed_head` score per shared-pool candidate (`[C, H]` -> `[C]`):
    /// how likely each candidate is to seed a `latent`-mode instance.
    pub fn latent_seed_scores(&self, candidate_states: &Tensor) -> Result<Vec<f32>> {
        self.latent_seed_head.forward(candidate_states)?.squeeze(1)?.to_vec1::<f32>()
    }

    /// `object_head` score per anchorless instance (`[I, H]` -> `[I]`).
    pub fn object_scores(&self, instance_states: &Tensor) -> Result<Vec<f32>> {
        self.object_head.forward(instance_states)?.squeeze(1)?.to_vec1::<f32>()
    }

    /// `_anchorless_states`: the learned instance queries, cross-attended
    /// over the shared candidate pool (`[C, H]` -> `[I, H]`).
    pub fn anchorless_instances(&self, candidate_states: &Tensor) -> Result<Tensor> {
        let inst = self.instance_embed.clone();
        if candidate_states.dim(0)? == 0 {
            return Ok(inst);
        }
        let q = self.q_proj.forward(&inst)?; // [I, D]
        let k = self.k_proj.forward(candidate_states)?; // [C, D]
        let v = self.v_proj.forward(candidate_states)?; // [C, H]
        let attn = (q.matmul(&k.t()?)? / (self.record_dim as f64).sqrt())?; // [I, C]
        let weights = candle_nn::ops::softmax(&attn, 1)?;
        inst + weights.matmul(&v)?
    }

    /// `_assign_logits`: `instance_states` (`[I, H]`) may be a different
    /// tensor than `candidate_states` (`[C, H]`) in `anchorless` mode; they
    /// are the same shared pool in `natural`/`latent` mode. `field_queries`:
    /// `[F, H]`.
    pub fn assign_logits(&self, instance_states: &Tensor, candidate_states: &Tensor, field_queries: &Tensor) -> Result<AssignLogits> {
        let inst = self.inst_proj.forward(instance_states)?; // [I, D]
        let field = self.field_proj.forward(field_queries)?; // [F, D]
        let query = inst.unsqueeze(0)?.broadcast_add(&field.unsqueeze(1)?)?; // [F, I, D]
        let null = query.broadcast_matmul(&self.null_embed.unsqueeze(1)?)?; // [F, I, 1]
        let cand = self.cand_proj.forward(candidate_states)?; // [C, D]
        let scores = query.broadcast_matmul(&cand.t()?)?; // [F, I, C]
        Tensor::cat(&[&null, &scores], 2)?.to_vec3::<f32>()
    }
}

#[derive(Debug, Clone)]
pub struct DecodedRecord {
    /// Field index -> selected `(candidate index, score)`.
    pub fields: HashMap<usize, Vec<(usize, f32)>>,
    /// Seed candidate index (`natural`/`latent`); `None` for `anchorless`
    /// instances, which aren't backed by a shared-pool span.
    pub anchor: Option<usize>,
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

/// `decode_group`. `object_logits` are, per mode: the anchor field's pair
/// logits over the shared pool (`Natural`), the head's `latent_seed_head`
/// score per shared-pool candidate (`Latent`), or the `object_head` score per
/// learned instance (`Anchorless`). `spans` are the shared-pool token spans
/// (`Natural`/`Latent`, same indexing as `object_logits`); `Anchorless`
/// instances aren't backed by a span, so pass `None`.
pub fn decode_group(
    mode: RecordMode,
    fields: &[RecordField],
    object_logits: &[f32],
    spans: Option<&[(usize, usize)]>,
    assign: &AssignLogits,
    threshold: f32,
    temperature: f32,
) -> Vec<DecodedRecord> {
    let ni = object_logits.len();
    if ni == 0 {
        return Vec::new();
    }
    let obj_prob: Vec<f32> = object_logits.iter().map(|&x| sigmoid(x / temperature)).collect();
    let mut order: Vec<usize> = (0..ni).collect();
    order.sort_by(|&a, &b| obj_prob[b].total_cmp(&obj_prob[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order.into_iter().filter(|&i| obj_prob[i] >= threshold).collect();
    let anchor_field = (mode == RecordMode::Natural).then(|| fields.iter().position(|f| f.is_anchor)).flatten();

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
        let anchor = (mode != RecordMode::Anchorless).then_some(inst);
        let mut rec = DecodedRecord { fields: HashMap::new(), anchor, score: obj_prob[inst] };
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
    match mode {
        RecordMode::Natural => records.sort_by_key(|r| spans.expect("natural mode has spans")[r.anchor.unwrap()]),
        RecordMode::Latent | RecordMode::Anchorless => records = dedup_by_fields(records),
    }
    records
}

/// `_dedup_key` + the `dict`-preserving-first-position update used for
/// `latent`/`anchorless`: identical field assignments collapse into the
/// highest-scoring occurrence, kept at its first position.
fn dedup_by_fields(records: Vec<DecodedRecord>) -> Vec<DecodedRecord> {
    fn key(rec: &DecodedRecord) -> Vec<(usize, Vec<usize>)> {
        let mut fields: Vec<(usize, Vec<usize>)> = rec
            .fields
            .iter()
            .map(|(&f, v)| {
                let mut idxs: Vec<usize> = v.iter().map(|&(c, _)| c).collect();
                idxs.sort_unstable();
                (f, idxs)
            })
            .collect();
        fields.sort_by_key(|(f, _)| *f);
        fields
    }
    let mut out: Vec<(Vec<(usize, Vec<usize>)>, DecodedRecord)> = Vec::new();
    for rec in records {
        let k = key(&rec);
        match out.iter_mut().find(|(existing, _)| *existing == k) {
            Some((_, existing)) if existing.score >= rec.score => {}
            Some((_, existing)) => *existing = rec,
            None => out.push((k, rec)),
        }
    }
    out.into_iter().map(|(_, rec)| rec).collect()
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
