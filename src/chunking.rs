//! Long-document extraction: split text into overlapping word chunks, extract
//! each chunk independently, remap spans back to document offsets, and merge
//! duplicate predictions across overlaps (mirrors `gliner2.inference.chunking`).

use std::collections::HashMap;

use anyhow::{ensure, Result};
use serde_json::{Map, Value};

use crate::decode::{resolve_overlaps, OverlapPolicy, ScoredSpan};
use crate::processor::WordSplitter;

/// Chunk sizing for long-document extraction, in word tokens (as counted by
/// the active [`WordSplitter`]).
#[derive(Debug, Clone, Copy)]
pub struct ChunkOptions {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self { chunk_size: 384, chunk_overlap: 64 }
    }
}

/// One overlapping window of a document: `text` is the chunk's own slice,
/// `start_char` is where it begins in the original document (in `char`s, the
/// same unit `extract`'s span offsets use).
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub start_char: usize,
}

/// Split `text` into overlapping word windows.
pub fn split_text_into_chunks(text: &str, opts: ChunkOptions, splitter: WordSplitter) -> Result<Vec<TextChunk>> {
    ensure!(opts.chunk_size > 0, "chunk_size must be greater than 0");
    ensure!(opts.chunk_overlap < opts.chunk_size, "chunk_overlap must be smaller than chunk_size");

    let words = splitter.split(text, false);
    if words.is_empty() {
        return Ok(vec![TextChunk { text: text.to_string(), start_char: 0 }]);
    }

    let mut chunks = Vec::new();
    let step = opts.chunk_size - opts.chunk_overlap;
    let mut start_word = 0;
    while start_word < words.len() {
        let end_word = (start_word + opts.chunk_size).min(words.len());
        let start_byte = words[start_word].byte_start;
        let end_byte = words[end_word - 1].byte_end;
        let start_char = words[start_word].char_start;
        chunks.push(TextChunk { text: text[start_byte..end_byte].to_string(), start_char });
        if end_word == words.len() {
            break;
        }
        start_word += step;
    }
    Ok(chunks)
}

/// Byte offset of each character index in `text` (length `char_count + 1`,
/// the last entry being `text.len()`), so a char-offset span can be sliced
/// back out of the original document.
fn char_byte_offsets(text: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    offsets.push(text.len());
    offsets
}

fn is_span_value(v: &Value) -> bool {
    v.as_object().is_some_and(|m| m.contains_key("text") && m.get("start").is_some_and(Value::is_u64) && m.get("end").is_some_and(Value::is_u64))
}

/// Recursively shift every span dict's `start`/`end` by `chunk.start_char`
/// and recompute `text` against the original document.
fn remap_spans(value: Value, chunk_start_char: usize, char_offsets: &[usize], original_text: &str) -> Value {
    match value {
        Value::Array(items) => {
            Value::Array(items.into_iter().map(|v| remap_spans(v, chunk_start_char, char_offsets, original_text)).collect())
        }
        Value::Object(map) => {
            let mut remapped: Map<String, Value> =
                map.into_iter().map(|(k, v)| (k, remap_spans(v, chunk_start_char, char_offsets, original_text))).collect();
            if is_span_value(&Value::Object(remapped.clone())) {
                let start = remapped["start"].as_u64().unwrap() as usize + chunk_start_char;
                let end = remapped["end"].as_u64().unwrap() as usize + chunk_start_char;
                remapped.insert("start".into(), Value::from(start));
                remapped.insert("end".into(), Value::from(end));
                if start <= end && end < char_offsets.len() {
                    remapped.insert("text".into(), Value::from(&original_text[char_offsets[start]..char_offsets[end]]));
                }
            }
            Value::Object(remapped)
        }
        other => other,
    }
}

fn as_list(v: &Value) -> Vec<Value> {
    match v {
        Value::Null => Vec::new(),
        Value::Array(a) => a.clone(),
        other => vec![other.clone()],
    }
}

fn is_classification_dict(v: &Value) -> bool {
    v.as_object().is_some_and(|m| m.contains_key("label") && m.contains_key("confidence"))
}

fn confidence_of(v: &Value) -> f64 {
    v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0)
}

/// Dedupe span dicts across chunk overlaps: exact-boundary duplicates keep
/// their highest-confidence copy, then `policy` resolves genuine overlaps.
fn dedupe_span_items(items: Vec<Value>, policy: OverlapPolicy) -> Vec<Value> {
    if items.is_empty() {
        return items;
    }
    let scored: Vec<ScoredSpan> = items
        .iter()
        .map(|v| ScoredSpan {
            score: confidence_of(v) as f32,
            start: v["start"].as_u64().unwrap() as usize,
            end: v["end"].as_u64().unwrap() as usize,
        })
        .collect();
    let selected = resolve_overlaps(&scored, policy);

    let mut used = vec![false; items.len()];
    let mut out = Vec::with_capacity(selected.len());
    for sel in selected {
        if let Some(idx) = (0..items.len()).find(|&i| {
            !used[i] && scored[i].start == sel.start && scored[i].end == sel.end && scored[i].score == sel.score
        }) {
            used[idx] = true;
            out.push(items[idx].clone());
        }
    }
    out.sort_by(|a, b| {
        (a["start"].as_u64(), a["end"].as_u64(), a.get("text").and_then(Value::as_str).unwrap_or(""))
            .cmp(&(b["start"].as_u64(), b["end"].as_u64(), b.get("text").and_then(Value::as_str).unwrap_or("")))
    });
    out
}

/// Canonical key for a non-span prediction (relation, structure instance,
/// classification dict), ignoring `confidence`, used to collapse the same
/// prediction seen across overlapping chunks.
fn canonical_key(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut pairs: Vec<(String, String)> =
                m.iter().filter(|(k, _)| k.as_str() != "confidence").map(|(k, v)| (k.clone(), canonical_key(v))).collect();
            pairs.sort();
            format!("{pairs:?}")
        }
        Value::Array(a) => format!("{:?}", a.iter().map(canonical_key).collect::<Vec<_>>()),
        other => format!("{other:?}"),
    }
}

fn representative_confidence(v: &Value) -> f64 {
    match v {
        Value::Object(m) => match m.get("confidence").and_then(Value::as_f64) {
            Some(c) => c,
            None => m.values().map(representative_confidence).fold(0.0, f64::max),
        },
        Value::Array(a) => a.iter().map(representative_confidence).fold(0.0, f64::max),
        _ => 0.0,
    }
}

fn dedupe_other_items(items: Vec<Value>) -> Vec<Value> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out: Vec<Value> = Vec::new();
    for item in items {
        let key = canonical_key(&item);
        match seen.get(&key) {
            None => {
                seen.insert(key, out.len());
                out.push(item);
            }
            Some(&idx) => {
                if representative_confidence(&item) > representative_confidence(&out[idx]) {
                    out[idx] = item;
                }
            }
        }
    }
    out
}

fn dedupe_items(items: Vec<Value>, policy: OverlapPolicy) -> Vec<Value> {
    let (span_items, other_items): (Vec<Value>, Vec<Value>) = items.into_iter().partition(is_span_value);
    let mut out = dedupe_span_items(span_items, policy);
    out.extend(dedupe_other_items(other_items));
    out
}

fn is_empty_value(v: &Value) -> bool {
    matches!(v, Value::Null) || matches!(v, Value::Object(m) if m.is_empty()) || matches!(v, Value::Array(a) if a.is_empty())
}

/// Generic merge for one non-`entities`/`relation_extraction` field, shared
/// by classification results, structure instance lists, and nested dicts.
fn merge_values(values: &[&Value], policy: OverlapPolicy) -> Value {
    let non_empty: Vec<&Value> = values.iter().copied().filter(|v| !is_empty_value(v)).collect();
    if non_empty.is_empty() {
        return values.first().map(|v| (*v).clone()).unwrap_or(Value::Null);
    }

    if non_empty.iter().all(|v| is_classification_dict(v)) {
        return (*non_empty.iter().max_by(|a, b| confidence_of(a).total_cmp(&confidence_of(b))).unwrap()).clone();
    }

    if non_empty.iter().all(|v| v.is_string()) {
        let strings: Vec<&str> = non_empty.iter().map(|v| v.as_str().unwrap()).collect();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for s in &strings {
            *counts.entry(s).or_insert(0) += 1;
        }
        let (best_idx, _) = strings
            .iter()
            .enumerate()
            .max_by_key(|&(i, s)| (counts[s], std::cmp::Reverse(i)))
            .unwrap();
        return Value::from(strings[best_idx]);
    }

    if non_empty.iter().all(|v| v.is_array()) {
        let mut items = Vec::new();
        for v in &non_empty {
            if let Value::Array(a) = v {
                items.extend(a.clone());
            }
        }
        return Value::Array(dedupe_items(items, policy));
    }

    if non_empty.iter().all(|v| v.is_object()) {
        return merge_nested_dicts(&non_empty, policy);
    }

    (*non_empty[0]).clone()
}

fn merge_nested_dicts(values: &[&Value], policy: OverlapPolicy) -> Value {
    let mut keys: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for v in values {
        if let Value::Object(m) = v {
            for k in m.keys() {
                if seen.insert(k.clone()) {
                    keys.push(k.clone());
                }
            }
        }
    }
    let mut merged = Map::new();
    for key in keys {
        let per_key: Vec<&Value> = values.iter().filter_map(|v| v.as_object().and_then(|m| m.get(&key))).collect();
        merged.insert(key, merge_values(&per_key, policy));
    }
    Value::Object(merged)
}

fn merge_entity_maps(values: &[&Value], scalar_labels: &std::collections::HashSet<String>, policy: OverlapPolicy) -> Value {
    let mut labels: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for v in values {
        if let Value::Object(m) = v {
            for k in m.keys() {
                if seen.insert(k.clone()) {
                    labels.push(k.clone());
                }
            }
        }
    }
    let mut out = Map::new();
    for label in labels {
        let mut items = Vec::new();
        for v in values {
            if let Some(entry) = v.as_object().and_then(|m| m.get(&label)) {
                items.extend(as_list(entry));
            }
        }
        let deduped = dedupe_span_items(items, policy);
        if scalar_labels.contains(&label) {
            out.insert(label, deduped.into_iter().next().unwrap_or(Value::Null));
        } else {
            out.insert(label, Value::Array(deduped));
        }
    }
    Value::Object(out)
}

fn merge_relation_maps(values: &[&Value]) -> Value {
    let mut labels: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for v in values {
        if let Value::Object(m) = v {
            for k in m.keys() {
                if seen.insert(k.clone()) {
                    labels.push(k.clone());
                }
            }
        }
    }
    let mut out = Map::new();
    for label in labels {
        let mut items = Vec::new();
        for v in values {
            if let Some(entry) = v.as_object().and_then(|m| m.get(&label)) {
                items.extend(as_list(entry));
            }
        }
        // Relations are always merged with "allow": chunk overlap only ever
        // produces exact re-detections, never genuine crossings to resolve.
        out.insert(label, Value::Array(dedupe_span_items(items, OverlapPolicy::Allow)));
    }
    Value::Object(out)
}

/// Merge one document's already span-remapped chunk results into a single
/// result, mirroring `gliner2.inference.chunking.merge_chunk_results`.
pub fn merge_chunk_results(results: Vec<Value>, scalar_entity_labels: &std::collections::HashSet<String>, policy: OverlapPolicy) -> Value {
    let mut keys: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for r in &results {
        if let Value::Object(m) = r {
            for k in m.keys() {
                if seen.insert(k.clone()) {
                    keys.push(k.clone());
                }
            }
        }
    }
    let mut merged = Map::new();
    for key in keys {
        let values: Vec<&Value> = results.iter().filter_map(|r| r.as_object().and_then(|m| m.get(&key))).collect();
        let merged_value = if key == "entities" {
            merge_entity_maps(&values, scalar_entity_labels, policy)
        } else if key == "relation_extraction" {
            merge_relation_maps(&values)
        } else {
            merge_values(&values, policy)
        };
        merged.insert(key, merged_value);
    }
    Value::Object(merged)
}

const SPAN_RESERVED: [&str; 4] = ["text", "confidence", "start", "end"];

/// Drop the confidence/span metadata this module needs internally but the
/// caller didn't ask for, mirroring `_strip_span_metadata`.
pub fn strip_span_metadata(value: Value, include_confidence: bool, include_spans: bool) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(|v| strip_span_metadata(v, include_confidence, include_spans)).collect()),
        Value::Object(map) => {
            if is_span_value(&Value::Object(map.clone())) {
                let extras: Map<String, Value> = map.iter().filter(|(k, _)| !SPAN_RESERVED.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
                if !include_confidence && !include_spans && extras.is_empty() {
                    return map.get("text").cloned().unwrap_or(Value::from(""));
                }
                let mut stripped = Map::new();
                stripped.insert("text".to_string(), map.get("text").cloned().unwrap_or(Value::from("")));
                if include_confidence {
                    if let Some(c) = map.get("confidence") {
                        stripped.insert("confidence".to_string(), c.clone());
                    }
                }
                if include_spans {
                    stripped.insert("start".to_string(), map["start"].clone());
                    stripped.insert("end".to_string(), map["end"].clone());
                }
                for (k, v) in extras {
                    stripped.insert(k, v);
                }
                return Value::Object(stripped);
            }

            if is_classification_dict(&Value::Object(map.clone())) {
                return if include_confidence {
                    Value::Object(map)
                } else {
                    map.get("label").cloned().unwrap_or(Value::Null)
                };
            }

            if map.contains_key("text") && map.contains_key("confidence") && !map.contains_key("start") && !map.contains_key("end") {
                return if include_confidence {
                    Value::Object(map)
                } else {
                    map.get("text").cloned().unwrap_or(Value::Null)
                };
            }

            Value::Object(map.into_iter().map(|(k, v)| (k, strip_span_metadata(v, include_confidence, include_spans))).collect())
        }
        other => other,
    }
}

/// Run `extract` per chunk, remap spans to document offsets, and merge
/// results across overlaps.
pub fn extract_chunked(
    text: &str,
    chunks: &[TextChunk],
    scalar_entity_labels: &std::collections::HashSet<String>,
    overlap_policy: OverlapPolicy,
    include_confidence: bool,
    include_spans: bool,
    per_chunk_extract: impl Fn(&str) -> Result<Value>,
) -> Result<Value> {
    let char_offsets = char_byte_offsets(text);
    let mut remapped = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let result = per_chunk_extract(&chunk.text)?;
        remapped.push(remap_spans(result, chunk.start_char, &char_offsets, text));
    }
    let merged = merge_chunk_results(remapped, scalar_entity_labels, overlap_policy);
    Ok(strip_span_metadata(merged, include_confidence, include_spans))
}
