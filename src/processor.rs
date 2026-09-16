//! Text splitting and prompt construction (mirrors `gliner2.processor.SchemaTransformer`
//! in inference mode).

use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use regex::Regex;
use tokenizers::Tokenizer;

use crate::schema::Schema;

pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
pub const SEP_TEXT: &str = "[SEP_TEXT]";
pub const P_TOKEN: &str = "[P]";
pub const C_TOKEN: &str = "[C]";
pub const E_TOKEN: &str = "[E]";
pub const R_TOKEN: &str = "[R]";
pub const L_TOKEN: &str = "[L]";
pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
pub const OUTPUT_TOKEN: &str = "[OUTPUT]";
pub const DESC_TOKEN: &str = "[DESCRIPTION]";

/// A word of the input text with byte and character (code point) offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub char_start: usize,
    pub char_end: usize,
}

/// Built-in word splitters (`gliner2.processing.word_splitter`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordSplitter {
    /// URLs, emails, @handles, `\w+(?:[-_]\w+)*`, or any other non-space char.
    #[default]
    Whitespace,
    /// Latin runs stay together; every other non-space char is a token (CJK).
    Char,
}

fn whitespace_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Python's Unicode `\w` is `str.isalnum() or "_"`, i.e. letters and
        // numbers but *not* combining marks, unlike Rust's `\w`.
        Regex::new(
            r"(?i)(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S",
        )
        .expect("valid regex")
    })
}

fn char_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9@._\-+]+|\S").expect("valid regex"))
}

impl WordSplitter {
    pub fn split(&self, text: &str, lower: bool) -> Vec<Word> {
        let re = match self {
            WordSplitter::Whitespace => whitespace_regex(),
            WordSplitter::Char => char_regex(),
        };
        let mut words = Vec::new();
        let mut byte_cursor = 0;
        let mut char_cursor = 0;
        for m in re.find_iter(text) {
            char_cursor += text[byte_cursor..m.start()].chars().count();
            let char_start = char_cursor;
            char_cursor += m.as_str().chars().count();
            byte_cursor = m.end();
            let token = if lower { m.as_str().to_lowercase() } else { m.as_str().to_string() };
            words.push(Word {
                text: token,
                byte_start: m.start(),
                byte_end: m.end(),
                char_start,
                char_end: char_cursor,
            });
        }
        words
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Structures,
    Entities,
    Relations,
    Classifications,
}

/// One schema group of the prompt (`( [P] prompt ( [X] field ... ) )`).
#[derive(Debug, Clone)]
pub struct PromptGroup {
    pub kind: TaskKind,
    /// Index into the matching `Schema` vector (`structures`/`relations`/`classifications`);
    /// unused for entities.
    pub spec_index: usize,
    pub name: String,
    pub fields: Vec<String>,
    pub tokens: Vec<String>,
    /// Subword positions of the child markers (`[E]`/`[R]`/`[L]`).
    pub marker_positions: Vec<usize>,
    /// Subword position of this group's `[P]` token. Unused by the boundary
    /// architecture; the span architecture reads the `[P]` state as the
    /// count-prediction/count-embedding input (`schema_emb[0]` upstream).
    pub prompt_position: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedInput {
    /// The text the model saw: the input with a final `.` appended when it does
    /// not already end in `.`, `!` or `?` (as the reference collator does).
    /// Word offsets index into this string.
    pub text: String,
    pub input_ids: Vec<u32>,
    /// Choice-field prefix words placed before the text (`( parent: field ( a | b ) )`).
    pub prefix_tokens: Vec<String>,
    pub words: Vec<Word>,
    /// Subword position of the first piece of every prefix token, then every word.
    /// Model token index `i` is prefix token `i` when `i < prefix_tokens.len()`,
    /// otherwise text word `i - prefix_tokens.len()`.
    pub word_positions: Vec<usize>,
    pub groups: Vec<PromptGroup>,
}

pub struct Processor {
    tokenizer: Tokenizer,
    pub word_splitter: WordSplitter,
}

fn transform_schema(
    parent: &str,
    fields: &[String],
    child_prefix: &str,
    prompt: Option<&str>,
    examples: &[(String, String)],
    label_descriptions: &[(String, String)],
    with_descriptions: bool,
    with_examples: bool,
) -> Vec<String> {
    let mut prompt_str = match prompt {
        Some(p) => format!("{parent}: {p}"),
        None => parent.to_string(),
    };
    if with_descriptions {
        for (label, desc) in label_descriptions {
            if fields.contains(label) {
                prompt_str.push_str(&format!(" {DESC_TOKEN} {label}: {desc}"));
            }
        }
    }
    if with_examples {
        for (input, output) in examples {
            if fields.contains(output) {
                prompt_str.push_str(&format!(" {EXAMPLE_TOKEN} {input} {OUTPUT_TOKEN} {output}"));
            }
        }
    }
    let mut tokens = vec!["(".to_string(), P_TOKEN.to_string(), prompt_str, "(".to_string()];
    for field in fields {
        tokens.push(child_prefix.to_string());
        tokens.push(field.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

impl Processor {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, word_splitter: WordSplitter::default() }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn build_groups(schema: &Schema) -> Vec<PromptGroup> {
        let mut groups = Vec::new();

        for (index, structure) in schema.structures.iter().enumerate() {
            if schema.structures[..index].iter().any(|s| s.name == structure.name) {
                continue;
            }
            let fields: Vec<String> = structure.fields.iter().map(|f| f.name.clone()).collect();
            let descs: Vec<(String, String)> = structure
                .fields
                .iter()
                .filter_map(|f| f.description.clone().map(|d| (f.name.clone(), d)))
                .collect();
            let tokens =
                transform_schema(&structure.name, &fields, C_TOKEN, None, &[], &descs, !descs.is_empty(), false);
            groups.push(PromptGroup {
                kind: TaskKind::Structures,
                spec_index: index,
                name: structure.name.clone(),
                fields,
                tokens,
                marker_positions: Vec::new(),
                prompt_position: 0,
            });
        }

        if !schema.entities.is_empty() {
            let fields: Vec<String> = schema.entities.iter().map(|e| e.name.clone()).collect();
            let descs: Vec<(String, String)> = schema
                .entities
                .iter()
                .filter_map(|e| e.description.clone().map(|d| (e.name.clone(), d)))
                .collect();
            let tokens =
                transform_schema("entities", &fields, E_TOKEN, None, &[], &descs, !descs.is_empty(), false);
            groups.push(PromptGroup {
                kind: TaskKind::Entities,
                spec_index: 0,
                name: "entities".to_string(),
                fields,
                tokens,
                marker_positions: Vec::new(),
                prompt_position: 0,
            });
        }

        let mut seen_relations: Vec<&str> = Vec::new();
        for (index, rel) in schema.relations.iter().enumerate() {
            if seen_relations.contains(&rel.name.as_str()) {
                continue;
            }
            seen_relations.push(&rel.name);
            let fields = vec!["head".to_string(), "tail".to_string()];
            let tokens = transform_schema(
                &rel.name,
                &fields,
                R_TOKEN,
                rel.description.as_deref(),
                &[],
                &[],
                true,
                true,
            );
            groups.push(PromptGroup {
                kind: TaskKind::Relations,
                spec_index: index,
                name: rel.name.clone(),
                fields,
                tokens,
                marker_positions: Vec::new(),
                prompt_position: 0,
            });
        }

        for (index, cls) in schema.classifications.iter().enumerate() {
            let tokens = transform_schema(
                &cls.task,
                &cls.labels,
                L_TOKEN,
                cls.prompt.as_deref(),
                &cls.examples,
                &cls.label_descriptions,
                true,
                true,
            );
            groups.push(PromptGroup {
                kind: TaskKind::Classifications,
                spec_index: index,
                name: cls.task.clone(),
                fields: cls.labels.clone(),
                tokens,
                marker_positions: Vec::new(),
                prompt_position: 0,
            });
        }
        groups
    }

    /// `_build_classification_prefix`: enumerate choice fields ahead of the text.
    fn choice_prefix(schema: &Schema) -> Vec<String> {
        let mut prefix = Vec::new();
        for structure in &schema.structures {
            let mut inner: Vec<String> = Vec::new();
            for field in &structure.fields {
                let Some(choices) = &field.choices else { continue };
                inner.push(field.name.clone());
                inner.push("(".into());
                for (i, choice) in choices.iter().enumerate() {
                    if i > 0 {
                        inner.push("|".into());
                    }
                    inner.push(choice.clone());
                }
                inner.push(")".into());
                inner.push(",".into());
            }
            if !inner.is_empty() {
                inner.pop();
                prefix.push("(".into());
                prefix.push(format!("{}:", structure.name));
                prefix.extend(inner);
                prefix.push(")".into());
            }
        }
        prefix
    }

    /// Build model inputs for one text (`_transform_record` + `_format_input_with_mapping`).
    pub fn prepare(&self, text: &str, schema: &Schema, max_words: Option<usize>) -> Result<PreparedInput> {
        let text = if text.is_empty() {
            ".".to_string()
        } else if text.ends_with(['.', '!', '?']) {
            text.to_string()
        } else {
            format!("{text}.")
        };
        let mut words = self.word_splitter.split(&text, true);
        if let Some(max) = max_words {
            words.truncate(max);
        }
        let mut groups = Self::build_groups(schema);
        let prefix_tokens = Self::choice_prefix(schema);

        // Combined word-level sequence.
        let mut combined: Vec<&str> = Vec::new();
        // `Some((group, is_prompt))`: `is_prompt` marks the `[P]` token (ti == 1),
        // otherwise it's a child marker (`[E]`/`[R]`/`[L]`/`[C]`).
        let mut marker_owner: Vec<Option<(usize, bool)>> = Vec::new();
        for (gi, group) in groups.iter().enumerate() {
            for (ti, token) in group.tokens.iter().enumerate() {
                combined.push(token);
                if ti == 1 {
                    marker_owner.push(Some((gi, true)));
                    continue;
                }
                // Child markers sit at 4, 6, ... < len - 2.
                let is_marker = ti >= 4 && ti < group.tokens.len() - 2 && ti % 2 == 0;
                marker_owner.push(is_marker.then_some((gi, false)));
            }
            if gi + 1 < groups.len() {
                combined.push(SEP_STRUCT);
                marker_owner.push(None);
            }
        }
        combined.push(SEP_TEXT);
        marker_owner.push(None);
        let text_start = combined.len();
        for token in &prefix_tokens {
            combined.push(token);
            marker_owner.push(None);
        }
        for w in &words {
            combined.push(&w.text);
            marker_owner.push(None);
        }

        let encodings = self
            .tokenizer
            .encode_batch(combined.clone(), false)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;

        let mut input_ids = Vec::new();
        let mut word_positions = Vec::with_capacity(words.len());
        for (index, encoding) in encodings.iter().enumerate() {
            let position = input_ids.len();
            if index >= text_start {
                word_positions.push(position);
            } else if let Some((gi, is_prompt)) = marker_owner[index] {
                if is_prompt {
                    groups[gi].prompt_position = position;
                } else {
                    groups[gi].marker_positions.push(position);
                }
            }
            input_ids.extend_from_slice(encoding.get_ids());
        }
        Ok(PreparedInput { text, input_ids, prefix_tokens, words, word_positions, groups })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_splitter_matches_reference_patterns() {
        let words = WordSplitter::Whitespace.split("Email me at Bob.Smith@example.com, see https://x.io/a?b ok-ish", true);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            texts,
            ["email", "me", "at", "bob.smith@example.com", ",", "see", "https://x.io/a?b", "ok-ish"]
        );
    }

    #[test]
    fn offsets_are_code_points() {
        let words = WordSplitter::Whitespace.split("héllo wörld", true);
        assert_eq!((words[1].char_start, words[1].char_end), (6, 11));
        assert_eq!((words[1].byte_start, words[1].byte_end), (7, 13));
    }

    #[test]
    fn char_splitter_splits_cjk() {
        let words = WordSplitter::Char.split("北京 GLiNER2", false);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["北", "京", "GLiNER2"]);
    }

    #[test]
    fn schema_tokens_layout() {
        let tokens = transform_schema(
            "entities",
            &["person".into(), "company".into()],
            E_TOKEN,
            None,
            &[],
            &[("person".into(), "a human".into())],
            true,
            false,
        );
        assert_eq!(
            tokens,
            ["(", "[P]", "entities [DESCRIPTION] person: a human", "(", "[E]", "person", "[E]", "company", ")", ")"]
        );
    }
}
