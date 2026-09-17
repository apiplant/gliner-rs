//! Classification settings shared by the one-shot CLI and the interactive shell,
//! plus result rendering.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use gliner_rs::{ClassActivation, ClassificationSpec, GLiNER2};
use serde_json::{json, Map, Value};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Human-readable, one line per text.
    Text,
    /// One JSON object per text.
    Jsonl,
    /// A single JSON array.
    Json,
    /// `text<TAB>task<TAB>label<TAB>confidence`, one row per predicted label.
    Tsv,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Activation {
    Auto,
    Softmax,
    Sigmoid,
}

/// A task as the user wrote it: `name`, multi-label flag, `a,b:desc,c`.
#[derive(Clone, Debug)]
pub struct TaskDef {
    pub name: String,
    pub multi: bool,
    pub labels: String,
}

impl TaskDef {
    /// Parse `name=a,b,c` (prefix the name with `+` for multi-label).
    pub fn parse(spec: &str) -> Result<Self> {
        let Some((name, labels)) = spec.split_once('=') else {
            bail!("expected name=label1,label2, got {spec:?}");
        };
        let (name, multi) = match name.trim().strip_prefix('+') {
            Some(n) => (n.trim().to_string(), true),
            None => (name.trim().to_string(), false),
        };
        if name.is_empty() {
            bail!("task name is empty in {spec:?}");
        }
        Ok(Self { name, multi, labels: labels.to_string() })
    }
}

impl std::fmt::Display for TaskDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}={}", if self.multi { "+" } else { "" }, self.name, self.labels)
    }
}

#[derive(Clone, Debug)]
pub struct Session {
    pub tasks: Vec<TaskDef>,
    /// Few-shot `(input, label)` examples, routed to tasks having that label.
    pub examples: Vec<(String, String)>,
    pub prompt: Option<String>,
    pub threshold: f32,
    pub activation: Activation,
    pub all: bool,
    pub top_k: Option<usize>,
    pub format: Format,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            tasks: Vec::new(),
            examples: Vec::new(),
            prompt: None,
            threshold: 0.5,
            activation: Activation::Auto,
            all: false,
            top_k: None,
            format: Format::Text,
        }
    }
}

pub fn parse_example(spec: &str) -> Result<(String, String)> {
    spec.rsplit_once("=>")
        .map(|(input, label)| (input.trim().to_string(), label.trim().to_string()))
        .filter(|(i, l)| !i.is_empty() && !l.is_empty())
        .with_context(|| format!("expected input=>label, got {spec:?}"))
}

fn parse_labels(spec: &str) -> (Vec<String>, Vec<(String, String)>) {
    let mut labels = Vec::new();
    let mut descriptions = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match item.split_once(':') {
            Some((label, desc)) => {
                labels.push(label.trim().to_string());
                descriptions.push((label.trim().to_string(), desc.trim().to_string()));
            }
            None => labels.push(item.to_string()),
        }
    }
    (labels, descriptions)
}

impl Session {
    /// Replace or add a task by name.
    pub fn set_task(&mut self, task: TaskDef) {
        match self.tasks.iter_mut().find(|t| t.name == task.name) {
            Some(existing) => *existing = task,
            None => self.tasks.push(task),
        }
    }

    pub fn specs(&self) -> Result<Vec<ClassificationSpec>> {
        if self.tasks.is_empty() {
            bail!("no labels: set some with --labels a,b,c or --task name=a,b,c");
        }
        self.tasks
            .iter()
            .map(|task| {
                let (labels, descriptions) = parse_labels(&task.labels);
                if labels.is_empty() || (labels.len() < 2 && !task.multi) {
                    bail!("task {:?} needs at least two labels (or make it multi-label)", task.name);
                }
                let mut spec = ClassificationSpec::new(task.name.clone(), labels);
                spec.multi_label = task.multi;
                spec.cls_threshold = self.threshold;
                spec.label_descriptions = descriptions;
                spec.prompt = self.prompt.clone();
                spec.activation = match self.activation {
                    Activation::Auto => ClassActivation::Auto,
                    Activation::Softmax => ClassActivation::Softmax,
                    Activation::Sigmoid => ClassActivation::Sigmoid,
                };
                spec.examples = self.examples.iter().filter(|(_, l)| spec.labels.contains(l)).cloned().collect();
                Ok(spec)
            })
            .collect()
    }

    /// Classify `texts` and print them in the session format.
    pub fn run(&self, model: &GLiNER2, texts: &[String]) -> Result<()> {
        let specs = self.specs()?;
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let all_results = model.classification_probabilities_batch(&refs, &specs)?;
        let mut json_rows = Vec::new();
        for (text, results) in texts.iter().zip(&all_results) {
            match self.format {
                Format::Text => {
                    let line = self.render_text(&specs, &results);
                    if texts.len() == 1 {
                        println!("{line}");
                    } else {
                        println!("{}\t{line}", one_line(text));
                    }
                }
                Format::Tsv => {
                    for (spec, (task, probs)) in specs.iter().zip(results.iter()) {
                        for (label, p) in predict(spec, probs) {
                            println!("{}\t{task}\t{label}\t{p:.4}", one_line(text));
                        }
                    }
                }
                Format::Jsonl => println!("{}", self.render_json(text, &specs, &results)),
                Format::Json => json_rows.push(self.render_json(text, &specs, &results)),
            }
        }
        if self.format == Format::Json {
            println!("{}", serde_json::to_string_pretty(&json_rows)?);
        }
        Ok(())
    }

    fn show_all(&self) -> bool {
        self.all || self.top_k.is_some()
    }

    fn render_text(&self, specs: &[ClassificationSpec], results: &[(String, Vec<(String, f32)>)]) -> String {
        specs
            .iter()
            .zip(results)
            .map(|(spec, (task, probs))| {
                let predicted = predict(spec, probs);
                let shown = if self.show_all() { ranked(probs, self.top_k) } else { predicted.clone() };
                let labels: Vec<String> = shown
                    .iter()
                    .map(|(l, p)| {
                        let mark = if self.show_all() && predicted.iter().any(|x| &x.0 == l) { "*" } else { "" };
                        format!("{mark}{l} ({p:.3})")
                    })
                    .collect();
                format!("{task}: {}", labels.join(", "))
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn render_json(&self, text: &str, specs: &[ClassificationSpec], results: &[(String, Vec<(String, f32)>)]) -> Value {
        let mut row = Map::new();
        row.insert("text".into(), json!(text));
        for (spec, (task, probs)) in specs.iter().zip(results) {
            let predicted = predict(spec, probs);
            let mut value = Map::new();
            if spec.multi_label {
                value.insert("labels".into(), json!(predicted.iter().map(|p| &p.0).collect::<Vec<_>>()));
                value.insert("confidences".into(), json!(predicted.iter().map(|p| p.1).collect::<Vec<_>>()));
            } else {
                value.insert("label".into(), json!(predicted[0].0));
                value.insert("confidence".into(), json!(predicted[0].1));
            }
            if self.show_all() {
                let probabilities: Map<String, Value> =
                    ranked(probs, self.top_k).into_iter().map(|(l, p)| (l, json!(p))).collect();
                value.insert("probabilities".into(), Value::Object(probabilities));
            }
            row.insert(task.clone(), Value::Object(value));
        }
        Value::Object(row)
    }
}

fn one_line(text: &str) -> String {
    text.replace(['\t', '\n', '\r'], " ")
}

/// Predicted labels, mirroring gliner2's decision rule: argmax for single-label;
/// every label above the threshold (or the argmax) for multi-label.
pub fn predict(spec: &ClassificationSpec, probs: &[(String, f32)]) -> Vec<(String, f32)> {
    let best = probs.iter().enumerate().fold(0, |best, (i, p)| if p.1 > probs[best].1 { i } else { best });
    if spec.multi_label {
        let chosen: Vec<_> = probs.iter().filter(|p| p.1 >= spec.cls_threshold).cloned().collect();
        if chosen.is_empty() {
            vec![probs[best].clone()]
        } else {
            chosen
        }
    } else {
        vec![probs[best].clone()]
    }
}

fn ranked(probs: &[(String, f32)], top_k: Option<usize>) -> Vec<(String, f32)> {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted.truncate(top_k.unwrap_or(usize::MAX));
    sorted
}
