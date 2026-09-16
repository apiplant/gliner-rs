//! Extraction schema builder (mirrors `gliner2.inference.schema.Schema`).

/// Output shape of an entity label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntityDtype {
    /// All non-overlapping mentions (default).
    #[default]
    List,
    /// Only the best mention (or `null`).
    Str,
}

#[derive(Debug, Clone)]
pub struct EntitySpec {
    pub name: String,
    pub description: Option<String>,
    pub threshold: Option<f32>,
    pub dtype: EntityDtype,
}

#[derive(Debug, Clone)]
pub struct RelationSpec {
    pub name: String,
    pub description: Option<String>,
    pub threshold: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassActivation {
    /// Sigmoid for multi-label tasks, softmax otherwise.
    #[default]
    Auto,
    Sigmoid,
    Softmax,
}

#[derive(Debug, Clone)]
pub struct ClassificationSpec {
    pub task: String,
    pub labels: Vec<String>,
    pub label_descriptions: Vec<(String, String)>,
    pub multi_label: bool,
    pub cls_threshold: f32,
    pub activation: ClassActivation,
    pub prompt: Option<String>,
    /// Few-shot `(input, output_label)` examples.
    pub examples: Vec<(String, String)>,
}

impl ClassificationSpec {
    pub fn new(task: impl Into<String>, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            task: task.into(),
            labels: labels.into_iter().map(Into::into).collect(),
            label_descriptions: Vec::new(),
            multi_label: false,
            cls_threshold: 0.5,
            activation: ClassActivation::Auto,
            prompt: None,
            examples: Vec::new(),
        }
    }

    pub fn multi_label(mut self, threshold: f32) -> Self {
        self.multi_label = true;
        self.cls_threshold = threshold;
        self
    }

    pub fn with_label_description(mut self, label: impl Into<String>, desc: impl Into<String>) -> Self {
        self.label_descriptions.push((label.into(), desc.into()));
        self
    }
}

/// Output shape of a structure field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldDtype {
    /// All values (default).
    #[default]
    List,
    /// A single value (or `null`).
    Str,
}

#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub dtype: FieldDtype,
    /// Closed set of allowed values.
    pub choices: Option<Vec<String>>,
    pub description: Option<String>,
    pub threshold: Option<f32>,
}

impl FieldSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), dtype: FieldDtype::List, choices: None, description: None, threshold: None }
    }

    /// Parse gliner2's compact field syntax: `name[::str|list][::[a|b|c]][::description]`.
    /// Choices without an explicit dtype default to `str`.
    pub fn parse(spec: &str) -> Self {
        let mut parts = spec.split("::");
        let mut field = FieldSpec::new(parts.next().unwrap_or_default());
        let mut dtype_explicit = false;
        for part in parts {
            match part {
                "str" => {
                    field.dtype = FieldDtype::Str;
                    dtype_explicit = true;
                }
                "list" => {
                    field.dtype = FieldDtype::List;
                    dtype_explicit = true;
                }
                p if p.starts_with('[') && p.ends_with(']') && p.len() >= 2 => {
                    field.choices = Some(p[1..p.len() - 1].split('|').map(|c| c.trim().to_string()).collect());
                    if !dtype_explicit {
                        field.dtype = FieldDtype::Str;
                    }
                }
                p => field.description = Some(p.to_string()),
            }
        }
        field
    }
}

/// How structure instances are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StructureMode {
    /// Record head in `natural` mode (first field is the anchor) when the
    /// checkpoint has one, otherwise the legacy decoder. Matches `extract_json`.
    #[default]
    Auto,
    /// One aggregate instance with every field (`Schema.structure(mode=None)`).
    Legacy,
}

#[derive(Debug, Clone)]
pub struct StructureSpec {
    pub name: String,
    pub fields: Vec<FieldSpec>,
    pub mode: StructureMode,
}

impl StructureSpec {
    pub fn new(name: impl Into<String>, fields: impl IntoIterator<Item = FieldSpec>) -> Self {
        Self { name: name.into(), fields: fields.into_iter().collect(), mode: StructureMode::Auto }
    }

    /// Build from compact field strings, e.g. `["name::str", "price::str::Product price"]`.
    pub fn parse(name: impl Into<String>, fields: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self::new(name, fields.into_iter().map(|f| FieldSpec::parse(f.as_ref())))
    }
}

/// A multi-task schema: structures, entities, relations and classifications in one pass.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub structures: Vec<StructureSpec>,
    pub entities: Vec<EntitySpec>,
    pub relations: Vec<RelationSpec>,
    pub classifications: Vec<ClassificationSpec>,
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entities(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for name in names {
            self.push_entity(EntitySpec {
                name: name.into(),
                description: None,
                threshold: None,
                dtype: EntityDtype::List,
            });
        }
        self
    }

    pub fn entity(mut self, spec: EntitySpec) -> Self {
        self.push_entity(spec);
        self
    }

    pub fn entity_with_description(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.entity(EntitySpec {
            name: name.into(),
            description: Some(description.into()),
            threshold: None,
            dtype: EntityDtype::List,
        })
    }

    fn push_entity(&mut self, spec: EntitySpec) {
        match self.entities.iter_mut().find(|e| e.name == spec.name) {
            Some(existing) => {
                existing.dtype = spec.dtype;
                existing.threshold = spec.threshold;
                if spec.description.is_some() {
                    existing.description = spec.description;
                }
            }
            None => self.entities.push(spec),
        }
    }

    pub fn relations(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for name in names {
            self.relations.push(RelationSpec {
                name: name.into(),
                description: None,
                threshold: None,
            });
        }
        self
    }

    pub fn relation(mut self, spec: RelationSpec) -> Self {
        self.relations.push(spec);
        self
    }

    pub fn structure(mut self, spec: StructureSpec) -> Self {
        self.structures.push(spec);
        self
    }

    pub fn classification(mut self, spec: ClassificationSpec) -> Self {
        self.classifications.push(spec);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compact_field_specs() {
        let f = FieldSpec::parse("status::[shipped|pending]::Order status");
        assert_eq!(f.name, "status");
        assert_eq!(f.dtype, FieldDtype::Str);
        assert_eq!(f.choices.as_deref(), Some(&["shipped".to_string(), "pending".to_string()][..]));
        assert_eq!(f.description.as_deref(), Some("Order status"));
        let f = FieldSpec::parse("tags::list::[a|b]");
        assert_eq!(f.dtype, FieldDtype::List);
        assert_eq!(FieldSpec::parse("name").dtype, FieldDtype::List);
    }
}
