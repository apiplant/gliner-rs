//! Extraction schema builder (mirrors `gliner2.inference.schema.Schema`).

use crate::records::Cardinality;

/// Regex post-filter on an extracted span's surface text
/// (`gliner2.inference.schema.RegexValidator`).
#[derive(Debug, Clone)]
pub struct RegexValidator {
    pub pattern: String,
    pub mode: ValidatorMode,
    pub exclude: bool,
    pub case_insensitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidatorMode {
    /// The whole surface text must match (default).
    #[default]
    Full,
    /// The pattern only needs to match somewhere in the surface text.
    Partial,
}

impl RegexValidator {
    pub fn new(pattern: impl Into<String>) -> Self {
        Self { pattern: pattern.into(), mode: ValidatorMode::Full, exclude: false, case_insensitive: true }
    }

    pub fn partial(mut self) -> Self {
        self.mode = ValidatorMode::Partial;
        self
    }

    pub fn exclude(mut self) -> Self {
        self.exclude = true;
        self
    }

    pub fn case_sensitive(mut self) -> Self {
        self.case_insensitive = false;
        self
    }

    pub fn validate(&self, text: &str) -> bool {
        let anchored = match self.mode {
            ValidatorMode::Full => format!("^(?:{})$", self.pattern),
            ValidatorMode::Partial => self.pattern.clone(),
        };
        let matched = regex::RegexBuilder::new(&anchored)
            .case_insensitive(self.case_insensitive)
            .build()
            .is_ok_and(|re| re.is_match(text));
        if self.exclude { !matched } else { matched }
    }
}

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
    pub validators: Vec<RegexValidator>,
    /// Set for the phantom entities `Schema::entity_attributes` injects for
    /// each attribute label: they get a query slot like any entity, but are
    /// excluded from the public entity output (`_entity_attribute_labels`).
    pub(crate) is_attribute: bool,
}

impl EntitySpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            threshold: None,
            dtype: EntityDtype::List,
            validators: Vec::new(),
            is_attribute: false,
        }
    }

    pub fn validator(mut self, validator: RegexValidator) -> Self {
        self.validators.push(validator);
        self
    }
}

/// Labels assigned as attributes of extracted entity spans
/// (`gliner2.inference.schema.AttributeGroup`).
#[derive(Debug, Clone)]
pub struct AttributeGroup {
    pub labels: Vec<String>,
    /// Independent sigmoid decisions instead of forcing one value.
    pub multi_label: bool,
    /// Selection cutoff for multi-label groups.
    pub threshold: f32,
    /// Entity types this group applies to; `None` means every entity.
    pub applies_to: Option<Vec<String>>,
    /// Prefix model-facing values with the group name to reduce ambiguity,
    /// while keeping returned values unqualified.
    pub qualify_labels: bool,
}

impl AttributeGroup {
    pub fn new(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            labels: labels.into_iter().map(Into::into).collect(),
            multi_label: false,
            threshold: 0.5,
            applies_to: None,
            qualify_labels: false,
        }
    }

    pub fn multi_label(mut self, threshold: f32) -> Self {
        self.multi_label = true;
        self.threshold = threshold;
        self
    }

    pub fn applies_to(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.applies_to = Some(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn qualify_labels(mut self) -> Self {
        self.qualify_labels = true;
        self
    }
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
    /// Overrides the dtype-derived default (`str` -> `required_one`, `list` ->
    /// `zero_or_more`) when set.
    pub cardinality: Option<Cardinality>,
    /// Whether a mention bound to this field cannot bind to another field in
    /// the same record instance. Defaults to `true`, matching `extract_json`.
    pub exclusive: Option<bool>,
    pub validators: Vec<RegexValidator>,
}

impl FieldSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype: FieldDtype::List,
            choices: None,
            description: None,
            threshold: None,
            cardinality: None,
            exclusive: None,
            validators: Vec::new(),
        }
    }

    pub fn cardinality(mut self, cardinality: Cardinality) -> Self {
        self.cardinality = Some(cardinality);
        self
    }

    pub fn exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = Some(exclusive);
        self
    }

    pub fn validator(mut self, validator: RegexValidator) -> Self {
        self.validators.push(validator);
        self
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
        // `extract_json`'s convenience default (`_json_schema`): every field
        // exclusive, cardinality by dtype alone (the anchor field included).
        field.cardinality = Some(match field.dtype {
            FieldDtype::Str => Cardinality::RequiredOne,
            FieldDtype::List => Cardinality::ZeroOrMore,
        });
        field.exclusive = Some(true);
        field
    }
}

/// How structure instances are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StructureMode {
    /// Record head in `natural` mode: the anchor field (first field, or
    /// `StructureSpec::anchor`) seeds one instance per detected mention, and
    /// other fields are assigned to it. Falls back to the legacy decoder when
    /// the checkpoint has no record head. Matches `extract_json`.
    #[default]
    Auto,
    /// One aggregate instance with every field (`Schema.structure(mode=None)`).
    Legacy,
    /// Record head in `latent` mode: no declared anchor: every field
    /// candidate is a potential instance seed, chosen by the head's learned
    /// selector rather than a fixed field.
    Latent,
    /// Record head in `anchorless` mode: document-conditioned learned
    /// instance queries predict object/no-object plus one candidate per
    /// field, independent of any single span.
    Anchorless,
}

#[derive(Debug, Clone)]
pub struct StructureSpec {
    pub name: String,
    pub fields: Vec<FieldSpec>,
    pub mode: StructureMode,
    /// Custom anchor field name for `StructureMode::Auto`. Defaults to the
    /// first field.
    pub anchor: Option<String>,
}

impl StructureSpec {
    pub fn new(name: impl Into<String>, fields: impl IntoIterator<Item = FieldSpec>) -> Self {
        Self { name: name.into(), fields: fields.into_iter().collect(), mode: StructureMode::Auto, anchor: None }
    }

    pub fn mode(mut self, mode: StructureMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn anchor(mut self, name: impl Into<String>) -> Self {
        self.anchor = Some(name.into());
        self
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
    pub entity_attribute_groups: Vec<(String, AttributeGroup)>,
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entities(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for name in names {
            self.push_entity(EntitySpec::new(name));
        }
        self
    }

    pub fn entity(mut self, spec: EntitySpec) -> Self {
        self.push_entity(spec);
        self
    }

    pub fn entity_with_description(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.entity(EntitySpec { description: Some(description.into()), ..EntitySpec::new(name) })
    }

    fn push_entity(&mut self, spec: EntitySpec) {
        match self.entities.iter_mut().find(|e| e.name == spec.name) {
            Some(existing) => {
                existing.dtype = spec.dtype;
                existing.threshold = spec.threshold;
                if spec.description.is_some() {
                    existing.description = spec.description;
                }
                if !spec.validators.is_empty() {
                    existing.validators = spec.validators;
                }
            }
            None => self.entities.push(spec),
        }
    }

    /// Attach attribute groups to already-declared entities
    /// (`Schema.entity_attributes`). Each label gets a phantom entity query
    /// slot; at decode time it is force-scored against every retained entity
    /// span instead of proposing spans of its own.
    pub fn entity_attributes(mut self, groups: impl IntoIterator<Item = (impl Into<String>, AttributeGroup)>) -> Self {
        let groups: Vec<(String, AttributeGroup)> = groups.into_iter().map(|(name, g)| (name.into(), g)).collect();
        // Matches `Schema.entity_attributes`: prompt labels are injected in
        // sorted order (a plain set in the reference impl), not declaration
        // order, since prompt order shifts the encoder's context.
        let mut prompt_labels: Vec<String> = Vec::new();
        for (group_name, group) in &groups {
            for label in &group.labels {
                let prompt_label = if group.qualify_labels { format!("{group_name}: {label}") } else { label.clone() };
                if !prompt_labels.contains(&prompt_label) {
                    prompt_labels.push(prompt_label);
                }
            }
        }
        prompt_labels.sort();
        for prompt_label in prompt_labels {
            if !self.entities.iter().any(|e| e.name == prompt_label) {
                self.entities.push(EntitySpec { is_attribute: true, ..EntitySpec::new(prompt_label) });
            }
        }
        self.entity_attribute_groups.extend(groups);
        self
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
