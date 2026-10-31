//! C boundary for serial use by the Apple demo. Metal errors never trigger CPU fallback.
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use gliner_rs::{
    AttributeGroup, Cardinality, ClassActivation, ClassificationSpec, EntityDtype, EntitySpec,
    ExtractOptions, FieldDtype, FieldSpec, GLiNER2, RegexValidator, RelationSpec, Schema,
    StructureMode, StructureSpec, ValidatorMode, WordSplitter,
};
use serde_json::Value;
use std::{
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string {key}"))
}
fn opt_string(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}
fn list<'a>(v: &'a Value, key: &str) -> Result<&'a [Value]> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        _ => bail!("{key} must be an array"),
    }
}
fn strings(v: &Value, key: &str) -> Result<Vec<String>> {
    list(v, key)?
        .iter()
        .map(|x| {
            x.as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("{key} must contain strings"))
        })
        .collect()
}
fn number(v: &Value, key: &str) -> Result<Option<f32>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let n = value
                .as_f64()
                .ok_or_else(|| anyhow!("{key} must be numeric"))? as f32;
            if !n.is_finite() {
                bail!("{key} must be finite and representable as Float32")
            }
            Ok(Some(n))
        }
    }
}
fn max_words(v: &Value) -> Result<Option<usize>> {
    match v.get("maxWords") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let n = value
                .as_u64()
                .ok_or_else(|| anyhow!("maxWords must be a nonnegative integer"))?;
            Ok(Some(usize::try_from(n).context("maxWords is too large")?))
        }
    }
}
fn validators(v: &Value) -> Result<Vec<RegexValidator>> {
    list(v, "validators")?
        .iter()
        .map(|x| {
            Ok(RegexValidator {
                pattern: text(x, "pattern")?.into(),
                mode: match x.get("mode").and_then(Value::as_str).unwrap_or("full") {
                    "full" => ValidatorMode::Full,
                    "partial" => ValidatorMode::Partial,
                    other => bail!("invalid validator mode {other}"),
                },
                exclude: x.get("exclude").and_then(Value::as_bool).unwrap_or(false),
                case_insensitive: x
                    .get("caseInsensitive")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            })
        })
        .collect()
}
fn schema(v: &Value) -> Result<Schema> {
    let mut s = Schema::new();
    for e in list(v, "entities")? {
        if e.get("isAttribute")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        let mut spec = EntitySpec::new(text(e, "name")?);
        spec.description = opt_string(e, "description");
        spec.threshold = number(e, "threshold")?;
        spec.dtype = match e.get("dtype").and_then(Value::as_str).unwrap_or("list") {
            "list" => EntityDtype::List,
            "str" => EntityDtype::Str,
            x => bail!("invalid entity dtype {x}"),
        };
        spec.validators = validators(e)?;
        s = s.entity(spec);
    }
    let mut groups = Vec::new();
    for named in list(v, "entityAttributes")? {
        let g = named.get("group").context("missing attribute group")?;
        let mut group = AttributeGroup::new(strings(g, "labels")?);
        group.multi_label = g
            .get("multiLabel")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        group.threshold = number(g, "threshold")?.unwrap_or(0.5);
        group.qualify_labels = g
            .get("qualifyLabels")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if g.get("appliesTo").is_some_and(|x| !x.is_null()) {
            group.applies_to = Some(strings(g, "appliesTo")?);
        }
        groups.push((text(named, "name")?.to_owned(), group));
    }
    if !groups.is_empty() {
        s = s.entity_attributes(groups);
    }
    for item in list(v, "relations")? {
        s = s.relation(RelationSpec {
            name: text(item, "name")?.into(),
            description: opt_string(item, "description"),
            threshold: number(item, "threshold")?,
        });
    }
    for item in list(v, "structures")? {
        let mut fields = Vec::new();
        for f in list(item, "fields")? {
            let mut field = FieldSpec::new(text(f, "name")?);
            field.dtype = match f.get("dtype").and_then(Value::as_str).unwrap_or("list") {
                "list" => FieldDtype::List,
                "str" => FieldDtype::Str,
                x => bail!("invalid field dtype {x}"),
            };
            if f.get("choices").is_some_and(|x| !x.is_null()) {
                field.choices = Some(strings(f, "choices")?);
            }
            field.description = opt_string(f, "description");
            field.threshold = number(f, "threshold")?;
            field.exclusive = f.get("exclusive").and_then(Value::as_bool);
            field.validators = validators(f)?;
            field.cardinality = match f.get("cardinality").and_then(Value::as_str) {
                None => None,
                Some("optional_one") => Some(Cardinality::OptionalOne),
                Some("required_one") => Some(Cardinality::RequiredOne),
                Some("zero_or_more") => Some(Cardinality::ZeroOrMore),
                Some("one_or_more") => Some(Cardinality::OneOrMore),
                Some(x) => bail!("invalid cardinality {x}"),
            };
            fields.push(field);
        }
        let mut spec = StructureSpec::new(text(item, "name")?, fields);
        spec.mode = match item.get("mode").and_then(Value::as_str).unwrap_or("auto") {
            "auto" => StructureMode::Auto,
            "legacy" => StructureMode::Legacy,
            "latent" => StructureMode::Latent,
            "anchorless" => StructureMode::Anchorless,
            x => bail!("invalid structure mode {x}"),
        };
        spec.anchor = opt_string(item, "anchor");
        s = s.structure(spec);
    }
    for item in list(v, "classifications")? {
        let mut cls = ClassificationSpec::new(text(item, "task")?, strings(item, "labels")?);
        cls.multi_label = item
            .get("multiLabel")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        cls.cls_threshold = number(item, "clsThreshold")?.unwrap_or(0.5);
        cls.prompt = opt_string(item, "prompt");
        cls.activation = match item
            .get("activation")
            .and_then(Value::as_str)
            .unwrap_or("auto")
        {
            "auto" => ClassActivation::Auto,
            "sigmoid" => ClassActivation::Sigmoid,
            "softmax" => ClassActivation::Softmax,
            x => bail!("invalid class activation {x}"),
        };
        for d in list(item, "labelDescriptions")? {
            cls.label_descriptions
                .push((text(d, "label")?.into(), text(d, "description")?.into()));
        }
        for ex in list(item, "examples")? {
            cls.examples
                .push((text(ex, "input")?.into(), text(ex, "output")?.into()));
        }
        s = s.classification(cls);
    }
    Ok(s)
}
fn read_cstr(p: *const c_char) -> Result<String> {
    if p.is_null() {
        bail!("null input pointer")
    };
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .context("invalid UTF-8")
        .map(str::to_owned)
}
fn put_error(out: *mut *mut c_char, message: &str) {
    if !out.is_null() {
        let sanitized = message.replace('\0', "\\0");
        unsafe { *out = CString::new(sanitized).unwrap().into_raw() }
    }
}
fn guarded<T>(err: *mut *mut c_char, op: impl FnOnce() -> Result<T>) -> Option<T> {
    if !err.is_null() {
        unsafe { *err = ptr::null_mut() }
    }
    match catch_unwind(AssertUnwindSafe(op)) {
        Ok(Ok(x)) => Some(x),
        Ok(Err(e)) => {
            put_error(err, &format!("{e:#}"));
            None
        }
        Err(_) => {
            put_error(err, "Rust inference panicked");
            None
        }
    }
}
#[no_mangle]
pub extern "C" fn gliner_candle_open(path: *const c_char, err: *mut *mut c_char) -> *mut GLiNER2 {
    guarded(err, || {
        let device = Device::new_metal(0).context("Metal device unavailable")?;
        let model = GLiNER2::load(read_cstr(path)?, &device, DType::F32)?;
        device.synchronize()?;
        Ok(Box::into_raw(Box::new(model)))
    })
    .unwrap_or(ptr::null_mut())
}
#[no_mangle]
pub extern "C" fn gliner_candle_extract(
    handle: *mut GLiNER2,
    request: *const c_char,
    err: *mut *mut c_char,
) -> *mut c_char {
    guarded(err, || {
        let model = unsafe { handle.as_mut() }.context("null model")?;
        let request: Value = serde_json::from_str(&read_cstr(request)?)?;
        let schema = schema(request.get("schema").context("missing schema")?)?;
        model.set_word_splitter(
            match request
                .get("wordSplitter")
                .and_then(Value::as_str)
                .unwrap_or("whitespace")
            {
                "whitespace" => WordSplitter::Whitespace,
                "char" => WordSplitter::Char,
                x => bail!("invalid splitter {x}"),
            },
        );
        let o = request.get("options").context("missing options")?;
        let opts = ExtractOptions {
            threshold: number(o, "threshold")?.unwrap_or(0.5),
            include_confidence: o
                .get("includeConfidence")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            include_spans: o
                .get("includeSpans")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            overlap_policy: opt_string(o, "overlapPolicy")
                .map(|x| x.parse())
                .transpose()?,
            max_words: max_words(o)?,
        };
        let result = model.extract(text(&request, "text")?, &schema, &opts)?;
        Ok(CString::new(serde_json::to_string(&result)?)?.into_raw())
    })
    .unwrap_or(ptr::null_mut())
}
#[no_mangle]
pub extern "C" fn gliner_candle_close(handle: *mut GLiNER2) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) }
    }
}
#[no_mangle]
pub extern "C" fn gliner_candle_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn schema_preserves_order_and_reconstructs_phantom_attributes() {
        let value = json!({"entities": [{"name":"person"}, {"name":"z", "isAttribute": true}], "entityAttributes": [{"name":"kind", "group":{"labels":["z"],"threshold":0.5}}], "classifications":[{"task":"sentiment","labels":["b","a"],"labelDescriptions":[{"label":"b","description":"second"}]}]});
        let result = schema(&value).unwrap();
        assert_eq!(
            result
                .entities
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            ["person", "z"]
        );
        assert_eq!(result.classifications[0].labels, ["b", "a"]);
        assert_eq!(result.classifications[0].label_descriptions[0].0, "b");
    }
    #[test]
    fn invalid_requests_return_errors_without_unwinding() {
        assert!(schema(
            &json!({"structures":[{"name":"x","fields":[{"name":"a","dtype":"bad"}]}]})
        )
        .is_err());
        assert!(number(&json!({"threshold":"bad"}), "threshold").is_err());
        assert!(number(&json!({"threshold":1e100}), "threshold").is_err());
        assert!(max_words(&json!({"maxWords":-1})).is_err());
        assert!(max_words(&json!({"maxWords":1.5})).is_err());
        assert_eq!(max_words(&json!({"maxWords":0})).unwrap(), Some(0));
        let mut error = ptr::null_mut();
        let request = CString::new("{}").unwrap();
        let result = gliner_candle_extract(ptr::null_mut(), request.as_ptr(), &mut error);
        assert!(result.is_null());
        assert!(!error.is_null());
        gliner_candle_string_free(error);
    }
}
