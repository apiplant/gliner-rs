//! Record cardinality/exclusivity/custom anchors, regex validators, and entity
//! attributes. Run with a checkpoint directory: `cargo run --release --example
//! records_and_attributes -- path/to/gliner2.5-multi-v1`.
use anyhow::Result;
use candle_core::{DType, Device};
use gliner_rs::{
    AttributeGroup, Cardinality, EntitySpec, ExtractOptions, FieldSpec, GLiNER2, RegexValidator, Schema,
    StructureSpec,
};

fn main() -> Result<()> {
    let model_dir = std::env::args().nth(1).expect("usage: records_and_attributes MODEL_DIR");
    let model = GLiNER2::load(&model_dir, &Device::Cpu, DType::F32)?;
    let opts = ExtractOptions { include_spans: true, include_confidence: true, ..Default::default() };

    // Custom anchor + cardinality + a non-exclusive field: `sku` (not `name`) seeds
    // each instance, `tag` may bind more than one mention, and `note` is allowed to
    // share a mention with another field instead of claiming it exclusively.
    let order = StructureSpec::new(
        "item",
        [
            FieldSpec::new("name"),
            FieldSpec::new("sku"),
            FieldSpec::new("tag").cardinality(Cardinality::ZeroOrMore),
            FieldSpec::new("note").exclusive(false),
        ],
    )
    .anchor("sku");
    let schema = Schema::new().structure(order);
    let result = model.extract(
        "SKU A100: wireless mouse, tags: electronics, accessory. SKU B200: desk lamp, tags: home.",
        &schema,
        &opts,
    )?;
    println!("custom anchor + cardinality:\n{}\n", serde_json::to_string_pretty(&result)?);

    // Regex validator: only keep "person" spans that look like a capitalized full name.
    let schema = Schema::new().entity(
        EntitySpec::new("person").validator(RegexValidator::new(r"^[A-Z][a-z]+ [A-Z][a-z]+$")),
    );
    let result = model.extract("alice met with Bob Smith and dr. jones yesterday.", &schema, &opts)?;
    println!("regex-validated entity:\n{}\n", serde_json::to_string_pretty(&result)?);

    // Entity attributes: score extra properties against an already-extracted span,
    // rather than proposing spans of their own.
    let schema = Schema::new()
        .entities(["product"])
        .entity_attributes([("color", AttributeGroup::new(["red", "blue", "green", "black"]).applies_to(["product"]))]);
    let result = model.extract("Apple released a red iPhone and a black MacBook this year.", &schema, &opts)?;
    println!("entity attributes:\n{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}
