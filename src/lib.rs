//! Rust inference for GLiNER2 boundary-architecture checkpoints
//! (`fastino/gliner2.5-multi-v1`), built on candle.
//!
//! ```no_run
//! use gliner_rs::{ExtractOptions, GLiNER2};
//! use candle_core::{DType, Device};
//!
//! let model = GLiNER2::load("path/to/gliner2.5-multi-v1", &Device::Cpu, DType::F32)?;
//! let result = model.extract_entities(
//!     "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
//!     &["company", "person", "product", "location"],
//!     &ExtractOptions::default(),
//! )?;
//! println!("{result}");
//! # anyhow::Ok(())
//! ```

pub mod config;
pub mod deberta;
pub mod decode;
pub mod download;
pub mod heads;
pub mod model;
pub mod model_path;
pub mod processor;
pub mod records;
pub mod schema;

pub use decode::OverlapPolicy;
pub use model::{ExtractOptions, GLiNER2};
pub use processor::WordSplitter;
pub use schema::{
    ClassActivation, ClassificationSpec, EntityDtype, EntitySpec, FieldDtype, FieldSpec, RelationSpec, Schema,
    StructureMode, StructureSpec,
};
