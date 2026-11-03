//! Checkpoint loading policies. The streaming path uses Candle's public Metal
//! storage API and never maps the complete checkpoint into the process.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Metal weight-loading policy, independently selectable from inference dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalWeightLoading {
    /// Read weights into exact-size persistent Metal buffers. On macOS, a
    /// bounded staging buffer feeds private GPU storage; iOS uses shared storage.
    #[default]
    Stream,
    /// Original Candle mmap-and-copy loader, retained for controlled comparisons.
    Mmap,
}

/// Options affecting model construction. CPU and CUDA retain their original
/// loaders regardless of the Metal-specific policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RustLoadOptions {
    pub metal_weight_loading: MetalWeightLoading,
    /// Fuse F32 relative-score gathers/arithmetic/masking and compute buckets
    /// once per distinct distance. Applied only to the Metal DeBERTa encoder;
    /// other dtypes retain the original attention implementation.
    pub metal_attention_optimizations: bool,
}

impl Default for RustLoadOptions {
    fn default() -> Self {
        Self {
            metal_weight_loading: MetalWeightLoading::Stream,
            metal_attention_optimizations: true,
        }
    }
}

/// Build a checkpoint variable source using the requested loading policy.
pub(crate) fn checkpoint_var_builder(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
    options: RustLoadOptions,
) -> Result<VarBuilder<'static>> {
    let path = path.as_ref();
    #[cfg(feature = "metal")]
    if let Device::Metal(metal) = device {
        if options.metal_weight_loading == MetalWeightLoading::Stream {
            let backend = streaming::StreamingMetalWeights::open(path, metal.clone())
                .with_context(|| format!("opening streamed Metal checkpoint {}", path.display()))?;
            return Ok(VarBuilder::from_backend(
                Box::new(backend),
                dtype,
                device.clone(),
            ));
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = options;
    // SAFETY: as for the pre-existing GLiNER2::load implementation, a checkpoint
    // must not be modified while its mmap-backed variable source is alive.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
        .with_context(|| format!("mapping {}", path.display()))
}

#[cfg(feature = "metal")]
mod streaming {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::Mutex;

    use anyhow::{anyhow, bail, Context, Result};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Device, MetalDevice, MetalStorage, Shape, Storage, Tensor};
    use candle_nn::var_builder::SimpleBackend;
    use candle_nn::Init;
    use serde::Deserialize;

    // Safetensors uses the same 100 MB header bound. No payload is read here.
    const MAX_HEADER_BYTES: u64 = 100_000_000;
    #[cfg(not(target_os = "ios"))]
    const STAGING_BYTES: usize = 16 * 1024 * 1024;

    #[derive(Debug, Deserialize)]
    struct HeaderTensor {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: (u64, u64),
    }
    #[derive(Debug)]
    struct TensorSpec {
        dtype: DType,
        shape: Shape,
        count: usize,
        file_offset: u64,
        bytes: usize,
    }
    struct State {
        file: File,
        loaded: HashMap<(String, DType), Tensor>,
        #[cfg(not(target_os = "ios"))]
        staging: Option<Tensor>,
    }

    pub(super) struct StreamingMetalWeights {
        specs: HashMap<String, TensorSpec>,
        device: MetalDevice,
        state: Mutex<State>,
    }

    fn stored_dtype(name: &str) -> Result<DType> {
        Ok(match name {
            "BOOL" | "U8" => DType::U8,
            "U32" => DType::U32,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            other => bail!("unsupported Metal checkpoint dtype {other:?}"),
        })
    }

    fn read_specs(file: &mut File) -> Result<HashMap<String, TensorSpec>> {
        let file_bytes = file.metadata()?.len();
        let mut length = [0u8; 8];
        file.read_exact(&mut length)
            .context("reading safetensors header length")?;
        let header_bytes = u64::from_le_bytes(length);
        let data_start = 8u64
            .checked_add(header_bytes)
            .context("safetensors header length overflow")?;
        if header_bytes > MAX_HEADER_BYTES || data_start > file_bytes {
            bail!("invalid safetensors header length {header_bytes} for {file_bytes}-byte file");
        }
        let mut header = vec![0u8; usize::try_from(header_bytes)?];
        file.read_exact(&mut header)
            .context("reading safetensors header")?;
        let entries: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&header).context("parsing safetensors header")?;
        let mut specs = HashMap::with_capacity(entries.len());
        let mut intervals = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            if name == "__metadata__" {
                continue;
            }
            let entry: HeaderTensor = serde_json::from_value(value)
                .with_context(|| format!("parsing safetensors tensor {name:?}"))?;
            let dtype = stored_dtype(&entry.dtype)?;
            let count = entry
                .shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .with_context(|| format!("tensor {name:?} element count overflows"))?;
            let bytes = count
                .checked_mul(dtype.size_in_bytes())
                .with_context(|| format!("tensor {name:?} byte count overflows"))?;
            let (start, end) = entry.data_offsets;
            if end < start || end - start != bytes as u64 || end > file_bytes - data_start {
                bail!("tensor {name:?} has invalid data offsets {start}..{end}");
            }
            intervals.push((start, end));
            specs.insert(
                name,
                TensorSpec {
                    dtype,
                    shape: Shape::from(entry.shape),
                    count,
                    file_offset: data_start + start,
                    bytes,
                },
            );
        }
        // Safetensors payloads are contiguous and non-overlapping, without holes.
        intervals.sort_unstable();
        let mut expected = 0u64;
        for (start, end) in intervals {
            if start != expected {
                bail!(
                    "safetensors tensor data overlaps or has an uncovered interval at {expected}"
                );
            }
            expected = end;
        }
        if expected != file_bytes - data_start {
            bail!("safetensors tensor data does not cover the file payload");
        }
        Ok(specs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::weights::tests::Checkpoint;
        use serde_json::json;
        use std::io::Write;

        fn parse(
            header: serde_json::Value,
            payload_bytes: usize,
        ) -> Result<HashMap<String, TensorSpec>> {
            let checkpoint = Checkpoint::new(header, &vec![0; payload_bytes]);
            read_specs(&mut File::open(&checkpoint.0)?)
        }

        #[test]
        fn header_accepts_contiguous_tensors_scalars_and_metadata() {
            let specs = parse(
                json!({
                    "__metadata__": {"format":"pt"},
                    "matrix": {"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
                    "scalar": {"dtype":"F32","shape":[],"data_offsets":[16,20]}
                }),
                20,
            )
            .unwrap();
            assert_eq!(specs.len(), 2);
            assert_eq!(specs["matrix"].bytes, 16);
            assert_eq!(specs["scalar"].count, 1);
        }

        #[test]
        fn header_rejects_unsafe_offsets_shapes_and_uncovered_data() {
            for header in [
                json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
                json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}),
                json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                    "y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
                json!({"x":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}),
                json!({"x":{"dtype":"F32","shape":[usize::MAX,2],"data_offsets":[0,8]}}),
                json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            ] {
                assert!(parse(header, 8).is_err());
            }
        }

        #[test]
        fn header_rejects_truncated_and_excessive_header_lengths() {
            let checkpoint = Checkpoint::new(json!({}), &[]);
            for length in [7u64, MAX_HEADER_BYTES + 1, u64::MAX] {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&checkpoint.0)
                    .unwrap();
                file.write_all(&length.to_le_bytes()).unwrap();
                assert!(read_specs(&mut File::open(&checkpoint.0).unwrap()).is_err());
            }
        }
    }

    impl StreamingMetalWeights {
        pub(super) fn open(path: &std::path::Path, device: MetalDevice) -> Result<Self> {
            let mut file = File::open(path)?;
            let specs = read_specs(&mut file)?;
            Ok(Self {
                specs,
                device,
                state: Mutex::new(State {
                    file,
                    loaded: HashMap::new(),
                    #[cfg(not(target_os = "ios"))]
                    staging: None,
                }),
            })
        }

        fn load(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
            let spec = self
                .specs
                .get(name)
                .with_context(|| format!("missing tensor {name:?}"))?;
            let Device::Metal(requested) = device else {
                bail!("Metal variable source used with a non-Metal device");
            };
            if requested.id() != self.device.id() {
                bail!("Metal variable source used with a different device");
            }
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("Metal weight-loader lock was poisoned"))?;
            let key = (name.to_owned(), dtype);
            if let Some(tensor) = state.loaded.get(&key) {
                return Ok(tensor.clone());
            }
            if spec.count == 0 {
                bail!("Metal checkpoint tensor {name:?} is empty");
            }
            state.file.seek(SeekFrom::Start(spec.file_offset))?;
            // Exact allocation outside Candle's reusable power-of-two pool. The
            // public API also registers the buffer with its Metal residency set.
            let destination = self
                .device
                .new_private_buffer(spec.count, spec.dtype, name)?;

            #[cfg(target_os = "ios")]
            {
                // Candle's PRIVATE_RESOURCE_OPTIONS is StorageModeShared on iOS.
                // SAFETY: this new buffer has spec.bytes bytes, is shared, has no
                // submitted GPU users, and is not exposed until read_exact ends.
                let data =
                    unsafe { std::slice::from_raw_parts_mut(destination.contents(), spec.bytes) };
                state
                    .file
                    .read_exact(data)
                    .with_context(|| format!("reading tensor {name:?}"))?;
            }
            #[cfg(not(target_os = "ios"))]
            {
                if state.staging.is_none() {
                    // Allocate one Shared buffer via Candle's public upload API.
                    // The initial zero vector is temporary, not checkpoint-sized.
                    state.staging = Some(Tensor::from_vec(
                        vec![0u8; STAGING_BYTES],
                        STAGING_BYTES,
                        device,
                    )?);
                }
                let staging = state.staging.as_ref().unwrap().clone();
                let (storage, _) = staging.storage_and_layout();
                let Storage::Metal(storage) = &*storage else {
                    unreachable!("Metal staging tensor")
                };
                let buffer = storage.buffer();
                let mut copied = 0usize;
                while copied < spec.bytes {
                    let count = STAGING_BYTES.min(spec.bytes - copied);
                    // SAFETY: the loader mutex gives exclusive CPU access, and
                    // the preceding chunk was synchronized before this refill.
                    let data = unsafe { std::slice::from_raw_parts_mut(buffer.contents(), count) };
                    state
                        .file
                        .read_exact(data)
                        .with_context(|| format!("reading tensor {name:?}"))?;
                    {
                        let mut blit = self.device.blit_command_encoder()?;
                        blit.copy_from_buffer(buffer, 0, &destination, copied, count);
                    }
                    // Bound live staging memory and prevent CPU overwrite while
                    // a previous GPU blit still reads the shared staging buffer.
                    self.device.flush_and_wait_current()?;
                    copied += count;
                }
            }
            let storage =
                MetalStorage::new(destination, self.device.clone(), spec.count, spec.dtype);
            let tensor = Tensor::from_storage(
                Storage::Metal(storage),
                spec.shape.clone(),
                BackpropOp::none(),
                false,
            )
            .to_dtype(dtype)?;
            if dtype != spec.dtype {
                self.device.flush_and_wait_current()?;
            }
            state.loaded.insert(key, tensor.clone());
            Ok(tensor)
        }
    }

    fn candle_error(error: anyhow::Error) -> candle_core::Error {
        candle_core::Error::Msg(format!("{error:#}")).bt()
    }
    impl SimpleBackend for StreamingMetalWeights {
        fn get(
            &self,
            shape: Shape,
            name: &str,
            _: Init,
            dtype: DType,
            device: &Device,
        ) -> candle_core::Result<Tensor> {
            let spec =
                self.specs
                    .get(name)
                    .ok_or_else(|| candle_core::Error::CannotFindTensor {
                        path: name.to_owned(),
                    })?;
            if spec.shape != shape {
                return Err(candle_core::Error::UnexpectedShape {
                    msg: format!("shape mismatch for {name}"),
                    expected: shape,
                    got: spec.shape.clone(),
                }
                .bt());
            }
            self.load(name, dtype, device).map_err(candle_error)
        }
        fn get_unchecked(
            &self,
            name: &str,
            dtype: DType,
            device: &Device,
        ) -> candle_core::Result<Tensor> {
            self.load(name, dtype, device).map_err(candle_error)
        }
        fn contains_tensor(&self, name: &str) -> bool {
            self.specs.contains_key(name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    pub(super) struct Checkpoint(pub std::path::PathBuf);
    impl Checkpoint {
        pub(super) fn new(header: serde_json::Value, payload: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "gliner-weights-{}-{}.safetensors",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed)
            ));
            let mut file = std::fs::File::create(&path).unwrap();
            let mut header = serde_json::to_vec(&header).unwrap();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.write_all(payload).unwrap();
            Self(path)
        }
    }
    impl Drop for Checkpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn metal_policy_does_not_change_cpu_loading_or_dtype_conversion() {
        let header = serde_json::json!({"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}});
        let payload: Vec<u8> = [1.25f32, -2.5, 0.0, 4.0]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect();
        let checkpoint = Checkpoint::new(header, &payload);
        for metal_weight_loading in [MetalWeightLoading::Stream, MetalWeightLoading::Mmap] {
            let vb = checkpoint_var_builder(
                &checkpoint.0,
                DType::F32,
                &Device::Cpu,
                RustLoadOptions {
                    metal_weight_loading,
                    ..RustLoadOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                vb.get((2, 2), "weight").unwrap().to_vec2::<f32>().unwrap(),
                vec![vec![1.25, -2.5], vec![0.0, 4.0]]
            );
            assert_eq!(
                vb.set_dtype(DType::F16)
                    .get((2, 2), "weight")
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                vec![vec![1.25, -2.5], vec![0.0, 4.0]]
            );
        }
    }
}
