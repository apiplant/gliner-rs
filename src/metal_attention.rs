//! F32 Metal relative-attention epilogue. This uses Candle's own command
//! encoder and allocation pool, including its cross-dispatch hazard tracking.

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp3, DType, Device, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLSize};

pub(crate) struct RelativeAttentionKernel {
    pipeline: ComputePipeline,
}

impl RelativeAttentionKernel {
    pub(crate) fn new(device: &candle_core::MetalDevice) -> Result<Self> {
        let options = MTLCompileOptions::new();
        // This setter works on all OS versions supported by Candle. Together
        // with the source pragmas, it disables reassociation and contraction.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .metal_device()
            .new_library_with_source(include_str!("metal_attention.metal"), Some(&options))
            .map_err(error)?;
        let function = library
            .get_function("gliner_relative_attention_f32", None)
            .map_err(error)?;
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(error)?;
        Ok(Self { pipeline })
    }

    pub(crate) fn forward(
        &self,
        content: &Tensor,
        qrel: &Tensor,
        krel: &Tensor,
        c2p: &Tensor,
        p2c: &Tensor,
        mask: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        content.apply_op3_no_bwd(
            qrel,
            krel,
            &RelativeAttentionOp {
                pipeline: self.pipeline.clone(),
                c2p: c2p.clone(),
                p2c: p2c.clone(),
                mask: mask.clone(),
                scale,
            },
        )
    }
}

fn error(error: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("fused Metal relative attention: {error}")).bt()
}

struct RelativeAttentionOp {
    pipeline: ComputePipeline,
    c2p: Tensor,
    p2c: Tensor,
    mask: Tensor,
    scale: f32,
}

#[repr(C)]
struct Params {
    dims: [u32; 4],
    strides: [[u64; 4]; 6],
}

fn strides(layout: &Layout) -> Result<[u64; 4]> {
    if layout.dims().len() != 4 {
        return Err(error("expected a rank-four tensor"));
    }
    Ok(std::array::from_fn(|i| layout.stride()[i] as u64))
}

impl CustomOp3 for RelativeAttentionOp {
    fn name(&self) -> &'static str {
        "gliner-relative-attention-f32"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(error("the Metal-only custom operation cannot run on CPU"))
    }

    fn metal_fwd(
        &self,
        content: &MetalStorage,
        cl: &Layout,
        qrel: &MetalStorage,
        ql: &Layout,
        krel: &MetalStorage,
        kl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, h, t, columns) = cl.shape().dims4()?;
        let (qb, qh, qt, r) = ql.shape().dims4()?;
        if t == 0
            || b == 0
            || h == 0
            || t != columns
            || (qb, qh, qt) != (b, h, t)
            || kl.dims() != ql.dims()
            || r == 0
        {
            return Err(error("inconsistent content/relative score shapes"));
        }
        let count = cl.shape().elem_count();
        if count > u32::MAX as usize || [b, h, t, r].iter().any(|&d| d > u32::MAX as usize) {
            return Err(error("attention dimensions exceed the kernel's u32 grid"));
        }
        let device = content.device();
        if content.dtype() != DType::F32
            || qrel.dtype() != DType::F32
            || krel.dtype() != DType::F32
            || qrel.device().id() != device.id()
            || krel.device().id() != device.id()
        {
            return Err(error(
                "content/relative scores must be F32 on the same Metal device",
            ));
        }
        if self.c2p.dims() != [1, 1, t, t]
            || self.p2c.dims() != [1, 1, t, t]
            || self.mask.dims() != [b, 1, t, t]
            || self.c2p.dtype() != DType::U32
            || self.p2c.dtype() != DType::U32
            || self.mask.dtype() != DType::U8
        {
            return Err(error(
                "invalid base relative-index or pair-mask shape/dtype",
            ));
        }
        for tensor in [&self.c2p, &self.p2c, &self.mask] {
            if !matches!(tensor.device(),Device::Metal(d) if d.id() == device.id()) {
                return Err(error(
                    "relative indices and mask must share the Metal device",
                ));
            }
        }
        let (cs, cil) = self.c2p.storage_and_layout();
        let (ps, pil) = self.p2c.storage_and_layout();
        let (ms, ml) = self.mask.storage_and_layout();
        let (Storage::Metal(ci), Storage::Metal(pi), Storage::Metal(mask)) = (&*cs, &*ps, &*ms)
        else {
            return Err(error("relative indices and mask require Metal storage"));
        };
        let params = Params {
            dims: [b as u32, h as u32, t as u32, r as u32],
            strides: [
                strides(cl)?,
                strides(ql)?,
                strides(kl)?,
                strides(cil)?,
                strides(pil)?,
                strides(ml)?,
            ],
        };
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("gliner-relative-attention")
            .build()?;
        let commands = device.command_encoder()?;
        let encoder = commands.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, (storage, layout)) in [
            (content, cl),
            (qrel, ql),
            (krel, kl),
            (ci, cil),
            (pi, pil),
            (mask, ml),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_input_buffer(
                index,
                Some(storage.buffer()),
                layout.start_offset() * storage.dtype().size_in_bytes(),
            );
        }
        encoder.set_output_buffer(6, Some(&output), 0);
        encoder.set_bytes(7, &params);
        encoder.set_bytes(8, &self.scale);
        let width = self.pipeline.max_total_threads_per_threadgroup().min(256);
        encoder.dispatch_threads(
            MTLSize {
                width: count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            cl.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offset_tensor(values: Vec<f32>, offset: usize, device: &Device) -> Result<Tensor> {
        let count = values.len();
        let mut storage = vec![1234.; offset];
        storage.extend(values);
        Tensor::from_vec(storage, count + offset, device)?.narrow(0, offset, count)
    }

    #[test]
    #[ignore = "requires an isolated Metal GPU validation window"]
    fn fused_f32_matches_unfused_batches_padding_offsets_and_strides() -> Result<()> {
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            unreachable!()
        };
        let kernel = RelativeAttentionKernel::new(metal)?;
        let (b, h, t, r) = (2, 3, 5, 8);
        let content = offset_tensor(
            (0..b * h * t * t)
                .map(|i| [16777216., -16777216., 0., -0., 1.25, -5.5, 31.][i % 7])
                .collect(),
            1,
            &device,
        )?
        .reshape((b, t, h, t))?
        .transpose(1, 2)?;
        let qrel = offset_tensor(
            (0..b * h * t * r)
                .map(|i| [-16777216., 16777216., 1., -0.5, 4., 0., -0.][i % 7])
                .collect(),
            3,
            &device,
        )?
        .reshape((b, t, h, r))?
        .transpose(1, 2)?;
        let krel = offset_tensor(
            (0..b * h * t * r)
                .map(|i| [1., -1., 0.25, 16777216., -16777216., 0., -0.][i % 7])
                .collect(),
            5,
            &device,
        )?
        .reshape((b, t, h, r))?
        .transpose(1, 2)?;
        let ids = |shift: usize| -> Result<Tensor> {
            let mut values = vec![0u32; shift];
            values.extend((0..t * t).map(|i| ((i * 3 + i / t + shift) % r) as u32));
            Tensor::from_vec(values, shift + t * t, &device)?
                .narrow(0, shift, t * t)?
                .reshape((1, 1, t, t))?
                .transpose(2, 3)
        };
        let ci = ids(2)?;
        let pi = ids(4)?;
        let mask = Tensor::from_vec(
            vec![1f32, 1., 0., 1., 0., 1., 0., 1., 0., 0.],
            (b, t),
            &device,
        )?;
        let pair = mask
            .unsqueeze(1)?
            .broadcast_mul(&mask.unsqueeze(2)?)?
            .unsqueeze(1)?
            .to_dtype(DType::U8)?;
        let mut mask_storage = vec![0u8; 3];
        mask_storage.extend(pair.flatten_all()?.to_vec1::<u8>()?);
        let pair = Tensor::from_vec(mask_storage, 3 + b * t * t, &device)?
            .narrow(0, 3, b * t * t)?
            .reshape((b, 1, t, t))?
            .transpose(2, 3)?;
        assert!(pair.layout().start_offset() > 0 && !pair.is_contiguous());
        assert!(content.layout().start_offset() > 0 && !content.is_contiguous());
        assert!(qrel.layout().start_offset() > 0 && !qrel.is_contiguous());
        assert!(!ci.is_contiguous());
        let scale = (1.0 / (192f64).sqrt()) as f32;
        let fused = kernel.forward(&content, &qrel, &krel, &ci, &pi, &pair, scale)?;
        let c = qrel
            .contiguous()?
            .gather(&ci.expand((b, h, t, t))?.contiguous()?, 3)?;
        let p = krel
            .contiguous()?
            .gather(&pi.expand((b, h, t, t))?.contiguous()?, 3)?
            .transpose(2, 3)?;
        let scores = ((&content + c)? + p)?.affine(scale as f64, 0.)?;
        let full = Tensor::full(f32::MIN, (b, h, t, t), &device)?;
        let reference = pair.expand((b, h, t, t))?.where_cond(&scores, &full)?;
        let actual = fused.flatten_all()?.to_vec1::<f32>()?;
        let expected = reference.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let cv = content.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let qv = qrel.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let kv = krel.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let civ = ci.contiguous()?.flatten_all()?.to_vec1::<u32>()?;
        let piv = pi.contiguous()?.flatten_all()?.to_vec1::<u32>()?;
        let mv = pair.flatten_all()?.to_vec1::<u8>()?;
        for batch in 0..b {
            for head in 0..h {
                for i in 0..t {
                    for j in 0..t {
                        let index = ((batch * h + head) * t + i) * t + j;
                        let expected = if mv[batch * t * t + i * t + j] == 0 {
                            f32::MIN
                        } else {
                            let q = qv[((batch * h + head) * t + i) * r + civ[i * t + j] as usize];
                            let k = kv[((batch * h + head) * t + j) * r + piv[j * t + i] as usize];
                            ((cv[index] + q) + k).mul_add(scale, 0.)
                        };
                        assert_eq!(
                            actual[index].to_bits(),
                            expected.to_bits(),
                            "b={batch},h={head},i={i},j={j}"
                        );
                    }
                }
            }
        }
        // Reject malformed base tensors before a GPU can read invalid metadata.
        assert!(kernel
            .forward(
                &content,
                &qrel,
                &krel,
                &ci,
                &pi,
                &pair.to_dtype(DType::U32)?,
                scale
            )
            .is_err());
        assert!(kernel
            .forward(
                &content,
                &qrel,
                &krel,
                &ci.narrow(2, 0, t - 1)?,
                &pi,
                &pair,
                scale
            )
            .is_err());
        Ok(())
    }
}
