//! Prepared kernels shared per [`goldy::Runtime`].
//!
//! Modules record dispatches into a [`goldy::Scheme`]. Exchanges stay on the
//! submitting root; these functions only bind retained parcels.

use crate::decode_attn::{
    KvCacheStoreKernel, RelativeAttentionCombineKernel, RelativeAttentionKernel,
    RelativeAttentionPartialKernel,
};
use crate::decode_ops::{
    DynamicShortConvKernel, HeadL2ClipKernel, ShortConvKernel, SigmoidGateKernel, SoftcapKernel,
    SoftcappedSwiGluKernel,
};
use crate::depth::{DepthCopyRowsKernel, DepthMixKernel, DepthStoreKernel};
use crate::kernels::{
    AttentionCombineKernel, AttentionPartialKernel, CacheStoreKernel, EmbedKernel,
    RmsnormInplaceKernel, RmsnormKernel, RmsnormUnweightedInplaceKernel, RmsnormUnweightedKernel,
    RopeStoreKernel, SwigluKernel, TensorKernels, ATTENTION_MAX_HEAD, ATTENTION_MAX_SPLITS,
    ATTENTION_SPLIT, DEFAULT_ROPE_THETA,
};
use crate::moe::MoeTopkKernel;
use anyhow::Context;
use goldy::{
    Buffer, GoldyError, Runtime, Scheme, Tensor, TensorDType, TensorLayout, TensorScalar,
    TensorShape, TensorView,
};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Prepared kernels for one runtime. Used while recording; schemes intern the pipelines.
pub(crate) struct Blocks {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    rope_store: RopeStoreKernel,
    cache_store: CacheStoreKernel,
    attention_partial: AttentionPartialKernel,
    attention_combine: AttentionCombineKernel,
    swiglu: SwigluKernel,
    tensors: TensorKernels,
    rmsnorm_unweighted: RmsnormUnweightedKernel,
    rmsnorm_unweighted_inplace: RmsnormUnweightedInplaceKernel,
    head_l2_clip: HeadL2ClipKernel,
    softcap: SoftcapKernel,
    sigmoid_gate: SigmoidGateKernel,
    softcapped_swiglu: SoftcappedSwiGluKernel,
    short_conv: ShortConvKernel,
    dynamic_short_conv: DynamicShortConvKernel,
    runtime: Runtime,
    kv_cache_store: KvCacheStoreKernel,
    /// Serial reference. The recorded path is the split-K pair.
    #[allow(dead_code)]
    relative_attention: RelativeAttentionKernel,
    relative_attention_partial: RelativeAttentionPartialKernel,
    relative_attention_combine: RelativeAttentionCombineKernel,
    moe_topk: MoeTopkKernel,
    depth_store: DepthStoreKernel,
    depth_copy_rows: DepthCopyRowsKernel,
    depth_mix: DepthMixKernel,
}

/// Activation and weight views for one attention block.
pub(crate) struct AttentionSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub xb2: TensorView<'a>,
    pub q: TensorView<'a>,
    /// Value projection scratch, at least as wide as the key projection.
    pub v: TensorView<'a>,
    /// `[q_heads, attention_splits(seq), head + 2]` flash-decoding scratch.
    pub partial: TensorView<'a>,
    pub norm: NormScratch<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub step: &'a Buffer,
    pub rms: TensorView<'a>,
    pub wq: TensorView<'a>,
    pub wk: TensorView<'a>,
    pub wv: TensorView<'a>,
    pub wo: TensorView<'a>,
    pub eps: f32,
}

/// Activation and weight views for one SwiGLU feed-forward block.
pub(crate) struct FfnSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub hb: TensorView<'a>,
    pub hb2: TensorView<'a>,
    /// `[hidden]` gate scratch for SwiGLU recorded as tensor operations.
    pub gate: TensorView<'a>,
    pub norm: NormScratch<'a>,
    pub rms: TensorView<'a>,
    pub w1: TensorView<'a>,
    pub w2: TensorView<'a>,
    pub w3: TensorView<'a>,
    pub eps: f32,
}

/// Scratch for RMSNorm recorded as tensor operations: the squares, `[dim]`, and
/// their mean and scale, `[1]`.
#[derive(Clone, Copy)]
pub(crate) struct NormScratch<'a> {
    pub square: TensorView<'a>,
    pub scale: TensorView<'a>,
}

impl Blocks {
    pub(crate) fn shared(runtime: &Runtime) -> anyhow::Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let mut cache = cache.lock().expect("blocks cache");
        let key = runtime.substrate_ptr() as usize;
        cache.retain(|(_, weak)| weak.strong_count() > 0);
        if let Some((_, weak)) = cache.iter().find(|(k, _)| *k == key) {
            if let Some(blocks) = weak.upgrade() {
                return Ok(blocks);
            }
        }
        let blocks = Arc::new(Self::prepare(runtime)?);
        cache.push((key, Arc::downgrade(&blocks)));
        Ok(blocks)
    }

    fn prepare(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            embed: EmbedKernel::prepare(runtime).context("prepare embed kernel")?,
            rmsnorm: RmsnormKernel::prepare(runtime).context("prepare rmsnorm kernel")?,
            rmsnorm_inplace: RmsnormInplaceKernel::prepare(runtime)
                .context("prepare rmsnorm_inplace kernel")?,
            rope_store: RopeStoreKernel::prepare(runtime).context("prepare rope_store kernel")?,
            cache_store: CacheStoreKernel::prepare(runtime)
                .context("prepare cache_store kernel")?,
            attention_partial: AttentionPartialKernel::prepare(runtime)
                .context("prepare attention_partial kernel")?,
            attention_combine: AttentionCombineKernel::prepare(runtime)
                .context("prepare attention_combine kernel")?,
            swiglu: SwigluKernel::prepare(runtime).context("prepare swiglu kernel")?,
            tensors: TensorKernels::prepare(runtime).context("prepare tensor kernels")?,
            rmsnorm_unweighted: RmsnormUnweightedKernel::prepare(runtime)
                .context("prepare rmsnorm_unweighted kernel")?,
            rmsnorm_unweighted_inplace: RmsnormUnweightedInplaceKernel::prepare(runtime)
                .context("prepare rmsnorm_unweighted_inplace kernel")?,
            head_l2_clip: HeadL2ClipKernel::prepare(runtime)
                .context("prepare head_l2_clip kernel")?,
            softcap: SoftcapKernel::prepare(runtime).context("prepare softcap kernel")?,
            sigmoid_gate: SigmoidGateKernel::prepare(runtime)
                .context("prepare sigmoid_gate kernel")?,
            softcapped_swiglu: SoftcappedSwiGluKernel::prepare(runtime)
                .context("prepare softcapped_swiglu kernel")?,
            short_conv: ShortConvKernel::prepare(runtime).context("prepare short_conv kernel")?,
            dynamic_short_conv: DynamicShortConvKernel::prepare(runtime)
                .context("prepare dynamic_short_conv kernel")?,
            runtime: runtime.clone(),
            kv_cache_store: KvCacheStoreKernel::prepare(runtime)
                .context("prepare kv_cache_store kernel")?,
            relative_attention: RelativeAttentionKernel::prepare(runtime)
                .context("prepare relative_attention kernel")?,
            relative_attention_partial: RelativeAttentionPartialKernel::prepare(runtime)
                .context("prepare relative_attention_partial kernel")?,
            relative_attention_combine: RelativeAttentionCombineKernel::prepare(runtime)
                .context("prepare relative_attention_combine kernel")?,
            moe_topk: MoeTopkKernel::prepare(runtime).context("prepare moe_topk kernel")?,
            depth_store: DepthStoreKernel::prepare(runtime)
                .context("prepare depth_store kernel")?,
            depth_copy_rows: DepthCopyRowsKernel::prepare(runtime)
                .context("prepare depth_copy_rows kernel")?,
            depth_mix: DepthMixKernel::prepare(runtime).context("prepare depth_mix kernel")?,
        })
    }

    /// Token-row gather into `x`.
    pub(crate) fn record_embed(
        &self,
        scheme: &mut Scheme,
        table: TensorView<'_>,
        step: &Buffer,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.embed
            .record(scheme, "embed", table, step, x)?
            .over_tensor(&x);
        Ok(())
    }

    /// RMSNorm into `out`, described as tensor operations. The kernel sums the mean
    /// square across 256 lanes, so the description is not exact and fuses only where
    /// contractions may reassociate.
    ///
    /// `weight` is the gain. `None` is the unweighted form (`out = x * rsqrt(mean(x²) + eps)`).
    fn norm(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: Option<TensorView<'_>>,
        out: TensorView<'_>,
        scratch: NormScratch<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        let node = match weight {
            Some(weight) => self
                .rmsnorm
                .record(scheme, "rmsnorm", x, weight, out, eps)?
                .groups([1, 1, 1])
                .node(),
            None => self
                .rmsnorm_unweighted
                .record(scheme, "rmsnorm", x, out, eps)?
                .groups([1, 1, 1])
                .node(),
        };
        let NormScratch { square, scale } = scratch;
        self.tensors.recorder(scheme).describe(node, false, |rec| {
            rec.mul_into("rms_square", x, x, square)?;
            rec.mean_into("rms_mean", square, 0, scale)?;
            rec.add_scalar_into("rms_eps", scale, eps, scale)?;
            rec.sqrt_into("rms_sqrt", scale, scale)?;
            rec.reciprocal_into("rms_scale", scale, scale)?;
            rec.mul_into("rms_normalize", scale, x, out)?;
            if let Some(weight) = weight {
                rec.mul_into("rms_weight", weight, out, out)?;
            }
            Ok(())
        })
    }

    /// `hb *= silu(hb) * hb2`, described as tensor operations in the kernel's order
    /// through `gate`, so it fuses into the gate and up projections.
    fn record_swiglu(
        &self,
        scheme: &mut Scheme,
        hb: TensorView<'_>,
        hb2: TensorView<'_>,
        gate: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let node = self
            .swiglu
            .record(scheme, "swiglu", hb, hb2)?
            .over_tensor(&hb)
            .node();
        self.tensors.recorder(scheme).describe(node, true, |rec| {
            rec.neg_into("silu_neg", hb, gate)?;
            rec.exp_into("silu_exp", gate, gate)?;
            rec.add_scalar_into("silu_denominator", gate, 1.0, gate)?;
            rec.reciprocal_into("silu_reciprocal", gate, gate)?;
            rec.mul_into("silu", hb, gate, gate)?;
            rec.mul_into("swiglu", gate, hb2, hb)
        })
    }

    /// RMSNorm, Q/K/V, RoPE, attention, output projection, residual into `x`.
    ///
    /// The three projections of the normalized input are recorded together: `wk` into
    /// the prefix of `xb2`, `wv` into `v`. [`RopeStoreKernel`] rotates Q in place and
    /// writes RoPE(K) into the cache. [`CacheStoreKernel`] copies V into the cache.
    /// `wo` reuses all of `xb2` after those stores.
    pub(crate) fn record_attention(
        &self,
        scheme: &mut Scheme,
        sites: AttentionSites<'_>,
    ) -> Result<(), GoldyError> {
        let head = sites.key.shape().dim(2)?;
        let q_heads = sites.q.shape().dim(0)?;
        let q = sites.q;
        let q_flat = q.reshape(&[q.numel_u32()])?;
        let xb_heads = sites.xb.reshape(&[q_heads, head])?;
        let kv_dim = sites.wk.shape().dim(0)?;
        if sites.wv.shape().dim(0)? != kv_dim {
            return Err(GoldyError::Validation(
                "attention: wk and wv row counts differ".into(),
            ));
        }
        if kv_dim > q.numel_u32() || kv_dim % 2 != 0 {
            return Err(GoldyError::Validation(
                "attention: kv width must be even and no wider than q".into(),
            ));
        }
        if sites.xb2.numel_u32() < kv_dim || sites.v.numel_u32() < kv_dim {
            return Err(GoldyError::Validation(
                "attention: xb2 or v is shorter than the kv projection".into(),
            ));
        }
        if head > ATTENTION_MAX_HEAD {
            return Err(GoldyError::Validation(format!(
                "attention: head size {head} exceeds {ATTENTION_MAX_HEAD}"
            )));
        }
        let splits = attention_splits(sites.key.shape().dim(0)?);
        let partial_dims = [q_heads, splits, head + 2];
        if sites.partial.shape().dims() != partial_dims {
            return Err(GoldyError::Validation(format!(
                "attention: partial scratch must be {partial_dims:?}"
            )));
        }
        let proj = sites.xb2.narrow(0, 0, kv_dim)?;
        let v = sites.v.narrow(0, 0, kv_dim)?;

        self.norm(
            scheme,
            sites.x,
            Some(sites.rms),
            sites.xb,
            sites.norm,
            sites.eps,
        )?;
        self.tensors
            .matmul_into(scheme, "wq", sites.wq, sites.xb, q_flat)?;
        self.tensors
            .matmul_into(scheme, "wk", sites.wk, sites.xb, proj)?;
        self.tensors
            .matmul_into(scheme, "wv", sites.wv, sites.xb, v)?;
        self.rope_store
            .record(
                scheme,
                "rope",
                q,
                proj,
                sites.key,
                sites.step,
                DEFAULT_ROPE_THETA,
            )?
            .over_1d((q.numel_u32() / 2).max(1));
        self.cache_store
            .record(scheme, "wv_store", v, sites.value, sites.step)?
            .over_1d(kv_dim);
        self.attention_partial
            .record(
                scheme,
                "attn_partial",
                q,
                sites.key,
                sites.value,
                sites.partial,
                sites.step,
            )?
            .groups([q_heads * splits, 1, 1]);
        self.attention_combine
            .record(scheme, "attn_combine", sites.partial, xb_heads, sites.step)?
            .groups([q_heads, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wo", sites.wo, sites.xb, sites.xb2)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb2, sites.x)?;
        Ok(())
    }

    /// RMSNorm, SwiGLU, down projection, residual into `x`.
    pub(crate) fn record_ffn(
        &self,
        scheme: &mut Scheme,
        sites: FfnSites<'_>,
    ) -> Result<(), GoldyError> {
        self.norm(
            scheme,
            sites.x,
            Some(sites.rms),
            sites.xb,
            sites.norm,
            sites.eps,
        )?;
        self.tensors
            .matmul_into(scheme, "w1", sites.w1, sites.xb, sites.hb)?;
        self.tensors
            .matmul_into(scheme, "w3", sites.w3, sites.xb, sites.hb2)?;
        self.record_swiglu(scheme, sites.hb, sites.hb2, sites.gate)?;
        self.tensors
            .matmul_into(scheme, "w2", sites.w2, sites.hb, sites.xb)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb, sites.x)?;
        Ok(())
    }

    pub(crate) fn record_rmsnorm(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        self.record_rms_algebra(scheme, x, Some(weight), out, eps)
    }

    pub(crate) fn record_rmsnorm_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        let tmp = self.vector(x.numel_u32())?;
        self.record_rms_algebra(scheme, x, Some(weight), tmp.view(), eps)?;
        self.tensors
            .recorder(scheme)
            .copy("rms_inplace", tmp.view(), x)?;
        scheme.keep_alive(tmp);
        Ok(())
    }

    pub(crate) fn record_rmsnorm_unweighted(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        self.record_rms_algebra(scheme, x, None, out, eps)
    }

    pub(crate) fn record_rmsnorm_unweighted_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        let tmp = self.vector(x.numel_u32())?;
        self.record_rms_algebra(scheme, x, None, tmp.view(), eps)?;
        self.tensors
            .recorder(scheme)
            .copy("rms_inplace", tmp.view(), x)?;
        scheme.keep_alive(tmp);
        Ok(())
    }

    /// `out = x * rsqrt(mean(x²) + eps) * weight`. Tensor algebra, so fusion can
    /// see it. `weight` is `None` for the unweighted form.
    fn record_rms_algebra(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: Option<TensorView<'_>>,
        out: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        let square = self.vector(x.numel_u32())?;
        let scale = self.vector(1)?;
        let mut rec = self.tensors.recorder(scheme);
        rec.mul_into("rms_square", x, x, square.view())?;
        rec.mean_into("rms_mean", square.view(), 0, scale.view())?;
        rec.add_scalar_into("rms_eps", scale.view(), eps, scale.view())?;
        rec.sqrt_into("rms_sqrt", scale.view(), scale.view())?;
        rec.reciprocal_into("rms_scale", scale.view(), scale.view())?;
        rec.mul_into("rms_normalize", scale.view(), x, out)?;
        if let Some(weight) = weight {
            rec.mul_into("rms_weight", weight, out, out)?;
        }
        scheme.keep_alive(square);
        scheme.keep_alive(scale);
        Ok(())
    }

    pub(crate) fn record_linear(
        &self,
        scheme: &mut Scheme,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors.matmul_into(scheme, label, weight, x, out)
    }

    pub(crate) fn record_linear_bf16(
        &self,
        scheme: &mut Scheme,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors
            .recorder(scheme)
            .matmul_bf16_into(label, weight, x, out)
    }

    pub(crate) fn record_nvfp4_indexed(
        &self,
        scheme: &mut Scheme,
        label: &str,
        codes: TensorView<'_>,
        scales: TensorView<'_>,
        factors: TensorView<'_>,
        ids: TensorView<'_>,
        slot: u32,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors.recorder(scheme).matmul_block_scaled_into(
            label,
            goldy::BlockScaled {
                codes,
                scales,
                elems: goldy::ops::PackedElem::E2M1,
                scale: goldy::ops::PackedScale::E4M3,
                block: [1, 16],
                global: Some(factors),
            },
            x,
            Some((ids, slot)),
            out,
        )
    }

    pub(crate) fn record_head_l2_clip(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        scale: TensorView<'_>,
        out: TensorView<'_>,
        step: &Buffer,
        threshold: f32,
        alpha: f32,
        floor: f32,
        use_scale: u32,
    ) -> Result<(), GoldyError> {
        let heads = x.shape().dim(0)?;
        if heads == 0 {
            return Err(GoldyError::Validation(
                "l2 clip: heads must be non-zero".into(),
            ));
        }
        let node = self
            .head_l2_clip
            .record(
                scheme, "l2_clip", x, scale, out, step, threshold, alpha, floor, use_scale,
            )?
            .groups([heads, 1, 1])
            .node();
        // The log-position factor reads the position as a number, which no site states.
        if alpha != 0.0 {
            return Ok(());
        }
        let scaled = Tensor::zeros(&self.runtime, x.shape(), TensorDType::F32)?;
        let square = Tensor::zeros(&self.runtime, x.shape(), TensorDType::F32)?;
        let factor = Tensor::zeros(
            &self.runtime,
            TensorShape::matrix(heads, 1),
            TensorDType::F32,
        )?;
        // The kernel sums each head across 256 lanes, so the sequential sum is not its order.
        self.tensors.recorder(scheme).describe(node, false, |rec| {
            let y = if use_scale != 0 {
                rec.mul_into("l2_scale", x, scale, scaled.view())?;
                scaled.view()
            } else {
                x
            };
            let f = factor.view();
            rec.mul_into("l2_square", y, y, square.view())?;
            rec.sum_into("l2_sum", square.view(), 1, f)?;
            rec.sqrt_into("l2_norm", f, f)?;
            rec.add_scalar_into("l2_eps", f, crate::decode_ops::L2_CLIP_NORM_EPS, f)?;
            rec.reciprocal_into("l2_inverse", f, f)?;
            rec.mul_scalar_into("l2_ratio", f, threshold, f)?;
            rec.min_scalar_into("l2_factor", f, 1.0, f)?;
            rec.mul_into("l2_clip", y, f, out)
        })?;
        for scratch in [scaled, square, factor] {
            scheme.keep_alive(scratch);
        }
        Ok(())
    }

    pub(crate) fn record_softcap(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
        cap: f32,
    ) -> Result<(), GoldyError> {
        // `cap == 1` is one tanh. Goldy owns that identity.
        let kept = self
            .tensors
            .recorder(scheme)
            .tanh_scaled_into("softcap", x, cap, out)?;
        for tensor in kept {
            scheme.keep_alive(tensor);
        }
        Ok(())
    }

    pub(crate) fn record_sigmoid_gate(
        &self,
        scheme: &mut Scheme,
        values: TensorView<'_>,
        logits: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let mut rec = self.tensors.recorder(scheme);
        let gate = rec.sigmoid("sigmoid_gate", logits)?;
        rec.mul_into("sigmoid_gate_mul", values, gate.view(), values)?;
        scheme.keep_alive(gate);
        Ok(())
    }

    pub(crate) fn record_softcapped_swiglu(
        &self,
        scheme: &mut Scheme,
        gate: TensorView<'_>,
        up: TensorView<'_>,
        out: TensorView<'_>,
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<(), GoldyError> {
        // Scratch takes the gate's shape, so a `[slots, cols]` gate stays one op each.
        let scratch = || Tensor::zeros(&self.runtime, gate.shape(), TensorDType::F32);
        let g = scratch()?;
        let u = scratch()?;
        let s = scratch()?;
        self.record_softcap(scheme, gate, g.view(), gate_cap)?;
        self.record_softcap(scheme, up, u.view(), up_cap)?;
        let mut rec = self.tensors.recorder(scheme);
        let sig = rec.sigmoid("swiglu_sigmoid", gate)?;
        rec.mul_into("swiglu_sig", g.view(), sig.view(), s.view())?;
        rec.mul_into("swiglu", s.view(), u.view(), out)?;
        for tensor in [g, u, s, sig] {
            scheme.keep_alive(tensor);
        }
        Ok(())
    }

    pub(crate) fn record_short_conv(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        state: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.short_conv
            .record(scheme, "short_conv", x, weight, state, out)?
            .over_tensor(&x);
        Ok(())
    }

    pub(crate) fn record_dynamic_short_conv(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        static_weight: TensorView<'_>,
        taps: TensorView<'_>,
        state: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.dynamic_short_conv
            .record(
                scheme,
                "dynamic_short_conv",
                x,
                static_weight,
                taps,
                state,
                out,
            )?
            .over_tensor(&x);
        Ok(())
    }

    /// Causal convolution over the cyclic window `state` at position `step`, as
    /// tensor operations: the row dot, the skip connection, and the scale.
    pub(crate) fn record_short_conv_at(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        state: TensorView<'_>,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let channels = x.shape().dim(0)?;
        let kernel = state.shape().dim(1)?;
        let position = TensorView::new(
            step,
            TensorLayout::packed(TensorDType::U32, TensorShape::vector(1), 1)?,
        )?;
        let dot = self.vector(channels)?;
        let summed = self.vector(channels)?;
        let mut rec = self.tensors.recorder(scheme);
        rec.cyclic_dot_at("short_conv", x, weight, state, position, dot.view())?;
        rec.add_into("short_conv_skip", dot.view(), x, summed.view())?;
        rec.div_scalar_into(
            "short_conv_scale",
            summed.view(),
            ((kernel as f32) + 1.0).sqrt(),
            out,
        )?;
        scheme.keep_alive(dot);
        scheme.keep_alive(summed);
        Ok(())
    }

    /// Dynamic causal convolution over the cyclic window `state` at position `step`,
    /// as tensor operations. Weight column `kernel - 1 - lag` is
    /// `static_weight[lag] + taps[head, lag]`, so the row dot reads the newest
    /// value against tap 0, the same lag order as the authored kernel.
    pub(crate) fn record_dynamic_short_conv_at(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        static_weight: TensorView<'_>,
        taps: TensorView<'_>,
        state: TensorView<'_>,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let channels = x.shape().dim(0)?;
        let kernel = state.shape().dim(1)?;
        let heads = taps.shape().dim(0)?;
        if static_weight.shape().dims() != [kernel, channels]
            || taps.shape().dims() != [heads, kernel]
            || heads == 0
            || channels % heads != 0
        {
            return Err(GoldyError::Validation(
                "dynamic conv: static weight is [kernel, channels] and taps [heads, kernel]".into(),
            ));
        }
        let head_dim = channels / heads;
        let position = TensorView::new(
            step,
            TensorLayout::packed(TensorDType::U32, TensorShape::vector(1), 1)?,
        )?;
        // Tap-major, so each tap's sum is one contiguous block.
        let weight = Tensor::zeros(
            &self.runtime,
            TensorShape::from_dims(&[kernel, heads, head_dim])?,
            TensorDType::F32,
        )?;
        let rows = [heads, head_dim];
        let row_shape = TensorShape::from_dims(&rows)?;
        let dot = Tensor::zeros(&self.runtime, row_shape, TensorDType::F32)?;
        let summed = Tensor::zeros(&self.runtime, row_shape, TensorDType::F32)?;
        let x = x.reshape(&rows)?;
        let state = state.reshape(&[heads, head_dim, kernel])?;
        let out = out.reshape(&rows)?;
        let mut rec = self.tensors.recorder(scheme);
        rec.cyclic_store_at("short_conv_store", x, state, position)?;
        for lag in 0..kernel {
            let column = weight
                .view()
                .narrow(0, kernel - 1 - lag, 1)?
                .reshape(&rows)?;
            let fixed = static_weight.narrow(0, lag, 1)?.reshape(&rows)?;
            let tap = taps.narrow(1, lag, 1)?;
            rec.add_into("short_conv_weight", fixed, tap, column)?;
        }
        let window = weight.view().permute(&[1, 2, 0])?;
        rec.cyclic_window_dot_at("short_conv_dot", window, state, position, dot.view())?;
        rec.add_into("short_conv_skip", dot.view(), x, summed.view())?;
        rec.div_scalar_into(
            "short_conv_scale",
            summed.view(),
            (2.0 * (kernel as f32) + 1.0).sqrt(),
            out,
        )?;
        for scratch in [weight, dot, summed] {
            scheme.keep_alive(scratch);
        }
        Ok(())
    }

    /// `out = weight[ids[slot]] @ x`. The returned slab must stay alive; it is
    /// retained on this `Blocks`.
    pub(crate) fn record_add(
        &self,
        scheme: &mut Scheme,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors.add_into(scheme, label, a, b, out)
    }

    pub(crate) fn record_indexed_matmul(
        &self,
        scheme: &mut Scheme,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        ids: TensorView<'_>,
        slot: u32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors
            .recorder(scheme)
            .matmul_indexed_into(label, weight, x, ids, slot, out)
    }

    /// `count` ids from `slot`. `out` is `[count * rows]`. `x_stride == 0` shares `x`.
    pub(crate) fn record_indexed_matmul_batch(
        &self,
        scheme: &mut Scheme,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        ids: TensorView<'_>,
        slot: u32,
        count: u32,
        x_stride: u32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors
            .recorder(scheme)
            .matmul_indexed_batch_into(label, weight, x, ids, slot, count, x_stride, out)
    }

    fn vector(&self, len: u32) -> Result<Tensor, GoldyError> {
        Tensor::zeros(
            &self.runtime,
            TensorShape::vector(len.max(1)),
            TensorDType::F32,
        )
    }

    pub(crate) fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub(crate) fn tensor_recorder<'a>(
        &'a self,
        scheme: &'a mut Scheme,
    ) -> goldy::TensorRecorder<'a> {
        self.tensors.recorder(scheme)
    }

    pub(crate) fn record_kv_cache_store(
        &self,
        scheme: &mut Scheme,
        k: TensorView<'_>,
        v: TensorView<'_>,
        key_cache: TensorView<'_>,
        value_cache: TensorView<'_>,
        step: &Buffer,
        window: u32,
    ) -> Result<(), GoldyError> {
        let node = self
            .kv_cache_store
            .record(
                scheme,
                "kv_cache_store",
                k,
                v,
                key_cache,
                value_cache,
                step,
                window,
            )?
            .over_tensor(&k)
            .node();
        // A full cache skips positions past its allocation, which a cyclic store would wrap.
        if window == 0 {
            return Ok(());
        }
        let position = TensorView::new(
            step,
            TensorLayout::packed(TensorDType::U32, TensorShape::vector(1), 1)?,
        )?;
        let keys = key_cache.permute(&[1, 2, 0])?;
        let values = value_cache.permute(&[1, 2, 0])?;
        self.tensors.recorder(scheme).describe(node, true, |rec| {
            rec.cyclic_store_at("kv_store_key", k, keys, position)?;
            rec.cyclic_store_at("kv_store_value", v, values, position)
        })
    }

    /// Serial reference for [`Self::record_relative_attention_partial`]. Not used
    /// on the recorded path.
    #[allow(dead_code)]
    pub(crate) fn record_relative_attention(
        &self,
        scheme: &mut Scheme,
        q: TensorView<'_>,
        key_cache: TensorView<'_>,
        value_cache: TensorView<'_>,
        rel: TensorView<'_>,
        sink: TensorView<'_>,
        out: TensorView<'_>,
        step: &Buffer,
        scale: f32,
        window: u32,
    ) -> Result<(), GoldyError> {
        let q_heads = q.shape().dim(0)?;
        self.relative_attention
            .record(
                scheme,
                "relative_attention",
                q,
                key_cache,
                value_cache,
                rel,
                sink,
                out,
                step,
                scale,
                window,
            )?
            .groups([q_heads, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_relative_attention_partial(
        &self,
        scheme: &mut Scheme,
        q: TensorView<'_>,
        key_cache: TensorView<'_>,
        value_cache: TensorView<'_>,
        rel: TensorView<'_>,
        partial: TensorView<'_>,
        step: &Buffer,
        scale: f32,
        window: u32,
    ) -> Result<(), GoldyError> {
        let q_heads = q.shape().dim(0)?;
        let splits = partial.shape().dim(1)?;
        self.relative_attention_partial
            .record(
                scheme,
                "relative_attention_partial",
                q,
                key_cache,
                value_cache,
                rel,
                partial,
                step,
                scale,
                window,
            )?
            .groups([q_heads * splits, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_relative_attention_combine(
        &self,
        scheme: &mut Scheme,
        partial: TensorView<'_>,
        sink: TensorView<'_>,
        out: TensorView<'_>,
        step: &Buffer,
        window: u32,
        capacity: u32,
    ) -> Result<(), GoldyError> {
        let q_heads = out.shape().dim(0)?;
        self.relative_attention_combine
            .record(
                scheme,
                "relative_attention_combine",
                partial,
                sink,
                out,
                step,
                window,
                capacity,
            )?
            .groups([q_heads, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_moe_topk(
        &self,
        scheme: &mut Scheme,
        logits: TensorView<'_>,
        bias: TensorView<'_>,
        ids: TensorView<'_>,
        weights: TensorView<'_>,
        route_scale: f32,
    ) -> Result<(), GoldyError> {
        self.moe_topk
            .record(scheme, "moe_topk", logits, bias, ids, weights, route_scale)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_depth_store(
        &self,
        scheme: &mut Scheme,
        src: TensorView<'_>,
        tape: TensorView<'_>,
        slot: u32,
    ) -> Result<(), GoldyError> {
        self.depth_store
            .record(scheme, "depth_store", src, tape, slot)?
            .over_tensor(&src);
        Ok(())
    }

    pub(crate) fn record_depth_copy_rows(
        &self,
        scheme: &mut Scheme,
        src: TensorView<'_>,
        dst: TensorView<'_>,
        src_slot: u32,
        n: u32,
    ) -> Result<(), GoldyError> {
        let hidden = src.shape().dim(1)?;
        self.depth_copy_rows
            .record(scheme, "depth_copy", src, dst, src_slot, n)?
            .over_1d(n.saturating_mul(hidden).max(1));
        Ok(())
    }

    pub(crate) fn record_depth_mix(
        &self,
        scheme: &mut Scheme,
        blocks: TensorView<'_>,
        partial: TensorView<'_>,
        queries: TensorView<'_>,
        out: TensorView<'_>,
        n_blocks: u32,
        eps: f32,
        scale: f32,
    ) -> Result<(), GoldyError> {
        let n_queries = queries.shape().dim(0)?;
        if n_queries == 0 {
            return Err(GoldyError::Validation(
                "depth mix needs at least one query".into(),
            ));
        }
        if n_blocks + 1 > crate::depth::DEPTH_MAX_KEYS {
            return Err(GoldyError::Validation(
                "depth mix has more keys than DEPTH_MAX_KEYS".into(),
            ));
        }
        self.depth_mix
            .record(
                scheme,
                "depth_mix",
                blocks,
                partial,
                queries,
                out,
                n_blocks,
                eps,
                scale,
            )?
            .groups([n_queries, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_zero(
        &self,
        scheme: &mut Scheme,
        label: &str,
        tensor: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors
            .recorder(scheme)
            .fill(label, tensor, TensorScalar::F32(0.0))
    }
}

/// Flash-decoding splits covering `seq_len` cached positions.
pub(crate) fn attention_splits(seq_len: u32) -> u32 {
    seq_len
        .div_ceil(ATTENTION_SPLIT)
        .max(1)
        .min(ATTENTION_MAX_SPLITS)
}

static CACHE: OnceLock<Mutex<Vec<(usize, Weak<Blocks>)>>> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_splits_cap_a_long_context() {
        assert_eq!(attention_splits(8), 1);
        assert_eq!(attention_splits(64), 8);
        assert_eq!(attention_splits(256), 32);
        assert_eq!(attention_splits(8_192), 32);
    }

    use goldy::{Tensor, TensorDType, TensorShape};

    #[test]
    fn fused_unweighted_rmsnorm_uses_configured_eps() {
        let runtime = crate::gpu::create_runtime().expect("goldy runtime");
        let ctx = runtime.create_context().unwrap();
        let blocks = Blocks::prepare(&runtime).unwrap();
        let eps = 1.0e-2_f32;
        let x_host = [0.25_f32, -0.5, 0.75, 0.1];
        let x = Tensor::from_f32(&runtime, TensorShape::vector(4), &x_host).unwrap();
        let out = Tensor::zeros(&runtime, TensorShape::vector(4), TensorDType::F32).unwrap();
        let square = Tensor::zeros(&runtime, TensorShape::vector(4), TensorDType::F32).unwrap();
        let scale = Tensor::zeros(&runtime, TensorShape::vector(1), TensorDType::F32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        scheme.set_automatic_fusion(true);
        blocks
            .norm(
                &mut scheme,
                x.view(),
                None,
                out.view(),
                NormScratch {
                    square: square.view(),
                    scale: scale.view(),
                },
                eps,
            )
            .unwrap();
        let mut sub = scheme.submit().unwrap();
        let bytes = (&mut sub >> out.buffer()).take::<u8>().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let mean = x_host
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / x_host.len() as f64;
        let inv = 1.0 / (mean + f64::from(eps)).sqrt();
        for (i, v) in x_host.iter().enumerate() {
            let want = (f64::from(*v) * inv) as f32;
            let err = (got[i] - want).abs();
            let tol = 1e-5 + 1e-4 * want.abs();
            assert!(
                err <= tol,
                "unweighted fused [{i}] = {}, reference {want}, err {err}",
                got[i]
            );
        }
    }
}
