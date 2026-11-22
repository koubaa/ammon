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
use crate::kernels::{
    AttentionCombineKernel, AttentionPartialKernel, CacheStoreKernel, EmbedKernel,
    RmsnormInplaceKernel, RmsnormKernel, RmsnormUnweightedInplaceKernel, RmsnormUnweightedKernel,
    RopeStoreKernel, SwigluKernel, TensorKernels, ATTENTION_MAX_HEAD, ATTENTION_SPLIT,
    DEFAULT_ROPE_THETA,
};
use anyhow::Context;
use goldy::{Buffer, GoldyError, Runtime, Scheme, TensorScalar, TensorView};
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
    kv_cache_store: KvCacheStoreKernel,
    /// Serial reference. The recorded path is the split-K pair.
    #[allow(dead_code)]
    relative_attention: RelativeAttentionKernel,
    relative_attention_partial: RelativeAttentionPartialKernel,
    relative_attention_combine: RelativeAttentionCombineKernel,
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
            kv_cache_store: KvCacheStoreKernel::prepare(runtime)
                .context("prepare kv_cache_store kernel")?,
            relative_attention: RelativeAttentionKernel::prepare(runtime)
                .context("prepare relative_attention kernel")?,
            relative_attention_partial: RelativeAttentionPartialKernel::prepare(runtime)
                .context("prepare relative_attention_partial kernel")?,
            relative_attention_combine: RelativeAttentionCombineKernel::prepare(runtime)
                .context("prepare relative_attention_combine kernel")?,
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

    /// RMSNorm into `out`. With `AMMON_SEMANTIC_NORM=1` under automatic fusion it is
    /// recorded as tensor operations, which fuse into the projections that read `out`.
    /// The mean square is then summed in order by one thread rather than across the
    /// kernel's 256 lanes, which is slower at every width Ammon runs.
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
        let semantic = std::env::var("AMMON_SEMANTIC_NORM").is_ok_and(|v| v == "1");
        if !(semantic && scheme.automatic_fusion()) {
            if let Some(weight) = weight {
                self.rmsnorm
                    .record(scheme, "rmsnorm", x, weight, out, eps)?
                    .groups([1, 1, 1]);
            } else {
                self.rmsnorm_unweighted
                    .record(scheme, "rmsnorm", x, out, eps)?
                    .groups([1, 1, 1]);
            }
            return Ok(());
        }
        let NormScratch { square, scale } = scratch;
        let mut rec = self.tensors.recorder(scheme);
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
    }

    /// `hb *= silu(hb) * hb2`. Under automatic fusion it is recorded as tensor
    /// operations in the kernel's order, which fuse into the gate and up projections.
    fn record_swiglu(
        &self,
        scheme: &mut Scheme,
        hb: TensorView<'_>,
        hb2: TensorView<'_>,
        gate: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if !scheme.automatic_fusion() {
            self.swiglu
                .record(scheme, "swiglu", hb, hb2)?
                .over_tensor(&hb);
            return Ok(());
        }
        let mut rec = self.tensors.recorder(scheme);
        rec.neg_into("silu_neg", hb, gate)?;
        rec.exp_into("silu_exp", gate, gate)?;
        rec.add_scalar_into("silu_denominator", gate, 1.0, gate)?;
        rec.reciprocal_into("silu_reciprocal", gate, gate)?;
        rec.mul_into("silu", hb, gate, gate)?;
        rec.mul_into("swiglu", gate, hb2, hb)
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
        self.rmsnorm
            .record(scheme, "rmsnorm", x, weight, out, eps)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_rmsnorm_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        self.rmsnorm_inplace
            .record(scheme, "rmsnorm", x, weight, eps)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_rmsnorm_unweighted(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        self.rmsnorm_unweighted
            .record(scheme, "rmsnorm", x, out, eps)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_rmsnorm_unweighted_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        eps: f32,
    ) -> Result<(), GoldyError> {
        self.rmsnorm_unweighted_inplace
            .record(scheme, "rmsnorm", x, eps)?
            .groups([1, 1, 1]);
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
        self.head_l2_clip
            .record(
                scheme, "l2_clip", x, scale, out, step, threshold, alpha, floor, use_scale,
            )?
            .groups([heads, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_softcap(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
        cap: f32,
    ) -> Result<(), GoldyError> {
        self.softcap
            .record(scheme, "softcap", x, out, cap)?
            .over_tensor(&x);
        Ok(())
    }

    pub(crate) fn record_sigmoid_gate(
        &self,
        scheme: &mut Scheme,
        values: TensorView<'_>,
        logits: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.sigmoid_gate
            .record(scheme, "sigmoid_gate", values, logits)?
            .over_tensor(&values);
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
        self.softcapped_swiglu
            .record(scheme, "softcapped_swiglu", gate, up, out, gate_cap, up_cap)?
            .over_tensor(&gate);
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
        self.kv_cache_store
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
            .over_tensor(&k);
        Ok(())
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
    seq_len.div_ceil(ATTENTION_SPLIT).max(1)
}

static CACHE: OnceLock<Mutex<Vec<(usize, Weak<Blocks>)>>> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use goldy::{Tensor, TensorDType, TensorShape};

    /// Clears `AMMON_SEMANTIC_NORM` on drop, including after a panic.
    struct SemanticNormFlag;

    impl SemanticNormFlag {
        fn enable() -> Self {
            // SAFETY: this lib test is the only writer, and the suite runs one test
            // at a time (`--test-threads=1`) so nothing else is reading the environment.
            unsafe { std::env::set_var("AMMON_SEMANTIC_NORM", "1") }
            Self
        }
    }

    impl Drop for SemanticNormFlag {
        fn drop(&mut self) {
            // SAFETY: same as `enable` — single-threaded test process.
            unsafe { std::env::remove_var("AMMON_SEMANTIC_NORM") }
        }
    }

    #[test]
    fn semantic_unweighted_rmsnorm_uses_configured_eps() {
        let _flag = SemanticNormFlag::enable();
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
                "unweighted semantic [{i}] = {}, reference {want}, err {err}",
                got[i]
            );
        }
    }
}
