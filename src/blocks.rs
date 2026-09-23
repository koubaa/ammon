//! Decoder blocks recorded as includable schemes.
//!
//! Each function writes dispatches into a grouped child [`goldy::Scheme`]. Exchanges
//! stay on the submitting root; these blocks only bind retained parcels.

use crate::kernels::{
    AttentionKernel, EmbedKernel, GemvKernel, RmsnormInplaceKernel, RmsnormKernel, RopeKernel,
    SwigluKernel, TensorKernels, DEFAULT_ROPE_THETA,
};
use anyhow::Context;
use goldy::{Buffer, GoldyError, Scheme, SchemeLabel, TensorView};

/// Prepared kernels for one runtime. Used while recording; schemes intern the pipelines.
pub struct Blocks {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    gemv: GemvKernel,
    rope: RopeKernel,
    attention: AttentionKernel,
    swiglu: SwigluKernel,
    tensors: TensorKernels,
}

/// Activation and weight views for one attention block.
pub(crate) struct AttentionSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub xb2: TensorView<'a>,
    pub q: TensorView<'a>,
    pub att: TensorView<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub step: &'a Buffer,
    pub rms: TensorView<'a>,
    pub wq: TensorView<'a>,
    pub wk: TensorView<'a>,
    pub wv: TensorView<'a>,
    pub wo: TensorView<'a>,
}

/// Activation and weight views for one SwiGLU feed-forward block.
pub(crate) struct FfnSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub hb: TensorView<'a>,
    pub hb2: TensorView<'a>,
    pub rms: TensorView<'a>,
    pub w1: TensorView<'a>,
    pub w2: TensorView<'a>,
    pub w3: TensorView<'a>,
}

impl Blocks {
    pub fn prepare(runtime: &goldy::Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            embed: EmbedKernel::prepare(runtime).context("prepare embed kernel")?,
            rmsnorm: RmsnormKernel::prepare(runtime).context("prepare rmsnorm kernel")?,
            rmsnorm_inplace: RmsnormInplaceKernel::prepare(runtime)
                .context("prepare rmsnorm_inplace kernel")?,
            gemv: GemvKernel::prepare(runtime).context("prepare gemv kernel")?,
            rope: RopeKernel::prepare(runtime).context("prepare rope kernel")?,
            attention: AttentionKernel::prepare(runtime).context("prepare attention kernel")?,
            swiglu: SwigluKernel::prepare(runtime).context("prepare swiglu kernel")?,
            tensors: TensorKernels::prepare(runtime).context("prepare tensor kernels")?,
        })
    }

    /// Token-row gather into `x`.
    pub fn record_embed(
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

    pub fn record_embed_group(
        &self,
        worker: &mut Scheme,
        label: impl Into<SchemeLabel>,
        table: TensorView<'_>,
        step: &Buffer,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record_embed(scheme, table, step, x))?;
        Ok(())
    }

    /// RMSNorm, Q/K/V, RoPE, attention, output projection, residual into `x`.
    ///
    /// `key` and `value` are one layer, `[seq, kv_heads, head]`. The cache GEMV
    /// sees that storage as `[seq, kv_dim]`; RoPE and attention keep the head view.
    /// `q` is `[q_heads, head]` and `att` is `[q_heads, seq]`; only the projection
    /// destination and cache-writer views are flattened for GEMV.
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
        let kv_width = sites.wk.shape().dim(0)?;
        let wv_rows = sites.wv.shape().dim(0)?;
        let key_rows = cache_rows(sites.key)?;
        let value_rows = cache_rows(sites.value)?;

        self.rmsnorm
            .record(scheme, "rmsnorm", sites.x, sites.rms, sites.xb)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wq", sites.wq, sites.xb, q_flat)?;
        self.gemv
            .record(scheme, "wk", sites.xb, sites.wk, key_rows, sites.step)?
            .over_1d(kv_width);
        self.gemv
            .record(scheme, "wv", sites.xb, sites.wv, value_rows, sites.step)?
            .over_1d(wv_rows);
        self.rope
            .record(scheme, "rope", q, sites.key, sites.step, DEFAULT_ROPE_THETA)?
            .over_1d((q.numel_u32() / 2).max(1));
        self.attention
            .record(
                scheme,
                "attn",
                q,
                sites.att,
                xb_heads,
                sites.key,
                sites.value,
                sites.step,
            )?
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
        self.rmsnorm
            .record(scheme, "rmsnorm", sites.x, sites.rms, sites.xb)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "w1", sites.w1, sites.xb, sites.hb)?;
        self.tensors
            .matmul_into(scheme, "w3", sites.w3, sites.xb, sites.hb2)?;
        self.swiglu
            .record(scheme, "swiglu", sites.hb, sites.hb2)?
            .over_tensor(&sites.hb);
        self.tensors
            .matmul_into(scheme, "w2", sites.w2, sites.hb, sites.xb)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb, sites.x)?;
        Ok(())
    }

    /// Final RMSNorm and classifier GEMV into `logits`.
    pub fn record_logits(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        rms_final: TensorView<'_>,
        classifier: TensorView<'_>,
        logits: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.rmsnorm_inplace
            .record(scheme, "rmsnorm", x, rms_final)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "classifier", classifier, x, logits)?;
        Ok(())
    }

    pub fn record_logits_group(
        &self,
        worker: &mut Scheme,
        label: impl Into<SchemeLabel>,
        x: TensorView<'_>,
        rms_final: TensorView<'_>,
        classifier: TensorView<'_>,
        logits: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| {
            self.record_logits(scheme, x, rms_final, classifier, logits)
        })?;
        Ok(())
    }
}

/// `[seq, kv_heads, head]` packed as the GEMV destination `[seq, kv_dim]`.
fn cache_rows(cache: TensorView<'_>) -> Result<TensorView<'_>, GoldyError> {
    let seq = cache.shape().dim(0)?;
    let kv_dim = cache.numel_u32() / seq;
    cache.reshape(&[seq, kv_dim])
}
