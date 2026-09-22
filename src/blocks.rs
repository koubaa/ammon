//! Decoder blocks recorded as includable schemes.
//!
//! Each function writes dispatches into a child [`goldy::Scheme`]. The architecture
//! crate [`Scheme::include`]s that child into the worker. Exchanges stay on the root:
//! these schemes only bind retained parcels.

use crate::kernels::{
    AttentionKernel, EmbedKernel, GemvKernel, RmsnormInplaceKernel, RmsnormKernel, RopeKernel,
    SwigluKernel, TensorKernels, DEFAULT_ROPE_THETA,
};
use anyhow::{Context, Result};
use goldy::{Buffer, Scheme, TensorView};

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
pub struct AttentionSites<'a> {
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
pub struct FfnSites<'a> {
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
    pub fn prepare(runtime: &goldy::Runtime) -> Result<Self> {
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
    ) -> Result<()> {
        self.embed
            .record(scheme, "embed", table, step, x)?
            .over_tensor(&x);
        Ok(())
    }

    /// RMSNorm, Q/K/V, RoPE, attention, output projection, residual into `x`.
    ///
    /// `key` and `value` are one layer, `[seq, kv_heads, head]`. The cache GEMV
    /// sees that storage as `[seq, kv_dim]`; RoPE and attention keep the head view.
    /// `q` and `att` are packed vectors, reshaped from the key-cache head size.
    pub fn record_attention(
        &self,
        scheme: &mut Scheme,
        sites: AttentionSites<'_>,
    ) -> Result<()> {
        let head = sites
            .key
            .shape()
            .dim(2)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let seq_len = sites
            .key
            .shape()
            .dim(0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let q = as_heads(sites.q, head)?;
        let xb_heads = as_heads(sites.xb, head)?;
        let att = as_heads(sites.att, seq_len)?;
        let kv_width = sites
            .wk
            .shape()
            .dim(0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let wv_rows = sites
            .wv
            .shape()
            .dim(0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let n_q_heads = q.shape().dim(0).map_err(|e| anyhow::anyhow!("{e}"))?;
        let key_rows = cache_rows(sites.key)?;
        let value_rows = cache_rows(sites.value)?;

        self.rmsnorm
            .record(scheme, "rmsnorm", sites.x, sites.rms, sites.xb)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wq", sites.wq, sites.xb, sites.q)?;
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
                att,
                xb_heads,
                sites.key,
                sites.value,
                sites.step,
            )?
            .groups([n_q_heads, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wo", sites.wo, sites.xb, sites.xb2)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb2, sites.x)?;
        Ok(())
    }

    /// RMSNorm, SwiGLU, down projection, residual into `x`.
    pub fn record_ffn(&self, scheme: &mut Scheme, sites: FfnSites<'_>) -> Result<()> {
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
    ) -> Result<()> {
        self.rmsnorm_inplace
            .record(scheme, "rmsnorm", x, rms_final)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "classifier", classifier, x, logits)?;
        Ok(())
    }
}

/// `[seq, kv_heads, head]` packed as the GEMV destination `[seq, kv_dim]`.
fn cache_rows(cache: TensorView<'_>) -> Result<TensorView<'_>> {
    let seq = cache.shape().dim(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let kv_dim = cache.numel_u32() / seq;
    cache
        .reshape(&[seq, kv_dim])
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn as_heads(packed: TensorView<'_>, inner: u32) -> Result<TensorView<'_>> {
    anyhow::ensure!(inner > 0, "head or sequence extent must be non-zero");
    let n = packed.numel_u32() / inner;
    packed
        .reshape(&[n, inner])
        .map_err(|e| anyhow::anyhow!("{e}"))
}
