//! Recordable inference modules. They own scratch tensors and prepared kernels;
//! the caller supplies activations, weights, and persistent cache. Modules never
//! submit schemes or bind exchanges.

use crate::blocks::{attention_splits, AttentionSites, Blocks, FfnSites, NormScratch};
use goldy::{
    Buffer, GoldyError, Runtime, Scheme, SchemeLabel, Tensor, TensorDType, TensorShape, TensorView,
};
use std::sync::Arc;

/// Pre-norm causal attention block weights. Scratch stays on [`CausalAttentionBlock`].
#[derive(Clone, Copy)]
pub struct CausalAttentionBlockWeights<'a> {
    pub norm: TensorView<'a>,
    pub query: TensorView<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub output: TensorView<'a>,
}

/// Pre-norm SwiGLU block weights. Scratch stays on [`SwiGluBlock`].
#[derive(Clone, Copy)]
pub struct SwiGluBlockWeights<'a> {
    pub norm: TensorView<'a>,
    pub gate: TensorView<'a>,
    pub up: TensorView<'a>,
    pub down: TensorView<'a>,
}

/// One layer of a causal KV cache: `[seq, kv_heads, head]`.
#[derive(Clone, Copy)]
pub struct KvLayer<'a> {
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
}

/// Persistent key/value cache: `[layers, seq, kv_heads, head]`.
pub struct KvCache {
    key: Tensor,
    value: Tensor,
}

impl KvCache {
    pub fn new(
        runtime: &Runtime,
        layers: u32,
        seq_len: u32,
        kv_heads: u32,
        head_size: u32,
    ) -> Result<Self, GoldyError> {
        let shape = TensorShape::from_dims(&[layers, seq_len, kv_heads, head_size])?;
        Ok(Self {
            key: Tensor::zeros(runtime, shape, TensorDType::F32)?,
            value: Tensor::zeros(runtime, shape, TensorDType::F32)?,
        })
    }

    pub fn layer(&self, layer: u32) -> Result<KvLayer<'_>, GoldyError> {
        Ok(KvLayer {
            key: squeeze_layer(self.key.view(), layer)?,
            value: squeeze_layer(self.value.view(), layer)?,
        })
    }
}

fn squeeze_layer(cache: TensorView<'_>, layer: u32) -> Result<TensorView<'_>, GoldyError> {
    let seq = cache.shape().dim(1)?;
    let kv_heads = cache.shape().dim(2)?;
    let head = cache.shape().dim(3)?;
    cache.narrow(0, layer, 1)?.reshape(&[seq, kv_heads, head])
}

fn blocks(runtime: &Runtime) -> anyhow::Result<Arc<Blocks>> {
    Blocks::shared(runtime)
}

/// Scratch of an RMSNorm recorded as tensor operations over `[hidden]`.
struct Norm {
    square: Tensor,
    scale: Tensor,
}

impl Norm {
    fn new(runtime: &Runtime, hidden: u32) -> Result<Self, GoldyError> {
        Ok(Self {
            square: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            scale: Tensor::zeros(runtime, TensorShape::vector(1), TensorDType::F32)?,
        })
    }

    fn sites(&self) -> NormScratch<'_> {
        NormScratch {
            square: self.square.view(),
            scale: self.scale.view(),
        }
    }
}

/// Token embedding table gather. No decode scratch.
pub struct Embedding {
    blocks: Arc<Blocks>,
}

impl Embedding {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        table: TensorView<'_>,
        step: &Buffer,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_embed(scheme, table, step, x)
    }

    pub fn record_group(
        &self,
        worker: &mut Scheme,
        label: impl Into<SchemeLabel>,
        table: TensorView<'_>,
        step: &Buffer,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record(scheme, table, step, x))?;
        Ok(())
    }
}

/// RMSNorm. Out-of-place writes `out`; [`Self::record_inplace`] overwrites `x`.
pub struct RmsNorm {
    blocks: Arc<Blocks>,
}

impl RmsNorm {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_rmsnorm(scheme, x, weight, out)
    }

    pub fn record_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_rmsnorm_inplace(scheme, x, weight)
    }
}

/// Dense projection: `out = weight @ x`.
pub struct Linear {
    blocks: Arc<Blocks>,
}

impl Linear {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_linear(scheme, "linear", weight, x, out)
    }
}

/// Pre-norm causal attention block: RMSNorm, QKV, RoPE, attention, projection, residual.
///
/// Owns decode scratch (`xb`, `xb2`, `q`, `v`, flash-decoding `partial`).
pub struct CausalAttentionBlock {
    blocks: Arc<Blocks>,
    xb: Tensor,
    xb2: Tensor,
    q: Tensor,
    v: Tensor,
    partial: Tensor,
    norm: Norm,
}

impl CausalAttentionBlock {
    pub fn new(
        runtime: &Runtime,
        hidden: u32,
        query_heads: u32,
        seq_len: u32,
    ) -> anyhow::Result<Self> {
        if query_heads == 0 || hidden % query_heads != 0 {
            anyhow::bail!("hidden {hidden} is not divisible by query_heads {query_heads}");
        }
        let head = hidden / query_heads;
        Ok(Self {
            blocks: blocks(runtime)?,
            xb: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            xb2: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            q: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[query_heads, head])?,
                TensorDType::F32,
            )?,
            v: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            partial: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[query_heads, attention_splits(seq_len), head + 2])?,
                TensorDType::F32,
            )?,
            norm: Norm::new(runtime, hidden)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weights: CausalAttentionBlockWeights<'_>,
        cache: KvLayer<'_>,
        step: &Buffer,
    ) -> Result<(), GoldyError> {
        self.blocks.record_attention(
            scheme,
            AttentionSites {
                x,
                xb: self.xb.view(),
                xb2: self.xb2.view(),
                q: self.q.view(),
                v: self.v.view(),
                partial: self.partial.view(),
                norm: self.norm.sites(),
                key: cache.key,
                value: cache.value,
                step,
                rms: weights.norm,
                wq: weights.query,
                wk: weights.key,
                wv: weights.value,
                wo: weights.output,
            },
        )
    }

    pub fn record_group(
        &self,
        worker: &mut Scheme,
        label: impl Into<SchemeLabel>,
        x: TensorView<'_>,
        weights: CausalAttentionBlockWeights<'_>,
        cache: KvLayer<'_>,
        step: &Buffer,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record(scheme, x, weights, cache, step))?;
        Ok(())
    }
}

/// Pre-norm SwiGLU block: RMSNorm, gated MLP, residual.
///
/// Owns decode scratch (`xb`, `hb`, `hb2`, `gate`).
pub struct SwiGluBlock {
    blocks: Arc<Blocks>,
    xb: Tensor,
    hb: Tensor,
    hb2: Tensor,
    gate: Tensor,
    norm: Norm,
}

impl SwiGluBlock {
    pub fn new(runtime: &Runtime, hidden: u32, intermediate: u32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            xb: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            hb: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            hb2: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            gate: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            norm: Norm::new(runtime, hidden)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weights: SwiGluBlockWeights<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_ffn(
            scheme,
            FfnSites {
                x,
                xb: self.xb.view(),
                hb: self.hb.view(),
                hb2: self.hb2.view(),
                gate: self.gate.view(),
                norm: self.norm.sites(),
                rms: weights.norm,
                w1: weights.gate,
                w2: weights.down,
                w3: weights.up,
            },
        )
    }

    pub fn record_group(
        &self,
        worker: &mut Scheme,
        label: impl Into<SchemeLabel>,
        x: TensorView<'_>,
        weights: SwiGluBlockWeights<'_>,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record(scheme, x, weights))?;
        Ok(())
    }
}
