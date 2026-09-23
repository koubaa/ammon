//! Recordable inference modules. They own scratch tensors and prepared kernels;
//! the caller supplies activations, weights, and persistent cache. Modules never
//! submit schemes or bind exchanges.

use crate::blocks::{AttentionSites, Blocks, FfnSites};
use goldy::{
    Buffer, GoldyError, Runtime, Scheme, SchemeLabel, Tensor, TensorDType, TensorShape, TensorView,
};
use std::sync::Arc;

/// Attention projection and pre-norm weights. Scratch stays on [`CausalSelfAttention`].
#[derive(Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub norm: TensorView<'a>,
    pub query: TensorView<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub output: TensorView<'a>,
}

/// SwiGLU feed-forward weights. Scratch stays on [`SwiGluMlp`].
#[derive(Clone, Copy)]
pub struct SwiGluWeights<'a> {
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

    pub fn layer(&self, layer: usize) -> Result<KvLayer<'_>, GoldyError> {
        let layer = u32::try_from(layer).map_err(|_| {
            GoldyError::Validation(format!("kv cache layer {layer} does not fit in u32"))
        })?;
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

/// Causal multi-head attention with owned decode scratch (`xb`, `xb2`, `q`, `att`).
pub struct CausalSelfAttention {
    blocks: Arc<Blocks>,
    xb: Tensor,
    xb2: Tensor,
    q: Tensor,
    att: Tensor,
}

impl CausalSelfAttention {
    pub fn new(
        runtime: &Runtime,
        hidden: u32,
        query_heads: u32,
        seq_len: u32,
    ) -> anyhow::Result<Self> {
        Self::with_blocks(
            runtime,
            Arc::new(Blocks::prepare(runtime)?),
            hidden,
            query_heads,
            seq_len,
        )
    }

    pub fn with_blocks(
        runtime: &Runtime,
        blocks: Arc<Blocks>,
        hidden: u32,
        query_heads: u32,
        seq_len: u32,
    ) -> anyhow::Result<Self> {
        if query_heads == 0 || hidden % query_heads != 0 {
            anyhow::bail!("hidden {hidden} is not divisible by query_heads {query_heads}");
        }
        let head = hidden / query_heads;
        Ok(Self {
            blocks,
            xb: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            xb2: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            q: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[query_heads, head])?,
                TensorDType::F32,
            )?,
            att: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[query_heads, seq_len])?,
                TensorDType::F32,
            )?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weights: AttentionWeights<'_>,
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
                att: self.att.view(),
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
        weights: AttentionWeights<'_>,
        cache: KvLayer<'_>,
        step: &Buffer,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record(scheme, x, weights, cache, step))?;
        Ok(())
    }
}

/// Pre-norm SwiGLU MLP with owned decode scratch (`xb`, `hb`, `hb2`).
pub struct SwiGluMlp {
    blocks: Arc<Blocks>,
    xb: Tensor,
    hb: Tensor,
    hb2: Tensor,
}

impl SwiGluMlp {
    pub fn new(runtime: &Runtime, hidden: u32, intermediate: u32) -> anyhow::Result<Self> {
        Self::with_blocks(
            runtime,
            Arc::new(Blocks::prepare(runtime)?),
            hidden,
            intermediate,
        )
    }

    pub fn with_blocks(
        runtime: &Runtime,
        blocks: Arc<Blocks>,
        hidden: u32,
        intermediate: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            blocks,
            xb: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            hb: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            hb2: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weights: SwiGluWeights<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_ffn(
            scheme,
            FfnSites {
                x,
                xb: self.xb.view(),
                hb: self.hb.view(),
                hb2: self.hb2.view(),
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
        weights: SwiGluWeights<'_>,
    ) -> Result<(), GoldyError> {
        worker.group(label, |scheme| self.record(scheme, x, weights))?;
        Ok(())
    }
}
