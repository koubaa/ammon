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

pub use crate::kernels::DEFAULT_RMS_EPS;

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
///
/// `out = weight * x * rsqrt(mean(x²) + eps)` with [`DEFAULT_RMS_EPS`] unless
/// [`Self::with_eps`] says otherwise. [`Self::record_unweighted`] omits the gain.
/// These records dispatch the RMSNorm kernel. Pre-norm blocks can lower the same
/// formula as tensor ops when `AMMON_SEMANTIC_NORM=1` and the scheme fuses automatically.
pub struct RmsNorm {
    blocks: Arc<Blocks>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Self::with_eps(runtime, DEFAULT_RMS_EPS)
    }

    pub fn with_eps(runtime: &Runtime, eps: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            eps,
        })
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_rmsnorm(scheme, x, weight, out, self.eps)
    }

    pub fn record_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks
            .record_rmsnorm_inplace(scheme, x, weight, self.eps)
    }

    /// `out = x * rsqrt(mean(x²) + eps)`, with no gain vector.
    pub fn record_unweighted(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks
            .record_rmsnorm_unweighted(scheme, x, out, self.eps)
    }

    /// In-place form of [`Self::record_unweighted`].
    pub fn record_unweighted_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks
            .record_rmsnorm_unweighted_inplace(scheme, x, self.eps)
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
    eps: f32,
}

impl CausalAttentionBlock {
    pub fn new(
        runtime: &Runtime,
        hidden: u32,
        query_heads: u32,
        seq_len: u32,
    ) -> anyhow::Result<Self> {
        Self::with_eps(runtime, hidden, query_heads, seq_len, DEFAULT_RMS_EPS)
    }

    pub fn with_eps(
        runtime: &Runtime,
        hidden: u32,
        query_heads: u32,
        seq_len: u32,
        eps: f32,
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
            eps,
        })
    }

    pub fn eps(&self) -> f32 {
        self.eps
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
                eps: self.eps,
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
    eps: f32,
}

impl SwiGluBlock {
    pub fn new(runtime: &Runtime, hidden: u32, intermediate: u32) -> anyhow::Result<Self> {
        Self::with_eps(runtime, hidden, intermediate, DEFAULT_RMS_EPS)
    }

    pub fn with_eps(
        runtime: &Runtime,
        hidden: u32,
        intermediate: u32,
        eps: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            xb: Tensor::zeros(runtime, TensorShape::vector(hidden), TensorDType::F32)?,
            hb: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            hb2: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            gate: Tensor::zeros(runtime, TensorShape::vector(intermediate), TensorDType::F32)?,
            norm: Norm::new(runtime, hidden)?,
            eps,
        })
    }

    pub fn eps(&self) -> f32 {
        self.eps
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
                eps: self.eps,
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

/// Log-position multiplier `1 + alpha * log(max((position + 1) / floor, 1))`.
///
/// `position` is [`crate::kernels::DecodeStep::position`]. `alpha == 0` disables it.
#[derive(Clone, Copy, Debug)]
pub struct LogPositionScale {
    pub alpha: f32,
    pub floor: f32,
}

/// Per-head L2 clip of a `[heads, dim]` vector.
///
/// Optional `scale` is `[heads, dim]` and is applied before the norm. Optional
/// [`LogPositionScale`] multiplies the clipped vector. The norm epsilon is
/// [`crate::kernels::L2_CLIP_NORM_EPS`] (`1e-6`), not an RMSNorm epsilon.
///
/// `y = s * min(1, threshold / (||s||_2 + 1e-6))` with `s = x` or `s = x * scale`,
/// then `y *= log_scale(position)` when a log scale is set.
pub struct HeadL2Clip {
    blocks: Arc<Blocks>,
    threshold: f32,
    log: Option<LogPositionScale>,
}

impl HeadL2Clip {
    pub fn new(runtime: &Runtime, threshold: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            threshold,
            log: None,
        })
    }

    pub fn with_log_scale(
        runtime: &Runtime,
        threshold: f32,
        log: LogPositionScale,
    ) -> anyhow::Result<Self> {
        if log.alpha != 0.0 && log.floor <= 0.0 {
            anyhow::bail!("log-position floor must be positive");
        }
        Ok(Self {
            blocks: blocks(runtime)?,
            threshold,
            log: Some(log),
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// `scale` is the per-element gain. `None` clips `x` directly; the unused
    /// scale binding is `x` itself and is not read. `step` is read only when
    /// this clip has a log-position scale. Otherwise any one-element
    /// [`crate::kernels::DecodeStep`] buffer is enough.
    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        scale: Option<TensorView<'_>>,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if x.shape().rank() != 2 || out.shape().dims() != x.shape().dims() {
            return Err(GoldyError::Validation(
                "l2 clip: x and out must be matching rank-2 [heads, dim] tensors".into(),
            ));
        }
        let (scale, use_scale) = match scale {
            Some(scale) => {
                if scale.shape().dims() != x.shape().dims() {
                    return Err(GoldyError::Validation("l2 clip: scale must match x".into()));
                }
                (scale, 1u32)
            }
            None => (x, 0u32),
        };
        let (alpha, floor) = match self.log {
            Some(log) => (log.alpha, log.floor),
            None => (0.0, 1.0),
        };
        self.blocks.record_head_l2_clip(
            scheme,
            x,
            scale,
            out,
            step,
            self.threshold,
            alpha,
            floor,
            use_scale,
        )
    }
}

/// `out = cap * tanh(x / cap)`.
pub struct Softcap {
    blocks: Arc<Blocks>,
    cap: f32,
}

impl Softcap {
    pub fn new(runtime: &Runtime, cap: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            cap,
        })
    }

    pub fn cap(&self) -> f32 {
        self.cap
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if out.shape().dims() != x.shape().dims() {
            return Err(GoldyError::Validation("softcap: out must match x".into()));
        }
        self.blocks.record_softcap(scheme, x, out, self.cap)
    }
}

/// In-place `values *= sigmoid(logits)`.
pub struct SigmoidGate {
    blocks: Arc<Blocks>,
}

impl SigmoidGate {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        values: TensorView<'_>,
        logits: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if logits.shape().dims() != values.shape().dims() {
            return Err(GoldyError::Validation(
                "sigmoid gate: logits must match values".into(),
            ));
        }
        self.blocks.record_sigmoid_gate(scheme, values, logits)
    }
}

/// `out = softcap(gate) * sigmoid(gate) * softcap(up)`.
///
/// `sigmoid` sees the raw gate. `gate_cap` bounds only the linear gate factor.
pub struct SoftcappedSwiGlu {
    blocks: Arc<Blocks>,
    gate_cap: f32,
    up_cap: f32,
}

impl SoftcappedSwiGlu {
    pub fn new(runtime: &Runtime, gate_cap: f32, up_cap: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            gate_cap,
            up_cap,
        })
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        gate: TensorView<'_>,
        up: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if up.shape().dims() != gate.shape().dims() || out.shape().dims() != gate.shape().dims() {
            return Err(GoldyError::Validation(
                "softcapped swiglu: gate, up, and out must match".into(),
            ));
        }
        self.blocks
            .record_softcapped_swiglu(scheme, gate, up, out, self.gate_cap, self.up_cap)
    }
}

fn require_conv_shape(channels: u32, kernel_size: u32) -> anyhow::Result<()> {
    if channels == 0 || kernel_size == 0 {
        anyhow::bail!("short conv needs a positive channel count and kernel size");
    }
    Ok(())
}

/// Static one-token causal depthwise convolution.
///
/// Owns `[channels, kernel]` of running state, oldest sample first. `weight` is
/// `[channels, kernel]` in conv1d order (the last tap multiplies the current
/// token). Each [`Self::record`] consumes one token and updates the state.
/// `out = (conv + x) / sqrt(kernel + 1)`. [`Self::record_reset`] zeroes the state.
///
/// One token per [`Self::record`]. A prefill is that many records in order.
/// The state lives on this module; it is not snapshotted or branched.
pub struct ShortConv {
    blocks: Arc<Blocks>,
    state: Tensor,
    channels: u32,
    kernel_size: u32,
}

impl ShortConv {
    pub fn new(runtime: &Runtime, channels: u32, kernel_size: u32) -> anyhow::Result<Self> {
        require_conv_shape(channels, kernel_size)?;
        Ok(Self {
            blocks: blocks(runtime)?,
            state: Tensor::zeros(
                runtime,
                TensorShape::matrix(channels, kernel_size),
                TensorDType::F32,
            )?,
            channels,
            kernel_size,
        })
    }

    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.blocks
            .record_zero(scheme, "conv_reset", self.state.view())
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if x.shape().dims() != [self.channels]
            || out.shape().dims() != [self.channels]
            || weight.shape().dims() != [self.channels, self.kernel_size]
        {
            return Err(GoldyError::Validation(
                "short conv: x and out are [channels], weight is [channels, kernel]".into(),
            ));
        }
        self.blocks
            .record_short_conv(scheme, x, weight, self.state.view(), out)
    }
}

/// Dynamic one-token causal depthwise convolution.
///
/// Owns the same running state as [`ShortConv`], plus tap scratch. `static_weight`
/// is `[kernel, channels]` with tap 0 on the current token. `proj` is
/// `[heads * kernel, input]` and maps `dynamic_input` to per-head taps shared by
/// that head's channels. `out = (conv + x) / sqrt(2 * kernel + 1)`.
///
/// One token per [`Self::record`], same as [`ShortConv`].
pub struct DynamicShortConv {
    blocks: Arc<Blocks>,
    state: Tensor,
    taps: Tensor,
    channels: u32,
    num_heads: u32,
    kernel_size: u32,
}

impl DynamicShortConv {
    pub fn new(
        runtime: &Runtime,
        channels: u32,
        num_heads: u32,
        kernel_size: u32,
    ) -> anyhow::Result<Self> {
        require_conv_shape(channels, kernel_size)?;
        if num_heads == 0 || channels % num_heads != 0 {
            anyhow::bail!(
                "dynamic short conv channels {channels} must divide into {num_heads} heads"
            );
        }
        Ok(Self {
            blocks: blocks(runtime)?,
            state: Tensor::zeros(
                runtime,
                TensorShape::matrix(channels, kernel_size),
                TensorDType::F32,
            )?,
            taps: Tensor::zeros(
                runtime,
                TensorShape::matrix(num_heads, kernel_size),
                TensorDType::F32,
            )?,
            channels,
            num_heads,
            kernel_size,
        })
    }

    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.blocks
            .record_zero(scheme, "conv_reset", self.state.view())
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        dynamic_input: TensorView<'_>,
        static_weight: TensorView<'_>,
        proj: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let tap_len = self.num_heads * self.kernel_size;
        if x.shape().dims() != [self.channels]
            || out.shape().dims() != [self.channels]
            || static_weight.shape().dims() != [self.kernel_size, self.channels]
            || proj.shape().rank() != 2
            || proj.shape().dim(0)? != tap_len
            || proj.shape().dim(1)? != dynamic_input.shape().dim(0)?
        {
            return Err(GoldyError::Validation(
                "dynamic short conv: x [channels], static_weight [kernel, channels], proj [heads * kernel, input]"
                    .into(),
            ));
        }
        let flat = self.taps.view().reshape(&[tap_len])?;
        self.blocks
            .record_linear(scheme, "conv_taps", proj, dynamic_input, flat)?;
        self.blocks.record_dynamic_short_conv(
            scheme,
            x,
            static_weight,
            self.taps.view(),
            self.state.view(),
            out,
        )
    }
}
