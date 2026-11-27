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
/// It is recorded as tensor operations, which automatic fusion composes with the
/// operations around them.
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

    /// `out = weight @ x` for a packed BF16 `[rows, k]` weight. The weight is
    /// not expanded; the matvec promotes each element inside the dot.
    pub fn record_bf16(
        &self,
        scheme: &mut Scheme,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks
            .record_linear_bf16(scheme, "linear_bf16", weight, x, out)
    }
}

/// NVFP4 expert matrix: U8 codes `[experts, rows, k/2]`, F8_E4M3 scales
/// `[experts, rows, k/16]`, and f32 per-expert factors `[experts]`.
#[derive(Clone, Copy)]
pub struct PackedExperts<'a> {
    pub codes: TensorView<'a>,
    pub scales: TensorView<'a>,
    pub factors: TensorView<'a>,
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

/// `out = cap * tanh(x / cap)`, using `goldy::gpu::tanh`.
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

/// In-place `values *= sigmoid(logits)`, using `goldy::gpu::sigmoid`.
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
/// `sigmoid` sees the raw gate and is `goldy::gpu::sigmoid`. `gate_cap` bounds
/// only the linear gate factor. Both caps use `goldy::gpu::tanh`.
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

    /// Like [`Self::record`], but the ring slot is `DecodeStep.position`.
    ///
    /// Submitting the same position again rewrites that slot and leaves the
    /// older taps where they are.
    pub fn record_at(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        step: &Buffer,
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
        self.record_at_state(scheme, x, weight, self.state.view(), step, out)
    }

    /// Position-addressed conv into `state`, which a visit owns.
    pub fn record_at_state(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        state: TensorView<'_>,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if x.shape().dims() != [self.channels]
            || out.shape().dims() != [self.channels]
            || weight.shape().dims() != [self.channels, self.kernel_size]
            || state.shape().dims() != [self.channels, self.kernel_size]
        {
            return Err(GoldyError::Validation(
                "short conv: x, out, and state are [channels], weight is [channels, kernel]".into(),
            ));
        }
        self.blocks
            .record_fusable_short_conv_at(scheme, x, weight, state, step, out)
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

/// Which keys a [`DecodeAttention`] reads. Capacity lives on [`DecodeKvCache`].
///
/// `Sliding` attends the last `window` positions (fewer before the window
/// fills): distances `0 .. window - 1`. `Full` attends `0 ..= position` up to
/// the cache capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeAttentionKind {
    Sliding { window: u32 },
    Full,
}

/// Key/value cache for one decode attention layer: `[capacity, kv_heads, head]`.
pub struct DecodeKvCache {
    blocks: Arc<Blocks>,
    key: Tensor,
    value: Tensor,
    /// `0` for a full cache. Otherwise the sliding window.
    window: u32,
    capacity: u32,
    kv_heads: u32,
    head_dim: u32,
}

impl DecodeKvCache {
    /// Ring of at least `window` slots. The attended span is still `window`;
    /// extra slots let a later chunked write avoid overwriting keys the same
    /// submission still reads. `capacity == window` is the decode ring.
    pub fn sliding(
        runtime: &Runtime,
        window: u32,
        capacity: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<Self> {
        if window == 0 {
            anyhow::bail!("sliding KV cache window must be non-zero");
        }
        if capacity < window {
            anyhow::bail!("sliding KV capacity {capacity} is below the window {window}");
        }
        Self::allocate(runtime, window, capacity, kv_heads, head_dim)
    }

    /// Preallocated prefix. Rows `0 ..= position` are occupied, and `position`
    /// must stay below `capacity`.
    pub fn full(
        runtime: &Runtime,
        capacity: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<Self> {
        if capacity == 0 {
            anyhow::bail!("full KV cache capacity must be non-zero");
        }
        Self::allocate(runtime, 0, capacity, kv_heads, head_dim)
    }

    fn allocate(
        runtime: &Runtime,
        window: u32,
        capacity: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<Self> {
        if kv_heads == 0 || head_dim == 0 {
            anyhow::bail!("KV cache needs a positive head count and head size");
        }
        let shape = TensorShape::from_dims(&[capacity, kv_heads, head_dim])?;
        Ok(Self {
            blocks: blocks(runtime)?,
            key: Tensor::zeros(runtime, shape, TensorDType::F32)?,
            value: Tensor::zeros(runtime, shape, TensorDType::F32)?,
            window,
            capacity,
            kv_heads,
            head_dim,
        })
    }

    pub fn kind(&self) -> DecodeAttentionKind {
        if self.window == 0 {
            DecodeAttentionKind::Full
        } else {
            DecodeAttentionKind::Sliding {
                window: self.window,
            }
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    /// Zeros the stored keys and values.
    ///
    /// Not required between sequences. Which rows are read depends only on
    /// `DecodeStep.position`, and a new sequence rewrites every row it reads.
    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.blocks
            .record_zero(scheme, "kv_reset_k", self.key.view())?;
        self.blocks
            .record_zero(scheme, "kv_reset_v", self.value.view())
    }

    /// Deposit `DecodeStep { token, position }` after [`Self::check_position`].
    ///
    /// This is the host check for a full cache. `record` does not see the
    /// position, because a consumer records once and deposits a new step per
    /// token. The kernels still refuse a position past `capacity`: the store
    /// skips the write, and attention writes NaN instead of reading past the
    /// allocation.
    pub fn deposit_position(
        &self,
        deposit: &goldy::DepositTransaction,
        token: u32,
        position: u32,
    ) -> Result<(), GoldyError> {
        self.check_position(position)?;
        deposit.write_data(0, &[crate::kernels::DecodeStep { token, position }])
    }

    pub fn keys(&self) -> TensorView<'_> {
        self.key.view()
    }

    pub fn values(&self) -> TensorView<'_> {
        self.value.view()
    }

    /// Host check for the position about to be deposited into `DecodeStep`.
    ///
    /// `record` runs once and cannot see later deposits. A full cache position
    /// must be `< capacity`. The kernels also refuse to index past the
    /// allocation if a replayed step violates that.
    pub fn check_position(&self, position: u32) -> Result<(), GoldyError> {
        if self.window == 0 && position >= self.capacity {
            return Err(GoldyError::Validation(format!(
                "full KV position {position} is outside capacity {}",
                self.capacity
            )));
        }
        Ok(())
    }
}

/// Persistent state for one attention visit: KV plus both K and V convolution
/// states.
///
/// A model that visits a layer more than once needs one of these per visit. The
/// attention module keeps only scratch, so several visits can share it.
/// Zeroing the KV rows is optional; zeroing the convolution state is what
/// starts a sequence.
pub struct DecodeLayerState {
    kv: DecodeKvCache,
    k_conv: Tensor,
    v_conv: Tensor,
    partial: Tensor,
    kernel: u32,
    /// Post-attention and post-MLP conv rings. Present after [`Self::with_post_conv`].
    ///
    /// `partial` is `[q_heads, splits, head_dim + 2]`. `splits` is
    /// `min(32, attended.div_ceil(8))`. `attended` is the sliding window, or
    /// the full-cache capacity. Past 256 positions a split loops 8-token
    /// chunks, so the scratch does not grow with context. The extra two lanes
    /// are the running max and the sum of exp.
    post_attn: Option<Tensor>,
    post_mlp: Option<Tensor>,
}

impl DecodeLayerState {
    pub fn sliding(
        runtime: &Runtime,
        window: u32,
        capacity: u32,
        kv_heads: u32,
        head_dim: u32,
        q_heads: u32,
        kernel: u32,
    ) -> anyhow::Result<Self> {
        Self::from_kv(
            runtime,
            DecodeKvCache::sliding(runtime, window, capacity, kv_heads, head_dim)?,
            q_heads,
            kernel,
            window,
        )
    }

    pub fn full(
        runtime: &Runtime,
        capacity: u32,
        kv_heads: u32,
        head_dim: u32,
        q_heads: u32,
        kernel: u32,
    ) -> anyhow::Result<Self> {
        Self::from_kv(
            runtime,
            DecodeKvCache::full(runtime, capacity, kv_heads, head_dim)?,
            q_heads,
            kernel,
            capacity,
        )
    }

    fn from_kv(
        runtime: &Runtime,
        kv: DecodeKvCache,
        q_heads: u32,
        kernel: u32,
        attended: u32,
    ) -> anyhow::Result<Self> {
        if kernel == 0 || q_heads == 0 {
            anyhow::bail!("decode state needs a kernel and query heads");
        }
        let kv_dim = kv.kv_heads * kv.head_dim;
        let splits = crate::blocks::attention_splits(attended);
        let f = TensorDType::F32;
        Ok(Self {
            k_conv: Tensor::zeros(runtime, TensorShape::matrix(kv_dim, kernel), f)?,
            v_conv: Tensor::zeros(runtime, TensorShape::matrix(kv_dim, kernel), f)?,
            partial: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[q_heads, splits, kv.head_dim + 2])?,
                f,
            )?,
            kv,
            kernel,
            post_attn: None,
            post_mlp: None,
        })
    }

    /// Own the post-attention and post-MLP conv rings on this visit.
    ///
    /// A module shared across visits must not keep these itself: a later pass
    /// would otherwise continue the previous pass's history.
    pub fn with_post_conv(mut self, runtime: &Runtime, hidden: u32) -> anyhow::Result<Self> {
        let shape = TensorShape::matrix(hidden, self.kernel);
        let f = TensorDType::F32;
        self.post_attn = Some(Tensor::zeros(runtime, shape, f)?);
        self.post_mlp = Some(Tensor::zeros(runtime, shape, f)?);
        Ok(self)
    }

    pub fn post_attn(&self) -> Option<TensorView<'_>> {
        self.post_attn.as_ref().map(Tensor::view)
    }

    pub fn post_mlp(&self) -> Option<TensorView<'_>> {
        self.post_mlp.as_ref().map(Tensor::view)
    }

    pub fn kv(&self) -> &DecodeKvCache {
        &self.kv
    }

    pub fn deposit_position(
        &self,
        deposit: &goldy::DepositTransaction,
        token: u32,
        position: u32,
    ) -> Result<(), GoldyError> {
        self.kv.deposit_position(deposit, token, position)
    }

    pub fn check_position(&self, position: u32) -> Result<(), GoldyError> {
        self.kv.check_position(position)
    }

    pub fn keys(&self) -> TensorView<'_> {
        self.kv.keys()
    }

    /// Zeros the convolution state. KV rows are left alone: a new sequence
    /// rewrites every key it reads.
    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.kv
            .blocks
            .record_zero(scheme, "k_conv_reset", self.k_conv.view())?;
        self.kv
            .blocks
            .record_zero(scheme, "v_conv_reset", self.v_conv.view())?;
        if let Some(state) = self.post_attn.as_ref() {
            self.kv
                .blocks
                .record_zero(scheme, "post_attn_reset", state.view())?;
        }
        if let Some(state) = self.post_mlp.as_ref() {
            self.kv
                .blocks
                .record_zero(scheme, "post_mlp_reset", state.view())?;
        }
        Ok(())
    }
}

/// Weights for one [`DecodeAttention`] step. Layout matches the bias-free
/// projections of the reference attention module.
pub struct DecodeAttentionWeights<'a> {
    pub wq: TensorView<'a>,
    pub wk: TensorView<'a>,
    pub wv: TensorView<'a>,
    pub wo: TensorView<'a>,
    pub wgate: TensorView<'a>,
    pub wsink: TensorView<'a>,
    pub wrel: TensorView<'a>,
    /// `[d_rel, rel_extent]`. Column `d` is the bias for distance `d`.
    pub rel_proj: TensorView<'a>,
    /// `[q_heads, head]` learned scale applied before the query L2 clip.
    pub q_scale: TensorView<'a>,
    /// `[q_heads, d_rel]` learned scale applied before the relative-state clip.
    pub rel_scale: TensorView<'a>,
    /// Static conv: `[kv_dim, kernel]`, last tap on the current token.
    /// Dynamic conv: `[kernel, kv_dim]`, tap 0 on the current token.
    pub k_conv: TensorView<'a>,
    pub v_conv: TensorView<'a>,
    /// Dynamic conv only: `[kv_heads * kernel, hidden]`.
    pub k_conv_proj: Option<TensorView<'a>>,
    pub v_conv_proj: Option<TensorView<'a>>,
}

/// Batch-1 decode attention: Q/K/relative L2 clip, V RMSNorm, short conv,
/// ring or full KV, relative bias, per-head sink, GQA, and a sigmoid output gate.
///
/// Query, key, and value streams are separate. A sliding layer passes the same
/// tensor three times. Full layers log-scale queries and relative states when
/// `log_scale` is set; keys are clipped with no scale and no log. The value
/// norm is unweighted RMSNorm over each head, with `rms_eps`. Head size divides
/// 128 and is at most 128.
///
/// Per-visit KV and convolution state is a [`DecodeLayerState`]. This module
/// holds scratch only, so one module can serve every visit. The position lives
/// in `step`. Call [`DecodeLayerState::deposit_position`] where that step is
/// deposited. A full cache that is replayed past its capacity does not read or
/// write outside the allocation, and the attention output is NaN.
pub struct DecodeAttention {
    blocks: Arc<Blocks>,
    linear: Linear,
    q_clip: HeadL2Clip,
    k_clip: HeadL2Clip,
    rel_clip: HeadL2Clip,
    v_norm: RmsNorm,
    gate: SigmoidGate,
    dynamic: bool,
    taps: Tensor,
    kernel: u32,
    kind: DecodeAttentionKind,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    hidden: u32,
    d_rel: u32,
    rel_extent: u32,
    scale: f32,
    q: Tensor,
    k_proj: Tensor,
    k_conv: Tensor,
    k: Tensor,
    v_proj: Tensor,
    v_conv: Tensor,
    v: Tensor,
    rel_proj: Tensor,
    rel: Tensor,
    rel_logits: Tensor,
    sink: Tensor,
    attn: Tensor,
    gate_logits: Tensor,
}

impl DecodeAttention {
    pub fn new(
        runtime: &Runtime,
        kind: DecodeAttentionKind,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        hidden: u32,
        d_rel: u32,
        rel_extent: u32,
        conv_kernel: u32,
        dynamic_conv: bool,
        qk_l2_clip: f32,
        rel_l2_clip: f32,
        rms_eps: f32,
        log_scale: Option<LogPositionScale>,
    ) -> anyhow::Result<Self> {
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            anyhow::bail!(
                "query heads {q_heads} must be a positive multiple of kv heads {kv_heads}"
            );
        }
        if head_dim == 0 || head_dim > 128 || 128 % head_dim != 0 {
            anyhow::bail!(
                "decode attention head size {head_dim} must divide 128 and be at most 128"
            );
        }
        if hidden == 0 || d_rel == 0 || rel_extent == 0 {
            anyhow::bail!("hidden, d_rel, and rel_extent must be non-zero");
        }
        match kind {
            DecodeAttentionKind::Sliding { window } if window == 0 => {
                anyhow::bail!("sliding attention window must be non-zero");
            }
            DecodeAttentionKind::Sliding { window } if rel_extent != window => {
                anyhow::bail!(
                    "sliding attention rel_extent {rel_extent} must equal the window {window}"
                );
            }
            _ => {}
        }
        if conv_kernel == 0 {
            anyhow::bail!("decode attention convolution kernel must be non-zero");
        }
        let q_dim = q_heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let tap_rows = if dynamic_conv { kv_heads } else { 1 };
        let tap_cols = if dynamic_conv { conv_kernel } else { 1 };
        let q_clip = match log_scale {
            Some(log) => HeadL2Clip::with_log_scale(runtime, qk_l2_clip, log)?,
            None => HeadL2Clip::new(runtime, qk_l2_clip)?,
        };
        let rel_clip = match log_scale {
            Some(log) => HeadL2Clip::with_log_scale(runtime, rel_l2_clip, log)?,
            None => HeadL2Clip::new(runtime, rel_l2_clip)?,
        };
        let f = TensorDType::F32;
        let matrix = |rows, cols| Tensor::zeros(runtime, TensorShape::matrix(rows, cols), f);
        let vector = |len| Tensor::zeros(runtime, TensorShape::vector(len), f);
        Ok(Self {
            blocks: blocks(runtime)?,
            linear: Linear::new(runtime)?,
            q_clip,
            k_clip: HeadL2Clip::new(runtime, qk_l2_clip)?,
            rel_clip,
            v_norm: RmsNorm::with_eps(runtime, rms_eps)?,
            gate: SigmoidGate::new(runtime)?,
            dynamic: dynamic_conv,
            taps: matrix(tap_rows, tap_cols)?,
            kernel: conv_kernel,
            kind,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            d_rel,
            rel_extent,
            scale: (f64::from(head_dim)).powf(-0.5) as f32,
            q: matrix(q_heads, head_dim)?,
            k_proj: vector(kv_dim)?,
            k_conv: matrix(kv_heads, head_dim)?,
            k: matrix(kv_heads, head_dim)?,
            v_proj: vector(kv_dim)?,
            v_conv: matrix(kv_heads, head_dim)?,
            v: matrix(kv_heads, head_dim)?,
            rel_proj: matrix(q_heads, d_rel)?,
            rel: matrix(q_heads, d_rel)?,
            rel_logits: matrix(q_heads, rel_extent)?,
            sink: vector(q_heads)?,
            attn: matrix(q_heads, head_dim)?,
            gate_logits: vector(q_dim)?,
        })
    }

    pub fn kind(&self) -> DecodeAttentionKind {
        self.kind
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        q_hidden: TensorView<'_>,
        k_hidden: TensorView<'_>,
        v_hidden: TensorView<'_>,
        weights: DecodeAttentionWeights<'_>,
        state: &DecodeLayerState,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let cache = &state.kv;
        if cache.kind() != self.kind
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
            || state.kernel != self.kernel
        {
            return Err(GoldyError::Validation(
                "decode attention cache does not match this layer".into(),
            ));
        }
        let q_dim = self.q_heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        let hidden = [self.hidden];
        if q_hidden.shape().dims() != hidden
            || k_hidden.shape().dims() != hidden
            || v_hidden.shape().dims() != hidden
            || out.shape().dims() != hidden
        {
            return Err(GoldyError::Validation(
                "decode attention streams and out are [hidden]".into(),
            ));
        }
        self.expect_matrix("wq", weights.wq, q_dim, self.hidden)?;
        self.expect_matrix("wk", weights.wk, kv_dim, self.hidden)?;
        self.expect_matrix("wv", weights.wv, kv_dim, self.hidden)?;
        self.expect_matrix("wo", weights.wo, self.hidden, q_dim)?;
        self.expect_matrix("wgate", weights.wgate, q_dim, self.hidden)?;
        self.expect_matrix("wsink", weights.wsink, self.q_heads, self.hidden)?;
        self.expect_matrix("wrel", weights.wrel, self.q_heads * self.d_rel, self.hidden)?;
        self.expect_matrix("rel_proj", weights.rel_proj, self.d_rel, self.rel_extent)?;
        self.expect_matrix("q_scale", weights.q_scale, self.q_heads, self.head_dim)?;
        self.expect_matrix("rel_scale", weights.rel_scale, self.q_heads, self.d_rel)?;
        let window = match self.kind {
            DecodeAttentionKind::Sliding { window } => window,
            DecodeAttentionKind::Full => 0,
        };

        let q_flat = self.q.view().reshape(&[q_dim])?;
        self.linear.record(scheme, weights.wq, q_hidden, q_flat)?;
        self.q_clip.record(
            scheme,
            self.q.view(),
            Some(weights.q_scale),
            step,
            self.q.view(),
        )?;

        self.project_kv(
            scheme,
            weights.wk,
            k_hidden,
            &self.k_proj,
            &self.k_conv,
            weights.k_conv,
            weights.k_conv_proj,
            state.k_conv.view(),
            step,
        )?;
        let k_conv = self.k_conv.view();
        self.k_clip
            .record(scheme, k_conv, None, step, self.k.view())?;
        self.project_kv(
            scheme,
            weights.wv,
            v_hidden,
            &self.v_proj,
            &self.v_conv,
            weights.v_conv,
            weights.v_conv_proj,
            state.v_conv.view(),
            step,
        )?;
        for h in 0..self.kv_heads {
            let src = self
                .v_conv
                .view()
                .narrow(0, h, 1)?
                .reshape(&[self.head_dim])?;
            let dst = self.v.view().narrow(0, h, 1)?.reshape(&[self.head_dim])?;
            self.v_norm.record_unweighted(scheme, src, dst)?;
        }

        self.blocks.record_kv_cache_store(
            scheme,
            self.k.view(),
            self.v.view(),
            cache.key.view(),
            cache.value.view(),
            step,
            window,
        )?;

        let rel_flat = self.rel_proj.view().reshape(&[self.q_heads * self.d_rel])?;
        self.linear
            .record(scheme, weights.wrel, q_hidden, rel_flat)?;
        self.rel_clip.record(
            scheme,
            self.rel_proj.view(),
            Some(weights.rel_scale),
            step,
            self.rel.view(),
        )?;
        self.linear.record(
            scheme,
            self.rel.view(),
            weights.rel_proj,
            self.rel_logits.view(),
        )?;
        self.linear
            .record(scheme, weights.wsink, q_hidden, self.sink.view())?;

        self.blocks.record_relative_attention_partial(
            scheme,
            self.q.view(),
            cache.key.view(),
            cache.value.view(),
            self.rel_logits.view(),
            state.partial.view(),
            step,
            self.scale,
            window,
        )?;
        self.blocks.record_relative_attention_combine(
            scheme,
            state.partial.view(),
            self.sink.view(),
            self.attn.view(),
            step,
            window,
            cache.capacity,
        )?;

        let attn_flat = self.attn.view().reshape(&[q_dim])?;
        self.linear
            .record(scheme, weights.wgate, q_hidden, self.gate_logits.view())?;
        self.gate
            .record(scheme, attn_flat, self.gate_logits.view())?;
        self.linear.record(scheme, weights.wo, attn_flat, out)?;
        Ok(())
    }

    fn project_kv(
        &self,
        scheme: &mut Scheme,
        weight: TensorView<'_>,
        hidden: TensorView<'_>,
        proj: &Tensor,
        conv_out: &Tensor,
        conv_weight: TensorView<'_>,
        conv_proj: Option<TensorView<'_>>,
        conv_state: TensorView<'_>,
        step: &Buffer,
    ) -> Result<(), GoldyError> {
        self.linear.record(scheme, weight, hidden, proj.view())?;
        let out = conv_out.view().reshape(&[proj.view().numel_u32()])?;
        if self.dynamic {
            let proj_w = conv_proj.ok_or_else(|| {
                GoldyError::Validation("dynamic conv requires a tap projection".into())
            })?;
            let tap_len = self.kv_heads * self.kernel;
            let flat = self.taps.view().reshape(&[tap_len])?;
            self.blocks
                .record_linear(scheme, "conv_taps", proj_w, hidden, flat)?;
            self.blocks.record_dynamic_short_conv_at(
                scheme,
                proj.view(),
                conv_weight,
                self.taps.view(),
                conv_state,
                step,
                out,
            )?;
        } else {
            self.blocks.record_short_conv_at(
                scheme,
                proj.view(),
                conv_weight,
                conv_state,
                step,
                out,
            )?;
        }
        Ok(())
    }

    fn expect_matrix(
        &self,
        name: &str,
        tensor: TensorView<'_>,
        rows: u32,
        cols: u32,
    ) -> Result<(), GoldyError> {
        if tensor.shape().dims() != [rows, cols] {
            return Err(GoldyError::Validation(format!(
                "decode attention {name} must be [{rows}, {cols}]"
            )));
        }
        Ok(())
    }
}

/// Batch-1 routed experts.
///
/// `logits = W x`, `scores = sigmoid(logits)`. Top-k runs on
/// `scores + bias` and the bias is not part of the weights:
/// `weights = scores[ids] / (sum + 1e-20) * route_scale`. Equal routing keys
/// keep the smaller expert index, then the ids are sorted ascending. Each
/// selected expert is a softcapped SwiGLU (`[gate | up]` concatenated) and a
/// down projection, accumulated in `out`. There is no shared expert.
///
/// The whole step is recorded. A replay only rewrites `x` and the step is not
/// read back to choose experts.
pub struct DecodeMoe {
    blocks: Arc<Blocks>,
    linear: Linear,
    swiglu: SoftcappedSwiGlu,
    n_experts: u32,
    top_k: u32,
    hidden: u32,
    intermediate: u32,
    route_scale: f32,
    logits: Tensor,
    ids: Tensor,
    weights: Tensor,
    activated: Tensor,
}

impl DecodeMoe {
    pub fn new(
        runtime: &Runtime,
        n_experts: u32,
        top_k: u32,
        hidden: u32,
        intermediate: u32,
        route_scale: f32,
        gate_cap: f32,
        up_cap: f32,
    ) -> anyhow::Result<Self> {
        if n_experts == 0 || hidden == 0 || intermediate == 0 {
            anyhow::bail!("MoE needs a positive expert count, hidden size, and intermediate size");
        }
        if top_k == 0 || top_k > n_experts {
            anyhow::bail!("MoE top-k {top_k} must be in 1..={n_experts}");
        }
        let f = TensorDType::F32;
        Ok(Self {
            blocks: blocks(runtime)?,
            linear: Linear::new(runtime)?,
            swiglu: SoftcappedSwiGlu::new(runtime, gate_cap, up_cap)?,
            n_experts,
            top_k,
            hidden,
            intermediate,
            route_scale,
            logits: Tensor::zeros(runtime, TensorShape::vector(n_experts), f)?,
            ids: Tensor::zeros(runtime, TensorShape::vector(top_k), TensorDType::U32)?,
            weights: Tensor::zeros(runtime, TensorShape::vector(top_k), f)?,
            activated: Tensor::zeros(runtime, TensorShape::vector(intermediate), f)?,
        })
    }

    pub fn n_experts(&self) -> u32 {
        self.n_experts
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }
}

/// Router matrix. BF16 stays packed and is promoted inside the dot.
#[derive(Clone, Copy)]
pub enum RouterWeight<'a> {
    F32(TensorView<'a>),
    Bf16(TensorView<'a>),
}

/// Expert stack. `Nvfp4` is E2M1 codes, E4M3 scales, block `[1, 16]`.
#[derive(Clone, Copy)]
pub enum ExpertWeight<'a> {
    F32(TensorView<'a>),
    Nvfp4(PackedExperts<'a>),
}

impl DecodeMoe {
    /// `router` is `[experts, hidden]`. `bias` is `[experts]`.
    /// `gate_up` is `[experts, 2 * intermediate, hidden]`.
    /// `down` is `[experts, hidden, intermediate]`.
    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        router: TensorView<'_>,
        bias: TensorView<'_>,
        gate_up: TensorView<'_>,
        down: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_experts(
            scheme,
            x,
            bias,
            RouterWeight::F32(router),
            ExpertWeight::F32(gate_up),
            ExpertWeight::F32(down),
            out,
        )
    }

    /// Same step as [`Self::record`] with a BF16 router and packed experts.
    pub fn record_nvfp4(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        router: TensorView<'_>,
        bias: TensorView<'_>,
        gate_up: PackedExperts<'_>,
        down: PackedExperts<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_experts(
            scheme,
            x,
            bias,
            RouterWeight::Bf16(router),
            ExpertWeight::Nvfp4(gate_up),
            ExpertWeight::Nvfp4(down),
            out,
        )
    }

    fn record_experts(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        bias: TensorView<'_>,
        router: RouterWeight<'_>,
        gate_up: ExpertWeight<'_>,
        down: ExpertWeight<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if x.shape().dims() != [self.hidden] || out.shape().dims() != [self.hidden] {
            return Err(GoldyError::Validation("MoE x and out are [hidden]".into()));
        }
        let router_view = match router {
            RouterWeight::F32(weight) | RouterWeight::Bf16(weight) => weight,
        };
        if router_view.shape().dims() != [self.n_experts, self.hidden]
            || bias.shape().dims() != [self.n_experts]
        {
            return Err(GoldyError::Validation(
                "MoE router is [experts, hidden] and bias is [experts]".into(),
            ));
        }
        if matches!(gate_up, ExpertWeight::Nvfp4(_))
            && (self.hidden % 16 != 0 || self.intermediate % 16 != 0)
        {
            return Err(GoldyError::Validation(
                "NVFP4 MoE hidden and intermediate sizes must be multiples of 16".into(),
            ));
        }
        if let ExpertWeight::F32(gate_up) = gate_up {
            let fused = 2 * self.intermediate;
            let down_view = match down {
                ExpertWeight::F32(weight) => weight,
                ExpertWeight::Nvfp4(_) => {
                    return Err(GoldyError::Validation(
                        "MoE gate_up and down must use the same packing".into(),
                    ))
                }
            };
            if gate_up.shape().dims() != [self.n_experts, fused, self.hidden]
                || down_view.shape().dims() != [self.n_experts, self.hidden, self.intermediate]
            {
                return Err(GoldyError::Validation(
                    "MoE gate_up is [experts, 2 * intermediate, hidden] and down is [experts, hidden, intermediate]"
                        .into(),
                ));
            }
        }
        self.blocks.record_zero(scheme, "moe_out", out)?;
        match router {
            RouterWeight::F32(weight) => {
                self.linear.record(scheme, weight, x, self.logits.view())?
            }
            RouterWeight::Bf16(weight) => {
                self.linear
                    .record_bf16(scheme, weight, x, self.logits.view())?
            }
        }
        self.blocks.record_moe_topk(
            scheme,
            self.logits.view(),
            bias,
            self.ids.view(),
            self.weights.view(),
            self.route_scale,
        )?;
        let fused_len = 2 * self.intermediate;
        if let (ExpertWeight::F32(gate_w), ExpertWeight::F32(down_w)) = (gate_up, down) {
            return self.record_f32_experts(scheme, x, gate_w, down_w, out);
        }
        for slot in 0..self.top_k {
            let fused = Tensor::zeros(
                self.blocks.runtime(),
                TensorShape::vector(fused_len),
                TensorDType::F32,
            )?;
            match gate_up {
                ExpertWeight::F32(weight) => self.blocks.record_indexed_matmul(
                    scheme,
                    "moe_gate_up",
                    weight,
                    x,
                    self.ids.view(),
                    slot,
                    fused.view(),
                )?,
                ExpertWeight::Nvfp4(weight) => self.blocks.record_nvfp4_indexed(
                    scheme,
                    "moe_gate_up",
                    weight.codes,
                    weight.scales,
                    weight.factors,
                    self.ids.view(),
                    slot,
                    x,
                    fused.view(),
                )?,
            }
            let rows = fused.view().reshape(&[2, self.intermediate])?;
            let gate = rows.narrow(0, 0, 1)?.reshape(&[self.intermediate])?;
            let up = rows.narrow(0, 1, 1)?.reshape(&[self.intermediate])?;
            self.swiglu
                .record(scheme, gate, up, self.activated.view())?;
            let projected = Tensor::zeros(
                self.blocks.runtime(),
                TensorShape::vector(self.hidden),
                TensorDType::F32,
            )?;
            match down {
                ExpertWeight::F32(weight) => self.blocks.record_indexed_matmul(
                    scheme,
                    "moe_down",
                    weight,
                    self.activated.view(),
                    self.ids.view(),
                    slot,
                    projected.view(),
                )?,
                ExpertWeight::Nvfp4(weight) => self.blocks.record_nvfp4_indexed(
                    scheme,
                    "moe_down",
                    weight.codes,
                    weight.scales,
                    weight.factors,
                    self.ids.view(),
                    slot,
                    self.activated.view(),
                    projected.view(),
                )?,
            }
            let weight = self.weights.view().narrow(0, slot, 1)?;
            let mut rec = self.blocks.tensor_recorder(scheme);
            let scaled = rec.mul("moe_scale", projected.view(), weight)?;
            rec.add_into("moe_accum", out, scaled.view(), out)?;
            for kept in [fused, projected, scaled] {
                scheme.keep_alive(kept);
            }
        }
        Ok(())
    }

    /// Gathered gate and up products over the two halves of each expert's stacked
    /// gate-up rows, and the gated product over `[k, I]`; then per slot a down
    /// projection, its scale, and the add onto `out`.
    ///
    /// Separate halves let one fused kernel compute every gate and up row once. Each
    /// slot's down reads one slab index, so the downs and the running sum stay one
    /// fused kernel; a batched down would leave the sum over slots to read
    /// projections another thread computes.
    fn record_f32_experts(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        gate_w: TensorView<'_>,
        down_w: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let k = self.top_k;
        let inter = self.intermediate;
        let scratch = || {
            Tensor::zeros(
                self.blocks.runtime(),
                TensorShape::vector(k * inter),
                TensorDType::F32,
            )
        };
        let (gate, up, activated) = (scratch()?, scratch()?, scratch()?);
        for (label, half, product) in [("moe_gate", 0, &gate), ("moe_up", 1, &up)] {
            self.blocks.record_indexed_matmul_batch(
                scheme,
                label,
                gate_w.narrow(1, half * inter, inter)?,
                x,
                self.ids.view(),
                0,
                k,
                0,
                product.view(),
            )?;
        }
        fn by_slot(t: &Tensor, k: u32, inter: u32) -> Result<TensorView<'_>, GoldyError> {
            t.view().reshape(&[k, inter])
        }
        self.swiglu.record(
            scheme,
            by_slot(&gate, k, inter)?,
            by_slot(&up, k, inter)?,
            by_slot(&activated, k, inter)?,
        )?;
        let projected = Tensor::zeros(
            self.blocks.runtime(),
            TensorShape::vector(k * self.hidden),
            TensorDType::F32,
        )?;
        let scaled = Tensor::zeros(
            self.blocks.runtime(),
            TensorShape::vector(k * self.hidden),
            TensorDType::F32,
        )?;
        for slot in 0..k {
            fn rows(t: &Tensor, slot: u32, len: u32) -> Result<TensorView<'_>, GoldyError> {
                t.view().narrow(0, slot * len, len)
            }
            let row = |t| rows(t, slot, self.hidden);
            self.blocks.record_indexed_matmul(
                scheme,
                "moe_down",
                down_w,
                activated
                    .view()
                    .narrow(0, slot * self.intermediate, self.intermediate)?,
                self.ids.view(),
                slot,
                row(&projected)?,
            )?;
            let weight = self.weights.view().narrow(0, slot, 1)?;
            let mut rec = self.blocks.tensor_recorder(scheme);
            rec.mul_into("moe_scale", row(&projected)?, weight, row(&scaled)?)?;
            rec.add_into("moe_accum", out, row(&scaled)?, out)?;
        }
        for kept in [gate, up, activated, projected, scaled] {
            scheme.keep_alive(kept);
        }
        Ok(())
    }
}

pub struct DepthMix {
    blocks: Arc<Blocks>,
}

impl DepthMix {
    pub fn new(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
        })
    }

    pub fn mix(
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
        self.blocks
            .record_depth_mix(scheme, blocks, partial, queries, out, n_blocks, eps, scale)
    }

    pub fn store_row(
        &self,
        scheme: &mut Scheme,
        src: TensorView<'_>,
        tape: TensorView<'_>,
        slot: u32,
    ) -> Result<(), GoldyError> {
        self.blocks.record_depth_store(scheme, src, tape, slot)
    }

    pub fn add(
        &self,
        scheme: &mut Scheme,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.blocks.record_add(scheme, "add", a, b, out)
    }
}
