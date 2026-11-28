//! Recordable inference modules. Stateless operator modules hold prepared kernels
//! and configuration only, declaring scheme-scoped temporary views so fused regions
//! elide intermediates into registers and unfused waves share the context's
//! transient pool; persistent KV and convolution state lives on explicit state
//! structs (`KvCache`, `DecodeKvCache`, `DecodeLayerState`, `ShortConv`,
//! `DynamicShortConv`). Modules never submit schemes or bind exchanges.

use crate::blocks::{attention_splits, AttentionSites, Blocks, FfnSites};
use goldy::{
    Buffer, GoldyError, Runtime, Scheme, SchemeLabel, Tensor, TensorDType, TensorShape, TensorView,
};
use std::sync::Arc;

/// Pre-norm causal attention block weights.
#[derive(Clone, Copy)]
pub struct CausalAttentionBlockWeights<'a> {
    pub norm: TensorView<'a>,
    pub query: TensorView<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub output: TensorView<'a>,
}

/// Pre-norm SwiGLU block weights.
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

fn require_len(what: &str, x: TensorView<'_>, len: u32) -> Result<(), GoldyError> {
    if x.numel_u32() != len {
        return Err(GoldyError::Validation(format!(
            "{what}: {} elements, but this module expects {len}",
            x.numel_u32()
        )));
    }
    Ok(())
}

/// Backing storage for persistent state tensors, either standalone or sliced out of
/// a shared contiguous slab.
enum StateTensor {
    Owned(Tensor),
    Slice {
        slab: Arc<Tensor>,
        offset: u32,
        len: u32,
        shape: TensorShape,
    },
}

impl StateTensor {
    fn zeros(runtime: &Runtime, shape: TensorShape) -> Result<Self, GoldyError> {
        Ok(Self::Owned(Tensor::zeros(
            runtime,
            shape,
            TensorDType::F32,
        )?))
    }

    fn slice(slab: &Arc<Tensor>, cursor: &mut u32, shape: TensorShape) -> Result<Self, GoldyError> {
        let offset = *cursor;
        let len = u32::try_from(shape.numel()?)
            .map_err(|_| GoldyError::Validation("state slice exceeds u32 elements".into()))?;
        *cursor = cursor
            .checked_add(len)
            .ok_or_else(|| GoldyError::Validation("state slab element count overflow".into()))?;
        Ok(Self::Slice {
            slab: Arc::clone(slab),
            offset,
            len,
            shape,
        })
    }

    fn view(&self) -> TensorView<'_> {
        match self {
            Self::Owned(tensor) => tensor.view(),
            Self::Slice {
                slab,
                offset,
                len,
                shape,
            } => slab
                .view()
                .narrow(0, *offset, *len)
                .expect("validated slab range")
                .reshape(shape.dims())
                .expect("validated slab shape"),
        }
    }
}

pub use crate::depth::DEPTH_RMS_EPS;
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
/// It is recorded as tensor operations over scheme-scoped temporary views, which
/// automatic fusion composes with surrounding operations and elides into registers.
pub struct RmsNorm {
    blocks: Arc<Blocks>,
    dim: u32,
    eps: f32,
}

impl RmsNorm {
    pub fn new(runtime: &Runtime, dim: u32) -> anyhow::Result<Self> {
        Self::with_eps(runtime, dim, DEFAULT_RMS_EPS)
    }

    pub fn with_eps(runtime: &Runtime, dim: u32, eps: f32) -> anyhow::Result<Self> {
        if dim == 0 {
            anyhow::bail!("RMSNorm width must be non-zero");
        }
        Ok(Self {
            blocks: blocks(runtime)?,
            dim,
            eps,
        })
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn record(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_out(scheme, x, Some(weight), out)
    }

    pub fn record_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_over(scheme, x, Some(weight))
    }

    /// `out = x * rsqrt(mean(x²) + eps)`, with no gain vector.
    pub fn record_unweighted(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_out(scheme, x, None, out)
    }

    /// In-place form of [`Self::record_unweighted`].
    pub fn record_unweighted_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.record_over(scheme, x, None)
    }

    fn record_out(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: Option<TensorView<'_>>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        require_len("rms norm", x, self.dim)?;
        self.blocks.record_rmsnorm(scheme, x, weight, out, self.eps)
    }

    fn record_over(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: Option<TensorView<'_>>,
    ) -> Result<(), GoldyError> {
        require_len("rms norm", x, self.dim)?;
        self.blocks
            .record_rmsnorm_inplace(scheme, x, weight, self.eps)
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
pub struct CausalAttentionBlock {
    blocks: Arc<Blocks>,
    hidden: u32,
    query_heads: u32,
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
        let _ = seq_len;
        Ok(Self {
            blocks: blocks(runtime)?,
            hidden,
            query_heads,
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
        require_len("attention", x, self.hidden)?;
        self.blocks.record_attention(
            scheme,
            AttentionSites {
                x,
                q_heads: self.query_heads,
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
pub struct SwiGluBlock {
    blocks: Arc<Blocks>,
    hidden: u32,
    intermediate: u32,
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
        if hidden == 0 || intermediate == 0 {
            anyhow::bail!("SwiGluBlock hidden and intermediate widths must be non-zero");
        }
        Ok(Self {
            blocks: blocks(runtime)?,
            hidden,
            intermediate,
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
        require_len("swiglu block", x, self.hidden)?;
        if weights.gate.shape().dim(0)? != self.intermediate {
            return Err(GoldyError::Validation(format!(
                "swiglu block: gate rows {} != {}",
                weights.gate.shape().dim(0)?,
                self.intermediate
            )));
        }
        self.blocks.record_ffn(
            scheme,
            FfnSites {
                x,
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
pub struct HeadL2Clip {
    blocks: Arc<Blocks>,
    threshold: f32,
    log: Option<LogPositionScale>,
    heads: u32,
    dim: u32,
}

impl HeadL2Clip {
    pub fn new(runtime: &Runtime, heads: u32, dim: u32, threshold: f32) -> anyhow::Result<Self> {
        Self::build(runtime, heads, dim, threshold, None)
    }

    pub fn with_log_scale(
        runtime: &Runtime,
        heads: u32,
        dim: u32,
        threshold: f32,
        log: LogPositionScale,
    ) -> anyhow::Result<Self> {
        if log.alpha != 0.0 && log.floor <= 0.0 {
            anyhow::bail!("log-position floor must be positive");
        }
        Self::build(runtime, heads, dim, threshold, Some(log))
    }

    fn build(
        runtime: &Runtime,
        heads: u32,
        dim: u32,
        threshold: f32,
        log: Option<LogPositionScale>,
    ) -> anyhow::Result<Self> {
        if heads == 0 || dim == 0 {
            anyhow::bail!("l2 clip needs a positive head count and width");
        }
        Ok(Self {
            blocks: blocks(runtime)?,
            threshold,
            log,
            heads,
            dim,
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
        if x.shape().dims() != [self.heads, self.dim] || out.shape().dims() != x.shape().dims() {
            return Err(GoldyError::Validation(format!(
                "l2 clip: x and out must be [{}, {}]",
                self.heads, self.dim
            )));
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

/// `out = cap * tanh(x / cap)` over `len` elements, using `goldy::gpu::tanh`.
pub struct Softcap {
    blocks: Arc<Blocks>,
    len: u32,
    cap: f32,
}

impl Softcap {
    pub fn new(runtime: &Runtime, len: u32, cap: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            len,
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
        require_len("softcap", x, self.len)?;
        self.blocks.record_softcap(scheme, x, out, self.cap)
    }
}

/// In-place `values *= sigmoid(logits)` over `len` elements, using
/// `goldy::gpu::sigmoid`.
pub struct SigmoidGate {
    blocks: Arc<Blocks>,
    len: u32,
}

impl SigmoidGate {
    pub fn new(runtime: &Runtime, len: u32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            len,
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
        require_len("sigmoid gate", values, self.len)?;
        self.blocks.record_sigmoid_gate(scheme, values, logits)
    }
}

/// `out = softcap(gate) * sigmoid(gate) * softcap(up)` over `len` elements.
pub struct SoftcappedSwiGlu {
    blocks: Arc<Blocks>,
    len: u32,
    gate_cap: f32,
    up_cap: f32,
}

impl SoftcappedSwiGlu {
    pub fn new(runtime: &Runtime, len: u32, gate_cap: f32, up_cap: f32) -> anyhow::Result<Self> {
        Ok(Self {
            blocks: blocks(runtime)?,
            len,
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
        require_len("softcapped swiglu", gate, self.len)?;
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
/// Owns `[channels, kernel]` of running state unless recording onto a caller-owned
/// ring via [`Self::record_at_state`] or [`Self::stateless`].
pub struct ShortConv {
    blocks: Arc<Blocks>,
    state: Option<Tensor>,
    channels: u32,
    kernel_size: u32,
}

impl ShortConv {
    pub fn new(runtime: &Runtime, channels: u32, kernel_size: u32) -> anyhow::Result<Self> {
        require_conv_shape(channels, kernel_size)?;
        Ok(Self {
            blocks: blocks(runtime)?,
            state: Some(Tensor::zeros(
                runtime,
                TensorShape::matrix(channels, kernel_size),
                TensorDType::F32,
            )?),
            channels,
            kernel_size,
        })
    }

    /// Stateless recorder for caller-owned rings passed to [`Self::record_at_state`].
    pub fn stateless(runtime: &Runtime, channels: u32, kernel_size: u32) -> anyhow::Result<Self> {
        require_conv_shape(channels, kernel_size)?;
        Ok(Self {
            blocks: blocks(runtime)?,
            state: None,
            channels,
            kernel_size,
        })
    }

    fn owned_state(&self) -> Result<TensorView<'_>, GoldyError> {
        self.state.as_ref().map(Tensor::view).ok_or_else(|| {
            GoldyError::Validation(
                "short conv was created stateless; pass an explicit state view to record_at_state"
                    .into(),
            )
        })
    }

    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.blocks
            .record_zero(scheme, "conv_reset", self.owned_state()?)
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
            .record_short_conv(scheme, x, weight, self.owned_state()?, out)
    }

    /// Like [`Self::record`], but the ring slot is `DecodeStep.position`.
    pub fn record_at(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let state = self.owned_state()?;
        self.record_at_state(scheme, x, weight, state, step, out)
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
            .record_short_conv_at(scheme, x, weight, state, step, out)
    }
}

/// Dynamic one-token causal depthwise convolution.
pub struct DynamicShortConv {
    blocks: Arc<Blocks>,
    state: Tensor,
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
        let taps = scheme.temporary_tensor(
            TensorShape::matrix(self.num_heads, self.kernel_size),
            TensorDType::F32,
        )?;
        let flat = taps.reshape(&[tap_len])?;
        self.blocks
            .record_linear(scheme, "conv_taps", proj, dynamic_input, flat)?;
        self.blocks.record_dynamic_short_conv(
            scheme,
            x,
            static_weight,
            taps,
            self.state.view(),
            out,
        )
    }
}

/// Which keys a [`DecodeAttention`] reads. Capacity lives on [`DecodeKvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeAttentionKind {
    Sliding { window: u32 },
    Full,
}

/// Key/value cache for one decode attention layer: `[capacity, kv_heads, head]`.
pub struct DecodeKvCache {
    blocks: Arc<Blocks>,
    key: StateTensor,
    value: StateTensor,
    /// `0` for a full cache. Otherwise the sliding window.
    window: u32,
    capacity: u32,
    kv_heads: u32,
    head_dim: u32,
}

impl DecodeKvCache {
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
            key: StateTensor::zeros(runtime, shape)?,
            value: StateTensor::zeros(runtime, shape)?,
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

    pub fn record_reset(&self, scheme: &mut Scheme) -> Result<(), GoldyError> {
        self.blocks
            .record_zero(scheme, "kv_reset_k", self.key.view())?;
        self.blocks
            .record_zero(scheme, "kv_reset_v", self.value.view())
    }

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

/// Shape specification for packing multiple [`DecodeLayerState`] visits onto one
/// contiguous persistent GPU slab via [`DecodeLayerState::allocate_packed`].
#[derive(Clone, Copy, Debug)]
pub struct DecodeLayerStateSpec {
    /// `0` for full attention; non-zero for a sliding window.
    pub window: u32,
    pub capacity: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_heads: u32,
    pub kernel: u32,
    pub post_conv_hidden: Option<u32>,
}

/// Persistent state for one attention visit: KV plus both K and V convolution
/// states (and optional post-attention / post-MLP convolution rings).
pub struct DecodeLayerState {
    kv: DecodeKvCache,
    k_conv: StateTensor,
    v_conv: StateTensor,
    /// Present on standalone states for split-identity inspection tests; `None`
    /// on slab-packed states so [`DecodeAttention::record`] uses a temporary view.
    partial: Option<Tensor>,
    splits: u32,
    q_heads: u32,
    kernel: u32,
    post_attn: Option<StateTensor>,
    post_mlp: Option<StateTensor>,
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

    /// Allocate persistent KV and convolution state for every visit in `specs` out
    /// of a single contiguous retained GPU [`Tensor`], leaving split-K scratch on
    /// scheme-scoped temporary views.
    pub fn allocate_packed(
        runtime: &Runtime,
        specs: &[DecodeLayerStateSpec],
    ) -> anyhow::Result<Vec<Self>> {
        let blocks = blocks(runtime)?;
        let mut total_elems = 0u64;
        for s in specs {
            if s.capacity == 0 || s.kv_heads == 0 || s.head_dim == 0 {
                anyhow::bail!("KV cache needs positive capacity, head count, and head size");
            }
            if s.window != 0 && s.capacity < s.window {
                anyhow::bail!(
                    "sliding KV capacity {} is below the window {}",
                    s.capacity,
                    s.window
                );
            }
            if s.kernel == 0 || s.q_heads == 0 {
                anyhow::bail!("decode state needs a kernel and query heads");
            }
            let kv_elems = u64::from(s.capacity) * u64::from(s.kv_heads) * u64::from(s.head_dim);
            let conv_elems = u64::from(s.kv_heads * s.head_dim) * u64::from(s.kernel);
            let post_elems = s
                .post_conv_hidden
                .map_or(0, |h| 2 * u64::from(h) * u64::from(s.kernel));
            total_elems = total_elems
                .checked_add(2 * kv_elems + 2 * conv_elems + post_elems)
                .ok_or_else(|| anyhow::anyhow!("packed state slab element count overflow"))?;
        }
        let total_u32 = u32::try_from(total_elems.max(1))
            .map_err(|_| anyhow::anyhow!("packed state slab exceeds u32 element index"))?;
        let slab = Arc::new(Tensor::zeros(
            runtime,
            TensorShape::vector(total_u32),
            TensorDType::F32,
        )?);
        let mut cursor = 0u32;
        let mut out = Vec::with_capacity(specs.len());
        for s in specs {
            let kv_shape = TensorShape::from_dims(&[s.capacity, s.kv_heads, s.head_dim])?;
            let conv_shape = TensorShape::matrix(s.kv_heads * s.head_dim, s.kernel);
            let key = StateTensor::slice(&slab, &mut cursor, kv_shape)?;
            let value = StateTensor::slice(&slab, &mut cursor, kv_shape)?;
            let k_conv = StateTensor::slice(&slab, &mut cursor, conv_shape)?;
            let v_conv = StateTensor::slice(&slab, &mut cursor, conv_shape)?;
            let (post_attn, post_mlp) = match s.post_conv_hidden {
                Some(hidden) => {
                    let post_shape = TensorShape::matrix(hidden, s.kernel);
                    (
                        Some(StateTensor::slice(&slab, &mut cursor, post_shape)?),
                        Some(StateTensor::slice(&slab, &mut cursor, post_shape)?),
                    )
                }
                None => (None, None),
            };
            let attended = if s.window == 0 { s.capacity } else { s.window };
            out.push(Self {
                kv: DecodeKvCache {
                    blocks: Arc::clone(&blocks),
                    key,
                    value,
                    window: s.window,
                    capacity: s.capacity,
                    kv_heads: s.kv_heads,
                    head_dim: s.head_dim,
                },
                k_conv,
                v_conv,
                partial: None,
                splits: attention_splits(attended),
                q_heads: s.q_heads,
                kernel: s.kernel,
                post_attn,
                post_mlp,
            });
        }
        Ok(out)
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
        let splits = attention_splits(attended);
        Ok(Self {
            k_conv: StateTensor::zeros(runtime, TensorShape::matrix(kv_dim, kernel))?,
            v_conv: StateTensor::zeros(runtime, TensorShape::matrix(kv_dim, kernel))?,
            partial: Some(Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[q_heads, splits, kv.head_dim + 2])?,
                TensorDType::F32,
            )?),
            splits,
            q_heads,
            kv,
            kernel,
            post_attn: None,
            post_mlp: None,
        })
    }

    pub fn with_post_conv(mut self, runtime: &Runtime, hidden: u32) -> anyhow::Result<Self> {
        let shape = TensorShape::matrix(hidden, self.kernel);
        self.post_attn = Some(StateTensor::zeros(runtime, shape)?);
        self.post_mlp = Some(StateTensor::zeros(runtime, shape)?);
        Ok(self)
    }

    pub fn post_attn(&self) -> Option<TensorView<'_>> {
        self.post_attn.as_ref().map(StateTensor::view)
    }

    pub fn post_mlp(&self) -> Option<TensorView<'_>> {
        self.post_mlp.as_ref().map(StateTensor::view)
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

    /// Split-attention scratch on standalone states, `[q_heads, splits, head_dim + 2]`.
    pub fn partial(&self) -> TensorView<'_> {
        self.partial
            .as_ref()
            .expect("slab-packed DecodeLayerState binds temporary split scratch on the scheme")
            .view()
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
/// Holds no retained scratch buffers: one instance can record every matching visit
/// across a model while declaring scheme-scoped temporary views.
pub struct DecodeAttention {
    blocks: Arc<Blocks>,
    linear: Linear,
    q_clip: HeadL2Clip,
    k_clip: HeadL2Clip,
    rel_clip: HeadL2Clip,
    rms_eps: f32,
    gate: SigmoidGate,
    dynamic_conv: bool,
    kernel: u32,
    kind: DecodeAttentionKind,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    hidden: u32,
    d_rel: u32,
    rel_extent: u32,
    scale: f32,
}

impl DecodeAttention {
    #[allow(clippy::too_many_arguments)]
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
        let q_clip = match log_scale {
            Some(log) => HeadL2Clip::with_log_scale(runtime, q_heads, head_dim, qk_l2_clip, log)?,
            None => HeadL2Clip::new(runtime, q_heads, head_dim, qk_l2_clip)?,
        };
        let rel_clip = match log_scale {
            Some(log) => HeadL2Clip::with_log_scale(runtime, q_heads, d_rel, rel_l2_clip, log)?,
            None => HeadL2Clip::new(runtime, q_heads, d_rel, rel_l2_clip)?,
        };
        Ok(Self {
            blocks: blocks(runtime)?,
            linear: Linear::new(runtime)?,
            q_clip,
            k_clip: HeadL2Clip::new(runtime, kv_heads, head_dim, qk_l2_clip)?,
            rel_clip,
            rms_eps,
            gate: SigmoidGate::new(runtime, q_dim)?,
            dynamic_conv,
            kernel: conv_kernel,
            kind,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            d_rel,
            rel_extent,
            scale: (f64::from(head_dim)).powf(-0.5) as f32,
        })
    }

    pub fn kind(&self) -> DecodeAttentionKind {
        self.kind
    }

    #[allow(clippy::too_many_arguments)]
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
            || state.q_heads != self.q_heads
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
        let f = TensorDType::F32;

        // Out-of-place `q_proj -> q` avoids aliasing `x` and `out` inside `q_clip`'s
        // semantic description so query clipping fuses cleanly.
        let q_proj =
            scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.head_dim), f)?;
        let q = scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.head_dim), f)?;
        self.linear
            .record(scheme, weights.wq, q_hidden, q_proj.reshape(&[q_dim])?)?;
        self.q_clip
            .record(scheme, q_proj, Some(weights.q_scale), step, q)?;

        let k_conv = self.project_kv(
            scheme,
            weights.wk,
            k_hidden,
            weights.k_conv,
            weights.k_conv_proj,
            state.k_conv.view(),
            step,
        )?;
        let k = scheme.temporary_tensor(TensorShape::matrix(self.kv_heads, self.head_dim), f)?;
        self.k_clip.record(scheme, k_conv, None, step, k)?;

        let v_conv = self.project_kv(
            scheme,
            weights.wv,
            v_hidden,
            weights.v_conv,
            weights.v_conv_proj,
            state.v_conv.view(),
            step,
        )?;
        let v = scheme.temporary_tensor(TensorShape::matrix(self.kv_heads, self.head_dim), f)?;
        for h in 0..self.kv_heads {
            let src = v_conv.narrow(0, h, 1)?.reshape(&[self.head_dim])?;
            let dst = v.narrow(0, h, 1)?.reshape(&[self.head_dim])?;
            self.blocks
                .record_rmsnorm(scheme, src, None, dst, self.rms_eps)?;
        }

        self.blocks.record_kv_cache_store(
            scheme,
            k,
            v,
            cache.key.view(),
            cache.value.view(),
            step,
            window,
        )?;

        let rel_proj = scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.d_rel), f)?;
        let rel = scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.d_rel), f)?;
        let rel_logits =
            scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.rel_extent), f)?;
        let sink = scheme.temporary_tensor(TensorShape::vector(self.q_heads), f)?;
        self.linear.record(
            scheme,
            weights.wrel,
            q_hidden,
            rel_proj.reshape(&[self.q_heads * self.d_rel])?,
        )?;
        self.rel_clip
            .record(scheme, rel_proj, Some(weights.rel_scale), step, rel)?;
        self.linear
            .record(scheme, rel, weights.rel_proj, rel_logits)?;
        self.linear.record(scheme, weights.wsink, q_hidden, sink)?;

        let partial = match state.partial.as_ref() {
            Some(buf) => buf.view(),
            None => scheme.temporary_tensor(
                TensorShape::from_dims(&[self.q_heads, state.splits, self.head_dim + 2])?,
                f,
            )?,
        };
        let attn = scheme.temporary_tensor(TensorShape::matrix(self.q_heads, self.head_dim), f)?;
        self.blocks.record_relative_attention_partial(
            scheme,
            q,
            cache.key.view(),
            cache.value.view(),
            rel_logits,
            partial,
            step,
            self.scale,
            window,
        )?;
        self.blocks.record_relative_attention_combine(
            scheme,
            partial,
            sink,
            attn,
            step,
            window,
            cache.capacity,
        )?;

        let gate_logits = scheme.temporary_tensor(TensorShape::vector(q_dim), f)?;
        self.linear
            .record(scheme, weights.wgate, q_hidden, gate_logits)?;
        self.gate
            .record(scheme, attn, gate_logits.reshape(attn.shape().dims())?)?;
        self.linear
            .record(scheme, weights.wo, attn.reshape(&[q_dim])?, out)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn project_kv(
        &self,
        scheme: &mut Scheme,
        weight: TensorView<'_>,
        hidden: TensorView<'_>,
        conv_weight: TensorView<'_>,
        conv_proj: Option<TensorView<'_>>,
        conv_state: TensorView<'_>,
        step: &Buffer,
    ) -> Result<TensorView<'static>, GoldyError> {
        let kv_dim = self.kv_heads * self.head_dim;
        let f = TensorDType::F32;
        let proj = scheme.temporary_tensor(TensorShape::vector(kv_dim), f)?;
        let conv_out =
            scheme.temporary_tensor(TensorShape::matrix(self.kv_heads, self.head_dim), f)?;
        self.linear.record(scheme, weight, hidden, proj)?;
        let out_flat = conv_out.reshape(&[kv_dim])?;
        if self.dynamic_conv {
            let proj_w = conv_proj.ok_or_else(|| {
                GoldyError::Validation("dynamic conv requires a tap projection".into())
            })?;
            let tap_len = self.kv_heads * self.kernel;
            let taps =
                scheme.temporary_tensor(TensorShape::matrix(self.kv_heads, self.kernel), f)?;
            self.blocks.record_linear(
                scheme,
                "conv_taps",
                proj_w,
                hidden,
                taps.reshape(&[tap_len])?,
            )?;
            self.blocks.record_dynamic_short_conv_at(
                scheme,
                proj,
                conv_weight,
                taps,
                conv_state,
                step,
                out_flat,
            )?;
        } else {
            self.blocks.record_short_conv_at(
                scheme,
                proj,
                conv_weight,
                conv_state,
                step,
                out_flat,
            )?;
        }
        Ok(conv_out)
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
/// Stateless across visits: declares only the scheme-scoped temporary views needed
/// by the active weight format (`record` vs `record_nvfp4`).
pub struct DecodeMoe {
    blocks: Arc<Blocks>,
    linear: Linear,
    n_experts: u32,
    top_k: u32,
    hidden: u32,
    intermediate: u32,
    route_scale: f32,
    batch_swiglu: SoftcappedSwiGlu,
    slot_swiglu: SoftcappedSwiGlu,
}

impl DecodeMoe {
    #[allow(clippy::too_many_arguments)]
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
        Ok(Self {
            blocks: blocks(runtime)?,
            linear: Linear::new(runtime)?,
            n_experts,
            top_k,
            hidden,
            intermediate,
            route_scale,
            batch_swiglu: SoftcappedSwiGlu::new(runtime, top_k * intermediate, gate_cap, up_cap)?,
            slot_swiglu: SoftcappedSwiGlu::new(runtime, intermediate, gate_cap, up_cap)?,
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
    #[allow(clippy::too_many_arguments)]
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
        let f = TensorDType::F32;
        let logits = scheme.temporary_tensor(TensorShape::vector(self.n_experts), f)?;
        let ids = scheme.temporary_tensor(TensorShape::vector(self.top_k), TensorDType::U32)?;
        let weights = scheme.temporary_tensor(TensorShape::vector(self.top_k), f)?;

        self.blocks.record_zero(scheme, "moe_out", out)?;
        match router {
            RouterWeight::F32(weight) => self.linear.record(scheme, weight, x, logits)?,
            RouterWeight::Bf16(weight) => self.linear.record_bf16(scheme, weight, x, logits)?,
        }
        self.blocks
            .record_moe_topk(scheme, logits, bias, ids, weights, self.route_scale)?;
        if let (ExpertWeight::F32(gate_w), ExpertWeight::F32(down_w)) = (gate_up, down) {
            return self.record_f32_experts(scheme, x, ids, weights, gate_w, down_w, out);
        }
        for slot in 0..self.top_k {
            let fused = scheme.temporary_tensor(TensorShape::vector(2 * self.intermediate), f)?;
            let activated = scheme.temporary_tensor(TensorShape::vector(self.intermediate), f)?;
            let projected = scheme.temporary_tensor(TensorShape::vector(self.hidden), f)?;
            let scaled = scheme.temporary_tensor(TensorShape::vector(self.hidden), f)?;
            match gate_up {
                ExpertWeight::F32(weight) => self.blocks.record_indexed_matmul(
                    scheme,
                    "moe_gate_up",
                    weight,
                    x,
                    ids,
                    slot,
                    fused,
                )?,
                ExpertWeight::Nvfp4(weight) => self.blocks.record_nvfp4_indexed(
                    scheme,
                    "moe_gate_up",
                    weight.codes,
                    weight.scales,
                    weight.factors,
                    ids,
                    slot,
                    x,
                    fused,
                )?,
            }
            let rows = fused.reshape(&[2, self.intermediate])?;
            let gate = rows.narrow(0, 0, 1)?.reshape(&[self.intermediate])?;
            let up = rows.narrow(0, 1, 1)?.reshape(&[self.intermediate])?;
            self.slot_swiglu.record(scheme, gate, up, activated)?;
            match down {
                ExpertWeight::F32(weight) => self.blocks.record_indexed_matmul(
                    scheme, "moe_down", weight, activated, ids, slot, projected,
                )?,
                ExpertWeight::Nvfp4(weight) => self.blocks.record_nvfp4_indexed(
                    scheme,
                    "moe_down",
                    weight.codes,
                    weight.scales,
                    weight.factors,
                    ids,
                    slot,
                    activated,
                    projected,
                )?,
            }
            let weight = weights.narrow(0, slot, 1)?;
            let mut rec = self.blocks.tensor_recorder(scheme);
            rec.mul_into("moe_scale", projected, weight, scaled)?;
            rec.add_into("moe_accum", out, scaled, out)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_f32_experts(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        ids: TensorView<'_>,
        weights: TensorView<'_>,
        gate_w: TensorView<'_>,
        down_w: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let k = self.top_k;
        let inter = self.intermediate;
        let batched = k * inter;
        let f = TensorDType::F32;
        let gate = scheme.temporary_tensor(TensorShape::vector(batched), f)?;
        let up = scheme.temporary_tensor(TensorShape::vector(batched), f)?;
        let activated = scheme.temporary_tensor(TensorShape::vector(batched), f)?;
        for (label, half, product) in [("moe_gate", 0, gate), ("moe_up", 1, up)] {
            self.blocks.record_indexed_matmul_batch(
                scheme,
                label,
                gate_w.narrow(1, half * inter, inter)?,
                x,
                ids,
                0,
                k,
                0,
                product,
            )?;
        }
        self.batch_swiglu.record(
            scheme,
            gate.reshape(&[k, inter])?,
            up.reshape(&[k, inter])?,
            activated.reshape(&[k, inter])?,
        )?;
        for slot in 0..k {
            let row_proj = scheme.temporary_tensor(TensorShape::vector(self.hidden), f)?;
            let row_scaled = scheme.temporary_tensor(TensorShape::vector(self.hidden), f)?;
            self.blocks.record_indexed_matmul(
                scheme,
                "moe_down",
                down_w,
                activated.narrow(0, slot * inter, inter)?,
                ids,
                slot,
                row_proj,
            )?;
            let weight = weights.narrow(0, slot, 1)?;
            let mut rec = self.blocks.tensor_recorder(scheme);
            rec.mul_into("moe_scale", row_proj, weight, row_scaled)?;
            rec.add_into("moe_accum", out, row_scaled, out)?;
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

    #[allow(clippy::too_many_arguments)]
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

    pub fn copy_rows(
        &self,
        scheme: &mut Scheme,
        src: TensorView<'_>,
        dst: TensorView<'_>,
        src_slot: u32,
        n: u32,
    ) -> Result<(), GoldyError> {
        self.blocks
            .record_depth_copy_rows(scheme, src, dst, src_slot, n)
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
