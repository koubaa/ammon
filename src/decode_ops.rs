//! Pointwise decode ops and one-token causal convolutions.
//!
//! Softcap, the sigmoid gate, and softcapped SwiGLU call `goldy::gpu::tanh`
//! and `goldy::gpu::sigmoid`.

#![allow(clippy::too_many_arguments)]

use crate::kernels::DecodeStep;

/// Added inside the L2 norm before comparing with the threshold.
///
/// This is the clip's own epsilon, not an RMSNorm epsilon.
pub const L2_CLIP_NORM_EPS: f32 = 1e-6;

/// Per-head L2 clip of `[heads, dim]`.
///
/// `y = x` when `use_scale` is 0, otherwise `y = x * scale` with `scale` the same
/// shape as `x`. Then `y *= min(1, threshold / (||y||_2 + 1e-6))`. When `alpha`
/// is not 0, that result is also multiplied by
/// `1 + alpha * log(max((position + 1) / floor, 1))`. `position` is
/// [`DecodeStep::position`]. One workgroup owns one head.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn head_l2_clip(
    #[tensor(shape = [heads, dim])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [heads, dim])] scale: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [heads, dim])] out: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
    threshold: f32,
    alpha: f32,
    floor: f32,
    use_scale: u32,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let h = goldy::gpu::workgroup_id().x;
    let dim = x.dim(1);
    let base = h * dim;
    let mut sumsq = 0.0;
    let mut j = local;
    while j < dim {
        let mut v = x[base + j];
        if use_scale != 0 {
            v = v * scale[base + j];
        }
        sumsq = sumsq + v * v;
        j = j + 256;
    }
    sumsq = goldy::gpu::workgroup_sum::<256>(sumsq, scratch);
    let norm = goldy::gpu::sqrt(sumsq);
    let mut factor = threshold / (norm + 1.0e-6);
    if factor > 1.0 {
        factor = 1.0;
    }
    if alpha != 0.0 {
        let mut t = (step[0].position as f32 + 1.0) / floor;
        if t < 1.0 {
            t = 1.0;
        }
        // Natural log. Switch to a dedicated helper only if Goldy grows one;
        // `gpu::log` is already the right primitive.
        factor = factor * (1.0 + alpha * goldy::gpu::log(t));
    }
    j = local;
    while j < dim {
        let mut v = x[base + j];
        if use_scale != 0 {
            v = v * scale[base + j];
        }
        out[base + j] = v * factor;
        j = j + 256;
    }
}

/// `out = cap * tanh(x / cap)` via [`goldy::gpu::tanh`].
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn softcap(
    #[tensor(shape = [n])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] out: goldy::gpu::TensorWrite<f32>,
    cap: f32,
) {
    let i = goldy::gpu::global_id().x;
    if i >= x.len() {
        return;
    }
    out[i] = goldy::gpu::tanh(x[i] / cap) * cap;
}

/// In-place `values *= sigmoid(logits)` via [`goldy::gpu::sigmoid`].
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn sigmoid_gate(
    #[tensor(shape = [n])] values: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [n])] logits: goldy::gpu::Tensor<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i >= values.len() {
        return;
    }
    values[i] = values[i] * goldy::gpu::sigmoid(logits[i]);
}

/// `out = softcap(gate) * sigmoid(gate) * softcap(up)`.
///
/// Sigmoid sees the raw gate. Both caps use [`goldy::gpu::tanh`].
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn softcapped_swiglu(
    #[tensor(shape = [n])] gate: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] up: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] out: goldy::gpu::TensorWrite<f32>,
    gate_cap: f32,
    up_cap: f32,
) {
    let i = goldy::gpu::global_id().x;
    if i >= gate.len() {
        return;
    }
    let g = gate[i];
    let u = up[i];
    out[i] = (goldy::gpu::tanh(g / gate_cap) * gate_cap)
        * goldy::gpu::sigmoid(g)
        * (goldy::gpu::tanh(u / up_cap) * up_cap);
}

/// One-token static causal depthwise convolution.
///
/// `weight` is `[channels, kernel]` in conv1d order: index 0 is the oldest tap
/// (`kernel - 1` steps back) and index `kernel - 1` is the current token.
/// `state` is `[channels, kernel]`, oldest first, and already holds the previous
/// tokens (zeros before the first). The kernel writes the new state.
/// `out = (conv + x) / sqrt(kernel + 1)`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn short_conv(
    #[tensor(shape = [channels])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] weight: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] state: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [channels])] out: goldy::gpu::TensorWrite<f32>,
) {
    let c = goldy::gpu::global_id().x;
    if c >= x.len() {
        return;
    }
    let kernel = state.dim(1);
    let base = c * kernel;
    let last = kernel - 1;
    let mut acc = 0.0;
    let mut k = 0u32;
    while k < last {
        acc = acc + weight[base + k] * state[base + k + 1];
        k = k + 1;
    }
    let cur = x[c];
    acc = acc + weight[base + last] * cur;
    k = 0;
    while k < last {
        state[base + k] = state[base + k + 1];
        k = k + 1;
    }
    state[base + last] = cur;
    out[c] = (acc + cur) / goldy::gpu::sqrt((kernel as f32) + 1.0);
}

/// Position-addressed static conv. Slot `position % kernel` is rewritten, so
/// submitting the same position again does not shift the history.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn short_conv_at(
    #[tensor(shape = [channels])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] weight: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] state: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [channels])] out: goldy::gpu::TensorWrite<f32>,
    step: &[crate::kernels::DecodeStep],
) {
    let c = goldy::gpu::global_id().x;
    if c >= x.len() {
        return;
    }
    let kernel = state.dim(1);
    let pos = step[0].position;
    let base = c * kernel;
    let cur = x[c];
    let mut acc = 0.0;
    let mut lag = 0u32;
    while lag < kernel {
        let mut token = cur;
        if lag > 0 {
            token = 0.0;
            if pos >= lag {
                let prev = (pos - lag) % kernel;
                token = state[base + prev];
            }
        }
        let tap = kernel - 1 - lag;
        acc = acc + weight[base + tap] * token;
        lag = lag + 1;
    }
    state[base + (pos % kernel)] = cur;
    out[c] = (acc + cur) / goldy::gpu::sqrt((kernel as f32) + 1.0);
}

/// One-token dynamic causal depthwise convolution.
///
/// `static_weight` is `[kernel, channels]` with tap 0 on the current token.
/// `taps` is `[heads, kernel]` in that same order, added to every channel of
/// the head. `state` matches [`short_conv`]. The normalizer is
/// `sqrt(2 * kernel + 1)`, not the static one.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn dynamic_short_conv(
    #[tensor(shape = [channels])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [kernel, channels])] static_weight: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [heads, kernel])] taps: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] state: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [channels])] out: goldy::gpu::TensorWrite<f32>,
) {
    let c = goldy::gpu::global_id().x;
    if c >= x.len() {
        return;
    }
    let kernel = state.dim(1);
    let channels = x.len();
    let heads = taps.dim(0);
    let head_dim = channels / heads;
    let head = c / head_dim;
    let base = c * kernel;
    let cur = x[c];
    let mut acc = (static_weight[c] + taps[head * kernel]) * cur;
    let mut lag = 1u32;
    while lag < kernel {
        let token = state[base + kernel - lag];
        let w = static_weight[lag * channels + c] + taps[head * kernel + lag];
        acc = acc + w * token;
        lag = lag + 1;
    }
    let mut k = 0u32;
    let last = kernel - 1;
    while k < last {
        state[base + k] = state[base + k + 1];
        k = k + 1;
    }
    state[base + last] = cur;
    out[c] = (acc + cur) / goldy::gpu::sqrt((2.0 * (kernel as f32)) + 1.0);
}

/// Position-addressed dynamic conv. Same ring rule as [`short_conv_at`].
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn dynamic_short_conv_at(
    #[tensor(shape = [channels])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [kernel, channels])] static_weight: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [heads, kernel])] taps: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [channels, kernel])] state: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [channels])] out: goldy::gpu::TensorWrite<f32>,
    step: &[crate::kernels::DecodeStep],
) {
    let c = goldy::gpu::global_id().x;
    if c >= x.len() {
        return;
    }
    let kernel = state.dim(1);
    let channels = x.len();
    let heads = taps.dim(0);
    let head_dim = channels / heads;
    let head = c / head_dim;
    let pos = step[0].position;
    let base = c * kernel;
    let cur = x[c];
    let mut acc = 0.0;
    let mut lag = 0u32;
    while lag < kernel {
        let mut token = cur;
        if lag > 0 {
            token = 0.0;
            if pos >= lag {
                token = state[base + ((pos - lag) % kernel)];
            }
        }
        let w = static_weight[lag * channels + c] + taps[head * kernel + lag];
        acc = acc + w * token;
        lag = lag + 1;
    }
    state[base + (pos % kernel)] = cur;
    out[c] = (acc + cur) / goldy::gpu::sqrt((2.0 * (kernel as f32)) + 1.0);
}

pub use dynamic_short_conv::Kernel as DynamicShortConvKernel;
pub use dynamic_short_conv_at::Kernel as DynamicShortConvAtKernel;
pub use head_l2_clip::Kernel as HeadL2ClipKernel;
pub use short_conv::Kernel as ShortConvKernel;
pub use short_conv_at::Kernel as ShortConvAtKernel;
pub use sigmoid_gate::Kernel as SigmoidGateKernel;
pub use softcap::Kernel as SoftcapKernel;
pub use softcapped_swiglu::Kernel as SoftcappedSwiGluKernel;
