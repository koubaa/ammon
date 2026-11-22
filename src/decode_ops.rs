//! Pointwise decode ops and one-token causal convolutions.
//!
//! `tanh` and `sigmoid` are written from `exp` so this crate does not wait on
//! Goldy builtins. Each site is marked. When `goldy::gpu::tanh` and
//! `goldy::gpu::sigmoid` exist, those calls replace the expansions below.

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

/// `out = cap * tanh(x / cap)`.
///
/// For `|z| < 0.5`, `tanh` is the odd Taylor series through `z^11`. The
/// `exp` form cancels near zero. Outside that, `z >= 0` uses
/// `(1 - e^{-2z}) / (1 + e^{-2z})` and `z < 0` uses `(e^{2z} - 1) / (e^{2z} + 1)`.
/// Replace this with `goldy::gpu::tanh` once that builtin matches libm for
/// small `z` (it does not, as of goldy `83052da`). Keep the three copies
/// (here and both softcaps in `softcapped_swiglu`) on the same series.
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
    let z = x[i] / cap;
    let mut az = z;
    if az < 0.0 {
        az = -az;
    }
    let mut t = 0.0;
    if az < 0.5 {
        let z2 = z * z;
        t = z
            * (1.0
                + z2 * ((-1.0 / 3.0)
                    + z2 * ((2.0 / 15.0)
                        + z2 * ((-17.0 / 315.0)
                            + z2 * ((62.0 / 2835.0) + z2 * (-1382.0 / 155925.0))))));
    } else if z >= 0.0 {
        let e = goldy::gpu::exp(-2.0 * z);
        t = (1.0 - e) / (1.0 + e);
    } else {
        let e = goldy::gpu::exp(2.0 * z);
        t = (e - 1.0) / (e + 1.0);
    }
    out[i] = t * cap;
}

/// `values *= sigmoid(logits)`, elementwise.
///
/// Stable sigmoid: `1 / (1 + e^{-z})` for `z >= 0`, else `e^{z} / (1 + e^{z})`.
/// Replace with `goldy::gpu::sigmoid` when that builtin exists.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn sigmoid_gate(
    #[tensor(shape = [n])] values: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [n])] logits: goldy::gpu::Tensor<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i >= values.len() {
        return;
    }
    let z = logits[i];
    let mut s = 0.0;
    if z >= 0.0 {
        s = 1.0 / (1.0 + goldy::gpu::exp(-z));
    } else {
        let e = goldy::gpu::exp(z);
        s = e / (1.0 + e);
    }
    values[i] = values[i] * s;
}

/// `out = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)`.
///
/// `sigmoid` sees the raw gate. Both softcaps use the same `tanh` as
/// [`softcap`], including the small-`z` series. Replace them with
/// `goldy::gpu::tanh` and `goldy::gpu::sigmoid` when those builtins are precise.
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
    let gz = g / gate_cap;
    let mut gaz = gz;
    if gaz < 0.0 {
        gaz = -gaz;
    }
    let mut gt = 0.0;
    if gaz < 0.5 {
        let z2 = gz * gz;
        gt = gz
            * (1.0
                + z2 * ((-1.0 / 3.0)
                    + z2 * ((2.0 / 15.0)
                        + z2 * ((-17.0 / 315.0)
                            + z2 * ((62.0 / 2835.0) + z2 * (-1382.0 / 155925.0))))));
    } else if gz >= 0.0 {
        let e = goldy::gpu::exp(-2.0 * gz);
        gt = (1.0 - e) / (1.0 + e);
    } else {
        let e = goldy::gpu::exp(2.0 * gz);
        gt = (e - 1.0) / (e + 1.0);
    }
    let uz = u / up_cap;
    let mut uaz = uz;
    if uaz < 0.0 {
        uaz = -uaz;
    }
    let mut ut = 0.0;
    if uaz < 0.5 {
        let z2 = uz * uz;
        ut = uz
            * (1.0
                + z2 * ((-1.0 / 3.0)
                    + z2 * ((2.0 / 15.0)
                        + z2 * ((-17.0 / 315.0)
                            + z2 * ((62.0 / 2835.0) + z2 * (-1382.0 / 155925.0))))));
    } else if uz >= 0.0 {
        let e = goldy::gpu::exp(-2.0 * uz);
        ut = (1.0 - e) / (1.0 + e);
    } else {
        let e = goldy::gpu::exp(2.0 * uz);
        ut = (e - 1.0) / (e + 1.0);
    }
    let mut s = 0.0;
    if g >= 0.0 {
        s = 1.0 / (1.0 + goldy::gpu::exp(-g));
    } else {
        let e = goldy::gpu::exp(g);
        s = e / (1.0 + e);
    }
    out[i] = (gt * gate_cap) * s * (ut * up_cap);
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

pub use dynamic_short_conv::Kernel as DynamicShortConvKernel;
pub use head_l2_clip::Kernel as HeadL2ClipKernel;
pub use short_conv::Kernel as ShortConvKernel;
pub use sigmoid_gate::Kernel as SigmoidGateKernel;
pub use softcap::Kernel as SoftcapKernel;
pub use softcapped_swiglu::Kernel as SoftcappedSwiGluKernel;
