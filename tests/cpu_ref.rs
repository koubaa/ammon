//! Host reference for kernel and block checks.
//!
//! Reductions run in f64. Callers compare the f32 rounding of that reference to
//! the GPU result at atol 1e-5 and rtol 1e-4.

#![allow(dead_code)]

use ammon::kernels::DEFAULT_ROPE_THETA;

pub fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

pub fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{what}[{i}] = {g}, reference {w}"
        );
        let reference = f64::from(*w);
        let err = (f64::from(*g) - reference).abs();
        let tol = 1e-5_f64 + 1e-4_f64 * reference.abs();
        assert!(
            err <= tol,
            "{what}[{i}] = {g}, reference {w}, abs err {err} > tol {tol}"
        );
    }
}

/// `weight = None` is the unweighted form.
pub fn rmsnorm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let mut y = f64::from(*v) * scale;
            if let Some(weight) = weight {
                y *= f64::from(weight[i]);
            }
            y as f32
        })
        .collect()
}

/// Row-major `out[i] = dot(w[i, :], x)`.
pub fn gemv(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            let mut sum = 0.0_f64;
            for j in 0..cols {
                sum += f64::from(w[i * cols + j]) * f64::from(x[j]);
            }
            sum as f32
        })
        .collect()
}

pub fn swiglu(hb: &[f32], hb2: &[f32]) -> Vec<f32> {
    hb.iter()
        .zip(hb2)
        .map(|(a, b)| {
            let a = f64::from(*a);
            let silu = a / (1.0 + (-a).exp());
            (silu * f64::from(*b)) as f32
        })
        .collect()
}

/// Pairwise RoPE. `k` may be shorter than `q` (the kv projection).
pub fn rope(q: &mut [f32], k: &mut [f32], head: usize, pos: u32) {
    let theta = DEFAULT_ROPE_THETA;
    let mut i = 0;
    while i < q.len() {
        let head_dim = (i % head) as f32;
        let freq = 1.0_f32 / theta.powf(head_dim / head as f32);
        let val = pos as f32 * freq;
        let (fcr, fci) = (val.cos(), val.sin());
        let (q0, q1) = (q[i], q[i + 1]);
        q[i] = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;
        if i < k.len() {
            let (k0, k1) = (k[i], k[i + 1]);
            k[i] = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }
        i += 2;
    }
}

pub fn attention(
    q: &[f32],
    key: &[f32],
    value: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head: usize,
    pos: u32,
) -> Vec<f32> {
    let kv_dim = kv_heads * head;
    let kv_mul = q_heads / kv_heads;
    let mut out = vec![0.0_f32; q_heads * head];
    for h in 0..q_heads {
        let kv_base = (h / kv_mul) * head;
        let scores: Vec<f64> = (0..=pos as usize)
            .map(|t| {
                let dot: f64 = (0..head)
                    .map(|i| f64::from(q[h * head + i]) * f64::from(key[t * kv_dim + kv_base + i]))
                    .sum();
                dot / (head as f64).sqrt()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        for i in 0..head {
            out[h * head + i] = (weights
                .iter()
                .enumerate()
                .map(|(t, w)| w * f64::from(value[t * kv_dim + kv_base + i]))
                .sum::<f64>()
                / total) as f32;
        }
    }
    out
}

/// One pre-norm attention step. `key` / `value` are `[seq, kv_heads, head]` and are updated.
pub fn attention_block_step(
    x: &mut [f32],
    norm_w: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    key: &mut [f32],
    value: &mut [f32],
    q_heads: usize,
    kv_heads: usize,
    head: usize,
    pos: u32,
    eps: f32,
) {
    let hidden = x.len();
    let kv_dim = kv_heads * head;
    let xn = rmsnorm(x, Some(norm_w), eps);
    let mut q = gemv(wq, hidden, hidden, &xn);
    let mut k = gemv(wk, kv_dim, hidden, &xn);
    let v = gemv(wv, kv_dim, hidden, &xn);
    rope(&mut q, &mut k, head, pos);
    let base = pos as usize * kv_dim;
    key[base..base + kv_dim].copy_from_slice(&k);
    value[base..base + kv_dim].copy_from_slice(&v);
    let attn = attention(&q, key, value, q_heads, kv_heads, head, pos);
    let y = gemv(wo, hidden, hidden, &attn);
    for i in 0..hidden {
        x[i] = (f64::from(x[i]) + f64::from(y[i])) as f32;
    }
}

pub fn swiglu_block(
    x: &[f32],
    norm_w: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
    eps: f32,
) -> Vec<f32> {
    let hidden = x.len();
    let intermediate = w1.len() / hidden;
    let xn = rmsnorm(x, Some(norm_w), eps);
    let hb = gemv(w1, intermediate, hidden, &xn);
    let hb2 = gemv(w3, intermediate, hidden, &xn);
    let gated = swiglu(&hb, &hb2);
    let y = gemv(w2, hidden, intermediate, &gated);
    x.iter()
        .zip(y)
        .map(|(a, b)| (f64::from(*a) + f64::from(b)) as f32)
        .collect()
}

pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Per-head L2 clip. `x` and `scale` are `[heads, dim]` row-major.
/// `alpha == 0` skips the log-position term. The norm epsilon is `1e-6`.
pub fn head_l2_clip(
    x: &[f32],
    heads: usize,
    dim: usize,
    scale: Option<&[f32]>,
    threshold: f32,
    alpha: f32,
    floor: f64,
    pos: u32,
) -> Vec<f32> {
    let mut out = vec![0.0; heads * dim];
    let mut log_mul = 1.0_f64;
    if alpha != 0.0 {
        let mut t = (f64::from(pos) + 1.0) / floor;
        if t < 1.0 {
            t = 1.0;
        }
        log_mul = 1.0 + f64::from(alpha) * t.ln();
    }
    for h in 0..heads {
        let mut sumsq = 0.0_f64;
        for j in 0..dim {
            let mut v = f64::from(x[h * dim + j]);
            if let Some(scale) = scale {
                v *= f64::from(scale[h * dim + j]);
            }
            sumsq += v * v;
        }
        let mut factor = f64::from(threshold) / (sumsq.sqrt() + 1e-6);
        if factor > 1.0 {
            factor = 1.0;
        }
        factor *= log_mul;
        for j in 0..dim {
            let mut v = f64::from(x[h * dim + j]);
            if let Some(scale) = scale {
                v *= f64::from(scale[h * dim + j]);
            }
            out[h * dim + j] = (v * factor) as f32;
        }
    }
    out
}

pub fn softcap(x: &[f32], cap: f64) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let z = f64::from(*v) / cap;
            (cap * z.tanh()) as f32
        })
        .collect()
}

pub fn sigmoid_gate(values: &[f32], logits: &[f32]) -> Vec<f32> {
    values
        .iter()
        .zip(logits)
        .map(|(v, z)| (f64::from(*v) * sigmoid(f64::from(*z))) as f32)
        .collect()
}

pub fn softcapped_swiglu(gate: &[f32], up: &[f32], gate_cap: f64, up_cap: f64) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let g = f64::from(*g);
            let u = f64::from(*u);
            let capped_g = gate_cap * (g / gate_cap).tanh();
            let capped_u = up_cap * (u / up_cap).tanh();
            (capped_g * sigmoid(g) * capped_u) as f32
        })
        .collect()
}

/// Static conv1d step. `weight` and `state` are `[channels, kernel]`, oldest tap first.
pub fn short_conv_step(
    state: &mut [f32],
    channels: usize,
    kernel: usize,
    x: &[f32],
    weight: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; channels];
    for c in 0..channels {
        let base = c * kernel;
        let last = kernel - 1;
        let mut acc = 0.0_f64;
        for k in 0..last {
            acc += f64::from(weight[base + k]) * f64::from(state[base + k + 1]);
        }
        let cur = f64::from(x[c]);
        acc += f64::from(weight[base + last]) * cur;
        for k in 0..last {
            state[base + k] = state[base + k + 1];
        }
        state[base + last] = x[c];
        out[c] = ((acc + cur) / ((kernel as f64) + 1.0).sqrt()) as f32;
    }
    out
}

/// Dynamic conv step. `static_weight` is `[kernel, channels]` with tap 0 on the
/// current token. `taps` is `[heads, kernel]` in that order.
pub fn dynamic_short_conv_step(
    state: &mut [f32],
    channels: usize,
    heads: usize,
    kernel: usize,
    x: &[f32],
    static_weight: &[f32],
    taps: &[f32],
) -> Vec<f32> {
    let head_dim = channels / heads;
    let mut out = vec![0.0; channels];
    for c in 0..channels {
        let head = c / head_dim;
        let base = c * kernel;
        let cur = f64::from(x[c]);
        let mut acc = (f64::from(static_weight[c]) + f64::from(taps[head * kernel])) * cur;
        for lag in 1..kernel {
            let token = f64::from(state[base + kernel - lag]);
            let w =
                f64::from(static_weight[lag * channels + c]) + f64::from(taps[head * kernel + lag]);
            acc += w * token;
        }
        let last = kernel - 1;
        for k in 0..last {
            state[base + k] = state[base + k + 1];
        }
        state[base + last] = x[c];
        out[c] = ((acc + cur) / ((2.0 * kernel as f64) + 1.0).sqrt()) as f32;
    }
    out
}
