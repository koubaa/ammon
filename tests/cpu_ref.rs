//! Host reference for kernel and block checks.
//!
//! Reductions run in f64. Callers compare the f32 rounding of that reference to
//! the GPU result at atol 1e-5 and rtol 1e-4.

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
