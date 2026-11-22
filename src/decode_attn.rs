//! Batch-1 decode attention with a ring or a full KV cache.
//!
//! Scores match eager Inkling 2 attention: `(Q Kᵀ) * scale + relative bias`,
//! then a per-head sink logit inside the softmax. The sink has no value. A
//! sliding layer attends `window` keys (distances `0 .. window-1`); a full
//! layer attends every key at `0 ..= position`, and a distance past
//! `rel_extent` adds zero bias. Head size is at most 128, the same cap as the
//! RoPE attention kernels. Q, K, the relative state, V's RMSNorm, and the
//! sigmoid output gate are recorded by the existing modules.

use crate::kernels::DecodeStep;

/// Write the current key and value into the cache slot for `step.position`.
///
/// `window == 0` stores at the absolute position (full cache). Otherwise the
/// cache length is the window and the slot is `position % window`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn kv_cache_store(
    #[tensor(shape = [kv_heads, head])] k: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [kv_heads, head])] v: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] key_cache: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] value_cache: goldy::gpu::TensorMut<f32>,
    step: &[DecodeStep],
    window: u32,
) {
    let i = goldy::gpu::global_id().x;
    let row = k.len();
    if i >= row {
        return;
    }
    let pos = step[0].position;
    let mut slot = pos;
    if window != 0 {
        slot = pos % key_cache.dim(0);
    }
    let dst = slot * row + i;
    key_cache[dst] = k[i];
    value_cache[dst] = v[i];
}

/// One query head against the keys visible at `step.position`.
///
/// Workgroup `h` owns query head `h`. `window == 0` reads keys `0 ..= position`.
/// A non-zero window reads the `window` logical positions ending at `position`
/// (fewer at the start of the sequence) through `position % capacity`.
/// `rel[h, d]` is added when `d = position - key < extent`; other distances add
/// nothing. The sink logit joins the softmax and is then dropped.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn relative_attention(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] key_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] value_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, extent])] rel: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads])] sink: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, head])] out: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
    scale: f32,
    window: u32,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let mut qs = goldy::gpu::workgroup_array::<f32, 128>();
    let h = goldy::gpu::workgroup_id().x;
    let local = goldy::gpu::local_id().x;
    let head = q.dim(1);
    let extent = rel.dim(1);
    let capacity = key_cache.dim(0);
    let kv_heads = key_cache.dim(1);
    let groups = q.dim(0) / kv_heads;
    let kv_head = h / groups;
    let pos = step[0].position;

    let mut n = pos + 1;
    let mut start = 0u32;
    if window != 0 {
        if n > window {
            start = pos + 1 - window;
            n = window;
        }
    }

    if local < head {
        qs[local] = q[h * head + local];
    }
    goldy::gpu::workgroup_barrier();

    let mut max_s = sink[h];
    let mut t = 0u32;
    while t < n {
        let logical = start + t;
        let mut slot = logical;
        if window != 0 {
            slot = logical % capacity;
        }
        let mut dot = 0.0;
        if local < head {
            let k = key_cache[(slot * kv_heads + kv_head) * head + local];
            dot = qs[local] * k;
        }
        dot = goldy::gpu::workgroup_sum::<256>(dot, scratch);
        let d = pos - logical;
        let mut bias = 0.0;
        if d < extent {
            bias = rel[h * extent + d];
        }
        let score = dot * scale + bias;
        if score > max_s {
            max_s = score;
        }
        t = t + 1;
    }

    let mut denom = goldy::gpu::exp(sink[h] - max_s);
    let mut acc = 0.0;
    t = 0;
    while t < n {
        let logical = start + t;
        let mut slot = logical;
        if window != 0 {
            slot = logical % capacity;
        }
        let mut dot = 0.0;
        if local < head {
            let k = key_cache[(slot * kv_heads + kv_head) * head + local];
            dot = qs[local] * k;
        }
        dot = goldy::gpu::workgroup_sum::<256>(dot, scratch);
        let d = pos - logical;
        let mut bias = 0.0;
        if d < extent {
            bias = rel[h * extent + d];
        }
        let w = goldy::gpu::exp(dot * scale + bias - max_s);
        denom = denom + w;
        if local < head {
            let v = value_cache[(slot * kv_heads + kv_head) * head + local];
            acc = acc + w * v;
        }
        t = t + 1;
    }

    if local < head {
        out[h * head + local] = acc / denom;
    }
}

pub use kv_cache_store::Kernel as KvCacheStoreKernel;
pub use relative_attention::Kernel as RelativeAttentionKernel;
