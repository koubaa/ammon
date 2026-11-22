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

/// Cache slot for a logical position.
///
/// Ring mode (`window != 0`) is `logical % capacity`, so the allocation can be
/// wider than the window. Full mode is the position itself, and positions at
/// or past `capacity` are not a slot. The shaders inline this; a device
/// function call is not available in `#[goldy::compute]`.
#[allow(dead_code)]
fn cache_slot(logical: u32, window: u32, capacity: u32) -> Option<u32> {
    if window != 0 {
        Some(logical % capacity)
    } else if logical < capacity {
        Some(logical)
    } else {
        None
    }
}

/// Write the current key and value into the cache slot for `step.position`.
///
/// `window == 0` stores at the absolute position (full cache) and leaves the
/// cache untouched when that position is past the allocation. Otherwise the
/// slot is `position % capacity`.
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
    let capacity = key_cache.dim(0);
    // Same rule as `cache_slot`. Inlined because a shader body cannot call it.
    // A full-cache position past the allocation does not write. The host check
    // is `DecodeKvCache::deposit_position`; this is the guard for a replay
    // that deposits the step after `record`.
    let mut slot = pos;
    if window != 0 {
        slot = pos % capacity;
    } else if slot >= capacity {
        return;
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

    // Past the full-cache allocation: do not read those rows. The head output
    // is NaN so the fault is visible after the output projection.
    if window == 0 {
        if pos >= capacity {
            if local < head {
                out[h * head + local] = 0.0 / 0.0;
            }
            return;
        }
    }

    let mut n = pos + 1;
    let mut start = 0u32;
    if window != 0 {
        if n > window {
            start = pos + 1 - window;
            n = window;
        }
    } else if n > capacity {
        // A replayed step can name a position past the allocation. Attend only
        // the keys that exist; do not index the cache with `pos`.
        n = capacity;
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

/// Split-K score of [`relative_attention`]. Workgroup `h * splits + s` owns
/// eight compact keys of query head `h`.
///
/// Compact key `i` is logical position `start + i`. A ring maps that through
/// `logical % capacity`. The bias uses the logical distance. Splits that start
/// at or past the visible count return without writing; the combine reads only
/// the live prefix. A full-cache position past `capacity` returns without
/// writing so it cannot read off the end.
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn relative_attention_partial(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] key_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, kv_heads, head])] value_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, extent])] rel: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, splits, slot])] partial: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
    scale: f32,
    window: u32,
) {
    let mut qs = goldy::gpu::workgroup_array::<f32, 128>();
    let mut lanes = goldy::gpu::workgroup_array::<f32, 128>();
    let mut scores = goldy::gpu::workgroup_array::<f32, 8>();
    let mut probs = goldy::gpu::workgroup_array::<f32, 8>();
    let local = goldy::gpu::local_id().x;
    let splits = partial.dim(1);
    let group = goldy::gpu::workgroup_id().x;
    let h = group / splits;
    let split = group % splits;
    let head = q.dim(1);
    let extent = rel.dim(1);
    let capacity = key_cache.dim(0);
    let kv_heads = key_cache.dim(1);
    let groups = q.dim(0) / kv_heads;
    let kv_head = h / groups;
    let pos = step[0].position;
    if window == 0 {
        if pos >= capacity {
            return;
        }
    }
    let mut n = pos + 1;
    let mut start = 0u32;
    if window != 0 {
        if n > window {
            start = pos + 1 - window;
            n = window;
        }
    } else if n > capacity {
        n = capacity;
    }
    // Eight tokens per split while that covers the visible span. A longer
    // context keeps at most ATTENTION_MAX_SPLITS columns and loops chunks.
    let natural = (n + 7) / 8;
    let bounded = natural > splits;
    let mut per = 8u32;
    if bounded {
        per = (n + splits - 1) / splits;
    }
    let t_begin = split * per;
    if t_begin >= n {
        if bounded {
            let out = (h * splits + split) * partial.dim(2);
            if local < head {
                partial[out + local] = 0.0;
            }
            if local == 0 {
                partial[out + head] = -1.0e30;
                partial[out + head + 1] = 0.0;
            }
        }
        return;
    }
    let mut t_end = t_begin + per;
    if t_end > n {
        t_end = n;
    }
    let mut m_run = 0.0;
    let mut l_run = 0.0;
    let mut acc_run = 0.0;
    let mut started = 0u32;
    let mut offset = t_begin;
    while offset < t_end {
        let t0 = offset;
        let mut count = t_end - t0;
        if count > 8 {
            count = 8;
        }
        if local < head {
            qs[local] = q[h * head + local];
        }
        goldy::gpu::workgroup_barrier();

        let p = local / 16;
        let mut dot = 0.0;
        if p < count {
            let logical = start + t0 + p;
            let mut slot = logical;
            if window != 0 {
                slot = logical % capacity;
            }
            let k_base = (slot * kv_heads + kv_head) * head;
            let mut i = local % 16;
            while i < head {
                dot = dot + qs[i] * key_cache[k_base + i];
                i = i + 16;
            }
        }
        lanes[local] = dot;
        goldy::gpu::workgroup_barrier();
        if local < count {
            let b = local * 16;
            let mut sum = 0.0;
            for l in 0..16 {
                sum = sum + lanes[b + l];
            }
            let logical = start + t0 + local;
            let d = pos - logical;
            let mut bias = 0.0;
            if d < extent {
                bias = rel[h * extent + d];
            }
            scores[local] = sum * scale + bias;
        }
        goldy::gpu::workgroup_barrier();

        let mut m = scores[0];
        for j in 1..count {
            let s = scores[j];
            if s > m {
                m = s;
            }
        }
        let mut m_new = m;
        if started != 0u32 && m_run > m_new {
            m_new = m_run;
        }
        if local < count {
            probs[local] = goldy::gpu::exp(scores[local] - m_new);
        }
        goldy::gpu::workgroup_barrier();

        let ways = 128 / head;
        let way = local / head;
        let i = local % head;
        let mut acc = 0.0;
        if way < ways {
            let mut j = way;
            while j < count {
                let logical = start + t0 + j;
                let mut slot = logical;
                if window != 0 {
                    slot = logical % capacity;
                }
                let v = value_cache[(slot * kv_heads + kv_head) * head + i];
                acc = acc + probs[j] * v;
                j = j + ways;
            }
        }
        lanes[local] = acc;
        goldy::gpu::workgroup_barrier();

        let mut total = 0.0;
        if local < head {
            total = lanes[local];
            for w in 1..ways {
                total = total + lanes[w * head + local];
            }
        }
        let mut rescale = 1.0;
        if started != 0u32 {
            rescale = goldy::gpu::exp(m_run - m_new);
        }
        if local < head {
            acc_run = acc_run * rescale + total;
        }
        if local == 0 {
            let mut chunk_l = 0.0;
            for j in 0..count {
                chunk_l = chunk_l + probs[j];
            }
            l_run = l_run * rescale + chunk_l;
            lanes[0] = m_new;
            lanes[1] = l_run;
        }
        goldy::gpu::workgroup_barrier();
        m_run = lanes[0];
        l_run = lanes[1];
        started = 1u32;
        goldy::gpu::workgroup_barrier();
        offset = offset + count;
    }
    let out = (h * splits + split) * partial.dim(2);
    if local < head {
        partial[out + local] = acc_run;
    }
    if local == 0 {
        partial[out + head] = m_run;
        partial[out + head + 1] = l_run;
    }
}

/// Split-K combine for [`relative_attention_partial`], with the sink logit.
///
/// `m = max(split maxes, sink)` and the denominator gains `exp(sink - m)`.
/// A full-cache position past `capacity` writes NaN and does not read partials
/// that were never stored.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn relative_attention_combine(
    #[tensor(shape = [q_heads, splits, slot])] partial: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads])] sink: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, head])] out: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
    window: u32,
    capacity: u32,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let mut lanes = goldy::gpu::workgroup_array::<f32, 256>();
    let mut sums = goldy::gpu::workgroup_array::<f32, 256>();
    let h = goldy::gpu::workgroup_id().x;
    let local = goldy::gpu::local_id().x;
    let head = out.dim(1);
    let slot = partial.dim(2);
    let pos = step[0].position;
    if window == 0 {
        if pos >= capacity {
            if local < head {
                out[h * head + local] = 0.0 / 0.0;
            }
            return;
        }
    }
    let mut n = pos + 1;
    if window != 0 {
        if n > window {
            n = window;
        }
    } else if n > capacity {
        n = capacity;
    }
    let natural = (n + 7) / 8;
    let mut live = natural;
    if natural > partial.dim(1) {
        live = partial.dim(1);
    }
    let base = h * partial.dim(1) * slot;

    let mut m = partial[base + head];
    let mut s = local;
    while s < live {
        let v = partial[base + s * slot + head];
        if v > m {
            m = v;
        }
        s = s + 256;
    }
    m = goldy::gpu::workgroup_max::<256>(m, scratch);
    let sink_logit = sink[h];
    if sink_logit > m {
        m = sink_logit;
    }

    let ways = 256 / head;
    let way = local / head;
    let i = local % head;
    let mut l = 0.0;
    let mut acc = 0.0;
    if way < ways {
        let mut j = way;
        while j < live {
            let o = base + j * slot;
            let w = goldy::gpu::exp(partial[o + head] - m);
            l = l + partial[o + head + 1] * w;
            acc = acc + partial[o + i] * w;
            j = j + ways;
        }
    }
    lanes[local] = acc;
    if i == 0 {
        sums[way] = l;
    }
    goldy::gpu::workgroup_barrier();

    if local < head {
        let mut total = lanes[local];
        let mut norm = sums[0];
        for w in 1..ways {
            total = total + lanes[w * head + local];
            norm = norm + sums[w];
        }
        norm = norm + goldy::gpu::exp(sink_logit - m);
        out[h * head + local] = total / norm;
    }
}

pub use kv_cache_store::Kernel as KvCacheStoreKernel;
pub use relative_attention::Kernel as RelativeAttentionKernel;
pub use relative_attention_combine::Kernel as RelativeAttentionCombineKernel;
pub use relative_attention_partial::Kernel as RelativeAttentionPartialKernel;

#[cfg(test)]
mod tests {
    use super::cache_slot;

    #[test]
    fn cache_slot_is_modulo_for_a_ring_and_bounded_for_a_full_cache() {
        assert_eq!(cache_slot(5, 4, 4), Some(1));
        assert_eq!(cache_slot(5, 4, 8), Some(5));
        assert_eq!(cache_slot(1, 0, 2), Some(1));
        assert_eq!(cache_slot(2, 0, 2), None);
    }
}
