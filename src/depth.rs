//! Depth attention over the block tape.
//!
//! Keys and values are an unweighted RMSNorm of `cat(blocks, partial)` with
//! the class default `eps = 1e-6` (not `rms_norm_eps`). Queries are learned and
//! are not normalized. There is no mask and no sink. The scale is
//! `hidden^{-0.5}`. Softmax is over the block axis.

/// Unweighted depth-norm epsilon. The class default, not `rms_norm_eps`.
pub const DEPTH_RMS_EPS: f32 = 1e-6;

/// Most block keys one mix reads, including the partial. Mini stays well under this.
pub const DEPTH_MAX_KEYS: u32 = 32;

/// Copy `src` into `tape[slot]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn depth_store(
    #[tensor(shape = [hidden])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, hidden])] tape: goldy::gpu::TensorMut<f32>,
    slot: u32,
) {
    let i = goldy::gpu::global_id().x;
    let hidden = src.len();
    if i >= hidden {
        return;
    }
    tape[slot * hidden + i] = src[i];
}

/// Copy the first `n` rows from `src` onto `dst`. Used to restore the pre-loop snapshot.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn depth_copy_rows(
    #[tensor(shape = [capacity, hidden])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [capacity, hidden])] dst: goldy::gpu::TensorMut<f32>,
    src_slot: u32,
    n: u32,
) {
    let hidden = src.dim(1);
    let i = goldy::gpu::global_id().x;
    let total = n * hidden;
    if i >= total {
        return;
    }
    let row = i / hidden;
    let col = i - row * hidden;
    dst[i] = src[(src_slot + row) * hidden + col];
}

/// One query head of a depth mix. Workgroup `h` owns query `h`.
///
/// `n_blocks` rows of `blocks` are valid. The partial is one more key. `n_blocks`
/// may be 0 (messages whose previous-block slice is empty).
///
/// Every key is reduced at once: the workgroup splits into `lanes`-wide groups, one per
/// key, and one tree folds each group's norm and query dot. The barrier count depends
/// on `lanes`, not on the number of keys.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn depth_mix(
    #[tensor(shape = [capacity, hidden])] blocks: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [hidden])] partial: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n_queries, hidden])] queries: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n_queries, hidden])] out: goldy::gpu::TensorWrite<f32>,
    n_blocks: u32,
    eps: f32,
    scale: f32,
) {
    let mut ss_part = goldy::gpu::workgroup_array::<f32, 256>();
    let mut dot_part = goldy::gpu::workgroup_array::<f32, 256>();
    let mut scores = goldy::gpu::workgroup_array::<f32, 32>();
    let mut invs = goldy::gpu::workgroup_array::<f32, 32>();
    let mut probs = goldy::gpu::workgroup_array::<f32, 32>();
    let h = goldy::gpu::workgroup_id().x;
    let local = goldy::gpu::local_id().x;
    let hidden = partial.len();
    let n_keys = n_blocks + 1;
    let q_base = h * hidden;

    let mut lanes = 256u32;
    while lanes * n_keys > 256 {
        lanes = lanes / 2;
    }
    let key = local / lanes;
    let lane = local - key * lanes;
    let mut ss = 0.0;
    let mut dot = 0.0;
    if key < n_keys {
        let mut j = lane;
        while j < hidden {
            let mut v = partial[j];
            if key < n_blocks {
                v = blocks[key * hidden + j];
            }
            ss = ss + v * v;
            dot = dot + queries[q_base + j] * v;
            j = j + lanes;
        }
    }
    ss_part[local] = ss;
    dot_part[local] = dot;
    goldy::gpu::workgroup_barrier();
    let mut stride = lanes / 2;
    while stride > 0 {
        if lane < stride {
            ss_part[local] = ss_part[local] + ss_part[local + stride];
            dot_part[local] = dot_part[local] + dot_part[local + stride];
        }
        goldy::gpu::workgroup_barrier();
        stride = stride / 2;
    }
    if local < n_keys {
        let inv = 1.0 / goldy::gpu::sqrt(ss_part[local * lanes] / (hidden as f32) + eps);
        scores[local] = dot_part[local * lanes] * inv * scale;
        invs[local] = inv;
    }
    goldy::gpu::workgroup_barrier();

    if local == 0 {
        let mut m = scores[0];
        let mut k = 1u32;
        while k < n_keys {
            if scores[k] > m {
                m = scores[k];
            }
            k = k + 1;
        }
        let mut sum = 0.0;
        k = 0;
        while k < n_keys {
            let e = goldy::gpu::exp(scores[k] - m);
            probs[k] = e;
            sum = sum + e;
            k = k + 1;
        }
        k = 0;
        while k < n_keys {
            probs[k] = probs[k] / sum;
            k = k + 1;
        }
    }
    goldy::gpu::workgroup_barrier();

    let mut j = local;
    while j < hidden {
        let mut acc = 0.0;
        let mut key = 0u32;
        while key < n_keys {
            let mut v = partial[j];
            if key < n_blocks {
                v = blocks[key * hidden + j];
            }
            acc = acc + probs[key] * (v * invs[key]);
            key = key + 1;
        }
        out[q_base + j] = acc;
        j = j + 256;
    }
}

pub use depth_copy_rows::Kernel as DepthCopyRowsKernel;
pub use depth_mix::Kernel as DepthMixKernel;
pub use depth_store::Kernel as DepthStoreKernel;
