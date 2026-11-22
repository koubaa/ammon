//! Batch-1 routed experts.
//!
//! The router is `sigmoid(W x)`. Top-k is taken on `sigmoid + bias`, but the
//! bias does not enter the weights. Ties keep the smaller expert index. The
//! selected ids are then sorted ascending. Each selected expert is
//! `down(softcap(gate) * sigmoid(gate) * softcap(up))`, and the outputs are
//! summed with the normalized route weights. There is no shared expert.

/// Select `k` experts on the device and write normalized route weights.
///
/// One workgroup. `ids` and `weights` are length `k`. A later token only
/// changes `logits`; the scheme does not read them back.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn moe_topk(
    #[tensor(shape = [n])] logits: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] bias: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [k])] ids: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [k])] weights: goldy::gpu::TensorMut<f32>,
    route_scale: f32,
) {
    let mut scores = goldy::gpu::workgroup_array::<f32, 256>();
    let mut index = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let n = logits.len();
    let k = ids.len();
    let mut pass = 0u32;
    while pass < k {
        let mut best = -1.0e30;
        let mut best_i = n;
        let mut i = local;
        while i < n {
            let mut taken = 0u32;
            let mut p = 0u32;
            while p < pass {
                if (ids[p] as u32) == i {
                    taken = 1;
                }
                p = p + 1;
            }
            if taken == 0 {
                let s = goldy::gpu::sigmoid(logits[i]) + bias[i];
                if best_i == n || s > best || (s == best && i < best_i) {
                    best = s;
                    best_i = i;
                }
            }
            i = i + 256;
        }
        scores[local] = best;
        index[local] = best_i as f32;
        goldy::gpu::workgroup_barrier();
        let mut step = 128u32;
        while step > 0 {
            if local < step {
                let other = local + step;
                let os = scores[other];
                let oi = index[other] as u32;
                let ms = scores[local];
                let mi = index[local] as u32;
                let mut replace = 0u32;
                if oi < n {
                    if mi >= n {
                        replace = 1;
                    } else if os > ms || (os == ms && oi < mi) {
                        replace = 1;
                    }
                }
                if replace == 1 {
                    scores[local] = os;
                    index[local] = oi as f32;
                }
            }
            goldy::gpu::workgroup_barrier();
            step = step / 2;
        }
        if local == 0 {
            ids[pass] = index[0];
        }
        goldy::gpu::workgroup_barrier();
        pass = pass + 1;
    }
    if local == 0 {
        let mut i = 0u32;
        while i < k {
            let id = ids[i] as u32;
            weights[i] = goldy::gpu::sigmoid(logits[id]);
            i = i + 1;
        }
        let mut a = 0u32;
        while a + 1 < k {
            let mut m = a;
            let mut b = a + 1;
            while b < k {
                if ids[b] < ids[m] {
                    m = b;
                }
                b = b + 1;
            }
            if m != a {
                let id_tmp = ids[a];
                ids[a] = ids[m];
                ids[m] = id_tmp;
                let w_tmp = weights[a];
                weights[a] = weights[m];
                weights[m] = w_tmp;
            }
            a = a + 1;
        }
        let mut sum = 0.0;
        i = 0;
        while i < k {
            sum = sum + weights[i];
            i = i + 1;
        }
        let scale = route_scale / (sum + 1.0e-20);
        i = 0;
        while i < k {
            weights[i] = weights[i] * scale;
            i = i + 1;
        }
    }
}

/// `gate` and `up` projections of expert `ids[slot]`.
///
/// `gate_up` is `[experts, 2 * intermediate, hidden]`, concatenated `[gate | up]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn moe_gate_up(
    #[tensor(shape = [hidden])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [experts, fused, hidden])] gate_up: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [k])] ids: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [intermediate])] gate: goldy::gpu::TensorWrite<f32>,
    #[tensor(shape = [intermediate])] up: goldy::gpu::TensorWrite<f32>,
    slot: u32,
) {
    let i = goldy::gpu::global_id().x;
    let inter = gate.len();
    if i >= inter {
        return;
    }
    let hidden = x.len();
    let expert = ids[slot] as u32;
    let fused = gate_up.dim(1);
    let gate_base = (expert * fused + i) * hidden;
    let up_base = (expert * fused + inter + i) * hidden;
    let mut g = 0.0;
    let mut u = 0.0;
    let mut j = 0u32;
    while j < hidden {
        let xv = x[j];
        g = g + gate_up[gate_base + j] * xv;
        u = u + gate_up[up_base + j] * xv;
        j = j + 1;
    }
    gate[i] = g;
    up[i] = u;
}

/// Add `weights[slot] * down[expert] @ activated` into `out`.
///
/// `down` is `[experts, hidden, intermediate]`. `out` must already be zero for
/// the first expert; later experts accumulate.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn moe_down_accum(
    #[tensor(shape = [intermediate])] activated: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [experts, hidden, intermediate])] down: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [k])] ids: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [k])] weights: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [hidden])] out: goldy::gpu::TensorMut<f32>,
    slot: u32,
) {
    let h = goldy::gpu::global_id().x;
    let hidden = out.len();
    if h >= hidden {
        return;
    }
    let inter = activated.len();
    let expert = ids[slot] as u32;
    let base = (expert * hidden + h) * inter;
    let mut dot = 0.0;
    let mut i = 0u32;
    while i < inter {
        dot = dot + down[base + i] * activated[i];
        i = i + 1;
    }
    out[h] = out[h] + weights[slot] * dot;
}

pub use moe_down_accum::Kernel as MoeDownAccumKernel;
pub use moe_gate_up::Kernel as MoeGateUpKernel;
pub use moe_topk::Kernel as MoeTopkKernel;
