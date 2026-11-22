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
    #[tensor(shape = [k])] ids: goldy::gpu::TensorMut<u32>,
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
                if ids[p] == i {
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
            ids[pass] = index[0] as u32;
        }
        goldy::gpu::workgroup_barrier();
        pass = pass + 1;
    }
    if local == 0 {
        let mut i = 0u32;
        while i < k {
            let id = ids[i];
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

pub use moe_topk::Kernel as MoeTopkKernel;
