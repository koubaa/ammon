//! Neural-net GPU kernels. Domain ops, not a specific architecture.

#![allow(clippy::too_many_arguments)]

pub const WORKGROUP: u32 = 256;

/// Llama-2 / llama2.c default RoPE base frequency.
pub const DEFAULT_ROPE_THETA: f32 = 10000.0;

/// Token and sequence position uploaded each decode step.
#[goldy::gpu]
pub struct DecodeStep {
    pub token: u32,
    pub position: u32,
}

impl DecodeStep {
    /// Retained scattered parcel for a worker deposit. Contents are overwritten each step.
    pub fn parcel(runtime: &goldy::Runtime) -> anyhow::Result<goldy::Buffer> {
        Ok(runtime.acquire_buffer_with_data(
            &[Self {
                token: 0,
                position: 0,
            }],
            goldy::BufferKind::Scattered,
        )?)
    }

    /// Deposit target covering one [`DecodeStep`] in `buffer`.
    pub fn deposit_target(buffer: &goldy::Buffer) -> goldy::DepositTarget<'_> {
        goldy::DepositTarget::buffer_elements::<Self>(buffer, 1)
    }
}

/// Gather one embedding row from a `[vocab, dim]` view into `x`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn embed(
    #[tensor(shape = [vocab, dim])] embed: goldy::gpu::Tensor<f32>,
    step: &[DecodeStep],
    #[tensor(shape = [dim])] x: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < x.len() {
        let token = step[0].token;
        x[i] = embed[token * x.len() + i];
    }
}

/// Position-strided serial GEMV: `[n] × [d, n] -> [seq, d]` at `step.position`.
///
/// The decode path does not use this. `wk` / `wv` are semantic matmuls into scratch,
/// then [`rope_store`] and [`cache_store`] write the cache. This kernel remains the
/// exact serial reference for tests.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn gemv(
    #[tensor(shape = [n])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [d, n])] w: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, d])] xout: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
) {
    let i = goldy::gpu::global_id().x;
    let n = x.len();
    let d = w.dim(0);
    if i < d {
        let pos = step[0].position;
        let out_i = pos * xout.dim(1) + i;
        let mut sum = 0.0;
        for j in 0..n {
            sum = sum + w[i * n + j] * x[j];
        }
        xout[out_i] = sum;
    }
}

/// RMSNorm into `o` (one workgroup, strided over `x.len()`).
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rmsnorm(
    #[tensor(shape = [dim])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [dim])] weight: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [dim])] o: goldy::gpu::TensorWrite<f32>,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let size = x.len();
    let mut ss = 0.0;
    let mut j = local;
    while j < size {
        let v = x[j];
        ss = ss + v * v;
        j = j + 256;
    }
    ss = goldy::gpu::workgroup_sum::<256>(ss, scratch);
    ss = ss / (size as f32);
    ss = ss + 1e-5;
    ss = 1.0 / goldy::gpu::sqrt(ss);
    j = local;
    while j < size {
        o[j] = weight[j] * (ss * x[j]);
        j = j + 256;
    }
}

/// In-place RMSNorm on `x`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rmsnorm_inplace(
    #[tensor(shape = [dim])] x: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [dim])] weight: goldy::gpu::Tensor<f32>,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let size = x.len();
    let mut ss = 0.0;
    let mut j = local;
    while j < size {
        let v = x[j];
        ss = ss + v * v;
        j = j + 256;
    }
    ss = goldy::gpu::workgroup_sum::<256>(ss, scratch);
    ss = ss / (size as f32);
    ss = ss + 1e-5;
    ss = 1.0 / goldy::gpu::sqrt(ss);
    j = local;
    while j < size {
        x[j] = weight[j] * (ss * x[j]);
        j = j + 256;
    }
}

/// Pairwise RoPE on Q and an already-written K cache row.
///
/// Decode uses [`rope_store`], which reads unrotated K from scratch. This kernel
/// rotates K in place and is the identity check at position 0.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rope(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [seq, kv_heads, head])] k: goldy::gpu::TensorMut<f32>,
    step: &[DecodeStep],
    theta: f32,
) {
    let i = goldy::gpu::global_id().x * 2;
    let dim = q.len();
    if i >= dim {
        return;
    }
    let pos = step[0].position;
    let head_size = q.dim(1);
    let kv_dim = k.dim(1) * k.dim(2);
    let k_base = pos * kv_dim;
    let head_dim = (i % head_size) as i32;
    let freq = 1.0 / goldy::gpu::pow(theta, (head_dim as f32) / (head_size as f32));
    let val = (pos as f32) * freq;
    let fcr = goldy::gpu::cos(val);
    let fci = goldy::gpu::sin(val);
    let mut rotn = 1;
    if i < kv_dim {
        rotn = 2;
    }
    for v in 0..rotn {
        if v == 0 {
            let v0 = q[i];
            let v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        } else {
            let v0 = k[k_base + i];
            let v1 = k[k_base + i + 1];
            k[k_base + i] = v0 * fcr - v1 * fci;
            k[k_base + i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

/// Rotate Q in place, and write RoPE(K) from a flat projection into `k[step.position]`.
///
/// One thread owns pair `(2i, 2i+1)`. Q pairs cover the full query width. K pairs
/// stop at `k_proj.len()`, which is `kv_dim` and is at most the query width.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rope_store(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [kv_dim])] k_proj: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, kv_heads, head])] k: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
    theta: f32,
) {
    let i = goldy::gpu::global_id().x * 2;
    let dim = q.len();
    if i >= dim {
        return;
    }
    let pos = step[0].position;
    let head_size = q.dim(1);
    let kv_dim = k_proj.len();
    let k_base = pos * kv_dim;
    let head_dim = (i % head_size) as i32;
    let freq = 1.0 / goldy::gpu::pow(theta, (head_dim as f32) / (head_size as f32));
    let val = (pos as f32) * freq;
    let fcr = goldy::gpu::cos(val);
    let fci = goldy::gpu::sin(val);
    let q0 = q[i];
    let q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    if i < kv_dim {
        let k0 = k_proj[i];
        let k1 = k_proj[i + 1];
        k[k_base + i] = k0 * fcr - k1 * fci;
        k[k_base + i + 1] = k0 * fci + k1 * fcr;
    }
}

/// Copy a flat projection into `dst[step.position]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn cache_store(
    #[tensor(shape = [kv_dim])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, kv_heads, head])] dst: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
) {
    let i = goldy::gpu::global_id().x;
    if i >= src.len() {
        return;
    }
    let row = dst.dim(1) * dst.dim(2);
    let out_i = step[0].position * row + i;
    dst[out_i] = src[i];
}

/// Cached positions reduced by one [`attention_partial`] workgroup.
pub const ATTENTION_SPLIT: u32 = 32;
/// Widest head [`attention_partial`] and [`attention_combine`] cover (128 threads).
pub const ATTENTION_MAX_HEAD: u32 = 128;

/// Flash-decoding pass 1. Workgroup `h * splits + s` scores positions
/// `[32 s, 32 s + 32)` clipped to `t <= pos` for query head `h`.
///
/// Writes `Σ exp(score - m) · v` to `partial[h, s, ..head]`, the split max `m` to
/// `partial[h, s, head]` and `Σ exp(score - m)` to `partial[h, s, head + 1]`.
/// Splits that start past `pos` return without writing; [`attention_combine`] reads
/// only live splits. Requires `head <= 128`.
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn attention_partial(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, kv_heads, head])] key_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, kv_heads, head])] value_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, splits, slot])] partial: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
) {
    let mut qs = goldy::gpu::workgroup_array::<f32, 128>();
    let mut lanes = goldy::gpu::workgroup_array::<f32, 128>();
    let mut scores = goldy::gpu::workgroup_array::<f32, 32>();
    let mut probs = goldy::gpu::workgroup_array::<f32, 32>();
    let local = goldy::gpu::local_id().x;
    let splits = partial.dim(1);
    let group = goldy::gpu::workgroup_id().x;
    let h = group / splits;
    let split = group % splits;
    let pos = step[0].position;
    let t0 = split * 32;
    if t0 > pos {
        return;
    }
    let mut n = pos + 1 - t0;
    if n > 32 {
        n = 32;
    }
    let head_size = q.dim(1);
    let kv_mul = q.dim(0) / key_cache.dim(1);
    let kv_dim = key_cache.dim(1) * head_size;
    let head_base = (h / kv_mul) * head_size;

    if local < head_size {
        qs[local] = q[h * head_size + local];
    }
    goldy::gpu::workgroup_barrier();

    // Four lanes per position; lane `l` sums dims `l, l + 4, ...`.
    let p = local / 4;
    let mut dot = 0.0;
    if p < n {
        let k_base = (t0 + p) * kv_dim + head_base;
        let mut i = local % 4;
        while i < head_size {
            dot = dot + qs[i] * key_cache[k_base + i];
            i = i + 4;
        }
    }
    lanes[local] = dot;
    goldy::gpu::workgroup_barrier();
    if local < n {
        let b = local * 4;
        let sum = lanes[b] + lanes[b + 1] + lanes[b + 2] + lanes[b + 3];
        scores[local] = sum / goldy::gpu::sqrt(head_size as f32);
    }
    goldy::gpu::workgroup_barrier();

    let mut m = scores[0];
    for j in 1..n {
        let s = scores[j];
        if s > m {
            m = s;
        }
    }
    if local < n {
        probs[local] = goldy::gpu::exp(scores[local] - m);
    }
    goldy::gpu::workgroup_barrier();

    // `ways` thread groups of `head` lanes split the positions, then fold.
    let ways = 128 / head_size;
    let way = local / head_size;
    let i = local % head_size;
    let mut acc = 0.0;
    if way < ways {
        let mut j = way;
        while j < n {
            acc = acc + probs[j] * value_cache[(t0 + j) * kv_dim + head_base + i];
            j = j + ways;
        }
    }
    lanes[local] = acc;
    goldy::gpu::workgroup_barrier();

    let out = (h * splits + split) * partial.dim(2);
    if local < head_size {
        let mut total = lanes[local];
        for w in 1..ways {
            total = total + lanes[w * head_size + local];
        }
        partial[out + local] = total;
    }
    if local == 0 {
        let mut l = 0.0;
        for j in 0..n {
            l = l + probs[j];
        }
        partial[out + head_size] = m;
        partial[out + head_size + 1] = l;
    }
}

/// Flash-decoding pass 2: rescale and sum the live [`attention_partial`] splits of
/// head `workgroup_id().x` into `xb[h]`.
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn attention_combine(
    #[tensor(shape = [q_heads, splits, slot])] partial: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, head])] xb: goldy::gpu::TensorWrite<f32>,
    step: &[DecodeStep],
) {
    let h = goldy::gpu::workgroup_id().x;
    let i = goldy::gpu::local_id().x;
    let head_size = xb.dim(1);
    if i >= head_size {
        return;
    }
    let slot = partial.dim(2);
    let base = h * partial.dim(1) * slot;
    let live = step[0].position / 32 + 1;
    let mut m = partial[base + head_size];
    for s in 1..live {
        let v = partial[base + s * slot + head_size];
        if v > m {
            m = v;
        }
    }
    let mut l = 0.0;
    let mut acc = 0.0;
    for s in 0..live {
        let o = base + s * slot;
        let w = goldy::gpu::exp(partial[o + head_size] - m);
        l = l + partial[o + head_size + 1] * w;
        acc = acc + partial[o + i] * w;
    }
    xb[h * head_size + i] = acc / l;
}

/// Serial multi-head attention (one workgroup per head). Inclusive over `t <= pos`.
///
/// Decode uses [`attention_partial`] plus [`attention_combine`]. This kernel remains
/// the exact reference for tests.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn attention(
    #[tensor(shape = [q_heads, head])] q: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [q_heads, seq])] att: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [q_heads, head])] xb: goldy::gpu::TensorWrite<f32>,
    #[tensor(shape = [seq, kv_heads, head])] key_cache: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [seq, kv_heads, head])] value_cache: goldy::gpu::Tensor<f32>,
    step: &[DecodeStep],
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let h = goldy::gpu::workgroup_id().x;
    let local = goldy::gpu::local_id().x;
    let pos = step[0].position;
    let head_size = q.dim(1);
    let seq_len = att.dim(1);
    let att_base = h * seq_len;
    let q_base = h * head_size;
    let kv_mul = q.dim(0) / key_cache.dim(1);
    let kv_head = h / kv_mul;
    let kv_dim = key_cache.dim(1) * key_cache.dim(2);

    let mut t = local;
    while t <= pos {
        let mut score = 0.0;
        let k_base = t * kv_dim + kv_head * head_size;
        for i in 0..head_size {
            score = score + q[q_base + i] * key_cache[k_base + i];
        }
        score = score / goldy::gpu::sqrt(head_size as f32);
        att[att_base + t] = score;
        t = t + 256;
    }
    goldy::gpu::workgroup_softmax_in_place::<256>(att, att_base, pos + 1, scratch);

    let mut i = local;
    while i < head_size {
        let mut acc = 0.0;
        for t in 0..(pos + 1) {
            let v_base = t * kv_dim + kv_head * head_size;
            acc = acc + att[att_base + t] * value_cache[v_base + i];
        }
        xb[q_base + i] = acc;
        i = i + 256;
    }
}

/// SwiGLU: `hb[i] *= silu(hb[i]) * hb2[i]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn swiglu(
    #[tensor(shape = [hidden])] hb: goldy::gpu::TensorMut<f32>,
    #[tensor(shape = [hidden])] hb2: goldy::gpu::Tensor<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i >= hb.len() {
        return;
    }
    let mut val = hb[i];
    val = val * (1.0 / (1.0 + goldy::gpu::exp(-val)));
    val = val * hb2[i];
    hb[i] = val;
}

pub use attention::Kernel as AttentionKernel;
pub use attention_combine::Kernel as AttentionCombineKernel;
pub use attention_partial::Kernel as AttentionPartialKernel;
pub use cache_store::Kernel as CacheStoreKernel;
pub use embed::Kernel as EmbedKernel;
pub use gemv::Kernel as GemvKernel;
pub use rmsnorm::Kernel as RmsnormKernel;
pub use rmsnorm_inplace::Kernel as RmsnormInplaceKernel;
pub use rope::Kernel as RopeKernel;
pub use rope_store::Kernel as RopeStoreKernel;
pub use swiglu::Kernel as SwigluKernel;

/// Goldy tensor add / semantic matmul, prepared once per runtime.
///
/// Architecture crates record through this instead of [`goldy::TensorKernels`].
/// Residuals use Goldy's portable binary kernel; packed GEMV/GEMM uses Goldy's
/// semantic matmul (cuBLAS / MPS / stdlib).
pub struct TensorKernels {
    ops: goldy::TensorKernels,
}

impl TensorKernels {
    pub fn prepare(runtime: &goldy::Runtime) -> Result<Self, goldy::GoldyError> {
        Ok(Self {
            ops: goldy::TensorKernels::new(runtime)?,
        })
    }

    pub fn add_into(
        &self,
        scheme: &mut goldy::Scheme,
        label: impl Into<String>,
        a: goldy::TensorView<'_>,
        b: goldy::TensorView<'_>,
        out: goldy::TensorView<'_>,
    ) -> Result<(), goldy::GoldyError> {
        let label = label.into();
        self.ops.recorder(scheme).add_into(&label, a, b, out)
    }

    pub fn matmul_into(
        &self,
        scheme: &mut goldy::Scheme,
        label: impl Into<String>,
        a: goldy::TensorView<'_>,
        b: goldy::TensorView<'_>,
        out: goldy::TensorView<'_>,
    ) -> Result<(), goldy::GoldyError> {
        let label = label.into();
        self.ops.recorder(scheme).matmul_into(&label, a, b, out)
    }
}
