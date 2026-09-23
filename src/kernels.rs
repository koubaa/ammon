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

/// Position-strided KV-cache writer: `[n] × [d, n] -> [seq, d]` at `step.position`.
/// Ordinary rank-1 GEMV stays Goldy semantic matmul.
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

/// Pairwise RoPE on Q heads and the current K cache row.
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

/// Multi-head attention (one workgroup per head). Inclusive over `t <= pos`.
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
pub use embed::Kernel as EmbedKernel;
pub use gemv::Kernel as GemvKernel;
pub use rmsnorm::Kernel as RmsnormKernel;
pub use rmsnorm_inplace::Kernel as RmsnormInplaceKernel;
pub use rope::Kernel as RopeKernel;
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
