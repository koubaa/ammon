//! Algebraic GPU kernel checks (not a CPU transformer).

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "cpu_ref.rs"]
mod cpu_ref;

use ammon::gpu::create_runtime;
use ammon::kernels::{
    AttentionCombineKernel, AttentionKernel, AttentionPartialKernel, CacheStoreKernel, DecodeStep,
    EmbedKernel, GemvKernel, RmsnormInplaceKernel, RmsnormKernel, RmsnormUnweightedInplaceKernel,
    RmsnormUnweightedKernel, RopeKernel, RopeStoreKernel, SwigluKernel, TensorKernels,
    ATTENTION_SPLIT, DEFAULT_RMS_EPS, DEFAULT_ROPE_THETA,
};
use ammon::{
    CausalAttentionBlock, CausalAttentionBlockWeights, KvCache, RmsNorm, SwiGluBlock,
    SwiGluBlockWeights,
};
use goldy::{BufferKind, MemoryExchange, Runtime, Scheme, Tensor, TensorDType, TensorShape};

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn step_buf(device: &Runtime, token: u32, position: u32) -> goldy::Buffer {
    device
        .acquire_buffer_with_data(&[DecodeStep { token, position }], BufferKind::Scattered)
        .unwrap()
}

fn expect_record_err<T>(r: Result<T, goldy::GoldyError>) -> goldy::GoldyError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected tensor shape contract error"),
    }
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn embed_gathers_selected_row() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed =
        Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let step = step_buf(&device, 1, 0);
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[0.0, 0.0]).unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "embed", embed.view(), &step, x.view())
        .unwrap()
        .over_1d(2);
    let got = read_f32(&mut scheme, x.buffer());
    assert_eq!(got, vec![3.0, 4.0]);
}

#[test]
fn rope_at_pos_zero_is_identity() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let q = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let k = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&[1, 2, 2]).unwrap(),
        &[5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let step = step_buf(&device, 0, 0);
    let kernel = RopeKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "rope",
            q.view(),
            k.view(),
            &step,
            DEFAULT_ROPE_THETA,
        )
        .unwrap()
        .over_1d(2);
    let q_out = read_f32(&mut scheme, q.buffer());
    assert_eq!(q_out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gemv_identity() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = Tensor::from_f32(&device, TensorShape::matrix(1, 2), &[0.0, 0.0]).unwrap();
    let step = step_buf(&device, 0, 0);
    let gemv = GemvKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    gemv.record(&mut scheme, "gemv", x.view(), w.view(), out.view(), &step)
        .unwrap()
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![1.0, 2.0]);
}

#[test]
fn gemv_writes_cache_row_at_position() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::matrix(4, 2), TensorDType::F32).unwrap();
    let step = step_buf(&device, 0, 2);
    let gemv = GemvKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    gemv.record(
        &mut scheme,
        "gemv_pos",
        x.view(),
        w.view(),
        out.view(),
        &step,
    )
    .unwrap()
    .over_1d(2);
    assert_eq!(
        read_f32(&mut scheme, out.buffer()),
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
    );
}

#[test]
fn gemv_rejects_rank1_output() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = Tensor::from_f32(&device, TensorShape::vector(2), &[0.0, 0.0]).unwrap();
    let step = step_buf(&device, 0, 0);
    let gemv = GemvKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let before = scheme.ir_node_count();
    let err = expect_record_err(gemv.record(
        &mut scheme,
        "gemv_rank1",
        x.view(),
        w.view(),
        out.view(),
        &step,
    ));
    assert!(err.to_string().contains("expected rank 2"), "{err}");
    assert_eq!(scheme.ir_node_count(), before);
}

#[test]
fn swiglu_of_zero_is_zero() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let hb = Tensor::from_f32(&device, TensorShape::vector(2), &[0.0, 0.0]).unwrap();
    let hb2 = Tensor::from_f32(&device, TensorShape::vector(2), &[5.0, 7.0]).unwrap();
    let kernel = SwigluKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "swiglu", hb.view(), hb2.view())
        .unwrap()
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, hb.buffer()), vec![0.0, 0.0]);
}

#[test]
fn rmsnorm_matches_llama3_cuda_formula() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let o = Tensor::from_f32(&device, TensorShape::vector(4), &[0.0; 4]).unwrap();
    let kernel = RmsnormKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "rms",
            x.view(),
            w.view(),
            o.view(),
            DEFAULT_RMS_EPS,
        )
        .unwrap()
        .groups([1, 1, 1]);
    let got = read_f32(&mut scheme, o.buffer());
    let ss = 1.0f32 + 1e-5;
    let scale = 1.0 / ss.sqrt();
    for v in got {
        assert!((v - scale).abs() < 1e-5, "{v} vs {scale}");
    }
}

#[test]
fn deposit_feeds_embed_without_rerecord() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed = Tensor::from_f32(
        &device,
        TensorShape::matrix(2, 2),
        &[10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let step = step_buf(&device, 0, 0);
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[0.0, 0.0]).unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut worker = Scheme::new(&ctx);
    kernel
        .record(&mut worker, "embed", embed.view(), &step, x.view())
        .unwrap()
        .over_1d(2);
    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(&mut upload, DecodeStep::deposit_target(&step))
        .unwrap();
    for token in [0u32, 1u32] {
        deposit
            .write_data(0, &[DecodeStep { token, position: 0 }])
            .unwrap();
        let _ = upload.submit().unwrap();
        let mut sub = worker.submit().unwrap();
        let got = (&mut sub >> x.buffer()).take::<f32>().unwrap().to_vec();
        if token == 0 {
            assert_eq!(got, vec![10.0, 20.0]);
        } else {
            assert_eq!(got, vec![30.0, 40.0]);
        }
    }
    // Promoting the node's layout variant is a params-only re-record; the deposit is not.
    let stats = worker.replay_stats();
    assert_eq!(stats.records - stats.specialization_promotions, 1);
}

#[test]
fn tensor_add_into_is_elementwise() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let a = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let b = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let kernels = TensorKernels::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .add_into(&mut scheme, "add", a.view(), b.view(), out.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![4.0, 6.0]);
}

#[test]
fn rope_store_writes_rotated_k_from_scratch() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    // head = 2, one query head. Position 0 is a no-op rotation (cos=1, sin=0).
    let q = Tensor::from_f32(&device, TensorShape::matrix(1, 2), &[1.0, 2.0]).unwrap();
    let k_proj = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let k = Tensor::zeros(
        &device,
        TensorShape::from_dims(&[2, 1, 2]).unwrap(),
        TensorDType::F32,
    )
    .unwrap();
    let step = step_buf(&device, 0, 1);
    let kernel = RopeStoreKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "rope_store",
            q.view(),
            k_proj.view(),
            k.view(),
            &step,
            DEFAULT_ROPE_THETA,
        )
        .unwrap()
        .over_1d(1);
    let q_out = read_f32(&mut scheme, q.buffer());
    let k_out = read_f32(&mut scheme, k.buffer());
    // pos 1, head_dim 0: freq = 1, val = 1, so this is a real rotation of both pairs.
    let (fcr, fci) = (1.0f32.cos(), 1.0f32.sin());
    let rotate = |a: f32, b: f32| (a * fcr - b * fci, a * fci + b * fcr);
    let (q0, q1) = rotate(1.0, 2.0);
    let (k0, k1) = rotate(3.0, 4.0);
    assert!(
        (q_out[0] - q0).abs() < 1e-5 && (q_out[1] - q1).abs() < 1e-5,
        "{q_out:?}"
    );
    assert_eq!(k_out[..2], [0.0, 0.0]);
    assert!(
        (k_out[2] - k0).abs() < 1e-5 && (k_out[3] - k1).abs() < 1e-5,
        "{k_out:?}"
    );
}

#[test]
fn cache_store_writes_row_at_position() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let src = Tensor::from_f32(&device, TensorShape::vector(2), &[9.0, 8.0]).unwrap();
    let dst = Tensor::zeros(
        &device,
        TensorShape::from_dims(&[3, 1, 2]).unwrap(),
        TensorDType::F32,
    )
    .unwrap();
    let step = step_buf(&device, 0, 2);
    let kernel = CacheStoreKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "store", src.view(), dst.view(), &step)
        .unwrap()
        .over_1d(2);
    assert_eq!(
        read_f32(&mut scheme, dst.buffer()),
        vec![0.0, 0.0, 0.0, 0.0, 9.0, 8.0]
    );
}

#[test]
fn semantic_gemv_then_cache_store_matches_serial_identity() {
    // Identity weights make the reduction a single product, so cuBLAS and the
    // serial kernel agree bit-for-bit. Wider reductions may not.
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let scratch = Tensor::zeros(&device, TensorShape::vector(4), TensorDType::F32).unwrap();
    let cache = Tensor::zeros(
        &device,
        TensorShape::from_dims(&[3, 1, 2]).unwrap(),
        TensorDType::F32,
    )
    .unwrap();
    let step = step_buf(&device, 0, 1);
    let tensors = TensorKernels::prepare(&device).unwrap();
    let store = CacheStoreKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let proj = scratch.view().narrow(0, 0, 2).unwrap();
    tensors
        .matmul_into(&mut scheme, "wv", w.view(), x.view(), proj)
        .unwrap();
    store
        .record(&mut scheme, "wv_store", proj, cache.view(), &step)
        .unwrap()
        .over_1d(2);
    assert_eq!(
        read_f32(&mut scheme, cache.buffer()),
        vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
    );
}

#[test]
fn tensor_matmul_into_is_gemv() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let kernels = TensorKernels::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .matmul_into(&mut scheme, "gemv", w.view(), x.view(), out.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![3.0, 4.0]);
}

#[test]
fn rope_rejects_rank2_kv_cache() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let q = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let k = Tensor::from_f32(&device, TensorShape::matrix(1, 4), &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let step = step_buf(&device, 0, 0);
    let kernel = RopeKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let before = scheme.ir_node_count();
    let err = expect_record_err(kernel.record(
        &mut scheme,
        "rope_rank",
        q.view(),
        k.view(),
        &step,
        DEFAULT_ROPE_THETA,
    ));
    assert!(err.to_string().contains("parameter `k`"), "{err}");
    assert!(err.to_string().contains("expected rank 3"), "{err}");
    assert_eq!(scheme.ir_node_count(), before);
}

fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
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

struct AttentionCase {
    q_heads: u32,
    kv_heads: u32,
    head: u32,
    seq: u32,
    positions: &'static [u32],
}

fn cpu_attention(
    case: &AttentionCase,
    q: &[f32],
    key: &[f32],
    value: &[f32],
    pos: u32,
) -> Vec<f64> {
    let (head, kv_dim) = (case.head as usize, (case.kv_heads * case.head) as usize);
    let kv_mul = (case.q_heads / case.kv_heads) as usize;
    let mut out = vec![0.0f64; case.q_heads as usize * head];
    for h in 0..case.q_heads as usize {
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
            out[h * head + i] = weights
                .iter()
                .enumerate()
                .map(|(t, w)| w * f64::from(value[t * kv_dim + kv_base + i]))
                .sum::<f64>()
                / total;
        }
    }
    out
}

fn assert_split_attention_matches_reference(case: AttentionCase) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let partial_kernel = AttentionPartialKernel::prepare(&device).unwrap();
    let combine_kernel = AttentionCombineKernel::prepare(&device).unwrap();
    let serial_kernel = AttentionKernel::prepare(&device).unwrap();
    let (q_heads, head, seq) = (case.q_heads, case.head, case.seq);
    let splits = seq.div_ceil(ATTENTION_SPLIT);
    let cache_dims = [seq, case.kv_heads, head];
    let cache_len = (seq * case.kv_heads * head) as usize;
    let q_host = pseudo_random((q_heads * head) as usize, 11);
    let key_host = pseudo_random(cache_len, 12);
    let value_host = pseudo_random(cache_len, 13);
    let q = Tensor::from_f32(&device, TensorShape::matrix(q_heads, head), &q_host).unwrap();
    let key = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&cache_dims).unwrap(),
        &key_host,
    )
    .unwrap();
    let value = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&cache_dims).unwrap(),
        &value_host,
    )
    .unwrap();
    for &pos in case.positions {
        let step = step_buf(&device, 0, pos);
        // Dead splits keep NaN; the combine must never read them.
        let partial_dims = [q_heads, splits, head + 2];
        let poison = vec![f32::NAN; (q_heads * splits * (head + 2)) as usize];
        let partial = Tensor::from_f32(
            &device,
            TensorShape::from_dims(&partial_dims).unwrap(),
            &poison,
        )
        .unwrap();
        let split_out = Tensor::zeros(
            &device,
            TensorShape::matrix(q_heads, head),
            TensorDType::F32,
        )
        .unwrap();
        let serial_out = Tensor::zeros(
            &device,
            TensorShape::matrix(q_heads, head),
            TensorDType::F32,
        )
        .unwrap();
        let att =
            Tensor::zeros(&device, TensorShape::matrix(q_heads, seq), TensorDType::F32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        partial_kernel
            .record(
                &mut scheme,
                "attn_partial",
                q.view(),
                key.view(),
                value.view(),
                partial.view(),
                &step,
            )
            .unwrap()
            .groups([q_heads * splits, 1, 1]);
        combine_kernel
            .record(
                &mut scheme,
                "attn_combine",
                partial.view(),
                split_out.view(),
                &step,
            )
            .unwrap()
            .groups([q_heads, 1, 1]);
        serial_kernel
            .record(
                &mut scheme,
                "attn_serial",
                q.view(),
                att.view(),
                serial_out.view(),
                key.view(),
                value.view(),
                &step,
            )
            .unwrap()
            .groups([q_heads, 1, 1]);
        let split = read_f32(&mut scheme, split_out.buffer());
        let serial = read_f32(&mut scheme, serial_out.buffer());
        let expected = cpu_attention(&case, &q_host, &key_host, &value_host, pos);
        for (i, want) in expected.iter().enumerate() {
            for (name, got) in [("split", split[i]), ("serial", serial[i])] {
                assert!(
                    (f64::from(got) - want).abs() < 1e-4,
                    "{name} attention[{i}] = {got}, expected {want} (pos {pos}, head {head}, q_heads {q_heads}, kv_heads {})",
                    case.kv_heads
                );
            }
        }
    }
}

#[test]
fn split_attention_matches_reference_across_split_boundaries() {
    assert_split_attention_matches_reference(AttentionCase {
        q_heads: 6,
        kv_heads: 6,
        head: 48,
        seq: 256,
        positions: &[0, 1, 7, 8, 9, 31, 32, 33, 100, 255],
    });
}

#[test]
fn split_attention_matches_reference_for_gqa_and_head_widths() {
    for case in [
        AttentionCase {
            q_heads: 4,
            kv_heads: 2,
            head: 64,
            seq: 80,
            positions: &[0, 47, 79],
        },
        AttentionCase {
            q_heads: 2,
            kv_heads: 1,
            head: 128,
            seq: 40,
            positions: &[39],
        },
        AttentionCase {
            q_heads: 3,
            kv_heads: 3,
            head: 20,
            seq: 33,
            positions: &[5, 32],
        },
        AttentionCase {
            q_heads: 1,
            kv_heads: 1,
            head: 32,
            seq: 2112,
            positions: &[2047, 2111],
        },
    ] {
        assert_split_attention_matches_reference(case);
    }
}

#[test]
fn embed_rejects_dim_mismatch() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed =
        Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let step = step_buf(&device, 1, 0);
    let x = Tensor::from_f32(&device, TensorShape::vector(3), &[0.0, 0.0, 0.0]).unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let before = scheme.ir_node_count();
    let err =
        expect_record_err(kernel.record(&mut scheme, "embed_dim", embed.view(), &step, x.view()));
    assert!(err.to_string().contains("parameter `x`"), "{err}");
    assert!(err.to_string().contains("`dim`"), "{err}");
    assert_eq!(scheme.ir_node_count(), before);
}

fn dispatch_rmsnorm(
    device: &goldy::Runtime,
    ctx: &goldy::Context,
    kernel: &RmsnormKernel,
    x_host: &[f32],
    w_host: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = x_host.len() as u32;
    let x = Tensor::from_f32(device, TensorShape::vector(n), x_host).unwrap();
    let w = Tensor::from_f32(device, TensorShape::vector(n), w_host).unwrap();
    let o = Tensor::zeros(device, TensorShape::vector(n), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(ctx);
    kernel
        .record(&mut scheme, "rms", x.view(), w.view(), o.view(), eps)
        .unwrap()
        .groups([1, 1, 1]);
    read_f32(&mut scheme, o.buffer())
}

#[test]
fn rmsnorm_matches_cpu_reference_for_eps_and_width() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let weighted = RmsnormKernel::prepare(&device).unwrap();
    let unweighted = RmsnormUnweightedKernel::prepare(&device).unwrap();
    let weighted_ip = RmsnormInplaceKernel::prepare(&device).unwrap();
    let unweighted_ip = RmsnormUnweightedInplaceKernel::prepare(&device).unwrap();
    let module = RmsNorm::with_eps(&device, 1.0e-6).unwrap();
    assert_eq!(RmsNorm::new(&device).unwrap().eps(), DEFAULT_RMS_EPS);

    for (dim, seed, eps) in [
        (1u32, 4u32, DEFAULT_RMS_EPS),
        (4, 1, DEFAULT_RMS_EPS),
        (7, 9, 1.0e-2),
        (300, 2, 1.0e-6),
    ] {
        let x_host = cpu_ref::pseudo_random(dim as usize, seed);
        let w_host = cpu_ref::pseudo_random(dim as usize, seed + 17);
        let got = dispatch_rmsnorm(&device, &ctx, &weighted, &x_host, &w_host, eps);
        cpu_ref::assert_close(
            &format!("rmsnorm dim {dim} eps {eps}"),
            &got,
            &cpu_ref::rmsnorm(&x_host, Some(&w_host), eps),
        );

        let x = Tensor::from_f32(&device, TensorShape::vector(dim), &x_host).unwrap();
        let o = Tensor::zeros(&device, TensorShape::vector(dim), TensorDType::F32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        unweighted
            .record(&mut scheme, "rms", x.view(), o.view(), eps)
            .unwrap()
            .groups([1, 1, 1]);
        cpu_ref::assert_close(
            &format!("rmsnorm unweighted dim {dim}"),
            &read_f32(&mut scheme, o.buffer()),
            &cpu_ref::rmsnorm(&x_host, None, eps),
        );

        let x_ip = Tensor::from_f32(&device, TensorShape::vector(dim), &x_host).unwrap();
        let w = Tensor::from_f32(&device, TensorShape::vector(dim), &w_host).unwrap();
        let mut scheme = Scheme::new(&ctx);
        weighted_ip
            .record(&mut scheme, "rms", x_ip.view(), w.view(), eps)
            .unwrap()
            .groups([1, 1, 1]);
        cpu_ref::assert_close(
            "rmsnorm inplace",
            &read_f32(&mut scheme, x_ip.buffer()),
            &cpu_ref::rmsnorm(&x_host, Some(&w_host), eps),
        );

        let x_ip = Tensor::from_f32(&device, TensorShape::vector(dim), &x_host).unwrap();
        let mut scheme = Scheme::new(&ctx);
        unweighted_ip
            .record(&mut scheme, "rms", x_ip.view(), eps)
            .unwrap()
            .groups([1, 1, 1]);
        cpu_ref::assert_close(
            "rmsnorm unweighted inplace",
            &read_f32(&mut scheme, x_ip.buffer()),
            &cpu_ref::rmsnorm(&x_host, None, eps),
        );
    }

    // Module records use the same kernels and the epsilon stored on the module.
    let eps = 1.0e-6;
    let x_host = cpu_ref::pseudo_random(8, 21);
    let w_host = cpu_ref::pseudo_random(8, 22);
    let x = Tensor::from_f32(&device, TensorShape::vector(8), &x_host).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::vector(8), &w_host).unwrap();
    let o = Tensor::zeros(&device, TensorShape::vector(8), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    module
        .record(&mut scheme, x.view(), w.view(), o.view())
        .unwrap();
    cpu_ref::assert_close(
        "RmsNorm::record",
        &read_f32(&mut scheme, o.buffer()),
        &cpu_ref::rmsnorm(&x_host, Some(&w_host), eps),
    );
    let x = Tensor::from_f32(&device, TensorShape::vector(8), &x_host).unwrap();
    let mut scheme = Scheme::new(&ctx);
    module
        .record_unweighted(&mut scheme, x.view(), o.view())
        .unwrap();
    cpu_ref::assert_close(
        "RmsNorm::record_unweighted",
        &read_f32(&mut scheme, o.buffer()),
        &cpu_ref::rmsnorm(&x_host, None, eps),
    );
    let x = Tensor::from_f32(&device, TensorShape::vector(8), &x_host).unwrap();
    let mut scheme = Scheme::new(&ctx);
    module
        .record_unweighted_inplace(&mut scheme, x.view())
        .unwrap();
    cpu_ref::assert_close(
        "RmsNorm::record_unweighted_inplace",
        &read_f32(&mut scheme, x.buffer()),
        &cpu_ref::rmsnorm(&x_host, None, eps),
    );
}

#[test]
fn swiglu_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let hb_host = cpu_ref::pseudo_random(6, 30);
    let hb2_host = cpu_ref::pseudo_random(6, 31);
    let hb = Tensor::from_f32(&device, TensorShape::vector(6), &hb_host).unwrap();
    let hb2 = Tensor::from_f32(&device, TensorShape::vector(6), &hb2_host).unwrap();
    let kernel = SwigluKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "swiglu", hb.view(), hb2.view())
        .unwrap()
        .over_1d(6);
    cpu_ref::assert_close(
        "swiglu",
        &read_f32(&mut scheme, hb.buffer()),
        &cpu_ref::swiglu(&hb_host, &hb2_host),
    );
}

#[test]
fn gemv_and_matmul_match_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let cols = 5usize;
    let rows = 3usize;
    let x_host = cpu_ref::pseudo_random(cols, 40);
    let w_host = cpu_ref::pseudo_random(rows * cols, 41);
    let want = cpu_ref::gemv(&w_host, rows, cols, &x_host);
    let x = Tensor::from_f32(&device, TensorShape::vector(cols as u32), &x_host).unwrap();
    let w = Tensor::from_f32(
        &device,
        TensorShape::matrix(rows as u32, cols as u32),
        &w_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::matrix(2, rows as u32),
        TensorDType::F32,
    )
    .unwrap();
    let step = step_buf(&device, 0, 1);
    let gemv = GemvKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    gemv.record(&mut scheme, "gemv", x.view(), w.view(), out.view(), &step)
        .unwrap()
        .over_1d(rows as u32);
    let got = read_f32(&mut scheme, out.buffer());
    cpu_ref::assert_close("gemv row", &got[rows..rows * 2], &want);

    let flat = Tensor::zeros(&device, TensorShape::vector(rows as u32), TensorDType::F32).unwrap();
    let tensors = TensorKernels::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    tensors
        .matmul_into(&mut scheme, "gemv", w.view(), x.view(), flat.view())
        .unwrap();
    cpu_ref::assert_close("matmul", &read_f32(&mut scheme, flat.buffer()), &want);
}

#[test]
fn rope_store_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let q_heads = 2usize;
    let head = 4usize;
    let kv_heads = 1usize;
    let kv_dim = kv_heads * head;
    let pos = 2u32;
    let mut q_host = cpu_ref::pseudo_random(q_heads * head, 50);
    let mut k_host = cpu_ref::pseudo_random(kv_dim, 51);
    let q = Tensor::from_f32(
        &device,
        TensorShape::matrix(q_heads as u32, head as u32),
        &q_host,
    )
    .unwrap();
    let k_proj = Tensor::from_f32(&device, TensorShape::vector(kv_dim as u32), &k_host).unwrap();
    let k = Tensor::zeros(
        &device,
        TensorShape::from_dims(&[3, kv_heads as u32, head as u32]).unwrap(),
        TensorDType::F32,
    )
    .unwrap();
    let step = step_buf(&device, 0, pos);
    let kernel = RopeStoreKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "rope_store",
            q.view(),
            k_proj.view(),
            k.view(),
            &step,
            DEFAULT_ROPE_THETA,
        )
        .unwrap()
        .over_1d((q_heads * head / 2) as u32);
    cpu_ref::rope(&mut q_host, &mut k_host, head, pos);
    cpu_ref::assert_close("rope q", &read_f32(&mut scheme, q.buffer()), &q_host);
    let k_out = read_f32(&mut scheme, k.buffer());
    let base = pos as usize * kv_dim;
    assert!(k_out[..base].iter().all(|v| *v == 0.0), "{k_out:?}");
    cpu_ref::assert_close("rope k", &k_out[base..base + kv_dim], &k_host);
}

#[test]
fn causal_attention_block_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let eps = 1.0e-6;
    let hidden = 8usize;
    let q_heads = 2usize;
    let kv_heads = 1usize;
    let head = hidden / q_heads;
    let seq = 16u32;
    let kv_dim = kv_heads * head;
    let block =
        CausalAttentionBlock::with_eps(&device, hidden as u32, q_heads as u32, seq, eps).unwrap();
    let cache = KvCache::new(&device, 1, seq, kv_heads as u32, head as u32).unwrap();
    let norm_h = cpu_ref::pseudo_random(hidden, 60);
    let wq_h = cpu_ref::pseudo_random(hidden * hidden, 61);
    let wk_h = cpu_ref::pseudo_random(kv_dim * hidden, 62);
    let wv_h = cpu_ref::pseudo_random(kv_dim * hidden, 63);
    let wo_h = cpu_ref::pseudo_random(hidden * hidden, 64);
    let norm = Tensor::from_f32(&device, TensorShape::vector(hidden as u32), &norm_h).unwrap();
    let wq = Tensor::from_f32(
        &device,
        TensorShape::matrix(hidden as u32, hidden as u32),
        &wq_h,
    )
    .unwrap();
    let wk = Tensor::from_f32(
        &device,
        TensorShape::matrix(kv_dim as u32, hidden as u32),
        &wk_h,
    )
    .unwrap();
    let wv = Tensor::from_f32(
        &device,
        TensorShape::matrix(kv_dim as u32, hidden as u32),
        &wv_h,
    )
    .unwrap();
    let wo = Tensor::from_f32(
        &device,
        TensorShape::matrix(hidden as u32, hidden as u32),
        &wo_h,
    )
    .unwrap();
    let mut x_host = cpu_ref::pseudo_random(hidden, 65);
    let x = Tensor::from_f32(&device, TensorShape::vector(hidden as u32), &x_host).unwrap();
    let mut key = vec![0.0_f32; seq as usize * kv_dim];
    let mut value = vec![0.0_f32; seq as usize * kv_dim];
    let weights = CausalAttentionBlockWeights {
        norm: norm.view(),
        query: wq.view(),
        key: wk.view(),
        value: wv.view(),
        output: wo.view(),
    };
    for pos in [0u32, 7, 8, 15] {
        let step = step_buf(&device, 3, pos);
        let mut scheme = Scheme::new(&ctx);
        block
            .record(
                &mut scheme,
                x.view(),
                weights,
                cache.layer(0).unwrap(),
                &step,
            )
            .unwrap();
        cpu_ref::attention_block_step(
            &mut x_host,
            &norm_h,
            &wq_h,
            &wk_h,
            &wv_h,
            &wo_h,
            &mut key,
            &mut value,
            q_heads,
            kv_heads,
            head,
            pos,
            eps,
        );
        cpu_ref::assert_close(
            &format!("attention block pos {pos}"),
            &read_f32(&mut scheme, x.buffer()),
            &x_host,
        );
    }
}

#[test]
fn swiglu_block_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let eps = 1.0e-6;
    let hidden = 8usize;
    let intermediate = 16usize;
    let block = SwiGluBlock::with_eps(&device, hidden as u32, intermediate as u32, eps).unwrap();
    let x_host = cpu_ref::pseudo_random(hidden, 70);
    let norm_h = cpu_ref::pseudo_random(hidden, 71);
    let w1_h = cpu_ref::pseudo_random(intermediate * hidden, 72);
    let w3_h = cpu_ref::pseudo_random(intermediate * hidden, 73);
    let w2_h = cpu_ref::pseudo_random(hidden * intermediate, 74);
    let x = Tensor::from_f32(&device, TensorShape::vector(hidden as u32), &x_host).unwrap();
    let norm = Tensor::from_f32(&device, TensorShape::vector(hidden as u32), &norm_h).unwrap();
    let w1 = Tensor::from_f32(
        &device,
        TensorShape::matrix(intermediate as u32, hidden as u32),
        &w1_h,
    )
    .unwrap();
    let w3 = Tensor::from_f32(
        &device,
        TensorShape::matrix(intermediate as u32, hidden as u32),
        &w3_h,
    )
    .unwrap();
    let w2 = Tensor::from_f32(
        &device,
        TensorShape::matrix(hidden as u32, intermediate as u32),
        &w2_h,
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    block
        .record(
            &mut scheme,
            x.view(),
            SwiGluBlockWeights {
                norm: norm.view(),
                gate: w1.view(),
                up: w3.view(),
                down: w2.view(),
            },
        )
        .unwrap();
    cpu_ref::assert_close(
        "swiglu block",
        &read_f32(&mut scheme, x.buffer()),
        &cpu_ref::swiglu_block(&x_host, &norm_h, &w1_h, &w3_h, &w2_h, eps),
    );
}
