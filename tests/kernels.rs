//! Algebraic GPU kernel checks (not a CPU transformer).

#![cfg(any(feature = "cuda", feature = "metal"))]

use ammon::gpu::create_runtime;
use ammon::kernels::{
    CacheStoreKernel, DecodeStep, EmbedKernel, GemvKernel, RmsnormKernel, RopeKernel,
    RopeStoreKernel, SwigluKernel, TensorKernels, DEFAULT_ROPE_THETA,
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
        .record(&mut scheme, "rms", x.view(), w.view(), o.view())
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
    assert_eq!(worker.replay_stats().records, 1);
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
