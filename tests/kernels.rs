//! Algebraic GPU kernel checks (not a CPU transformer).

#![cfg(any(feature = "cuda", feature = "metal"))]

use ammon::gpu::create_runtime;
use ammon::kernels::{
    DecodeStep, EmbedKernel, GemvKernel, RmsnormKernel, RopeKernel, SwigluKernel, TensorKernels,
    DEFAULT_ROPE_THETA,
};
use goldy::{BufferKind, DepositTarget, MemoryExchange, Runtime, Scheme, Tensor, TensorDType, TensorShape};

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn step_buf(device: &Runtime, token: u32, position: u32) -> goldy::Buffer {
    device
        .acquire_buffer_with_data(&[DecodeStep { token, position }], BufferKind::Scattered)
        .unwrap()
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(scheme, buf)
        .expect("withdraw");
    let mut sub = scheme.submit().expect("submit");
    let bytes = grant
        .claim(&mut sub)
        .expect("claim")
        .consume()
        .expect("consume");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn embed_gathers_selected_row() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed = device
        .acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0], BufferKind::Scattered)
        .unwrap();
    let step = step_buf(&device, 1, 0);
    let x = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "embed", &embed, &step, &x, 2)
        .over_1d(2);
    let got = read_f32(&mut scheme, &x);
    assert_eq!(got, vec![3.0, 4.0]);
}

#[test]
fn rope_at_pos_zero_is_identity() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let q = device
        .acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0], BufferKind::Scattered)
        .unwrap();
    let k = device
        .acquire_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0], BufferKind::Scattered)
        .unwrap();
    let step = step_buf(&device, 0, 0);
    let kernel = RopeKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "rope",
            &q,
            &k,
            &step,
            4,
            2,
            4,
            0,
            DEFAULT_ROPE_THETA,
        )
        .over_1d(2);
    let q_out = read_f32(&mut scheme, &q);
    assert_eq!(q_out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gemv_identity() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = device
        .acquire_buffer_with_data(&[1.0f32, 2.0], BufferKind::Scattered)
        .unwrap();
    let w = device
        .acquire_buffer_with_data(&[1.0f32, 0.0, 0.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let out = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let step = step_buf(&device, 0, 0);
    let gemv = GemvKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    gemv.record(&mut scheme, "gemv", &x, &w, &out, &step, 2, 2, 0, 0, 0)
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, &out), vec![1.0, 2.0]);
}

#[test]
fn swiglu_of_zero_is_zero() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let hb = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let hb2 = device
        .acquire_buffer_with_data(&[5.0f32, 7.0], BufferKind::Scattered)
        .unwrap();
    let kernel = SwigluKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "swiglu", &hb, &hb2, 2)
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, &hb), vec![0.0, 0.0]);
}

#[test]
fn rmsnorm_matches_llama3_cuda_formula() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = device
        .acquire_buffer_with_data(&[1.0f32, 1.0, 1.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let w = device
        .acquire_buffer_with_data(&[1.0f32, 1.0, 1.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let o = device
        .acquire_buffer_with_data(&[0.0f32; 4], BufferKind::Scattered)
        .unwrap();
    let kernel = RmsnormKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "rms", &x, &w, &o, 4, 0)
        .groups([1, 1, 1]);
    let got = read_f32(&mut scheme, &o);
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
    let embed = device
        .acquire_buffer_with_data(&[10.0f32, 20.0, 30.0, 40.0], BufferKind::Scattered)
        .unwrap();
    let step = step_buf(&device, 0, 0);
    let x = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut worker = Scheme::new(&ctx);
    kernel
        .record(&mut worker, "embed", &embed, &step, &x, 2)
        .over_1d(2);
    let grant = MemoryExchange::new(&ctx)
        .bind_withdraw(&mut worker, &x)
        .unwrap();
    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(
            &mut upload,
            DepositTarget::buffer_elements::<DecodeStep>(&step, 1),
        )
        .unwrap();
    for token in [0u32, 1u32] {
        deposit
            .write_data(0, &[DecodeStep { token, position: 0 }])
            .unwrap();
        let _ = upload.submit().unwrap();
        let mut sub = worker.submit().unwrap();
        let bytes = grant.claim(&mut sub).unwrap().consume().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
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
    let mut kernels = TensorKernels::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .add_into(&mut scheme, "add", a.view(), b.view(), out.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![4.0, 6.0]);
}

#[test]
fn tensor_matmul_into_is_gemv() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let mut kernels = TensorKernels::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .matmul_into(&mut scheme, "gemv", w.view(), x.view(), out.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![3.0, 4.0]);
}
