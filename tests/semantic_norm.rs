//! Pre-norm blocks with `AMMON_SEMANTIC_NORM=1` and automatic fusion.
//!
//! The tensor-op RMSNorm path is selected only when that variable is `1`. This
//! binary sets it for every test so the kernel-test process, which expects the
//! kernel path, never observes the variable.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "cpu_ref.rs"]
mod cpu_ref;

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    CausalAttentionBlock, CausalAttentionBlockWeights, KvCache, SwiGluBlock, SwiGluBlockWeights,
};
use goldy::{BufferKind, Runtime, Scheme, Tensor, TensorShape};
use std::sync::Once;

fn enable_semantic_norm() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: set once before any GPU work in this process. Run with
        // `--test-threads=1` so other threads are not reading the environment.
        unsafe { std::env::set_var("AMMON_SEMANTIC_NORM", "1") }
    });
}

fn runtime() -> Runtime {
    enable_semantic_norm();
    create_runtime().expect("goldy runtime")
}

fn step_buf(device: &Runtime, token: u32, position: u32) -> goldy::Buffer {
    device
        .acquire_buffer_with_data(&[DecodeStep { token, position }], BufferKind::Scattered)
        .unwrap()
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn fused_attention_block_norm_matches_cpu_reference() {
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
    let norm_h = cpu_ref::pseudo_random(hidden, 80);
    let wq_h = cpu_ref::pseudo_random(hidden * hidden, 81);
    let wk_h = cpu_ref::pseudo_random(kv_dim * hidden, 82);
    let wv_h = cpu_ref::pseudo_random(kv_dim * hidden, 83);
    let wo_h = cpu_ref::pseudo_random(hidden * hidden, 84);
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
    let mut x_host = cpu_ref::pseudo_random(hidden, 85);
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
    for pos in [0u32, 9] {
        let step = step_buf(&device, 1, pos);
        let mut scheme = Scheme::new(&ctx);
        scheme.set_automatic_fusion(true);
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
            &format!("semantic attention pos {pos}"),
            &read_f32(&mut scheme, x.buffer()),
            &x_host,
        );
    }
}

#[test]
fn fused_swiglu_block_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let eps = 1.0e-2;
    let hidden = 8usize;
    let intermediate = 12usize;
    let block = SwiGluBlock::with_eps(&device, hidden as u32, intermediate as u32, eps).unwrap();
    let x_host = cpu_ref::pseudo_random(hidden, 90);
    let norm_h = cpu_ref::pseudo_random(hidden, 91);
    let w1_h = cpu_ref::pseudo_random(intermediate * hidden, 92);
    let w3_h = cpu_ref::pseudo_random(intermediate * hidden, 93);
    let w2_h = cpu_ref::pseudo_random(hidden * intermediate, 94);
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
    scheme.set_automatic_fusion(true);
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
        "semantic swiglu block",
        &read_f32(&mut scheme, x.buffer()),
        &cpu_ref::swiglu_block(&x_host, &norm_h, &w1_h, &w3_h, &w2_h, eps),
    );
}
