//! Pre-norm blocks under automatic fusion. With inference arithmetic they match the
//! CPU reference; under `MatchRecorded` the fused graph stores what the unfused one
//! does, bit for bit.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "cpu_ref.rs"]
mod cpu_ref;

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    configure_inference_scheme, CausalAttentionBlock, CausalAttentionBlockWeights, DepthMix,
    KvCache, SwiGluBlock, SwiGluBlockWeights,
};
use goldy::{BufferKind, Runtime, Scheme, Tensor, TensorShape};

fn runtime() -> Runtime {
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
    // 0.1 is large next to mean(x²), so a constructor that drops eps fails.
    let eps = 0.1;
    let hidden = 8usize;
    let q_heads = 2usize;
    let kv_heads = 1usize;
    let head = hidden / q_heads;
    let seq = 16u32;
    let kv_dim = kv_heads * head;
    let block =
        CausalAttentionBlock::with_eps(&device, hidden as u32, q_heads as u32, seq, eps).unwrap();
    assert_eq!(block.eps(), eps);
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
        configure_inference_scheme(&mut scheme);
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
            &format!("fused attention pos {pos}"),
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
    assert_eq!(block.eps(), eps);
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
    configure_inference_scheme(&mut scheme);
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
        "fused swiglu block",
        &read_f32(&mut scheme, x.buffer()),
        &cpu_ref::swiglu_block(&x_host, &norm_h, &w1_h, &w3_h, &w2_h, eps),
    );
}

/// Submits until the structure has replayed and no fused region is still compiling.
fn settle_fusion(scheme: &mut Scheme) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let _ = scheme.submit().unwrap();
        let report = scheme.fusion_report();
        let open = report
            .regions
            .iter()
            .any(|region| matches!(region.status, goldy::FusionRegionStatus::Compiling));
        if scheme.replay_stats().clean_submits > 0 && !scheme.compiles_pending() && !open {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "fusion did not settle: {report:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// The SwiGLU block recorded with automatic fusion on and off under `MatchRecorded`:
/// the gated product fuses with the projections and stores the same bytes.
#[test]
fn fused_swiglu_block_matches_unfused_bits() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let (hidden, intermediate) = (64u32, 96u32);
    let block = SwiGluBlock::new(&device, hidden, intermediate).unwrap();
    let tensor = |rows: u32, cols: u32, seed: u32| {
        let data = cpu_ref::pseudo_random((rows * cols) as usize, seed);
        Tensor::from_f32(&device, TensorShape::matrix(rows, cols), &data).unwrap()
    };
    let norm_h = cpu_ref::pseudo_random(hidden as usize, 101);
    let norm = Tensor::from_f32(&device, TensorShape::vector(hidden), &norm_h).unwrap();
    let w1 = tensor(intermediate, hidden, 102);
    let w3 = tensor(intermediate, hidden, 103);
    let w2 = tensor(hidden, intermediate, 104);
    let x_host = cpu_ref::pseudo_random(hidden as usize, 105);
    let run = |fusion: bool| {
        let x = Tensor::from_f32(&device, TensorShape::vector(hidden), &x_host).unwrap();
        let mut scheme = Scheme::new(&ctx);
        scheme.set_automatic_fusion(fusion);
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
        if fusion {
            settle_fusion(&mut scheme);
            let report = scheme.fusion_report();
            assert!(
                report.regions.iter().any(|region| {
                    matches!(region.status, goldy::FusionRegionStatus::Promoted)
                        && region.labels.iter().any(|l| l.to_string() == "swiglu")
                }),
                "{report:?}"
            );
        }
        // One step from the initial input: every earlier submit started from it too.
        x.buffer().write(0, bytemuck::cast_slice(&x_host)).unwrap();
        read_f32(&mut scheme, x.buffer())
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<u32>>()
    };
    assert_eq!(run(true), run(false));
}

/// A tape row store is described as a copy into its row, so under `MatchRecorded` it
/// fuses with the add that produces the row and stores the same bytes.
#[test]
fn tape_row_store_fuses_with_its_producer() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let (hidden, rows, slot) = (64u32, 4u32, 2u32);
    let depth = DepthMix::new(&device).unwrap();
    let a_h = cpu_ref::pseudo_random(hidden as usize, 111);
    let b_h = cpu_ref::pseudo_random(hidden as usize, 112);
    let tape_h = cpu_ref::pseudo_random((rows * hidden) as usize, 113);
    let run = |fusion: bool| {
        let vector =
            |data: &[f32]| Tensor::from_f32(&device, TensorShape::vector(hidden), data).unwrap();
        let (a, b, sum) = (
            vector(&a_h),
            vector(&b_h),
            vector(&vec![0.0; hidden as usize]),
        );
        let tape = Tensor::from_f32(&device, TensorShape::matrix(rows, hidden), &tape_h).unwrap();
        let mut scheme = Scheme::new(&ctx);
        scheme.set_automatic_fusion(fusion);
        depth
            .add(&mut scheme, a.view(), b.view(), sum.view())
            .unwrap();
        depth
            .store_row(&mut scheme, sum.view(), tape.view(), slot)
            .unwrap();
        if fusion {
            settle_fusion(&mut scheme);
            let report = scheme.fusion_report();
            assert!(
                report.regions.iter().any(|region| {
                    matches!(region.status, goldy::FusionRegionStatus::Promoted)
                        && region.labels.iter().any(|l| l.to_string() == "depth_store")
                }),
                "{report:?}"
            );
            assert_eq!(scheme.executed_node_count(), 1, "{report:?}");
        }
        read_f32(&mut scheme, tape.buffer())
    };
    let fused = run(true);
    assert_eq!(
        fused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        run(false).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for (row, got) in fused.chunks(hidden as usize).enumerate() {
        let want: Vec<f32> = if row == slot as usize {
            a_h.iter().zip(&b_h).map(|(a, b)| a + b).collect()
        } else {
            tape_h[row * hidden as usize..][..hidden as usize].to_vec()
        };
        assert_eq!(got, want, "row {row}");
    }
}
