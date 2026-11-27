#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

use ammon::gpu::create_runtime;
use ammon::{Linear, SigmoidGate};
use goldy::{Scheme, Tensor, TensorDType, TensorShape};

fn settle_fusion(scheme: &mut Scheme) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        drop(scheme.submit().unwrap());
        let compiling = scheme
            .fusion_report()
            .regions
            .iter()
            .any(|region| matches!(region.status, goldy::FusionRegionStatus::Compiling));
        if !scheme.compiles_pending() && !compiling {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "fusion did not settle: {:?}",
            scheme.fusion_report()
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

fn read_f32(scheme: &mut Scheme, tensor: &Tensor) -> Vec<f32> {
    let mut sub = scheme.submit().unwrap();
    (&mut sub >> tensor.buffer())
        .take::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
fn out_of_place_gate_and_output_projection_fuse_exactly() {
    let runtime = create_runtime().unwrap();
    let ctx = runtime.create_context().unwrap();
    let n = 32u32;
    let vector = TensorShape::vector(n);
    let matrix = TensorShape::matrix(n, n);
    let data = |seed: u32, len: usize| {
        (0..len)
            .map(|i| {
                let word = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
                (word % 257) as f32 / 128.0 - 1.0
            })
            .collect::<Vec<_>>()
    };
    let hidden = Tensor::from_f32(&runtime, vector, &data(1, n as usize)).unwrap();
    let values = Tensor::from_f32(&runtime, vector, &data(2, n as usize)).unwrap();
    let gate_weight = Tensor::from_f32(&runtime, matrix, &data(3, (n * n) as usize)).unwrap();
    let out_weight = Tensor::from_f32(&runtime, matrix, &data(4, (n * n) as usize)).unwrap();
    let linear = Linear::new(&runtime).unwrap();
    let gate = SigmoidGate::new(&runtime).unwrap();

    let record = |automatic: bool| {
        let logits = Tensor::zeros(&runtime, vector, TensorDType::F32).unwrap();
        let gated = Tensor::zeros(&runtime, vector, TensorDType::F32).unwrap();
        let out = Tensor::zeros(&runtime, vector, TensorDType::F32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        scheme.set_automatic_fusion(automatic);
        linear
            .record(
                &mut scheme,
                gate_weight.view(),
                hidden.view(),
                logits.view(),
            )
            .unwrap();
        gate.record_into(&mut scheme, values.view(), logits.view(), gated.view())
            .unwrap();
        linear
            .record(&mut scheme, out_weight.view(), gated.view(), out.view())
            .unwrap();
        (scheme, logits, gated, out)
    };

    let (mut fused, _fused_logits, _fused_gated, fused_out) = record(true);
    let (mut plain, _plain_logits, _plain_gated, plain_out) = record(false);
    settle_fusion(&mut fused);
    settle_fusion(&mut plain);

    let report = fused.fusion_report();
    assert!(
        report.regions.iter().any(|region| {
            region.labels.len() == 4
                && region
                    .labels
                    .iter()
                    .any(|label| label.to_string() == "sigmoid_gate_mul")
                && matches!(region.status, goldy::FusionRegionStatus::Promoted)
        }),
        "{report:?}"
    );
    assert_eq!(fused.executed_node_count(), 1, "{report:?}");
    assert_eq!(plain.executed_node_count(), 4);

    let got = read_f32(&mut fused, &fused_out);
    let want = read_f32(&mut plain, &plain_out);
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}
