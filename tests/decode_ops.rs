//! Batch-1 decode ops against an f64 reference (atol 1e-5, rtol 1e-4).

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "cpu_ref.rs"]
mod cpu_ref;
#[path = "transformers_goldens.rs"]
mod goldens;

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    DynamicShortConv, HeadL2Clip, LogPositionScale, ShortConv, SigmoidGate, Softcap,
    SoftcappedSwiGlu,
};
use goldy::{BufferKind, Runtime, Scheme, Tensor, TensorDType, TensorShape};

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn step_buf(device: &Runtime, position: u32) -> goldy::Buffer {
    device
        .acquire_buffer_with_data(&[DecodeStep { token: 0, position }], BufferKind::Scattered)
        .unwrap()
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}

fn vector(device: &Runtime, values: &[f32]) -> Tensor {
    Tensor::from_f32(device, TensorShape::vector(values.len() as u32), values).unwrap()
}

#[test]
fn head_l2_clip_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let plain = HeadL2Clip::new(&device, 2.0).unwrap();
    let logged = HeadL2Clip::with_log_scale(
        &device,
        2.0,
        LogPositionScale {
            alpha: 0.5,
            floor: 4.0,
        },
    )
    .unwrap();

    // ||[3, 4]|| = 5, so the threshold of 2 actually clips.
    let x_host = [3.0_f32, 4.0, 0.1, -0.2];
    let x = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &x_host).unwrap();
    let out = Tensor::zeros(&device, TensorShape::matrix(2, 2), TensorDType::F32).unwrap();
    let step = step_buf(&device, 0);
    let mut scheme = Scheme::new(&ctx);
    plain
        .record(&mut scheme, x.view(), None, &step, out.view())
        .unwrap();
    let want = cpu_ref::head_l2_clip(&x_host, 2, 2, None, 2.0, 0.0, 1.0, 0);
    assert!((f64::from(want[0]) - 3.0 * 2.0 / (5.0 + 1e-6)).abs() < 1e-6);
    assert!(
        (want[2] - 0.1).abs() < 1e-6,
        "below-threshold head stays put"
    );
    cpu_ref::assert_close("l2 clip", &read_f32(&mut scheme, out.buffer()), &want);

    let scale_host = [2.0_f32, 0.5, 1.0, 1.0];
    let scale = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &scale_host).unwrap();
    let mut scheme = Scheme::new(&ctx);
    plain
        .record(&mut scheme, x.view(), Some(scale.view()), &step, out.view())
        .unwrap();
    cpu_ref::assert_close(
        "scaled l2 clip",
        &read_f32(&mut scheme, out.buffer()),
        &cpu_ref::head_l2_clip(&x_host, 2, 2, Some(&scale_host), 2.0, 0.0, 1.0, 0),
    );

    // pos 7, floor 4: (7 + 1) / 4 = 2, so the log term is live.
    // pos 1 stays at scale 1 because (1 + 1) / 4 < 1.
    for pos in [1u32, 7] {
        let step = step_buf(&device, pos);
        let mut scheme = Scheme::new(&ctx);
        logged
            .record(&mut scheme, x.view(), Some(scale.view()), &step, out.view())
            .unwrap();
        let want = cpu_ref::head_l2_clip(&x_host, 2, 2, Some(&scale_host), 2.0, 0.5, 4.0, pos);
        if pos == 7 {
            let bare = cpu_ref::head_l2_clip(&x_host, 2, 2, Some(&scale_host), 2.0, 0.0, 4.0, pos);
            assert!(
                (f64::from(want[0]) - f64::from(bare[0]) * (1.0 + 0.5 * 2.0_f64.ln())).abs() < 1e-5
            );
        }
        let got = read_f32(&mut scheme, out.buffer());
        cpu_ref::assert_close(&format!("log-scaled l2 clip pos {pos}"), &got, &want);
        if pos == 7 {
            // libm golden for head 0, not the cpu_ref transcription:
            // scale [2, 0.5] on [3, 4], threshold 2, then 1+0.5*ln(2).
            assert!((f64::from(got[0]) - 2.554943345417038).abs() < 1e-5);
            assert!((f64::from(got[1]) - 0.8516477818056793).abs() < 1e-5);
        }
    }
}

#[test]
fn softcap_and_sigmoid_gate_match_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x_host = [0.0_f32, 2.0, 100.0, -100.0, -0.5];
    let x = vector(&device, &x_host);
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(x_host.len() as u32),
        TensorDType::F32,
    )
    .unwrap();
    let cap = Softcap::new(&device, 4.0).unwrap();
    let mut scheme = Scheme::new(&ctx);
    cap.record(&mut scheme, x.view(), out.view()).unwrap();
    let want = cpu_ref::softcap(&x_host, 4.0);
    assert!(
        (want[2] - 4.0).abs() < 1e-5,
        "large positive saturates at the cap"
    );
    assert!(
        (want[3] + 4.0).abs() < 1e-5,
        "large negative saturates at -cap"
    );
    cpu_ref::assert_close("softcap", &read_f32(&mut scheme, out.buffer()), &want);

    // Inkling softcap widths, including |x| <= 0.3 at caps 25 and 30.
    // The reference is libm tanh via cpu_ref.
    for cap in [25.0_f64, 30.0] {
        let x_host: Vec<f32> = (0..1001).map(|i| (i as f32 / 1000.0) * 0.6 - 0.3).collect();
        let x = vector(&device, &x_host);
        let out = Tensor::zeros(&device, TensorShape::vector(1001), TensorDType::F32).unwrap();
        let op = Softcap::new(&device, cap as f32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        op.record(&mut scheme, x.view(), out.view()).unwrap();
        let want = cpu_ref::softcap(&x_host, cap);
        let got = read_f32(&mut scheme, out.buffer());
        let max = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
            .fold(0.0_f64, f64::max);
        assert!(max < 1e-6, "softcap cap {cap} near zero max abs {max}");
    }

    let values_host = [2.0_f32, 3.0, 4.0, 1.0];
    let logits_host = [0.0_f32, 20.0, -20.0, 1.0];
    let values = vector(&device, &values_host);
    let logits = vector(&device, &logits_host);
    let gate = SigmoidGate::new(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    gate.record(&mut scheme, values.view(), logits.view())
        .unwrap();
    cpu_ref::assert_close(
        "sigmoid gate",
        &read_f32(&mut scheme, values.buffer()),
        &cpu_ref::sigmoid_gate(&values_host, &logits_host),
    );
}

#[test]
fn softcapped_swiglu_matches_cpu_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    // gate_cap 1 makes softcap(gate) differ from gate, so sigmoid-on-the-capped-gate fails.
    let gate_host = [2.0_f32, -1.5, 0.25, 8.0];
    let up_host = [3.0_f32, 0.5, -4.0, 1.0];
    let gate = vector(&device, &gate_host);
    let up = vector(&device, &up_host);
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(gate_host.len() as u32),
        TensorDType::F32,
    )
    .unwrap();
    let act = SoftcappedSwiGlu::new(&device, 1.0, 10.0).unwrap();
    let mut scheme = Scheme::new(&ctx);
    act.record(&mut scheme, gate.view(), up.view(), out.view())
        .unwrap();
    let want = cpu_ref::softcapped_swiglu(&gate_host, &up_host, 1.0, 10.0);
    let g = 2.0_f64;
    let u = 3.0_f64;
    let right = (g / 1.0).tanh() * cpu_ref::sigmoid(g) * (10.0 * (u / 10.0).tanh());
    let wrong = (g / 1.0).tanh() * cpu_ref::sigmoid((g / 1.0).tanh()) * (10.0 * (u / 10.0).tanh());
    assert!(
        (right - wrong).abs() > 1e-2,
        "sigmoid must see the raw gate"
    );
    assert!((f64::from(want[0]) - right).abs() < 1e-6);
    cpu_ref::assert_close(
        "softcapped swiglu",
        &read_f32(&mut scheme, out.buffer()),
        &want,
    );
}

#[test]
fn short_conv_matches_cpu_reference_and_resets() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let conv = ShortConv::new(&device, 1, 2).unwrap();
    let weight = Tensor::from_f32(&device, TensorShape::matrix(1, 2), &[3.0, 5.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(1), TensorDType::F32).unwrap();
    let mut state = vec![0.0_f32; 2];
    for (step, token) in [2.0_f32, 7.0].into_iter().enumerate() {
        let x = vector(&device, &[token]);
        let mut scheme = Scheme::new(&ctx);
        conv.record(&mut scheme, x.view(), weight.view(), out.view())
            .unwrap();
        let want = cpu_ref::short_conv_step(&mut state, 1, 2, &[token], &[3.0, 5.0]);
        let closed = if step == 0 {
            12.0 / 3.0_f64.sqrt()
        } else {
            48.0 / 3.0_f64.sqrt()
        };
        assert!((f64::from(want[0]) - closed).abs() < 1e-6);
        let got = read_f32(&mut scheme, out.buffer());
        // Formula goldens: (5*2+2)/sqrt(3), then (3*2+5*7+7)/sqrt(3).
        let golden = if step == 0 {
            6.92820323027551
        } else {
            27.71281292110204
        };
        assert!((f64::from(got[0]) - golden).abs() < 1e-5, "{got:?}");
        cpu_ref::assert_close("static conv", &got, &want);
    }

    let mut scheme = Scheme::new(&ctx);
    conv.record_reset(&mut scheme).unwrap();
    let _ = scheme.submit().unwrap();
    let x = vector(&device, &[2.0]);
    let mut scheme = Scheme::new(&ctx);
    conv.record(&mut scheme, x.view(), weight.view(), out.view())
        .unwrap();
    let mut fresh = vec![0.0_f32; 2];
    let want = cpu_ref::short_conv_step(&mut fresh, 1, 2, &[2.0], &[3.0, 5.0]);
    cpu_ref::assert_close(
        "static conv after reset",
        &read_f32(&mut scheme, out.buffer()),
        &want,
    );

    let channels = 5usize;
    let kernel = 4usize;
    let conv = ShortConv::new(&device, channels as u32, kernel as u32).unwrap();
    let weight_host = cpu_ref::pseudo_random(channels * kernel, 11);
    let weight = Tensor::from_f32(
        &device,
        TensorShape::matrix(channels as u32, kernel as u32),
        &weight_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(channels as u32),
        TensorDType::F32,
    )
    .unwrap();
    let mut state = vec![0.0_f32; channels * kernel];
    for step in 0..6 {
        let x_host = cpu_ref::pseudo_random(channels, 20 + step);
        let x = vector(&device, &x_host);
        let mut scheme = Scheme::new(&ctx);
        conv.record(&mut scheme, x.view(), weight.view(), out.view())
            .unwrap();
        let want = cpu_ref::short_conv_step(&mut state, channels, kernel, &x_host, &weight_host);
        cpu_ref::assert_close(
            &format!("static conv step {step}"),
            &read_f32(&mut scheme, out.buffer()),
            &want,
        );
    }
}

#[test]
fn dynamic_short_conv_matches_cpu_reference_and_resets() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    // One head, two channels, kernel 2. proj is identity so the input is the taps.
    let conv = DynamicShortConv::new(&device, 2, 1, 2).unwrap();
    let static_w = [1.0_f32, 2.0, 3.0, 4.0];
    let static_weight = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &static_w).unwrap();
    let proj = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let mut state = vec![0.0_f32; 4];
    let steps = [
        ([5.0_f32, 6.0], [10.0_f32, 20.0]),
        ([7.0, 8.0], [1.0, -1.0]),
    ];
    for (x_host, taps) in steps {
        let x = vector(&device, &x_host);
        let taps_in = vector(&device, &taps);
        let mut scheme = Scheme::new(&ctx);
        conv.record(
            &mut scheme,
            x.view(),
            taps_in.view(),
            static_weight.view(),
            proj.view(),
            out.view(),
        )
        .unwrap();
        let want = cpu_ref::dynamic_short_conv_step(&mut state, 2, 1, 2, &x_host, &static_w, &taps);
        let got = read_f32(&mut scheme, out.buffer());
        cpu_ref::assert_close("dynamic conv", &got, &want);
        if taps[0] == 10.0 {
            assert!((f64::from(got[0]) - 26.832815729997474).abs() < 1e-5);
            assert!((f64::from(got[1]) - 34.88266044899672).abs() < 1e-5);
        }
    }
    let first = {
        let mut fresh = vec![0.0_f32; 4];
        cpu_ref::dynamic_short_conv_step(&mut fresh, 2, 1, 2, &[5.0, 6.0], &static_w, &[10.0, 20.0])
    };
    assert!((f64::from(first[0]) - 60.0 / 5.0_f64.sqrt()).abs() < 1e-5);
    assert!((f64::from(first[1]) - 78.0 / 5.0_f64.sqrt()).abs() < 1e-5);

    let mut scheme = Scheme::new(&ctx);
    conv.record_reset(&mut scheme).unwrap();
    let _ = scheme.submit().unwrap();
    let x = vector(&device, &[5.0, 6.0]);
    let taps_in = vector(&device, &[10.0, 20.0]);
    let mut scheme = Scheme::new(&ctx);
    conv.record(
        &mut scheme,
        x.view(),
        taps_in.view(),
        static_weight.view(),
        proj.view(),
        out.view(),
    )
    .unwrap();
    cpu_ref::assert_close(
        "dynamic conv after reset",
        &read_f32(&mut scheme, out.buffer()),
        &first,
    );

    let channels = 4usize;
    let heads = 2usize;
    let kernel = 3usize;
    let conv =
        DynamicShortConv::new(&device, channels as u32, heads as u32, kernel as u32).unwrap();
    let static_host = cpu_ref::pseudo_random(kernel * channels, 30);
    let proj_host = cpu_ref::pseudo_random((heads * kernel) * 5, 31);
    let static_weight = Tensor::from_f32(
        &device,
        TensorShape::matrix(kernel as u32, channels as u32),
        &static_host,
    )
    .unwrap();
    let proj = Tensor::from_f32(
        &device,
        TensorShape::matrix((heads * kernel) as u32, 5),
        &proj_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(channels as u32),
        TensorDType::F32,
    )
    .unwrap();
    let mut state = vec![0.0_f32; channels * kernel];
    for step in 0..5 {
        let x_host = cpu_ref::pseudo_random(channels, 40 + step);
        let dyn_host = cpu_ref::pseudo_random(5, 50 + step);
        let x = vector(&device, &x_host);
        let dynamic = vector(&device, &dyn_host);
        let mut scheme = Scheme::new(&ctx);
        conv.record(
            &mut scheme,
            x.view(),
            dynamic.view(),
            static_weight.view(),
            proj.view(),
            out.view(),
        )
        .unwrap();
        let taps = cpu_ref::gemv(&proj_host, heads * kernel, 5, &dyn_host);
        let want = cpu_ref::dynamic_short_conv_step(
            &mut state,
            channels,
            heads,
            kernel,
            &x_host,
            &static_host,
            &taps,
        );
        cpu_ref::assert_close(
            &format!("dynamic conv step {step}"),
            &read_f32(&mut scheme, out.buffer()),
            &want,
        );
    }
}

#[test]
fn wide_clip_and_conv_match_reference() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let heads = 2usize;
    let dim = 128usize;
    let x_host = cpu_ref::pseudo_random(heads * dim, 80);
    let scale_host = cpu_ref::pseudo_random(heads * dim, 81);
    let x = Tensor::from_f32(
        &device,
        TensorShape::matrix(heads as u32, dim as u32),
        &x_host,
    )
    .unwrap();
    let scale = Tensor::from_f32(
        &device,
        TensorShape::matrix(heads as u32, dim as u32),
        &scale_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::matrix(heads as u32, dim as u32),
        TensorDType::F32,
    )
    .unwrap();
    let clip = HeadL2Clip::with_log_scale(
        &device,
        1.5,
        LogPositionScale {
            alpha: 0.3,
            floor: 4.0,
        },
    )
    .unwrap();
    let step = step_buf(&device, 9);
    let mut scheme = Scheme::new(&ctx);
    clip.record(&mut scheme, x.view(), Some(scale.view()), &step, out.view())
        .unwrap();
    cpu_ref::assert_close(
        "wide l2 clip",
        &read_f32(&mut scheme, out.buffer()),
        &cpu_ref::head_l2_clip(&x_host, heads, dim, Some(&scale_host), 1.5, 0.3, 4.0, 9),
    );

    let empty = Tensor::zeros(&device, TensorShape::matrix(0, 4), TensorDType::F32).unwrap();
    let plain = HeadL2Clip::new(&device, 1.0).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let err = plain
        .record(&mut scheme, empty.view(), None, &step, empty.view())
        .unwrap_err();
    assert!(err.to_string().contains("heads"), "{err}");

    let channels = 1000usize;
    let kernel = 4usize;
    let conv = ShortConv::new(&device, channels as u32, kernel as u32).unwrap();
    let weight_host = cpu_ref::pseudo_random(channels * kernel, 82);
    let weight = Tensor::from_f32(
        &device,
        TensorShape::matrix(channels as u32, kernel as u32),
        &weight_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(channels as u32),
        TensorDType::F32,
    )
    .unwrap();
    let mut state = vec![0.0_f32; channels * kernel];
    for step_i in 0..3 {
        let x_host = cpu_ref::pseudo_random(channels, 90 + step_i);
        let x = vector(&device, &x_host);
        let mut scheme = Scheme::new(&ctx);
        conv.record(&mut scheme, x.view(), weight.view(), out.view())
            .unwrap();
        let want = cpu_ref::short_conv_step(&mut state, channels, kernel, &x_host, &weight_host);
        cpu_ref::assert_close(
            "wide static conv",
            &read_f32(&mut scheme, out.buffer()),
            &want,
        );
    }

    let channels = 512usize;
    let heads = 4usize;
    let conv =
        DynamicShortConv::new(&device, channels as u32, heads as u32, kernel as u32).unwrap();
    let static_host = cpu_ref::pseudo_random(kernel * channels, 83);
    let proj_host = cpu_ref::pseudo_random(heads * kernel * 3, 84);
    let static_weight = Tensor::from_f32(
        &device,
        TensorShape::matrix(kernel as u32, channels as u32),
        &static_host,
    )
    .unwrap();
    let proj = Tensor::from_f32(
        &device,
        TensorShape::matrix((heads * kernel) as u32, 3),
        &proj_host,
    )
    .unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(channels as u32),
        TensorDType::F32,
    )
    .unwrap();
    let mut state = vec![0.0_f32; channels * kernel];
    let x_host = cpu_ref::pseudo_random(channels, 85);
    let dyn_host = cpu_ref::pseudo_random(3, 86);
    let x = vector(&device, &x_host);
    let dynamic = vector(&device, &dyn_host);
    let mut scheme = Scheme::new(&ctx);
    conv.record(
        &mut scheme,
        x.view(),
        dynamic.view(),
        static_weight.view(),
        proj.view(),
        out.view(),
    )
    .unwrap();
    let taps = cpu_ref::gemv(&proj_host, heads * kernel, 3, &dyn_host);
    let want = cpu_ref::dynamic_short_conv_step(
        &mut state,
        channels,
        heads,
        kernel,
        &x_host,
        &static_host,
        &taps,
    );
    cpu_ref::assert_close(
        "wide dynamic conv",
        &read_f32(&mut scheme, out.buffer()),
        &want,
    );
}

/// Outputs of the pinned transformers classes (`eeee1109`), not of `cpu_ref`.
#[test]
fn matches_pinned_transformers_modules() {
    let device = runtime();
    let ctx = device.create_context().unwrap();

    for (cap, y) in [
        (25.0_f32, &goldens::SOFTCAP_25_Y),
        (30.0, &goldens::SOFTCAP_30_Y),
    ] {
        let x = vector(&device, &goldens::SOFTCAP_25_X);
        let out = Tensor::zeros(&device, TensorShape::vector(300), TensorDType::F32).unwrap();
        let op = Softcap::new(&device, cap).unwrap();
        let mut scheme = Scheme::new(&ctx);
        op.record(&mut scheme, x.view(), out.view()).unwrap();
        let err = max_abs(&read_f32(&mut scheme, out.buffer()), y);
        assert!(err < 1e-6, "transformers softcap {cap} max abs {err}");
    }

    let gate = vector(&device, &goldens::SWIGLU_GATE);
    let up = vector(&device, &goldens::SWIGLU_UP);
    let out = Tensor::zeros(&device, TensorShape::vector(280), TensorDType::F32).unwrap();
    let sw = SoftcappedSwiGlu::new(&device, 4.0, 25.0).unwrap();
    let mut scheme = Scheme::new(&ctx);
    sw.record(&mut scheme, gate.view(), up.view(), out.view())
        .unwrap();
    let err = max_abs(&read_f32(&mut scheme, out.buffer()), &goldens::SWIGLU_Y);
    assert!(err < 1e-6, "transformers swiglu max abs {err}");

    let x = Tensor::from_f32(&device, TensorShape::matrix(3, 8), &goldens::L2_X).unwrap();
    let scale = Tensor::from_f32(&device, TensorShape::matrix(3, 8), &goldens::L2_SCALE).unwrap();
    let out = Tensor::zeros(&device, TensorShape::matrix(3, 8), TensorDType::F32).unwrap();
    let clip = HeadL2Clip::with_log_scale(
        &device,
        1.5,
        LogPositionScale {
            alpha: 0.3,
            floor: 4.0,
        },
    )
    .unwrap();
    let ys = [
        &goldens::L2_Y_0,
        &goldens::L2_Y_1,
        &goldens::L2_Y_2,
        &goldens::L2_Y_3,
        &goldens::L2_Y_4,
    ];
    for (pos, y) in goldens::L2_POSITIONS.iter().zip(ys) {
        let step = step_buf(&device, *pos);
        let mut scheme = Scheme::new(&ctx);
        clip.record(&mut scheme, x.view(), Some(scale.view()), &step, out.view())
            .unwrap();
        let err = max_abs(&read_f32(&mut scheme, out.buffer()), y);
        assert!(err < 1e-5, "transformers l2 pos {pos} max abs {err}");
    }

    let weight =
        Tensor::from_f32(&device, TensorShape::matrix(8, 4), &goldens::STATIC_WEIGHT).unwrap();
    let conv = ShortConv::new(&device, 8, 4).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(8), TensorDType::F32).unwrap();
    let xs = [
        &goldens::STATIC_X_0,
        &goldens::STATIC_X_1,
        &goldens::STATIC_X_2,
        &goldens::STATIC_X_3,
        &goldens::STATIC_X_4,
        &goldens::STATIC_X_5,
        &goldens::STATIC_X_6,
    ];
    let ys = [
        &goldens::STATIC_Y_0,
        &goldens::STATIC_Y_1,
        &goldens::STATIC_Y_2,
        &goldens::STATIC_Y_3,
        &goldens::STATIC_Y_4,
        &goldens::STATIC_Y_5,
        &goldens::STATIC_Y_6,
    ];
    for (i, (x_host, y)) in xs.iter().zip(ys).enumerate() {
        let x = vector(&device, &x_host[..]);
        let mut scheme = Scheme::new(&ctx);
        conv.record(&mut scheme, x.view(), weight.view(), out.view())
            .unwrap();
        let err = max_abs(&read_f32(&mut scheme, out.buffer()), y);
        assert!(
            err < 1e-5,
            "transformers static conv step {i} max abs {err}"
        );
    }

    let static_w = Tensor::from_f32(
        &device,
        TensorShape::matrix(4, 8),
        &goldens::DYN_STATIC_WEIGHT,
    )
    .unwrap();
    let proj = Tensor::from_f32(&device, TensorShape::matrix(8, 5), &goldens::DYN_PROJ).unwrap();
    let conv = DynamicShortConv::new(&device, 8, 2, 4).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(8), TensorDType::F32).unwrap();
    let xs = [
        &goldens::DYN_X_0,
        &goldens::DYN_X_1,
        &goldens::DYN_X_2,
        &goldens::DYN_X_3,
        &goldens::DYN_X_4,
        &goldens::DYN_X_5,
        &goldens::DYN_X_6,
    ];
    let ins = [
        &goldens::DYN_IN_0,
        &goldens::DYN_IN_1,
        &goldens::DYN_IN_2,
        &goldens::DYN_IN_3,
        &goldens::DYN_IN_4,
        &goldens::DYN_IN_5,
        &goldens::DYN_IN_6,
    ];
    let ys = [
        &goldens::DYN_Y_0,
        &goldens::DYN_Y_1,
        &goldens::DYN_Y_2,
        &goldens::DYN_Y_3,
        &goldens::DYN_Y_4,
        &goldens::DYN_Y_5,
        &goldens::DYN_Y_6,
    ];
    for (i, ((x_host, din), y)) in xs.iter().zip(ins).zip(ys).enumerate() {
        let x = vector(&device, &x_host[..]);
        let dynamic = vector(&device, &din[..]);
        let mut scheme = Scheme::new(&ctx);
        conv.record(
            &mut scheme,
            x.view(),
            dynamic.view(),
            static_w.view(),
            proj.view(),
            out.view(),
        )
        .unwrap();
        let err = max_abs(&read_f32(&mut scheme, out.buffer()), y);
        assert!(
            err < 1e-5,
            "transformers dynamic conv step {i} max abs {err}"
        );
    }
}
