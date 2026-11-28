//! Recording scratch belongs to the module that records it, not to the scheme.
//!
//! Each case records once and submits twice: a buffer dropped when `record`
//! returns fails the first submit with `StaleResource`. Then it drops the module and
//! expects the next submit to find the module's scratch gone, so a scheme that
//! kept its own reference to the scratch fails the case too.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{DecodeMoe, RmsNorm, ShortConv, SigmoidGate, Softcap, SoftcappedSwiGlu};
use goldy::{BufferKind, Context, GoldyError, Runtime, Scheme, Tensor, TensorDType, TensorShape};

fn vector(device: &Runtime, len: u32) -> Tensor {
    let values: Vec<f32> = (0..len).map(|i| 0.25 + 0.125 * i as f32).collect();
    Tensor::from_f32(device, TensorShape::vector(len), &values).unwrap()
}

fn zeros(device: &Runtime, dims: &[u32]) -> Tensor {
    Tensor::zeros(
        device,
        TensorShape::from_dims(dims).unwrap(),
        TensorDType::F32,
    )
    .unwrap()
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Records `module` once, replays it, then drops it and expects a stale submit.
fn assert_module_owns_scratch<M>(
    name: &str,
    ctx: &Context,
    module: M,
    out: &goldy::Buffer,
    record: impl FnOnce(&M, &mut Scheme) -> Result<(), GoldyError>,
) {
    let mut scheme = Scheme::new(ctx);
    record(&module, &mut scheme).unwrap_or_else(|e| panic!("{name}: record: {e:?}"));
    // In-place records rewrite their input, so the replay's values may differ.
    for _ in 0..2 {
        let got = read_f32(&mut scheme, out);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}: non-finite output {got:?}"
        );
    }
    drop(module);
    assert!(
        matches!(scheme.submit(), Err(GoldyError::StaleResource)),
        "{name}: the scheme still holds scratch after its module dropped"
    );
}

#[test]
fn rms_norm_owns_its_scratch() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let x = vector(&device, 8);
    let weight = vector(&device, 8);
    let out = zeros(&device, &[8]);
    assert_module_owns_scratch(
        "RmsNorm::record",
        &ctx,
        RmsNorm::new(&device, 8).unwrap(),
        out.buffer(),
        |norm, scheme| norm.record(scheme, x.view(), weight.view(), out.view()),
    );
    assert_module_owns_scratch(
        "RmsNorm::record_unweighted",
        &ctx,
        RmsNorm::new(&device, 8).unwrap(),
        out.buffer(),
        |norm, scheme| norm.record_unweighted(scheme, x.view(), out.view()),
    );
    let inplace = vector(&device, 8);
    assert_module_owns_scratch(
        "RmsNorm::record_inplace",
        &ctx,
        RmsNorm::new(&device, 8).unwrap(),
        inplace.buffer(),
        |norm, scheme| norm.record_inplace(scheme, inplace.view(), weight.view()),
    );
    let inplace = vector(&device, 8);
    assert_module_owns_scratch(
        "RmsNorm::record_unweighted_inplace",
        &ctx,
        RmsNorm::new(&device, 8).unwrap(),
        inplace.buffer(),
        |norm, scheme| norm.record_unweighted_inplace(scheme, inplace.view()),
    );
}

#[test]
fn rms_norm_refuses_a_width_it_has_no_scratch_for() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let norm = RmsNorm::new(&device, 8).unwrap();
    let x = vector(&device, 4);
    let out = zeros(&device, &[4]);
    let mut scheme = Scheme::new(&ctx);
    assert!(matches!(
        norm.record_unweighted(&mut scheme, x.view(), out.view()),
        Err(GoldyError::Validation(_))
    ));
}

#[test]
fn elementwise_modules_own_their_scratch() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let x = vector(&device, 6);
    let up = vector(&device, 6);
    let out = zeros(&device, &[6]);
    assert_module_owns_scratch(
        "Softcap",
        &ctx,
        Softcap::new(&device, 6, 4.0).unwrap(),
        out.buffer(),
        |cap, scheme| cap.record(scheme, x.view(), out.view()),
    );
    let values = vector(&device, 6);
    assert_module_owns_scratch(
        "SigmoidGate",
        &ctx,
        SigmoidGate::new(&device, 6).unwrap(),
        values.buffer(),
        |gate, scheme| gate.record(scheme, values.view(), x.view()),
    );
    assert_module_owns_scratch(
        "SoftcappedSwiGlu",
        &ctx,
        SoftcappedSwiGlu::new(&device, 6, 4.0, 25.0).unwrap(),
        out.buffer(),
        |swiglu, scheme| swiglu.record(scheme, x.view(), up.view(), out.view()),
    );
    // A `[rows, cols]` gate binds the same scratch reshaped.
    let rows = x.view().reshape(&[2, 3]).unwrap();
    let up_rows = up.view().reshape(&[2, 3]).unwrap();
    let out_rows = out.view().reshape(&[2, 3]).unwrap();
    assert_module_owns_scratch(
        "SoftcappedSwiGlu [2, 3]",
        &ctx,
        SoftcappedSwiGlu::new(&device, 6, 4.0, 25.0).unwrap(),
        out.buffer(),
        |swiglu, scheme| swiglu.record(scheme, rows, up_rows, out_rows),
    );
}

#[test]
fn short_conv_owns_its_position_addressed_scratch() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let (channels, kernel) = (4, 3);
    let x = vector(&device, channels);
    let weight = vector(&device, channels * kernel);
    let weight = weight.view().reshape(&[channels, kernel]).unwrap();
    // The ring lives outside the module, so only its scratch goes stale.
    let state = zeros(&device, &[channels, kernel]);
    let out = zeros(&device, &[channels]);
    let step = device
        .acquire_buffer_with_data(
            &[DecodeStep {
                token: 0,
                position: 2,
            }],
            BufferKind::Scattered,
        )
        .unwrap();
    assert_module_owns_scratch(
        "ShortConv::record_at_state",
        &ctx,
        ShortConv::new(&device, channels, kernel).unwrap(),
        out.buffer(),
        |conv, scheme| {
            conv.record_at_state(scheme, x.view(), weight, state.view(), &step, out.view())
        },
    );
}

#[test]
fn decode_moe_replays_its_f32_experts() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let (experts, top_k, hidden, inter) = (4, 2, 8, 6);
    let router = vector(&device, experts * hidden);
    let router = router.view().reshape(&[experts, hidden]).unwrap();
    let bias = zeros(&device, &[experts]);
    let gate_up = vector(&device, experts * 2 * inter * hidden);
    let gate_up = gate_up
        .view()
        .reshape(&[experts, 2 * inter, hidden])
        .unwrap();
    let down = vector(&device, experts * hidden * inter);
    let down = down.view().reshape(&[experts, hidden, inter]).unwrap();
    let x = vector(&device, hidden);
    let out = zeros(&device, &[hidden]);
    assert_module_owns_scratch(
        "DecodeMoe::record",
        &ctx,
        DecodeMoe::new(&device, experts, top_k, hidden, inter, 1.0, 4.0, 25.0).unwrap(),
        out.buffer(),
        |moe, scheme| {
            moe.record(
                scheme,
                x.view(),
                router,
                bias.view(),
                gate_up,
                down,
                out.view(),
            )
        },
    );
}
