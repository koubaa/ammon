//! Routed experts against one forward of the pinned `Inkling2SparseMoeBlock`.
//!
//! Tokens are independent. The scheme is recorded once and each token is
//! written into the same buffer. Max absolute error is at most 1e-5. The
//! generator keeps every token's top-k margin at least 1e-4, with one margin
//! near that floor.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "moe_goldens.rs"]
mod goldens;

use ammon::gpu::create_runtime;
use ammon::DecodeMoe;
use goldy::{Runtime, Scheme, Tensor, TensorDType, TensorShape};

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
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

fn tensor3(device: &Runtime, dims: &[u32], data: &[f32]) -> Tensor {
    assert_eq!(data.len(), dims.iter().product::<u32>() as usize);
    Tensor::from_f32(device, TensorShape::from_dims(dims).unwrap(), data).unwrap()
}

struct Spec {
    n_experts: u32,
    top_k: u32,
    hidden: u32,
    intermediate: u32,
    tokens: usize,
    router: &'static [f32],
    bias: &'static [f32],
    gate_up: &'static [f32],
    down: &'static [f32],
    x: &'static [f32],
    y: &'static [f32],
}

fn replay(spec: Spec) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let moe = DecodeMoe::new(
        &device,
        spec.n_experts,
        spec.top_k,
        spec.hidden,
        spec.intermediate,
        1.5,
        4.0,
        25.0,
    )
    .unwrap();
    let router = tensor3(&device, &[spec.n_experts, spec.hidden], spec.router);
    let bias = Tensor::from_f32(&device, TensorShape::vector(spec.n_experts), spec.bias).unwrap();
    let gate_up = tensor3(
        &device,
        &[spec.n_experts, 2 * spec.intermediate, spec.hidden],
        spec.gate_up,
    );
    let down = tensor3(
        &device,
        &[spec.n_experts, spec.hidden, spec.intermediate],
        spec.down,
    );
    let x = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let mut recorded = Scheme::new(&ctx);
    moe.record(
        &mut recorded,
        x.view(),
        router.view(),
        bias.view(),
        gate_up.view(),
        down.view(),
        out.view(),
    )
    .unwrap();
    let h = spec.hidden as usize;
    for t in 0..spec.tokens {
        let token = &spec.x[t * h..(t + 1) * h];
        let want = &spec.y[t * h..(t + 1) * h];
        x.buffer().write(0, bytemuck::cast_slice(token)).unwrap();
        let _ = recorded.submit().unwrap();
        let mut read = Scheme::new(&ctx);
        let got = read_f32(&mut read, out.buffer());
        let err = max_abs(&got, want);
        assert!(
            err <= 1e-5,
            "token {t} max abs {err} got {got:?} want {want:?}"
        );
    }
    // Specialization compiles finish after the token loop on a cold runner, and each
    // promotion re-records. Replay the last token until one submit is clean.
    for _ in 0..12 {
        if recorded.replay_stats().clean_submits > 0 {
            break;
        }
        let _ = recorded.submit().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let stats = recorded.replay_stats();
    assert_eq!(stats.topology_records, 0, "{stats:?}");
    assert!(stats.clean_submits > 0, "{stats:?}");
}

#[test]
fn near_tie_tokens_match_the_pinned_moe_block() {
    replay(Spec {
        n_experts: 16,
        top_k: 4,
        hidden: 32,
        intermediate: 16,
        tokens: 4,
        router: &goldens::NEAR_ROUTER,
        bias: &goldens::NEAR_BIAS,
        gate_up: &goldens::NEAR_GATE_UP,
        down: &goldens::NEAR_DOWN,
        x: &goldens::NEAR_X,
        y: &goldens::NEAR_Y,
    });
}

#[test]
fn wide_router_matches_the_pinned_moe_block() {
    replay(Spec {
        n_experts: 128,
        top_k: 8,
        hidden: 24,
        intermediate: 8,
        tokens: 2,
        router: &goldens::WIDE_ROUTER,
        bias: &goldens::WIDE_BIAS,
        gate_up: &goldens::WIDE_GATE_UP,
        down: &goldens::WIDE_DOWN,
        x: &goldens::WIDE_X,
        y: &goldens::WIDE_Y,
    });
}

#[test]
fn mini_and_small_shapes_construct() {
    let device = runtime();
    let mini = DecodeMoe::new(&device, 128, 8, 2304, 8, 1.0, 4.0, 25.0).unwrap();
    assert_eq!(mini.n_experts(), 128);
    assert_eq!(mini.top_k(), 8);
    let small = DecodeMoe::new(&device, 384, 8, 2304, 8, 1.0, 4.0, 25.0).unwrap();
    assert_eq!(small.n_experts(), 384);
}
