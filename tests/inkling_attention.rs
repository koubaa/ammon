//! Decode attention against forwards of the pinned transformers attention classes.
//!
//! Tolerance is a max absolute error of 1e-5. Sliding cases run past the window
//! so the ring drops the oldest key. The full case attends the whole prefix and
//! zeroes relative bias past `rel_extent`.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

#[path = "inkling_attention_goldens.rs"]
mod goldens;

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    configure_inference_scheme, inference_arithmetic_policy, DecodeAttention, DecodeAttentionKind,
    DecodeAttentionWeights, DecodeLayerState, LogPositionScale,
};
use goldy::{
    ArithmeticPolicy, BufferKind, MemoryExchange, Runtime, Scheme, Tensor, TensorDType, TensorShape,
};

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

fn matrix(device: &Runtime, rows: u32, cols: u32, data: &[f32]) -> Tensor {
    assert_eq!(data.len(), (rows * cols) as usize);
    Tensor::from_f32(device, TensorShape::matrix(rows, cols), data).unwrap()
}

struct Spec {
    kind: DecodeAttentionKind,
    /// Cache rows. For a sliding layer this is at least the window.
    capacity: u32,
    q_heads: u32,
    kv_heads: u32,
    head: u32,
    hidden: u32,
    d_rel: u32,
    rel_extent: u32,
    kernel: u32,
    dynamic: bool,
    log: Option<LogPositionScale>,
}

type Step<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32]);

/// One attention layer, its state, weights and activations.
struct Layer {
    attn: DecodeAttention,
    state: DecodeLayerState,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    wgate: Tensor,
    wsink: Tensor,
    wrel: Tensor,
    rel_proj: Tensor,
    q_scale: Tensor,
    rel_scale: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    k_proj: Option<Tensor>,
    v_proj: Option<Tensor>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    step: goldy::Buffer,
}

impl Layer {
    fn new(device: &Runtime, spec: &Spec, w: &Weights<'_>) -> Self {
        let attn = DecodeAttention::new(
            device,
            spec.kind,
            spec.q_heads,
            spec.kv_heads,
            spec.head,
            spec.hidden,
            spec.d_rel,
            spec.rel_extent,
            spec.kernel,
            spec.dynamic,
            1.25,
            0.9,
            1e-5,
            spec.log,
        )
        .unwrap();
        let state = match spec.kind {
            DecodeAttentionKind::Sliding { window } => DecodeLayerState::sliding(
                device,
                window,
                spec.capacity,
                spec.kv_heads,
                spec.head,
                spec.q_heads,
                spec.kernel,
            ),
            DecodeAttentionKind::Full => DecodeLayerState::full(
                device,
                spec.capacity,
                spec.kv_heads,
                spec.head,
                spec.q_heads,
                spec.kernel,
            ),
        }
        .unwrap();
        let q_dim = spec.q_heads * spec.head;
        let kv_dim = spec.kv_heads * spec.head;
        let (k_conv, v_conv, k_proj, v_proj) = if spec.dynamic {
            (
                matrix(device, spec.kernel, kv_dim, w.k_conv),
                matrix(device, spec.kernel, kv_dim, w.v_conv),
                Some(matrix(
                    device,
                    spec.kv_heads * spec.kernel,
                    spec.hidden,
                    w.k_conv_proj,
                )),
                Some(matrix(
                    device,
                    spec.kv_heads * spec.kernel,
                    spec.hidden,
                    w.v_conv_proj,
                )),
            )
        } else {
            (
                matrix(device, kv_dim, spec.kernel, w.k_conv),
                matrix(device, kv_dim, spec.kernel, w.v_conv),
                None,
                None,
            )
        };
        let activation =
            || Tensor::zeros(device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
        Self {
            attn,
            state,
            wq: matrix(device, q_dim, spec.hidden, w.wq),
            wk: matrix(device, kv_dim, spec.hidden, w.wk),
            wv: matrix(device, kv_dim, spec.hidden, w.wv),
            wo: matrix(device, spec.hidden, q_dim, w.wo),
            wgate: matrix(device, q_dim, spec.hidden, w.wgate),
            wsink: matrix(device, spec.q_heads, spec.hidden, w.wsink),
            wrel: matrix(device, spec.q_heads * spec.d_rel, spec.hidden, w.wrel),
            rel_proj: matrix(device, spec.d_rel, spec.rel_extent, w.rel_proj),
            q_scale: matrix(device, spec.q_heads, spec.head, w.q_scale),
            rel_scale: matrix(device, spec.q_heads, spec.d_rel, w.rel_scale),
            k_conv,
            v_conv,
            k_proj,
            v_proj,
            q: activation(),
            k: activation(),
            v: activation(),
            out: activation(),
            step: step_buf(device, 0),
        }
    }

    fn record(&self, scheme: &mut Scheme) {
        let weights = DecodeAttentionWeights {
            wq: self.wq.view(),
            wk: self.wk.view(),
            wv: self.wv.view(),
            wo: self.wo.view(),
            wgate: self.wgate.view(),
            wsink: self.wsink.view(),
            wrel: self.wrel.view(),
            rel_proj: self.rel_proj.view(),
            q_scale: self.q_scale.view(),
            rel_scale: self.rel_scale.view(),
            k_conv: self.k_conv.view(),
            v_conv: self.v_conv.view(),
            k_conv_proj: self.k_proj.as_ref().map(Tensor::view),
            v_conv_proj: self.v_proj.as_ref().map(Tensor::view),
        };
        self.attn
            .record(
                scheme,
                self.q.view(),
                self.k.view(),
                self.v.view(),
                weights,
                &self.state,
                &self.step,
                self.out.view(),
            )
            .unwrap();
    }

    /// Writes step `i`'s activations and position.
    fn load(&self, deposit: &goldy::DepositTransaction, i: usize, (q, k, v, _): Step<'_>) {
        self.q.buffer().write(0, bytemuck::cast_slice(q)).unwrap();
        self.k.buffer().write(0, bytemuck::cast_slice(k)).unwrap();
        self.v.buffer().write(0, bytemuck::cast_slice(v)).unwrap();
        self.state.deposit_position(deposit, 0, i as u32).unwrap();
    }
}

/// Inference arithmetic against the reference, then the exact graph fused and unfused.
fn run(spec: Spec, w: Weights<'_>, steps: &[Step<'_>]) {
    matches_reference(&spec, &w, steps, false);
    matches_reference(&spec, &w, steps, true);
    fused_matches_unfused(&spec, &w, steps);
}

/// The inference scheme against the reference. With `fusion`, the scheme settles on
/// the first step, and the key clip and the cache store run inside fused regions.
fn matches_reference(spec: &Spec, w: &Weights<'_>, steps: &[Step<'_>], fusion: bool) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let layer = Layer::new(&device, spec, w);
    let mut recorded = Scheme::new(&ctx);
    configure_inference_scheme(&mut recorded);
    recorded.set_automatic_fusion(fusion);
    // Specialization re-records while it warms up, so a short step list may
    // never observe a clean submit. The replay check is about command buffers.
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    layer.record(&mut recorded);
    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(&mut upload, DecodeStep::deposit_target(&layer.step))
        .unwrap();
    for (i, &step) in steps.iter().enumerate() {
        layer.load(&deposit, i, step);
        let _ = upload.submit().unwrap();
        if i == 0 && fusion {
            settle_fusion(&mut recorded);
        }
        let _ = recorded.submit().unwrap();
        let mut read = Scheme::new(&ctx);
        let got = read_f32(&mut read, layer.out.buffer());
        let y = step.3;
        let err = max_abs(&got, y);
        assert!(err <= 1e-5, "step {i} max abs {err} got {got:?} want {y:?}");
        if !fusion {
            unused_splits_hold_the_identity(&ctx, spec, &layer, i as u32);
        }
    }
    if fusion {
        let report = recorded.fusion_report();
        let fused = |name: &str| {
            report.regions.iter().any(|region| {
                matches!(region.status, goldy::FusionRegionStatus::Promoted)
                    && region.labels.iter().any(|label| label.to_string() == name)
            })
        };
        // Every clip without a log scale: query, key and relative on a sliding layer.
        // Clips and the combine describe reassociated sums, so they fuse only
        // under relaxed arithmetic; the cache store is exact.
        let relaxed = inference_arithmetic_policy().unwrap() == ArithmeticPolicy::Relaxed;
        let logged = spec.log.is_some_and(|log| log.alpha != 0.0);
        let clips: usize = report
            .regions
            .iter()
            .filter(|region| matches!(region.status, goldy::FusionRegionStatus::Promoted))
            .map(|region| {
                region
                    .labels
                    .iter()
                    .filter(|label| label.to_string() == "l2_clip")
                    .count()
            })
            .sum();
        // Metal cannot realize the reduction order of the native projection that
        // feeds an unscaled relative clip, so that clip stays a dispatch of its own.
        let metal = device.backend_type() == goldy::BackendType::Metal;
        let expected = match (relaxed, logged) {
            (false, _) => 0,
            (true, true) => 1,
            (true, false) if metal => 2,
            (true, false) => 3,
        };
        assert_eq!(clips, expected, "fused clips: {report:?}");
        assert_eq!(
            fused("relative_attention_combine"),
            relaxed,
            "combine fused: {report:?}"
        );
        assert!(fused("kv_cache_store"), "no fused cache store: {report:?}");
    }
    let stats = recorded.replay_stats();
    assert_eq!(
        stats.topology_records, 0,
        "replay changed scheme topology: {stats:?}"
    );
    assert!(
        stats.clean_submits > 0,
        "no clean replay submits: {stats:?}"
    );
}

/// Every split past the keys visible at `position` stores the combine's identity, so
/// the combine may reduce over all of them.
fn unused_splits_hold_the_identity(
    ctx: &goldy::Context,
    spec: &Spec,
    layer: &Layer,
    position: u32,
) {
    let partial = layer.state.partial();
    let dims = partial.shape().dims().to_vec();
    let (splits, slot) = (dims[1], dims[2] as usize);
    let visible = match spec.kind {
        DecodeAttentionKind::Sliding { window } => (position + 1).min(window),
        DecodeAttentionKind::Full => (position + 1).min(spec.capacity),
    };
    let per = if visible.div_ceil(8) > splits {
        visible.div_ceil(splits)
    } else {
        8
    };
    let mut read = Scheme::new(ctx);
    let host = read_f32(&mut read, partial.buffer());
    let mut unused = 0;
    for h in 0..dims[0] as usize {
        for s in (0..splits).filter(|s| s * per >= visible) {
            let at = (h * splits as usize + s as usize) * slot;
            let (sums, max, weight) = (
                &host[at..at + slot - 2],
                host[at + slot - 2],
                host[at + slot - 1],
            );
            assert!(
                sums.iter().all(|&v| v == 0.0) && max == -1.0e30 && weight == 0.0,
                "position {position} split {s} of head {h}: {:?}",
                &host[at..at + slot]
            );
            unused += 1;
        }
    }
    if position == 0 && splits > 1 {
        assert!(unused > 0, "no unused split at position 0");
    }
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

/// The layer recorded under `MatchRecorded` with automatic fusion on and off stores
/// the same bytes at every step. A step's replay is idempotent, so the fused scheme
/// settles on the first step's inputs.
fn fused_matches_unfused(spec: &Spec, w: &Weights<'_>, steps: &[Step<'_>]) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let mut runs: Vec<(Layer, Scheme, Scheme, goldy::DepositTransaction)> = [true, false]
        .into_iter()
        .map(|fusion| {
            let layer = Layer::new(&device, spec, w);
            let mut scheme = Scheme::new(&ctx);
            scheme.set_automatic_fusion(fusion);
            layer.record(&mut scheme);
            let mut upload = Scheme::new(&ctx);
            let deposit = MemoryExchange::new(&ctx)
                .bind_deposit(&mut upload, DecodeStep::deposit_target(&layer.step))
                .unwrap();
            (layer, scheme, upload, deposit)
        })
        .collect();
    for (i, &step) in steps.iter().enumerate() {
        let mut outputs = Vec::new();
        for (layer, scheme, upload, deposit) in &mut runs {
            layer.load(deposit, i, step);
            let _ = upload.submit().unwrap();
            if i == 0 && scheme.automatic_fusion() {
                settle_fusion(scheme);
            }
            let _ = scheme.submit().unwrap();
            let mut read = Scheme::new(&ctx);
            let out: Vec<u32> = read_f32(&mut read, layer.out.buffer())
                .iter()
                .map(|v| v.to_bits())
                .collect();
            outputs.push(out);
        }
        assert_eq!(outputs[0], outputs[1], "step {i}: fused output differs");
    }
    let fused = &runs[0].1;
    let report = fused.fusion_report();
    assert!(
        !report.regions.is_empty()
            && report
                .regions
                .iter()
                .all(|region| matches!(region.status, goldy::FusionRegionStatus::Promoted)),
        "{report:?}"
    );
}

struct Weights<'a> {
    wq: &'a [f32],
    wk: &'a [f32],
    wv: &'a [f32],
    wo: &'a [f32],
    wgate: &'a [f32],
    wsink: &'a [f32],
    wrel: &'a [f32],
    rel_proj: &'a [f32],
    q_scale: &'a [f32],
    rel_scale: &'a [f32],
    k_conv: &'a [f32],
    v_conv: &'a [f32],
    k_conv_proj: &'a [f32],
    v_conv_proj: &'a [f32],
}

#[test]
fn sliding_window_matches_transformers_including_wrap() {
    let steps = [
        (
            &goldens::SLIDING_Q_0[..],
            &goldens::SLIDING_K_0[..],
            &goldens::SLIDING_V_0[..],
            &goldens::SLIDING_Y_0[..],
        ),
        (
            &goldens::SLIDING_Q_1[..],
            &goldens::SLIDING_K_1[..],
            &goldens::SLIDING_V_1[..],
            &goldens::SLIDING_Y_1[..],
        ),
        (
            &goldens::SLIDING_Q_2[..],
            &goldens::SLIDING_K_2[..],
            &goldens::SLIDING_V_2[..],
            &goldens::SLIDING_Y_2[..],
        ),
        (
            &goldens::SLIDING_Q_3[..],
            &goldens::SLIDING_K_3[..],
            &goldens::SLIDING_V_3[..],
            &goldens::SLIDING_Y_3[..],
        ),
        (
            &goldens::SLIDING_Q_4[..],
            &goldens::SLIDING_K_4[..],
            &goldens::SLIDING_V_4[..],
            &goldens::SLIDING_Y_4[..],
        ),
        (
            &goldens::SLIDING_Q_5[..],
            &goldens::SLIDING_K_5[..],
            &goldens::SLIDING_V_5[..],
            &goldens::SLIDING_Y_5[..],
        ),
        (
            &goldens::SLIDING_Q_6[..],
            &goldens::SLIDING_K_6[..],
            &goldens::SLIDING_V_6[..],
            &goldens::SLIDING_Y_6[..],
        ),
        (
            &goldens::SLIDING_Q_7[..],
            &goldens::SLIDING_K_7[..],
            &goldens::SLIDING_V_7[..],
            &goldens::SLIDING_Y_7[..],
        ),
    ];
    run(
        Spec {
            kind: DecodeAttentionKind::Sliding { window: 4 },
            capacity: 4,
            q_heads: 4,
            kv_heads: 2,
            head: 8,
            hidden: 16,
            d_rel: 4,
            rel_extent: 4,
            kernel: 3,
            dynamic: false,
            log: None,
        },
        Weights {
            wq: &goldens::SLIDING_WQ,
            wk: &goldens::SLIDING_WK,
            wv: &goldens::SLIDING_WV,
            wo: &goldens::SLIDING_WO,
            wgate: &goldens::SLIDING_WGATE,
            wsink: &goldens::SLIDING_WSINK,
            wrel: &goldens::SLIDING_WREL,
            rel_proj: &goldens::SLIDING_REL_PROJ,
            q_scale: &goldens::SLIDING_Q_SCALE,
            rel_scale: &goldens::SLIDING_REL_SCALE,
            k_conv: &goldens::SLIDING_K_CONV,
            v_conv: &goldens::SLIDING_V_CONV,
            k_conv_proj: &[],
            v_conv_proj: &[],
        },
        &steps,
    );
}

#[test]
fn wide_gqa_head_matches_transformers_past_the_window() {
    let steps = [
        (
            &goldens::WIDE_Q_0[..],
            &goldens::WIDE_K_0[..],
            &goldens::WIDE_V_0[..],
            &goldens::WIDE_Y_0[..],
        ),
        (
            &goldens::WIDE_Q_1[..],
            &goldens::WIDE_K_1[..],
            &goldens::WIDE_V_1[..],
            &goldens::WIDE_Y_1[..],
        ),
        (
            &goldens::WIDE_Q_2[..],
            &goldens::WIDE_K_2[..],
            &goldens::WIDE_V_2[..],
            &goldens::WIDE_Y_2[..],
        ),
        (
            &goldens::WIDE_Q_3[..],
            &goldens::WIDE_K_3[..],
            &goldens::WIDE_V_3[..],
            &goldens::WIDE_Y_3[..],
        ),
        (
            &goldens::WIDE_Q_4[..],
            &goldens::WIDE_K_4[..],
            &goldens::WIDE_V_4[..],
            &goldens::WIDE_Y_4[..],
        ),
        (
            &goldens::WIDE_Q_5[..],
            &goldens::WIDE_K_5[..],
            &goldens::WIDE_V_5[..],
            &goldens::WIDE_Y_5[..],
        ),
    ];
    run(
        Spec {
            kind: DecodeAttentionKind::Sliding { window: 4 },
            capacity: 6,
            q_heads: 4,
            kv_heads: 2,
            head: 128,
            hidden: 8,
            d_rel: 4,
            rel_extent: 4,
            kernel: 2,
            dynamic: false,
            log: None,
        },
        Weights {
            wq: &goldens::WIDE_WQ,
            wk: &goldens::WIDE_WK,
            wv: &goldens::WIDE_WV,
            wo: &goldens::WIDE_WO,
            wgate: &goldens::WIDE_WGATE,
            wsink: &goldens::WIDE_WSINK,
            wrel: &goldens::WIDE_WREL,
            rel_proj: &goldens::WIDE_REL_PROJ,
            q_scale: &goldens::WIDE_Q_SCALE,
            rel_scale: &goldens::WIDE_REL_SCALE,
            k_conv: &goldens::WIDE_K_CONV,
            v_conv: &goldens::WIDE_V_CONV,
            k_conv_proj: &[],
            v_conv_proj: &[],
        },
        &steps,
    );
}

#[test]
fn full_attention_matches_transformers_with_log_scale_and_short_bias() {
    let steps = [
        (
            &goldens::FULL_Q_0[..],
            &goldens::FULL_K_0[..],
            &goldens::FULL_V_0[..],
            &goldens::FULL_Y_0[..],
        ),
        (
            &goldens::FULL_Q_1[..],
            &goldens::FULL_K_1[..],
            &goldens::FULL_V_1[..],
            &goldens::FULL_Y_1[..],
        ),
        (
            &goldens::FULL_Q_2[..],
            &goldens::FULL_K_2[..],
            &goldens::FULL_V_2[..],
            &goldens::FULL_Y_2[..],
        ),
        (
            &goldens::FULL_Q_3[..],
            &goldens::FULL_K_3[..],
            &goldens::FULL_V_3[..],
            &goldens::FULL_Y_3[..],
        ),
        (
            &goldens::FULL_Q_4[..],
            &goldens::FULL_K_4[..],
            &goldens::FULL_V_4[..],
            &goldens::FULL_Y_4[..],
        ),
        (
            &goldens::FULL_Q_5[..],
            &goldens::FULL_K_5[..],
            &goldens::FULL_V_5[..],
            &goldens::FULL_Y_5[..],
        ),
    ];
    run(
        Spec {
            kind: DecodeAttentionKind::Full,
            capacity: 16,
            q_heads: 4,
            kv_heads: 2,
            head: 8,
            hidden: 16,
            d_rel: 4,
            rel_extent: 3,
            kernel: 3,
            dynamic: true,
            log: Some(LogPositionScale {
                alpha: 0.3,
                floor: 2.0,
            }),
        },
        Weights {
            wq: &goldens::FULL_WQ,
            wk: &goldens::FULL_WK,
            wv: &goldens::FULL_WV,
            wo: &goldens::FULL_WO,
            wgate: &goldens::FULL_WGATE,
            wsink: &goldens::FULL_WSINK,
            wrel: &goldens::FULL_WREL,
            rel_proj: &goldens::FULL_REL_PROJ,
            q_scale: &goldens::FULL_Q_SCALE,
            rel_scale: &goldens::FULL_REL_SCALE,
            k_conv: &goldens::FULL_K_CONV,
            v_conv: &goldens::FULL_V_CONV,
            k_conv_proj: &goldens::FULL_K_CONV_PROJ,
            v_conv_proj: &goldens::FULL_V_CONV_PROJ,
        },
        &steps,
    );
}

#[test]
fn mini_head_counts_construct() {
    let device = runtime();
    let attn = DecodeAttention::new(
        &device,
        DecodeAttentionKind::Sliding { window: 128 },
        24,
        8,
        128,
        256,
        8,
        128,
        4,
        false,
        22.627,
        8.0,
        1e-5,
        None,
    )
    .unwrap();
    assert_eq!(attn.kind(), DecodeAttentionKind::Sliding { window: 128 });
    let _state = DecodeLayerState::sliding(&device, 128, 128, 8, 128, 24, 4).unwrap();
}

#[test]
fn full_cache_replay_past_capacity_does_not_write() {
    full_cache_replay_past_capacity(false);
    full_cache_replay_past_capacity(true);
}

/// With `fusion`, the scheme settles on a valid position first, so the past-capacity
/// replay runs the fused regions.
fn full_cache_replay_past_capacity(fusion: bool) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let attn = DecodeAttention::new(
        &device,
        DecodeAttentionKind::Full,
        2,
        2,
        4,
        4,
        2,
        2,
        2,
        false,
        1.0,
        1.0,
        1e-5,
        None,
    )
    .unwrap();
    let cache = DecodeLayerState::full(&device, 2, 2, 4, 2, 2).unwrap();
    let err = cache.check_position(2).unwrap_err();
    assert!(err.to_string().contains("outside capacity"), "{err}");
    cache.check_position(1).unwrap();

    let hidden = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 0.0, 0.0, 0.0]).unwrap();
    let wq = Tensor::zeros(&device, TensorShape::matrix(8, 4), TensorDType::F32).unwrap();
    let mut wk_host = vec![0.0_f32; 8 * 4];
    for row in 0..8 {
        wk_host[row * 4] = 1.0;
    }
    let wk = Tensor::from_f32(&device, TensorShape::matrix(8, 4), &wk_host).unwrap();
    let wo = Tensor::zeros(&device, TensorShape::matrix(4, 8), TensorDType::F32).unwrap();
    let wrel = Tensor::zeros(&device, TensorShape::matrix(4, 4), TensorDType::F32).unwrap();
    let rel_proj = Tensor::zeros(&device, TensorShape::matrix(2, 2), TensorDType::F32).unwrap();
    let q_scale = Tensor::from_f32(
        &device,
        TensorShape::matrix(2, 4),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let rel_scale =
        Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let wsink = Tensor::zeros(&device, TensorShape::matrix(2, 4), TensorDType::F32).unwrap();
    let k_conv = Tensor::zeros(&device, TensorShape::matrix(8, 2), TensorDType::F32).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(4), TensorDType::F32).unwrap();
    let step = step_buf(&device, 0);
    let weights = DecodeAttentionWeights {
        wq: wq.view(),
        wk: wk.view(),
        wv: wk.view(),
        wo: wo.view(),
        wgate: wq.view(),
        wsink: wsink.view(),
        wrel: wrel.view(),
        rel_proj: rel_proj.view(),
        q_scale: q_scale.view(),
        rel_scale: rel_scale.view(),
        k_conv: k_conv.view(),
        v_conv: k_conv.view(),
        k_conv_proj: None,
        v_conv_proj: None,
    };
    let mut recorded = Scheme::new(&ctx);
    configure_inference_scheme(&mut recorded);
    recorded.set_automatic_fusion(fusion);
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    attn.record(
        &mut recorded,
        hidden.view(),
        hidden.view(),
        hidden.view(),
        weights,
        &cache,
        &step,
        out.view(),
    )
    .unwrap();
    let _ = recorded.submit().unwrap();
    if fusion {
        settle_fusion(&mut recorded);
        let report = recorded.fusion_report();
        let combined = report.regions.iter().any(|region| {
            matches!(region.status, goldy::FusionRegionStatus::Promoted)
                && region
                    .labels
                    .iter()
                    .any(|label| label.to_string() == "relative_attention_combine")
        });
        if inference_arithmetic_policy().unwrap() == ArithmeticPolicy::Relaxed {
            assert!(combined, "no fused combine: {report:?}");
        } else {
            assert!(!combined, "strict policy fused combine: {report:?}");
        }
        assert!(
            report.regions.iter().any(|region| {
                matches!(region.status, goldy::FusionRegionStatus::Promoted)
                    && region
                        .labels
                        .iter()
                        .any(|label| label.to_string() == "kv_cache_store")
            }),
            "no fused cache store: {report:?}"
        );
    }
    let mut read = Scheme::new(&ctx);
    let after_zero = read_f32(&mut read, cache.keys().buffer());
    assert!(after_zero.iter().any(|v| *v != 0.0), "{after_zero:?}");

    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(&mut upload, DecodeStep::deposit_target(&step))
        .unwrap();
    for position in [5u32, 1u32] {
        deposit
            .write_data(0, &[DecodeStep { token: 0, position }])
            .unwrap();
        let _ = upload.submit().unwrap();
        let _ = recorded.submit().unwrap();
        let mut read = Scheme::new(&ctx);
        let got = read_f32(&mut read, cache.keys().buffer());
        if position == 5 {
            assert_eq!(got, after_zero, "a replay past capacity must not store");
            let mut read = Scheme::new(&ctx);
            let out_got = read_f32(&mut read, out.buffer());
            assert!(
                out_got.iter().all(|v| v.is_nan()),
                "past-capacity attention writes NaN, got {out_got:?}"
            );
        } else {
            assert_ne!(
                got, after_zero,
                "position 1 is inside the cache and must store"
            );
        }
    }
}
