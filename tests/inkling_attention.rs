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
    DecodeAttention, DecodeAttentionKind, DecodeAttentionWeights, DecodeKvCache, LogPositionScale,
};
use goldy::{BufferKind, MemoryExchange, Runtime, Scheme, Tensor, TensorDType, TensorShape};

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

fn run(spec: Spec, w: Weights<'_>, steps: &[(&[f32], &[f32], &[f32], &[f32])]) {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let attn = DecodeAttention::new(
        &device,
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
    let cache = DecodeKvCache::new(&device, spec.kind, spec.kv_heads, spec.head).unwrap();
    let q_dim = spec.q_heads * spec.head;
    let kv_dim = spec.kv_heads * spec.head;
    let wq = matrix(&device, q_dim, spec.hidden, w.wq);
    let wk = matrix(&device, kv_dim, spec.hidden, w.wk);
    let wv = matrix(&device, kv_dim, spec.hidden, w.wv);
    let wo = matrix(&device, spec.hidden, q_dim, w.wo);
    let wgate = matrix(&device, q_dim, spec.hidden, w.wgate);
    let wsink = matrix(&device, spec.q_heads, spec.hidden, w.wsink);
    let wrel = matrix(&device, spec.q_heads * spec.d_rel, spec.hidden, w.wrel);
    let rel_proj = matrix(&device, spec.d_rel, spec.rel_extent, w.rel_proj);
    let q_scale = matrix(&device, spec.q_heads, spec.head, w.q_scale);
    let rel_scale = matrix(&device, spec.q_heads, spec.d_rel, w.rel_scale);
    let (k_conv, v_conv, k_proj, v_proj) = if spec.dynamic {
        (
            matrix(&device, spec.kernel, kv_dim, w.k_conv),
            matrix(&device, spec.kernel, kv_dim, w.v_conv),
            Some(matrix(
                &device,
                spec.kv_heads * spec.kernel,
                spec.hidden,
                w.k_conv_proj,
            )),
            Some(matrix(
                &device,
                spec.kv_heads * spec.kernel,
                spec.hidden,
                w.v_conv_proj,
            )),
        )
    } else {
        (
            matrix(&device, kv_dim, spec.kernel, w.k_conv),
            matrix(&device, kv_dim, spec.kernel, w.v_conv),
            None,
            None,
        )
    };
    let q_t = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let k_t = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let v_t = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(spec.hidden), TensorDType::F32).unwrap();
    let step = step_buf(&device, 0);
    let mut recorded = Scheme::new(&ctx);
    let weights = DecodeAttentionWeights {
        wq: wq.view(),
        wk: wk.view(),
        wv: wv.view(),
        wo: wo.view(),
        wgate: wgate.view(),
        wsink: wsink.view(),
        wrel: wrel.view(),
        rel_proj: rel_proj.view(),
        q_scale: q_scale.view(),
        rel_scale: rel_scale.view(),
        k_conv: k_conv.view(),
        v_conv: v_conv.view(),
        k_conv_proj: k_proj.as_ref().map(Tensor::view),
        v_conv_proj: v_proj.as_ref().map(Tensor::view),
    };
    attn.record(
        &mut recorded,
        q_t.view(),
        k_t.view(),
        v_t.view(),
        weights,
        &cache,
        &step,
        out.view(),
    )
    .unwrap();
    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(&mut upload, DecodeStep::deposit_target(&step))
        .unwrap();
    for (i, (q, k, v, y)) in steps.iter().enumerate() {
        q_t.buffer().write(0, bytemuck::cast_slice(q)).unwrap();
        k_t.buffer().write(0, bytemuck::cast_slice(k)).unwrap();
        v_t.buffer().write(0, bytemuck::cast_slice(v)).unwrap();
        cache.deposit_position(&deposit, 0, i as u32).unwrap();
        let _ = upload.submit().unwrap();
        let _ = recorded.submit().unwrap();
        let mut read = Scheme::new(&ctx);
        let got = read_f32(&mut read, out.buffer());
        let err = max_abs(&got, y);
        assert!(err <= 1e-5, "step {i} max abs {err} got {got:?} want {y:?}");
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
            kind: DecodeAttentionKind::Full { capacity: 8 },
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
    let _cache = DecodeKvCache::new(&device, attn.kind(), 8, 128).unwrap();
}

#[test]
fn full_cache_replay_past_capacity_does_not_write() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let attn = DecodeAttention::new(
        &device,
        DecodeAttentionKind::Full { capacity: 2 },
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
    let cache =
        DecodeKvCache::new(&device, DecodeAttentionKind::Full { capacity: 2 }, 2, 4).unwrap();
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
