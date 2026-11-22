//! Depth mixes and the loop block tape, replayed from one recorded scheme.
//!
//! Goldens are `tests/data/depth_goldens.bin`, produced by
//! `tests/gen_depth_goldens.py` from the pinned loop modules. Max abs error
//! is at most 1e-5.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

use ammon::gpu::create_runtime;
use ammon::{DepthPlan, DepthRunner};
use goldy::{Scheme, Tensor, TensorDType, TensorShape};

const GOLDEN: &[u8] = include_bytes!("data/depth_goldens.bin");

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let bytes: [u8; 4] = self.data[self.at..self.at + 4].try_into().unwrap();
        self.at += 4;
        u32::from_le_bytes(bytes)
    }

    fn f32s(&mut self) -> Vec<f32> {
        let n = self.u32() as usize;
        let bytes = &self.data[self.at..self.at + n * 4];
        self.at += n * 4;
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
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

fn tensor(device: &goldy::Runtime, dims: &[u32], data: &[f32]) -> Tensor {
    assert_eq!(data.len() as u32, dims.iter().product::<u32>());
    Tensor::from_f32(device, TensorShape::from_dims(dims).unwrap(), data).unwrap()
}

fn zeros(device: &goldy::Runtime, dims: &[u32]) -> Tensor {
    Tensor::zeros(
        device,
        TensorShape::from_dims(dims).unwrap(),
        TensorDType::F32,
    )
    .unwrap()
}

fn write(buf: &Tensor, data: &[f32]) {
    buf.buffer().write(0, bytemuck::cast_slice(data)).unwrap();
}

#[test]
fn depth_schedule_matches_pinned_modules_on_replay() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let mut reader = Reader {
        data: GOLDEN,
        at: 0,
    };
    assert_eq!(reader.u32(), 0x4854_5044);
    assert_eq!(reader.u32(), 1);
    let n_plans = reader.u32();
    for _ in 0..n_plans {
        let layers = reader.u32();
        let block = reader.u32();
        let loop_start = reader.u32();
        let loop_end = reader.u32();
        let passes = reader.u32();
        let hidden = reader.u32();
        let n_tokens = reader.u32();
        let plan = DepthPlan {
            layers,
            block,
            loop_start,
            loop_end,
            passes,
        };
        if layers == 24 {
            let mini = DepthPlan::mini();
            assert_eq!(
                (
                    plan.layers,
                    plan.block,
                    plan.loop_start,
                    plan.loop_end,
                    plan.passes
                ),
                (
                    mini.layers,
                    mini.block,
                    mini.loop_start,
                    mini.loop_end,
                    mini.passes
                )
            );
        }
        let visits = plan.visits();
        let n_full = plan.n_full();
        let n_loop = plan.n_loop();
        let n_msg = plan.message_visits();
        let q_single = tensor(&device, &[layers, hidden], &reader.f32s());
        let q_full = tensor(&device, &[n_full, 3, hidden], &reader.f32s());
        let q_mlp = tensor(&device, &[layers, hidden], &reader.f32s());
        let q_msg = tensor(&device, &[n_loop, 4, hidden], &reader.f32s());
        let q_final = tensor(&device, &[hidden], &reader.f32s());
        let mut tokens = Vec::new();
        for _ in 0..n_tokens {
            let entries = reader.f32s();
            let mlp_partial = reader.f32s();
            let pass_exit = reader.f32s();
            let final_partial = reader.f32s();
            let attn = reader.f32s();
            let mlp = reader.f32s();
            let msg = reader.f32s();
            let final_y = reader.f32s();
            let tape = reader.f32s();
            tokens.push((
                entries,
                mlp_partial,
                pass_exit,
                final_partial,
                attn,
                mlp,
                msg,
                final_y,
                tape,
            ));
        }
        let entries = zeros(&device, &[visits, hidden]);
        let mlp_partial = zeros(&device, &[visits, hidden]);
        let pass_exit = zeros(&device, &[passes, hidden]);
        let final_partial = zeros(&device, &[hidden]);
        let attn_out = zeros(&device, &[visits, 3, hidden]);
        let mlp_out = zeros(&device, &[visits, hidden]);
        let msg_out = zeros(&device, &[n_msg.max(1), 4, hidden]);
        let final_out = zeros(&device, &[hidden]);
        let runner = DepthRunner::new(&device, hidden, 8).unwrap();
        let mut recorded = Scheme::new(&ctx);
        runner
            .record(
                &mut recorded,
                plan,
                entries.view(),
                mlp_partial.view(),
                pass_exit.view(),
                q_single.view(),
                q_full.view(),
                q_mlp.view(),
                q_msg.view(),
                q_final.view(),
                final_partial.view(),
                attn_out.view(),
                mlp_out.view(),
                msg_out.view(),
                final_out.view(),
            )
            .unwrap();
        for (t, token) in tokens.iter().enumerate() {
            write(&entries, &token.0);
            write(&mlp_partial, &token.1);
            write(&pass_exit, &token.2);
            write(&final_partial, &token.3);
            let _ = recorded.submit().unwrap();
            let mut read = Scheme::new(&ctx);
            let got_attn = read_f32(&mut read, attn_out.buffer());
            let mut read = Scheme::new(&ctx);
            let got_mlp = read_f32(&mut read, mlp_out.buffer());
            let mut read = Scheme::new(&ctx);
            let got_msg = read_f32(&mut read, msg_out.buffer());
            let mut read = Scheme::new(&ctx);
            let got_final = read_f32(&mut read, final_out.buffer());
            let mut read = Scheme::new(&ctx);
            let got_tape = read_f32(&mut read, runner.tape().buffer());
            let err = max_abs(&got_attn, &token.4)
                .max(max_abs(&got_mlp, &token.5))
                .max(max_abs(&got_msg[..token.6.len()], &token.6))
                .max(max_abs(&got_final, &token.7))
                .max(max_abs(&got_tape[..token.8.len()], &token.8));
            assert!(err <= 1e-5, "plan {layers} token {t} max abs {err}");
        }
        for _ in 0..12 {
            if recorded.replay_stats().clean_submits > 0 {
                break;
            }
            let _ = recorded.submit().unwrap();
        }
        let stats = recorded.replay_stats();
        assert_eq!(stats.topology_records, 0, "{stats:?}");
        assert!(stats.clean_submits > 0, "{stats:?}");
    }
    assert_eq!(reader.at, reader.data.len());
}
