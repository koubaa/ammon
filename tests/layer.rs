//! One recorded decoder layer, replayed per token against the pinned module.
//!
//! `tests/data/decoder_layer.safetensors` is the layer `state_dict`.
//! `tests/data/layer_goldens.bin` is the activations. Max abs error is at most 1e-5.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    load_decode_layer, DecodeLayer, DecodeLayerScalars, DecodeLayerSpec, DecodeLayerState,
    DecodeLayerWeights,
};
use goldy::{BufferKind, MemoryExchange, Scheme, Tensor, TensorDType, TensorShape};

const WEIGHTS: &[u8] = include_bytes!("data/decoder_layer.safetensors");
const GOLDEN: &[u8] = include_bytes!("data/layer_goldens.bin");

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

    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.u32().to_le_bytes())
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

fn zeros(device: &goldy::Runtime, dims: &[u32]) -> Tensor {
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

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn decoder_layer_matches_pinned_module_on_replay() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let mut reader = Reader {
        data: GOLDEN,
        at: 0,
    };
    assert_eq!(reader.u32(), 0x5259_414C);
    assert_eq!(reader.u32(), 1);
    let n_cases = reader.u32();
    for case in 0..n_cases {
        let full = reader.u32() != 0;
        let layer_index = reader.u32();
        let boundary = reader.u32() != 0;
        let messages = reader.u32() != 0;
        let hidden = reader.u32();
        let q_heads = reader.u32();
        let kv_heads = reader.u32();
        let head_dim = reader.u32();
        let d_rel = reader.u32();
        let rel_extent = reader.u32();
        let conv_kernel = reader.u32();
        let post_kernel = reader.u32();
        let experts = reader.u32();
        let top_k = reader.u32();
        let intermediate = reader.u32();
        let gate_rank = reader.u32();
        let window = reader.u32();
        let n_tokens = reader.u32();
        let scalars = DecodeLayerScalars {
            qk_clip: reader.f32(),
            rel_clip: reader.f32(),
            rms_eps: reader.f32(),
            route_scale: reader.f32(),
            gate_cap: reader.f32(),
            up_cap: reader.f32(),
            log_alpha: reader.f32(),
            log_floor: reader.f32(),
        };
        let n_blocks = reader.u32();
        let blocks_host = reader.f32s();
        let mut tokens = Vec::new();
        for _ in 0..n_tokens {
            let x = reader.f32s();
            let msg = reader.f32s();
            let y = reader.f32s();
            tokens.push((x, msg, y));
        }
        let spec = DecodeLayerSpec {
            full,
            dynamic_conv: full,
            layer_index,
            boundary,
            messages,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            d_rel,
            rel_extent,
            conv_kernel,
            post_kernel,
            experts,
            top_k,
            intermediate,
            gate_rank,
            window,
        };
        let prefix = if full { "full." } else { "sliding." };
        let host = load_decode_layer(WEIGHTS, prefix, spec).unwrap();
        let weights = DecodeLayerWeights::from_spec(&device, spec, &host).unwrap();
        let layer = DecodeLayer::new(&device, spec, scalars).unwrap();
        let state = if full {
            DecodeLayerState::full(&device, 8, kv_heads, head_dim, q_heads, conv_kernel).unwrap()
        } else {
            DecodeLayerState::sliding(&device, window, 8, kv_heads, head_dim, q_heads, conv_kernel)
                .unwrap()
        };
        let x = zeros(&device, &[hidden]);
        let blocks = zeros(&device, &[n_blocks + 1, hidden]);
        blocks
            .buffer()
            .write(0, bytemuck::cast_slice(&blocks_host))
            .unwrap();
        let msg = zeros(&device, &[4, hidden]);
        let msg_views = [
            msg.view()
                .narrow(0, 0, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 1, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 2, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 3, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
        ];
        let out = zeros(&device, &[hidden]);
        let step = device
            .acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )
            .unwrap();
        let mut recorded = Scheme::new(&ctx);
        layer
            .record(
                &mut recorded,
                x.view(),
                blocks.view(),
                n_blocks,
                Some(msg_views),
                &weights,
                &state,
                &step,
                out.view(),
            )
            .unwrap();
        let mut upload = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit(&mut upload, DecodeStep::deposit_target(&step))
            .unwrap();
        for (pos, (token, message, want)) in tokens.iter().enumerate() {
            x.buffer().write(0, bytemuck::cast_slice(token)).unwrap();
            msg.buffer()
                .write(0, bytemuck::cast_slice(message))
                .unwrap();
            state.deposit_position(&deposit, 0, pos as u32).unwrap();
            let _ = upload.submit().unwrap();
            let _ = recorded.submit().unwrap();
            let mut read = Scheme::new(&ctx);
            let got = read_f32(&mut read, out.buffer());
            let err = max_abs(&got, want);
            assert!(err <= 1e-5, "case {case} token {pos} max abs {err}");
        }
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
    assert_eq!(reader.at, reader.data.len());
}
