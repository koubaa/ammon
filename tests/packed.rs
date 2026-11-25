//! Packed uploads and matvecs. CI uses a synthetic safetensors file.
//! Real Mini rows are the opt-in script, not this test.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

use ammon::checkpoint::MappedCheckpoint;
use ammon::gpu::create_runtime;
use ammon::{DecodeMoe, Linear, PackedExperts};
use goldy::{BlockScaled, Runtime, Scheme, Tensor, TensorDType, TensorKernels, TensorShape};
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn bf16_exact(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        let bits = value.to_bits();
        assert_eq!(bits & 0xffff, 0, "{value} is not an exact bf16");
        out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    out
}

fn write_checkpoint(name: &str, views: Vec<(&str, TensorView<'_>)>) -> std::path::PathBuf {
    let bytes = serialize(views, None).unwrap();
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn mapped_bf16_matvec_matches_promotion() {
    let weight = bf16_exact(&[1.0, 0.0, 0.0, 2.0]);
    let path = write_checkpoint(
        "ammon-bf16-matvec.safetensors",
        vec![(
            "w",
            TensorView::new(Dtype::BF16, vec![2, 2], &weight).unwrap(),
        )],
    );
    let mapped = MappedCheckpoint::open(&path).unwrap();
    assert_eq!(mapped.bytes_named("w").unwrap().len(), 8);

    let device = runtime();
    let ctx = device.create_context().unwrap();
    let uploaded = mapped.upload(&device, "w").unwrap();
    assert_eq!(uploaded.dtype(), TensorDType::BF16);
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let linear = Linear::new(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    linear
        .record_bf16(&mut scheme, uploaded.view(), x.view(), out.view())
        .unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    assert_eq!(got, vec![3.0, 8.0]);
    let _ = std::fs::remove_file(path);
}

fn put_nibble(
    codes: &mut [u8],
    rows: usize,
    k: usize,
    expert: usize,
    row: usize,
    col: usize,
    nib: u8,
) {
    let packed = k / 2;
    let index = ((expert * rows + row) * packed) + col / 2;
    if col % 2 == 0 {
        codes[index] = (codes[index] & 0xf0) | (nib & 0x0f);
    } else {
        codes[index] = (codes[index] & 0x0f) | ((nib & 0x0f) << 4);
    }
}

#[test]
fn mapped_nvfp4_moe_matches_dequant_reference() {
    let experts = 2usize;
    let hidden = 16usize;
    let intermediate = 16usize;
    let fused = intermediate * 2;
    let mut gate_codes = vec![0u8; experts * fused * (hidden / 2)];
    let mut down_codes = vec![0u8; experts * hidden * (intermediate / 2)];
    // Concatenated [gate | up]. Value 1 at gate[0], value 2 at up[0].
    put_nibble(&mut gate_codes, fused, hidden, 0, 0, 0, 2);
    put_nibble(&mut gate_codes, fused, hidden, 0, intermediate, 0, 4);
    put_nibble(&mut down_codes, hidden, intermediate, 0, 0, 0, 2);
    let gate_scales = vec![0x38u8; experts * fused * (hidden / 16)];
    let down_scales = vec![0x38u8; experts * hidden * (intermediate / 16)];
    let factors = [1.0f32, 1.0];
    let mut router_f = vec![0.0f32; experts * hidden];
    router_f[0] = 10.0;
    let router = bf16_exact(&router_f);

    let path = write_checkpoint(
        "ammon-nvfp4-moe.safetensors",
        vec![
            (
                "router",
                TensorView::new(Dtype::BF16, vec![experts, hidden], &router).unwrap(),
            ),
            (
                "gate_codes",
                TensorView::new(Dtype::U8, vec![experts, fused, hidden / 2], &gate_codes).unwrap(),
            ),
            (
                "gate_scales",
                TensorView::new(
                    Dtype::F8_E4M3,
                    vec![experts, fused, hidden / 16],
                    &gate_scales,
                )
                .unwrap(),
            ),
            (
                "down_codes",
                TensorView::new(
                    Dtype::U8,
                    vec![experts, hidden, intermediate / 2],
                    &down_codes,
                )
                .unwrap(),
            ),
            (
                "down_scales",
                TensorView::new(
                    Dtype::F8_E4M3,
                    vec![experts, hidden, intermediate / 16],
                    &down_scales,
                )
                .unwrap(),
            ),
            (
                "factors",
                TensorView::new(Dtype::F32, vec![experts], bytemuck::cast_slice(&factors)).unwrap(),
            ),
        ],
    );
    let mapped = MappedCheckpoint::open(&path).unwrap();
    assert_eq!(
        mapped.bytes_named("gate_codes").unwrap().len(),
        gate_codes.len()
    );

    let device = runtime();
    let ctx = device.create_context().unwrap();
    let router_t = mapped.upload(&device, "router").unwrap();
    let gate_codes_t = mapped.upload(&device, "gate_codes").unwrap();
    let gate_scales_t = mapped.upload(&device, "gate_scales").unwrap();
    let down_codes_t = mapped.upload(&device, "down_codes").unwrap();
    let down_scales_t = mapped.upload(&device, "down_scales").unwrap();
    let factors_t = mapped.upload(&device, "factors").unwrap();
    let mut x_host = vec![0.0f32; hidden];
    x_host[0] = 1.0;
    let x = Tensor::from_f32(&device, TensorShape::vector(hidden as u32), &x_host).unwrap();
    let bias = Tensor::from_f32(&device, TensorShape::vector(experts as u32), &[0.0, 0.0]).unwrap();
    let out = Tensor::zeros(
        &device,
        TensorShape::vector(hidden as u32),
        TensorDType::F32,
    )
    .unwrap();
    let moe = DecodeMoe::new(
        &device,
        experts as u32,
        1,
        hidden as u32,
        intermediate as u32,
        1.0,
        30.0,
        30.0,
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    moe.record_nvfp4(
        &mut scheme,
        x.view(),
        router_t.view(),
        bias.view(),
        PackedExperts {
            codes: gate_codes_t.view(),
            scales: gate_scales_t.view(),
            factors: factors_t.view(),
        },
        PackedExperts {
            codes: down_codes_t.view(),
            scales: down_scales_t.view(),
            factors: factors_t.view(),
        },
        out.view(),
    )
    .unwrap();
    let got = read_f32(&mut scheme, out.buffer());

    let score = 1.0 / (1.0 + (-10.0f32).exp());
    let route = score / (score + 1.0e-20);
    let activated = (1.0f32 / 30.0).tanh()
        * 30.0
        * (1.0 / (1.0 + (-1.0f32).exp()))
        * (2.0f32 / 30.0).tanh()
        * 30.0;
    let expect = route * activated;
    assert!(
        (got[0] - expect).abs() < 1e-4,
        "got {} expect {expect} route {route}",
        got[0]
    );
    assert!(got[1..].iter().all(|v| v.abs() < 1e-5), "{got:?}");
    let _ = std::fs::remove_file(path);
}

fn e2m1(nib: u8) -> f32 {
    let mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nib & 7) as usize];
    if nib & 8 == 0 {
        mag
    } else {
        -mag
    }
}

fn e4m3(b: u8) -> f32 {
    let exp = (b >> 3) & 0xF;
    let mant = b & 7;
    let val = if exp == 0 {
        mant as f32 * 2f32.powi(-9)
    } else if exp == 15 && mant == 7 {
        return f32::NAN;
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if b & 0x80 == 0 {
        val
    } else {
        -val
    }
}

#[test]
fn random_block_scaled_row_matches_host_dequant() {
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    let experts = 2usize;
    let rows = 2usize;
    let k = 16usize;
    let mut codes = vec![0u8; experts * rows * k / 2];
    let mut scales = vec![0u8; experts * rows];
    for b in &mut codes {
        *b = (next() & 0xff) as u8;
    }
    for s in &mut scales {
        let mut v = (next() & 0xff) as u8;
        if (v & 0x7f) == 0x7f {
            v = 0x38;
        }
        *s = v;
    }
    let factors_host = [0.2f32, 0.37];
    let x_host: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let id = 1usize;
    let mut expect = vec![0.0f32; rows];
    for row in 0..rows {
        for col in 0..k {
            let byte = codes[(id * rows + row) * (k / 2) + col / 2];
            let nib = if col % 2 == 0 { byte & 0xf } else { byte >> 4 };
            expect[row] +=
                e2m1(nib) * e4m3(scales[id * rows + row]) * factors_host[id] * x_host[col];
        }
    }
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let kernels = TensorKernels::new(&device).unwrap();
    let codes_t = Tensor::from_bytes(
        &device,
        TensorShape::from_dims(&[experts as u32, rows as u32, (k / 2) as u32]).unwrap(),
        TensorDType::U8,
        &codes,
    )
    .unwrap();
    let scales_t = Tensor::from_bytes(
        &device,
        TensorShape::from_dims(&[experts as u32, rows as u32, 1]).unwrap(),
        TensorDType::F8E4M3,
        &scales,
    )
    .unwrap();
    let factors =
        Tensor::from_f32(&device, TensorShape::vector(experts as u32), &factors_host).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(k as u32), &x_host).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(1), &[id as u32]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(rows as u32), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .recorder(&mut scheme)
        .matmul_block_scaled_into(
            "rand",
            BlockScaled {
                codes: codes_t.view(),
                scales: scales_t.view(),
                elems: goldy::ops::PackedElem::E2M1,
                scale: goldy::ops::PackedScale::E4M3,
                block: [1, 16],
                global: Some(factors.view()),
            },
            x.view(),
            Some((ids.view(), 0)),
            out.view(),
        )
        .unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!((g - e).abs() < 1e-4, "row {i} got {g} expect {e}");
    }
}
