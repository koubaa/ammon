//! Safetensors checkpoints for one Inkling 2 / Inkling 2 Loop decoder layer.
//!
//! Names are the parameter names of the pinned transformers 5.24.0 classes
//! (`eeee1109`), optionally under a prefix such as `model.layers.0.`. Values
//! are FP32. A conv1d weight stored as `[channels, 1, kernel]` is squeezed to
//! `[channels, kernel]`.

use anyhow::{bail, Result};
use safetensors::Dtype;
use safetensors::SafeTensors;

/// Shapes and which optional submodules a decoder-layer checkpoint contains.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLayerSpec {
    pub full: bool,
    pub dynamic_conv: bool,
    pub layer_index: u32,
    pub boundary: bool,
    pub messages: bool,
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub rel_extent: u32,
    pub conv_kernel: u32,
    pub post_kernel: u32,
    pub experts: u32,
    pub top_k: u32,
    pub intermediate: u32,
    pub gate_rank: u32,
    /// Sliding window. Zero when `full` is set.
    pub window: u32,
}

/// Host FP32 weights of one decoder layer, in Ammon's layouts.
#[derive(Clone, Debug)]
pub struct DecodeLayerTensors {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wgate: Vec<f32>,
    pub wsink: Vec<f32>,
    pub wrel: Vec<f32>,
    pub rel_proj: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub k_conv_proj: Option<Vec<f32>>,
    pub v_conv_proj: Option<Vec<f32>>,
    pub input_norm: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub v_norm: Option<Vec<f32>>,
    pub post_norm: Vec<f32>,
    pub post_attn_conv: Vec<f32>,
    pub post_mlp_conv: Vec<f32>,
    pub attn_query: Option<Vec<f32>>,
    pub mlp_query: Vec<f32>,
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub down: Vec<f32>,
    pub q_gate: Option<MessageGateTensors>,
    pub k_gate: Option<MessageGateTensors>,
    pub v_gate: Option<MessageGateTensors>,
    pub mlp_gate: Option<MessageGateTensors>,
}

#[derive(Clone, Debug)]
pub struct MessageGateTensors {
    pub down: Vec<f32>,
    pub forget: Vec<f32>,
    pub message: Vec<f32>,
}

struct Owned {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Load one layer from a safetensors blob.
///
/// `prefix` is stripped from names (`model.layers.3.`). Keys that do not start
/// with `prefix` are left alone, so a full model file can be sliced. Every key
/// under the prefix must be a known FP32 parameter of this spec.
pub fn load_decode_layer(
    bytes: &[u8],
    prefix: &str,
    spec: DecodeLayerSpec,
) -> Result<DecodeLayerTensors> {
    let file =
        SafeTensors::deserialize(bytes).map_err(|err| anyhow::anyhow!("safetensors: {err}"))?;
    let mut have: Vec<(String, Owned)> = Vec::new();
    let mut errors = Vec::new();
    for (name, view) in file.iter() {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        if rest.is_empty() {
            errors.push(format!("empty parameter name under prefix `{prefix}`"));
            continue;
        }
        if view.dtype() != Dtype::F32 {
            errors.push(format!("`{name}` is {:?}, expected F32", view.dtype()));
            continue;
        }
        if view.data().len() % 4 != 0 {
            errors.push(format!("`{name}` byte length is not a multiple of 4"));
            continue;
        }
        let data = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        have.push((
            rest.to_string(),
            Owned {
                shape: view.shape().to_vec(),
                data,
            },
        ));
    }

    let mut used = vec![false; have.len()];
    let take = |errors: &mut Vec<String>, used: &mut [bool], key: &str, shape: &[usize]| {
        take_named(&have, used, prefix, key, shape, errors)
    };
    let take_conv =
        |errors: &mut Vec<String>, used: &mut [bool], key: &str, channels: u32, kernel: u32| {
            take_conv_named(&have, used, prefix, key, channels, kernel, errors)
        };

    let hidden = spec.hidden as usize;
    let q_dim = (spec.q_heads * spec.head_dim) as usize;
    let kv_dim = (spec.kv_heads * spec.head_dim) as usize;
    let q_heads = spec.q_heads as usize;
    let d_rel = spec.d_rel as usize;
    let experts = spec.experts as usize;
    let intermediate = spec.intermediate as usize;
    let rank = spec.gate_rank as usize;

    let wq = take(
        &mut errors,
        &mut used,
        "self_attn.q_proj.linear.weight",
        &[q_dim, hidden],
    );
    let wk = take(
        &mut errors,
        &mut used,
        "self_attn.k_proj.linear.weight",
        &[kv_dim, hidden],
    );
    let wv = take(
        &mut errors,
        &mut used,
        "self_attn.v_proj.linear.weight",
        &[kv_dim, hidden],
    );
    let wo = take(
        &mut errors,
        &mut used,
        "self_attn.o_proj.linear.weight",
        &[hidden, q_dim],
    );
    let wgate = take(
        &mut errors,
        &mut used,
        "self_attn.o_proj.gate_proj.weight",
        &[q_dim, hidden],
    );
    let wsink = take(
        &mut errors,
        &mut used,
        "self_attn.sink_proj.weight",
        &[q_heads, hidden],
    );
    let wrel = take(
        &mut errors,
        &mut used,
        "self_attn.relative_position_bias.linear.weight",
        &[q_heads * d_rel, hidden],
    );
    let rel_proj = take(
        &mut errors,
        &mut used,
        "self_attn.relative_position_bias.proj",
        &[d_rel, spec.rel_extent as usize],
    );
    let q_scale = take(
        &mut errors,
        &mut used,
        "self_attn.q_proj.norm.scale",
        &[q_heads, spec.head_dim as usize],
    );
    let rel_scale = take(
        &mut errors,
        &mut used,
        "self_attn.relative_position_bias.norm.scale",
        &[q_heads, d_rel],
    );

    let (k_conv, k_conv_proj, v_conv, v_conv_proj) = if spec.dynamic_conv {
        let taps = (spec.kv_heads * spec.conv_kernel) as usize;
        (
            take(
                &mut errors,
                &mut used,
                "self_attn.k_proj.sconv.static_weight",
                &[spec.conv_kernel as usize, kv_dim],
            ),
            take(
                &mut errors,
                &mut used,
                "self_attn.k_proj.sconv.weight_proj.weight",
                &[taps, hidden],
            ),
            take(
                &mut errors,
                &mut used,
                "self_attn.v_proj.sconv.static_weight",
                &[spec.conv_kernel as usize, kv_dim],
            ),
            take(
                &mut errors,
                &mut used,
                "self_attn.v_proj.sconv.weight_proj.weight",
                &[taps, hidden],
            ),
        )
    } else {
        (
            take_conv(
                &mut errors,
                &mut used,
                "self_attn.k_proj.sconv.conv1d.weight",
                spec.kv_heads * spec.head_dim,
                spec.conv_kernel,
            ),
            None,
            take_conv(
                &mut errors,
                &mut used,
                "self_attn.v_proj.sconv.conv1d.weight",
                spec.kv_heads * spec.head_dim,
                spec.conv_kernel,
            ),
            None,
        )
    };

    let (input_norm, q_norm, k_norm, v_norm) = if spec.full {
        (
            None,
            take(
                &mut errors,
                &mut used,
                "q_input_layernorm.weight",
                &[hidden],
            ),
            take(
                &mut errors,
                &mut used,
                "k_input_layernorm.weight",
                &[hidden],
            ),
            take(
                &mut errors,
                &mut used,
                "v_input_layernorm.weight",
                &[hidden],
            ),
        )
    } else {
        (
            take(&mut errors, &mut used, "input_layernorm.weight", &[hidden]),
            None,
            None,
            None,
        )
    };
    let post_norm = take(
        &mut errors,
        &mut used,
        "post_attention_layernorm.weight",
        &[hidden],
    );
    let post_attn_conv = take_conv(
        &mut errors,
        &mut used,
        "post_attention_convolution.conv1d.weight",
        spec.hidden,
        spec.post_kernel,
    );
    let post_mlp_conv = take_conv(
        &mut errors,
        &mut used,
        "post_mlp_convolution.conv1d.weight",
        spec.hidden,
        spec.post_kernel,
    );
    let attn_query = if spec.layer_index == 0 {
        None
    } else if spec.full {
        take(
            &mut errors,
            &mut used,
            "attention_input.query",
            &[3, hidden],
        )
    } else {
        take_row(
            &mut errors,
            &mut used,
            &have,
            prefix,
            "attention_input.query",
            hidden,
        )
    };
    let mlp_query = take_row(
        &mut errors,
        &mut used,
        &have,
        prefix,
        "mlp_input.query",
        hidden,
    );
    let router = take(
        &mut errors,
        &mut used,
        "mlp.gate.weight",
        &[experts, hidden],
    );
    let router_bias = take(
        &mut errors,
        &mut used,
        "mlp.gate.e_score_correction_bias",
        &[experts],
    );
    let gate_up = take(
        &mut errors,
        &mut used,
        "mlp.experts.gate_up_proj",
        &[experts, 2 * intermediate, hidden],
    );
    let down = take(
        &mut errors,
        &mut used,
        "mlp.experts.down_proj",
        &[experts, hidden, intermediate],
    );

    let gate = |errors: &mut Vec<String>, used: &mut [bool], which: &str| {
        if !spec.messages {
            return None;
        }
        let down = take(
            errors,
            used,
            &format!("{which}_message_gate.down_proj.weight"),
            &[rank, 2 * hidden],
        )?;
        let forget = take(
            errors,
            used,
            &format!("{which}_message_gate.forget_gate_proj.weight"),
            &[hidden, rank],
        )?;
        let message = take(
            errors,
            used,
            &format!("{which}_message_gate.message_gate_proj.weight"),
            &[hidden, rank],
        )?;
        Some(MessageGateTensors {
            down,
            forget,
            message,
        })
    };
    let q_gate = gate(&mut errors, &mut used, "q");
    let k_gate = gate(&mut errors, &mut used, "k");
    let v_gate = gate(&mut errors, &mut used, "v");
    let mlp_gate = gate(&mut errors, &mut used, "mlp");

    for (i, (name, _)) in have.iter().enumerate() {
        if !used[i] {
            errors.push(format!("unexpected `{prefix}{name}`"));
        }
    }
    if !errors.is_empty() {
        errors.sort();
        bail!(
            "safetensors decoder layer (`{prefix}`):\n{}",
            errors.join("\n")
        );
    }

    let need = |name: &str, value: Option<Vec<f32>>| -> Result<Vec<f32>> {
        value.ok_or_else(|| anyhow::anyhow!("internal: `{name}` missing after validation"))
    };
    Ok(DecodeLayerTensors {
        wq: need("wq", wq)?,
        wk: need("wk", wk)?,
        wv: need("wv", wv)?,
        wo: need("wo", wo)?,
        wgate: need("wgate", wgate)?,
        wsink: need("wsink", wsink)?,
        wrel: need("wrel", wrel)?,
        rel_proj: need("rel_proj", rel_proj)?,
        q_scale: need("q_scale", q_scale)?,
        rel_scale: need("rel_scale", rel_scale)?,
        k_conv: need("k_conv", k_conv)?,
        v_conv: need("v_conv", v_conv)?,
        k_conv_proj,
        v_conv_proj,
        input_norm,
        q_norm,
        k_norm,
        v_norm,
        post_norm: need("post_norm", post_norm)?,
        post_attn_conv: need("post_attn_conv", post_attn_conv)?,
        post_mlp_conv: need("post_mlp_conv", post_mlp_conv)?,
        attn_query,
        mlp_query: need("mlp_query", mlp_query)?,
        router: need("router", router)?,
        router_bias: need("router_bias", router_bias)?,
        gate_up: need("gate_up", gate_up)?,
        down: need("down", down)?,
        q_gate,
        k_gate,
        v_gate,
        mlp_gate,
    })
}

fn take_row(
    errors: &mut Vec<String>,
    used: &mut [bool],
    have: &[(String, Owned)],
    prefix: &str,
    key: &str,
    width: usize,
) -> Option<Vec<f32>> {
    let Some(index) = have.iter().position(|(name, _)| name == key) else {
        errors.push(format!("missing `{prefix}{key}` [{width}] or [1, {width}]"));
        return None;
    };
    used[index] = true;
    let owned = &have[index].1;
    if owned.shape == [width] || owned.shape == [1, width] {
        return Some(owned.data.clone());
    }
    errors.push(format!(
        "`{prefix}{key}` has shape {:?}, expected [{width}] or [1, {width}]",
        owned.shape
    ));
    None
}

fn take_named(
    have: &[(String, Owned)],
    used: &mut [bool],
    prefix: &str,
    key: &str,
    shape: &[usize],
    errors: &mut Vec<String>,
) -> Option<Vec<f32>> {
    let Some(index) = have.iter().position(|(name, _)| name == key) else {
        errors.push(format!("missing `{prefix}{key}` {shape:?}"));
        return None;
    };
    used[index] = true;
    let owned = &have[index].1;
    if owned.shape != shape {
        errors.push(format!(
            "`{prefix}{key}` has shape {:?}, expected {shape:?}",
            owned.shape
        ));
        return None;
    }
    Some(owned.data.clone())
}

fn take_conv_named(
    have: &[(String, Owned)],
    used: &mut [bool],
    prefix: &str,
    key: &str,
    channels: u32,
    kernel: u32,
    errors: &mut Vec<String>,
) -> Option<Vec<f32>> {
    let flat = [channels as usize, kernel as usize];
    let conv = [channels as usize, 1, kernel as usize];
    let Some(index) = have.iter().position(|(name, _)| name == key) else {
        errors.push(format!(
            "missing `{prefix}{key}` {flat:?} (or conv1d {conv:?})"
        ));
        return None;
    };
    used[index] = true;
    let owned = &have[index].1;
    if owned.shape == flat || owned.shape == conv {
        return Some(owned.data.clone());
    }
    errors.push(format!(
        "`{prefix}{key}` has shape {:?}, expected {flat:?} or {conv:?}",
        owned.shape
    ));
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn spec() -> DecodeLayerSpec {
        DecodeLayerSpec {
            full: false,
            dynamic_conv: false,
            layer_index: 1,
            boundary: false,
            messages: false,
            hidden: 4,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
            d_rel: 2,
            rel_extent: 4,
            conv_kernel: 2,
            post_kernel: 2,
            experts: 2,
            top_k: 1,
            intermediate: 2,
            gate_rank: 2,
            window: 4,
        }
    }

    fn f32_tensor<'a>(shape: &'a [usize], data: &'a [f32]) -> TensorView<'a> {
        TensorView::new(Dtype::F32, shape.to_vec(), bytemuck::cast_slice(data)).unwrap()
    }

    fn fill(spec: DecodeLayerSpec) -> Vec<(String, Vec<f32>)> {
        let h = spec.hidden as usize;
        let q = (spec.q_heads * spec.head_dim) as usize;
        let kv = (spec.kv_heads * spec.head_dim) as usize;
        let e = spec.experts as usize;
        let i = spec.intermediate as usize;
        let mut rows = vec![
            ("self_attn.q_proj.linear.weight", vec![0.0; q * h]),
            ("self_attn.k_proj.linear.weight", vec![0.0; kv * h]),
            ("self_attn.v_proj.linear.weight", vec![0.0; kv * h]),
            ("self_attn.o_proj.linear.weight", vec![0.0; h * q]),
            ("self_attn.o_proj.gate_proj.weight", vec![0.0; q * h]),
            (
                "self_attn.sink_proj.weight",
                vec![0.0; spec.q_heads as usize * h],
            ),
            (
                "self_attn.relative_position_bias.linear.weight",
                vec![0.0; spec.q_heads as usize * spec.d_rel as usize * h],
            ),
            (
                "self_attn.relative_position_bias.proj",
                vec![0.0; spec.d_rel as usize * spec.rel_extent as usize],
            ),
            (
                "self_attn.q_proj.norm.scale",
                vec![0.0; spec.q_heads as usize * spec.head_dim as usize],
            ),
            (
                "self_attn.relative_position_bias.norm.scale",
                vec![0.0; spec.q_heads as usize * spec.d_rel as usize],
            ),
            (
                "self_attn.k_proj.sconv.conv1d.weight",
                vec![0.0; kv * spec.conv_kernel as usize],
            ),
            (
                "self_attn.v_proj.sconv.conv1d.weight",
                vec![0.0; kv * spec.conv_kernel as usize],
            ),
            ("input_layernorm.weight", vec![0.0; h]),
            ("post_attention_layernorm.weight", vec![0.0; h]),
            (
                "post_attention_convolution.conv1d.weight",
                vec![0.0; h * spec.post_kernel as usize],
            ),
            (
                "post_mlp_convolution.conv1d.weight",
                vec![0.0; h * spec.post_kernel as usize],
            ),
            ("attention_input.query", vec![0.0; h]),
            ("mlp_input.query", vec![0.0; h]),
            ("mlp.gate.weight", vec![0.0; e * h]),
            ("mlp.gate.e_score_correction_bias", vec![0.0; e]),
            ("mlp.experts.gate_up_proj", vec![0.0; e * 2 * i * h]),
            ("mlp.experts.down_proj", vec![0.0; e * h * i]),
        ];
        let _ = &mut rows;
        rows.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn shapes_of(spec: DecodeLayerSpec) -> Vec<(&'static str, Vec<usize>)> {
        let h = spec.hidden as usize;
        let q = (spec.q_heads * spec.head_dim) as usize;
        let kv = (spec.kv_heads * spec.head_dim) as usize;
        let qh = spec.q_heads as usize;
        let e = spec.experts as usize;
        let i = spec.intermediate as usize;
        vec![
            ("self_attn.q_proj.linear.weight", vec![q, h]),
            ("self_attn.k_proj.linear.weight", vec![kv, h]),
            ("self_attn.v_proj.linear.weight", vec![kv, h]),
            ("self_attn.o_proj.linear.weight", vec![h, q]),
            ("self_attn.o_proj.gate_proj.weight", vec![q, h]),
            ("self_attn.sink_proj.weight", vec![qh, h]),
            (
                "self_attn.relative_position_bias.linear.weight",
                vec![qh * spec.d_rel as usize, h],
            ),
            (
                "self_attn.relative_position_bias.proj",
                vec![spec.d_rel as usize, spec.rel_extent as usize],
            ),
            (
                "self_attn.q_proj.norm.scale",
                vec![qh, spec.head_dim as usize],
            ),
            (
                "self_attn.relative_position_bias.norm.scale",
                vec![qh, spec.d_rel as usize],
            ),
            (
                "self_attn.k_proj.sconv.conv1d.weight",
                vec![kv, 1, spec.conv_kernel as usize],
            ),
            (
                "self_attn.v_proj.sconv.conv1d.weight",
                vec![kv, spec.conv_kernel as usize],
            ),
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            (
                "post_attention_convolution.conv1d.weight",
                vec![h, 1, spec.post_kernel as usize],
            ),
            (
                "post_mlp_convolution.conv1d.weight",
                vec![h, spec.post_kernel as usize],
            ),
            ("attention_input.query", vec![1, h]),
            ("mlp_input.query", vec![h]),
            ("mlp.gate.weight", vec![e, h]),
            ("mlp.gate.e_score_correction_bias", vec![e]),
            ("mlp.experts.gate_up_proj", vec![e, 2 * i, h]),
            ("mlp.experts.down_proj", vec![e, h, i]),
        ]
    }

    fn pack(rows: &[(&str, &[f32], &[usize])]) -> Vec<u8> {
        let views: Vec<(&str, TensorView<'_>)> = rows
            .iter()
            .map(|(name, data, shape)| (*name, f32_tensor(shape, data)))
            .collect();
        safetensors::tensor::serialize(views, None).unwrap()
    }

    #[test]
    fn loads_fp32_layer_and_squeezes_conv1d() {
        let spec = spec();
        let owned = fill(spec);
        let shape_list = shapes_of(spec);
        let rows: Vec<(&str, &[f32], &[usize])> = owned
            .iter()
            .zip(shape_list.iter())
            .map(|((name, data), (expect, shape))| {
                assert_eq!(name, expect);
                (name.as_str(), data.as_slice(), shape.as_slice())
            })
            .collect();
        let bytes = pack(&rows);
        let loaded = load_decode_layer(&bytes, "", spec).unwrap();
        assert_eq!(loaded.k_conv.len(), 8 * 2);
        assert!(loaded.q_gate.is_none());
        assert!(loaded.attn_query.is_some());
    }

    #[test]
    fn reports_missing_shape_dtype_and_unexpected_names() {
        let spec = spec();
        let mut owned = fill(spec);
        owned.retain(|(name, _)| name != "mlp.gate.weight");
        let mut shape_list = shapes_of(spec);
        shape_list.retain(|(name, _)| *name != "mlp.gate.weight");
        // Wrong rank on the sink, plus an extra tensor, plus a non-fp32 sibling file is separate.
        for (shape_name, shape) in &mut shape_list {
            if *shape_name == "self_attn.sink_proj.weight" {
                *shape = vec![1, spec.hidden as usize];
            }
        }
        // Rebuild sink data to match the wrong shape so the blob is valid safetensors.
        for (name, data) in &mut owned {
            if name == "self_attn.sink_proj.weight" {
                *data = vec![0.0; spec.hidden as usize];
            }
        }
        let rows: Vec<(&str, &[f32], &[usize])> = owned
            .iter()
            .zip(shape_list.iter())
            .map(|((name, data), (_, shape))| (name.as_str(), data.as_slice(), shape.as_slice()))
            .collect();
        let mut views: Vec<(&str, TensorView<'_>)> = rows
            .iter()
            .map(|(name, data, shape)| (*name, f32_tensor(shape, data)))
            .collect();
        let extra = [1.0f32, 2.0, 3.0, 4.0];
        views.push(("not.a.parameter", f32_tensor(&[4], &extra)));
        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        let err = load_decode_layer(&bytes, "model.layers.0.", spec).unwrap_err();
        let text = format!("{err:#}");
        // Prefix does not match, so every required name is missing and the file's
        // keys are ignored. Use an empty prefix for the real mismatch test.
        let _ = text;
        let bytes = {
            let views: Vec<(&str, TensorView<'_>)> = rows
                .iter()
                .map(|(name, data, shape)| (*name, f32_tensor(shape, data)))
                .chain(std::iter::once((
                    "not.a.parameter",
                    f32_tensor(&[4], &extra),
                )))
                .collect();
            safetensors::tensor::serialize(views, None).unwrap()
        };
        let err = load_decode_layer(&bytes, "", spec).unwrap_err().to_string();
        assert!(err.contains("missing `mlp.gate.weight`"), "{err}");
        assert!(err.contains("sink_proj.weight` has shape"), "{err}");
        assert!(err.contains("unexpected `not.a.parameter`"), "{err}");

        let f64_data = [0.0f64; 4];
        let bad = TensorView::new(Dtype::F64, vec![4], bytemuck::cast_slice(&f64_data)).unwrap();
        let only = safetensors::tensor::serialize([("input_layernorm.weight", bad)], None).unwrap();
        let err = load_decode_layer(&only, "", spec).unwrap_err().to_string();
        assert!(err.contains("expected F32"), "{err}");
    }
}
