//! One Inkling 2 / loop decoder layer, recorded once and replayed per token.
//!
//! Order matches the pinned decoder layer: depth mix (skipped on layer 0),
//! optional message gates, input RMSNorm, self-attention, post-attention short
//! conv, residual (the layer input is added only off a block boundary), MLP
//! depth mix, optional MLP message gate, post-attention RMSNorm, routed
//! experts, post-MLP short conv, and the residual add. KV and the key/value
//! convolutions live on [`DecodeLayerState`]. The post-attention and post-MLP
//! convolutions keep their own state on this module.

use crate::blocks::Blocks;
use crate::checkpoint::{DecodeLayerSpec, DecodeLayerTensors, MessageGateTensors};
use crate::depth::DEPTH_RMS_EPS;
use crate::modules::{
    DecodeAttention, DecodeAttentionKind, DecodeAttentionWeights, DecodeLayerState, DecodeMoe,
    Linear, LogPositionScale, RmsNorm, ShortConv, Softcap,
};
use anyhow::Context;
use goldy::{Buffer, GoldyError, Runtime, Scheme, Tensor, TensorDType, TensorShape, TensorView};
use std::sync::Arc;

/// Scalar hyperparameters that are config fields, not checkpoint tensors.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLayerScalars {
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub rms_eps: f32,
    pub route_scale: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    /// `0` disables log-position scaling. Full layers in the reference set it.
    pub log_alpha: f32,
    pub log_floor: f32,
}

struct GateWeights {
    down: Tensor,
    forget: Tensor,
    message: Tensor,
}

/// Device weights for [`DecodeLayer::record`].
pub struct DecodeLayerWeights {
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
    k_conv_proj: Option<Tensor>,
    v_conv_proj: Option<Tensor>,
    input_norm: Option<Tensor>,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    v_norm: Option<Tensor>,
    post_norm: Tensor,
    post_attn_conv: Tensor,
    post_mlp_conv: Tensor,
    attn_query: Option<Tensor>,
    mlp_query: Tensor,
    router: Tensor,
    router_bias: Tensor,
    gate_up: Tensor,
    down: Tensor,
    q_gate: Option<GateWeights>,
    k_gate: Option<GateWeights>,
    v_gate: Option<GateWeights>,
    mlp_gate: Option<GateWeights>,
}

impl DecodeLayerWeights {
    pub fn from_spec(
        runtime: &Runtime,
        spec: DecodeLayerSpec,
        host: &DecodeLayerTensors,
    ) -> anyhow::Result<Self> {
        let hidden = spec.hidden;
        let q_dim = spec.q_heads * spec.head_dim;
        let kv_dim = spec.kv_heads * spec.head_dim;
        let queries = if spec.full { 3 } else { 1 };
        let gate = |g: &MessageGateTensors| -> anyhow::Result<GateWeights> {
            Ok(GateWeights {
                down: upload(runtime, &g.down, &[spec.gate_rank, 2 * hidden])?,
                forget: upload(runtime, &g.forget, &[hidden, spec.gate_rank])?,
                message: upload(runtime, &g.message, &[hidden, spec.gate_rank])?,
            })
        };
        let conv = |data: &[f32]| -> anyhow::Result<Tensor> {
            if spec.dynamic_conv {
                upload(runtime, data, &[spec.conv_kernel, kv_dim])
            } else {
                upload(runtime, data, &[kv_dim, spec.conv_kernel])
            }
        };
        Ok(Self {
            wq: upload(runtime, &host.wq, &[q_dim, hidden])?,
            wk: upload(runtime, &host.wk, &[kv_dim, hidden])?,
            wv: upload(runtime, &host.wv, &[kv_dim, hidden])?,
            wo: upload(runtime, &host.wo, &[hidden, q_dim])?,
            wgate: upload(runtime, &host.wgate, &[q_dim, hidden])?,
            wsink: upload(runtime, &host.wsink, &[spec.q_heads, hidden])?,
            wrel: upload(runtime, &host.wrel, &[spec.q_heads * spec.d_rel, hidden])?,
            rel_proj: upload(runtime, &host.rel_proj, &[spec.d_rel, spec.rel_extent])?,
            q_scale: upload(runtime, &host.q_scale, &[spec.q_heads, spec.head_dim])?,
            rel_scale: upload(runtime, &host.rel_scale, &[spec.q_heads, spec.d_rel])?,
            k_conv: conv(&host.k_conv)?,
            v_conv: conv(&host.v_conv)?,
            k_conv_proj: host
                .k_conv_proj
                .as_ref()
                .map(|d| upload(runtime, d, &[spec.kv_heads * spec.conv_kernel, hidden]))
                .transpose()?,
            v_conv_proj: host
                .v_conv_proj
                .as_ref()
                .map(|d| upload(runtime, d, &[spec.kv_heads * spec.conv_kernel, hidden]))
                .transpose()?,
            input_norm: host
                .input_norm
                .as_ref()
                .map(|d| upload(runtime, d, &[hidden]))
                .transpose()?,
            q_norm: host
                .q_norm
                .as_ref()
                .map(|d| upload(runtime, d, &[hidden]))
                .transpose()?,
            k_norm: host
                .k_norm
                .as_ref()
                .map(|d| upload(runtime, d, &[hidden]))
                .transpose()?,
            v_norm: host
                .v_norm
                .as_ref()
                .map(|d| upload(runtime, d, &[hidden]))
                .transpose()?,
            post_norm: upload(runtime, &host.post_norm, &[hidden])?,
            post_attn_conv: upload(runtime, &host.post_attn_conv, &[hidden, spec.post_kernel])?,
            post_mlp_conv: upload(runtime, &host.post_mlp_conv, &[hidden, spec.post_kernel])?,
            attn_query: host
                .attn_query
                .as_ref()
                .map(|d| upload(runtime, d, &[queries, hidden]))
                .transpose()?,
            mlp_query: upload(runtime, &host.mlp_query, &[1, hidden])?,
            router: upload(runtime, &host.router, &[spec.experts, hidden])?,
            router_bias: upload(runtime, &host.router_bias, &[spec.experts])?,
            gate_up: upload(
                runtime,
                &host.gate_up,
                &[spec.experts, 2 * spec.intermediate, hidden],
            )?,
            down: upload(
                runtime,
                &host.down,
                &[spec.experts, hidden, spec.intermediate],
            )?,
            q_gate: host.q_gate.as_ref().map(gate).transpose()?,
            k_gate: host.k_gate.as_ref().map(gate).transpose()?,
            v_gate: host.v_gate.as_ref().map(gate).transpose()?,
            mlp_gate: host.mlp_gate.as_ref().map(gate).transpose()?,
        })
    }

    fn views(&self) -> DecodeAttentionWeights<'_> {
        DecodeAttentionWeights {
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
            k_conv_proj: self.k_conv_proj.as_ref().map(Tensor::view),
            v_conv_proj: self.v_conv_proj.as_ref().map(Tensor::view),
        }
    }
}

fn upload(runtime: &Runtime, data: &[f32], dims: &[u32]) -> anyhow::Result<Tensor> {
    let shape = TensorShape::from_dims(dims).context("decode layer weight shape")?;
    Tensor::from_f32(runtime, shape, data).context("upload decode layer weight")
}

fn zeros(runtime: &Runtime, dims: &[u32]) -> anyhow::Result<Tensor> {
    Ok(Tensor::zeros(
        runtime,
        TensorShape::from_dims(dims)?,
        TensorDType::F32,
    )?)
}

/// Scratch and child modules for one decoder layer.
pub struct DecodeLayer {
    spec: DecodeLayerSpec,
    blocks: Arc<Blocks>,
    attn: DecodeAttention,
    moe: DecodeMoe,
    norm: RmsNorm,
    linear: Linear,
    tanh: Softcap,
    post_attn: ShortConv,
    post_mlp: ShortConv,
    scale: f32,
    depth: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    qn: Tensor,
    kn: Tensor,
    vn: Tensor,
    attn_out: Tensor,
    conv_attn: Tensor,
    residual: Tensor,
    mlp: Tensor,
    mlp_norm: Tensor,
    moe_out: Tensor,
    conv_mlp: Tensor,
    n1: Tensor,
    n2: Tensor,
    cat: Tensor,
    rank: Tensor,
    forget_logits: Tensor,
    forget: Tensor,
    gate_logits: Tensor,
    gate: Tensor,
}

impl DecodeLayer {
    pub fn new(
        runtime: &Runtime,
        spec: DecodeLayerSpec,
        scalars: DecodeLayerScalars,
    ) -> anyhow::Result<Self> {
        let kind = if spec.full {
            DecodeAttentionKind::Full
        } else {
            if spec.window == 0 {
                anyhow::bail!("sliding decoder layer needs a window");
            }
            DecodeAttentionKind::Sliding {
                window: spec.window,
            }
        };
        let log = if scalars.log_alpha != 0.0 {
            Some(LogPositionScale {
                alpha: scalars.log_alpha,
                floor: scalars.log_floor,
            })
        } else {
            None
        };
        let hidden = spec.hidden;
        let h = |n| zeros(runtime, &[n]);
        Ok(Self {
            attn: DecodeAttention::new(
                runtime,
                kind,
                spec.q_heads,
                spec.kv_heads,
                spec.head_dim,
                hidden,
                spec.d_rel,
                spec.rel_extent,
                spec.conv_kernel,
                spec.dynamic_conv,
                scalars.qk_clip,
                scalars.rel_clip,
                scalars.rms_eps,
                log,
            )?,
            moe: DecodeMoe::new(
                runtime,
                spec.experts,
                spec.top_k,
                hidden,
                spec.intermediate,
                scalars.route_scale,
                scalars.gate_cap,
                scalars.up_cap,
            )?,
            norm: RmsNorm::with_eps(runtime, scalars.rms_eps)?,
            linear: Linear::new(runtime)?,
            tanh: Softcap::new(runtime, 1.0)?,
            post_attn: ShortConv::new(runtime, hidden, spec.post_kernel)?,
            post_mlp: ShortConv::new(runtime, hidden, spec.post_kernel)?,
            blocks: Blocks::shared(runtime)?,
            scale: (f64::from(hidden)).powf(-0.5) as f32,
            depth: zeros(runtime, &[3, hidden])?,
            q: h(hidden)?,
            k: h(hidden)?,
            v: h(hidden)?,
            qn: h(hidden)?,
            kn: h(hidden)?,
            vn: h(hidden)?,
            attn_out: h(hidden)?,
            conv_attn: h(hidden)?,
            residual: h(hidden)?,
            mlp: h(hidden)?,
            mlp_norm: h(hidden)?,
            moe_out: h(hidden)?,
            conv_mlp: h(hidden)?,
            n1: h(hidden)?,
            n2: h(hidden)?,
            cat: zeros(runtime, &[2, hidden])?,
            rank: h(spec.gate_rank.max(1))?,
            forget_logits: h(hidden)?,
            forget: h(hidden)?,
            gate_logits: h(hidden)?,
            gate: h(hidden)?,
            spec,
        })
    }

    pub fn spec(&self) -> DecodeLayerSpec {
        self.spec
    }

    /// Record the layer. `blocks` is `[capacity, hidden]` and the first
    /// `n_blocks` rows are valid. A boundary layer writes the layer input into
    /// row `n_blocks` so the MLP mix can see it; that write stays in `blocks`.
    ///
    /// `messages`, when this layer was built with them, is query, key, value,
    /// and MLP, each `[hidden]`. `step.position` is the token position.
    pub fn record(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        blocks: TensorView<'_>,
        n_blocks: u32,
        messages: Option<[TensorView<'_>; 4]>,
        weights: &DecodeLayerWeights,
        state: &DecodeLayerState,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let h = self.spec.hidden;
        if hidden.shape().dims() != [h] || out.shape().dims() != [h] {
            return Err(GoldyError::Validation(
                "decoder layer hidden and out are [hidden]".into(),
            ));
        }
        if blocks.shape().rank() != 2 || blocks.shape().dim(1)? != h {
            return Err(GoldyError::Validation(
                "decoder layer blocks are [capacity, hidden]".into(),
            ));
        }
        let capacity = blocks.shape().dim(0)?;
        let mlp_blocks = n_blocks + u32::from(self.spec.boundary);
        if mlp_blocks > capacity {
            return Err(GoldyError::Validation(
                "decoder layer block tape is shorter than the boundary append".into(),
            ));
        }
        if self.spec.messages != messages.is_some() {
            return Err(GoldyError::Validation(
                "decoder layer messages do not match the checkpoint".into(),
            ));
        }
        self.prepare_attention_streams(scheme, hidden, blocks, n_blocks, messages, weights)?;
        self.attn.record(
            scheme,
            self.qn.view(),
            self.kn.view(),
            self.vn.view(),
            weights.views(),
            state,
            step,
            self.attn_out.view(),
        )?;
        self.post_attn.record(
            scheme,
            self.attn_out.view(),
            weights.post_attn_conv.view(),
            self.conv_attn.view(),
        )?;
        let mlp_partial = if self.spec.boundary {
            self.blocks
                .record_depth_store(scheme, hidden, blocks, n_blocks)?;
            self.conv_attn.view()
        } else {
            self.blocks.record_add(
                scheme,
                "attn_residual",
                hidden,
                self.conv_attn.view(),
                self.residual.view(),
            )?;
            self.residual.view()
        };
        self.mix(
            scheme,
            blocks,
            mlp_partial,
            weights.mlp_query.view(),
            self.depth.view().narrow(0, 0, 1)?,
            mlp_blocks,
        )?;
        let mixed = self.row(0)?;
        if let Some(msgs) = messages {
            let gate = weights.mlp_gate.as_ref().ok_or_else(|| {
                GoldyError::Validation("decoder layer is missing the MLP message gate".into())
            })?;
            self.record_gate(scheme, mixed, msgs[3], gate, self.mlp.view())?;
        }
        let mlp_src = if self.spec.messages {
            self.mlp.view()
        } else {
            mixed
        };
        self.norm.record(
            scheme,
            mlp_src,
            weights.post_norm.view(),
            self.mlp_norm.view(),
        )?;
        self.moe.record(
            scheme,
            self.mlp_norm.view(),
            weights.router.view(),
            weights.router_bias.view(),
            weights.gate_up.view(),
            weights.down.view(),
            self.moe_out.view(),
        )?;
        self.post_mlp.record(
            scheme,
            self.moe_out.view(),
            weights.post_mlp_conv.view(),
            self.conv_mlp.view(),
        )?;
        let left = if self.spec.boundary {
            self.conv_attn.view()
        } else {
            self.residual.view()
        };
        self.blocks
            .record_add(scheme, "mlp_residual", left, self.conv_mlp.view(), out)?;
        Ok(())
    }

    fn prepare_attention_streams(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        blocks: TensorView<'_>,
        n_blocks: u32,
        messages: Option<[TensorView<'_>; 4]>,
        weights: &DecodeLayerWeights,
    ) -> Result<(), GoldyError> {
        if self.spec.layer_index > 0 {
            let queries = weights.attn_query.as_ref().ok_or_else(|| {
                GoldyError::Validation("decoder layer is missing attention_input.query".into())
            })?;
            let nq = if self.spec.full { 3 } else { 1 };
            self.mix(
                scheme,
                blocks,
                hidden,
                queries.view().narrow(0, 0, nq)?,
                self.depth.view().narrow(0, 0, nq)?,
                n_blocks,
            )?;
        }
        if let Some(msgs) = messages {
            let src = |index: u32| -> Result<TensorView<'_>, GoldyError> {
                if self.spec.layer_index == 0 {
                    Ok(hidden)
                } else if self.spec.full {
                    self.row(index)
                } else {
                    self.row(0)
                }
            };
            self.gate_into(
                scheme,
                src(0)?,
                msgs[0],
                weights.q_gate.as_ref(),
                self.q.view(),
            )?;
            self.gate_into(
                scheme,
                src(1)?,
                msgs[1],
                weights.k_gate.as_ref(),
                self.k.view(),
            )?;
            self.gate_into(
                scheme,
                src(2)?,
                msgs[2],
                weights.v_gate.as_ref(),
                self.v.view(),
            )?;
            self.norm_stream(
                scheme,
                self.q.view(),
                self.q_weight(weights)?,
                self.qn.view(),
            )?;
            self.norm_stream(
                scheme,
                self.k.view(),
                self.k_weight(weights)?,
                self.kn.view(),
            )?;
            self.norm_stream(
                scheme,
                self.v.view(),
                self.v_weight(weights)?,
                self.vn.view(),
            )?;
            return Ok(());
        }
        if self.spec.layer_index == 0 {
            if self.spec.full {
                self.norm_stream(scheme, hidden, self.q_weight(weights)?, self.qn.view())?;
                self.norm_stream(scheme, hidden, self.k_weight(weights)?, self.kn.view())?;
                self.norm_stream(scheme, hidden, self.v_weight(weights)?, self.vn.view())?;
            } else {
                let w = self.q_weight(weights)?;
                self.norm_stream(scheme, hidden, w, self.qn.view())?;
                self.norm_stream(scheme, hidden, w, self.kn.view())?;
                self.norm_stream(scheme, hidden, w, self.vn.view())?;
            }
            return Ok(());
        }
        if self.spec.full {
            self.norm_stream(
                scheme,
                self.row(0)?,
                self.q_weight(weights)?,
                self.qn.view(),
            )?;
            self.norm_stream(
                scheme,
                self.row(1)?,
                self.k_weight(weights)?,
                self.kn.view(),
            )?;
            self.norm_stream(
                scheme,
                self.row(2)?,
                self.v_weight(weights)?,
                self.vn.view(),
            )?;
        } else {
            let w = self.q_weight(weights)?;
            let src = self.row(0)?;
            self.norm_stream(scheme, src, w, self.qn.view())?;
            self.norm_stream(scheme, src, w, self.kn.view())?;
            self.norm_stream(scheme, src, w, self.vn.view())?;
        }
        Ok(())
    }

    fn q_weight<'a>(
        &'a self,
        weights: &'a DecodeLayerWeights,
    ) -> Result<TensorView<'a>, GoldyError> {
        if self.spec.full {
            weights.q_norm.as_ref().map(Tensor::view).ok_or_else(|| {
                GoldyError::Validation("full decoder layer is missing q_input_layernorm".into())
            })
        } else {
            weights
                .input_norm
                .as_ref()
                .map(Tensor::view)
                .ok_or_else(|| {
                    GoldyError::Validation(
                        "sliding decoder layer is missing input_layernorm".into(),
                    )
                })
        }
    }

    fn k_weight<'a>(
        &'a self,
        weights: &'a DecodeLayerWeights,
    ) -> Result<TensorView<'a>, GoldyError> {
        if self.spec.full {
            weights.k_norm.as_ref().map(Tensor::view).ok_or_else(|| {
                GoldyError::Validation("full decoder layer is missing k_input_layernorm".into())
            })
        } else {
            self.q_weight(weights)
        }
    }

    fn v_weight<'a>(
        &'a self,
        weights: &'a DecodeLayerWeights,
    ) -> Result<TensorView<'a>, GoldyError> {
        if self.spec.full {
            weights.v_norm.as_ref().map(Tensor::view).ok_or_else(|| {
                GoldyError::Validation("full decoder layer is missing v_input_layernorm".into())
            })
        } else {
            self.q_weight(weights)
        }
    }

    fn norm_stream(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.norm.record(scheme, x, weight, out)
    }

    fn gate_into(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        message: TensorView<'_>,
        gate: Option<&GateWeights>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let gate = gate.ok_or_else(|| {
            GoldyError::Validation("decoder layer is missing a message gate".into())
        })?;
        self.record_gate(scheme, hidden, message, gate, out)
    }

    fn record_gate(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        message: TensorView<'_>,
        gate: &GateWeights,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.norm
            .record_unweighted(scheme, hidden, self.n1.view())?;
        self.norm
            .record_unweighted(scheme, message, self.n2.view())?;
        self.blocks
            .record_depth_store(scheme, self.n1.view(), self.cat.view(), 0)?;
        self.blocks
            .record_depth_store(scheme, self.n2.view(), self.cat.view(), 1)?;
        let flat = self.cat.view().reshape(&[self.spec.hidden * 2])?;
        self.linear
            .record(scheme, gate.down.view(), flat, self.rank.view())?;
        self.linear.record(
            scheme,
            gate.forget.view(),
            self.rank.view(),
            self.forget_logits.view(),
        )?;
        self.tanh
            .record(scheme, self.forget_logits.view(), self.forget.view())?;
        self.linear.record(
            scheme,
            gate.message.view(),
            self.rank.view(),
            self.gate_logits.view(),
        )?;
        self.tanh
            .record(scheme, self.gate_logits.view(), self.gate.view())?;
        self.blocks.record_message_mix(
            scheme,
            hidden,
            message,
            self.forget.view(),
            self.gate.view(),
            out,
        )
    }

    fn mix(
        &self,
        scheme: &mut Scheme,
        blocks: TensorView<'_>,
        partial: TensorView<'_>,
        queries: TensorView<'_>,
        out: TensorView<'_>,
        n_blocks: u32,
    ) -> Result<(), GoldyError> {
        self.blocks.record_depth_mix(
            scheme,
            blocks,
            partial,
            queries,
            out,
            n_blocks,
            DEPTH_RMS_EPS,
            self.scale,
        )
    }

    fn row(&self, index: u32) -> Result<TensorView<'_>, GoldyError> {
        self.depth
            .view()
            .narrow(0, index, 1)?
            .reshape(&[self.spec.hidden])
    }
}
