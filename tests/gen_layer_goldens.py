#!/usr/bin/env python3
"""One sliding and one full loop decoder layer, saved as safetensors.

The classes are the pinned transformers 5.24.0 `inkling2_loop` modules
(commit eeee1109). `safetensors.torch.save_file` is the blob `save_pretrained`
writes. Activations for several decode tokens are a compact binary.

    python3 tests/gen_layer_goldens.py /path/to/modeling_inkling2_loop.py
"""

import struct
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

PIN = "eeee11099135592f1d03c2871fbd3d8132e5a4d7"


def load(path):
    lines = open(path).read().splitlines()
    chunk = []
    for line in lines[66:1173]:
        if line.lstrip().startswith("@"):
            continue
        chunk.append(line)
    ns = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": __import__("math"),
        "Callable": object,
        "Cache": object,
        "Inkling2LoopTextConfig": object,
        "Inkling2LoopConfig": object,
        "Unpack": dict,
        "TransformersKwargs": dict,
        "GradientCheckpointingLayer": nn.Module,
        "ACT2FN": {"sigmoid": torch.sigmoid},
        "ALL_ATTENTION_FUNCTIONS": type(
            "Attn",
            (),
            {"get_interface": staticmethod(lambda _name, default: default)},
        ),
    }
    exec("\n".join(chunk), ns)
    return ns


class Cache:
    def __init__(self, window):
        self.window = window
        self.conv_states = {}
        self.ready = set()
        self.seen = set()
        self.kernel = {}
        self.keys = None
        self.values = None

    def is_recording_past(self, _layer):
        return False

    def get_conv_state(self, layer, state_idx=0):
        return self.conv_states.get((layer, state_idx))

    def update_conv_state(self, conv_states, layer, state_idx=0, conv_kernel_size=None):
        key = (layer, state_idx)
        if key not in self.ready:
            k = conv_kernel_size if conv_kernel_size is not None else conv_states.shape[-1]
            self.kernel[key] = k
            self.conv_states[key] = torch.zeros(*conv_states.shape[:-1], k)
            self.ready.add(key)
        if key not in self.seen:
            full = conv_states
            self.seen.add(key)
            if full.shape[-1] < self.kernel[key]:
                full = F.pad(full, (self.kernel[key] - full.shape[-1], 0))
        else:
            full = torch.cat([self.conv_states[key], conv_states], dim=-1)
        self.conv_states[key] = full[..., -self.kernel[key] :].contiguous()
        return full

    def update(self, key_states, value_states, _layer, **_kwargs):
        if self.keys is None:
            self.keys, self.values = key_states, value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        if self.window is None:
            return self.keys, self.values
        return self.keys[..., -self.window :, :], self.values[..., -self.window :, :]


def config(full):
    types = ["hybrid_sliding"] * 12
    types[5] = "full_attention"
    types[11] = "full_attention"
    return SimpleNamespace(
        hidden_size=32,
        rms_norm_eps=1e-5,
        loop_gate_rank=4,
        layer_types=types,
        swa_head_dim=16,
        head_dim=16,
        swa_num_attention_heads=4,
        num_attention_heads=4,
        swa_num_key_value_heads=2,
        num_key_value_heads=2,
        sliding_window_size=4,
        rel_extent=8 if full else 4,
        attention_dropout=0.0,
        qk_l2_clip_threshold=2.0,
        rel_l2_clip_threshold=2.0,
        d_rel=4,
        log_scaling_alpha=0.25,
        log_scaling_n_floor=2.0,
        sconv_kernel_size=4,
        conv_kernel_size=4,
        n_layers_per_attn_res_block=6,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        n_routed_experts=4,
        moe_intermediate_size=8,
        num_experts_per_tok=2,
        num_local_experts=4,
        scoring_func="sigmoid",
        routed_scaling_factor=1.5,
        _attn_implementation="eager",
    )


def fill(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.15)
        for buf in module.buffers():
            if buf.is_floating_point():
                buf.copy_(torch.randn_like(buf) * 0.05)


def margin(module, hidden):
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, module.mlp.gate.weight.float())
        scores = torch.sigmoid(logits) + module.mlp.gate.e_score_correction_bias
        top = torch.topk(scores, module.mlp.gate.top_k + 1, dim=-1).values
        return (top[:, module.mlp.gate.top_k - 1] - top[:, module.mlp.gate.top_k]).min().item()


def run_case(ns, full, layer_idx, seed):
    cfg = config(full)
    cls = ns["Inkling2LoopFullLoopDecoderLayer" if full else "Inkling2LoopSlidingLoopDecoderLayer"]
    layer = cls(cfg, layer_idx)
    fill(layer, seed)
    hidden = cfg.hidden_size
    n_blocks = 2
    tokens = 5
    blocks = torch.randn(1, 1, n_blocks, hidden)
    xs = []
    messages = []
    ys = []
    cache = Cache(None if full else cfg.sliding_window_size)
    window = None if full else cfg.sliding_window_size
    for pos in range(tokens):
        x = torch.randn(1, 1, hidden)
        msg = torch.randn(1, 1, 4, hidden)
        if margin(layer, x.reshape(1, hidden)) < 1e-4:
            raise SystemExit(f"topk margin too small at seed {seed} pos {pos}")
        n_kv = pos + 1 if window is None else min(pos + 1, window)
        distance = torch.arange(n_kv - 1, -1, -1).view(1, n_kv)
        y = layer(
            x,
            blocks,
            msg,
            attention_mask=None,
            relative_distance=distance,
            cache=cache,
            position_ids=torch.tensor([[pos]]),
        )
        xs.append(x.reshape(-1).float())
        messages.append(msg.reshape(-1).float())
        ys.append(y.reshape(-1).float())
    return layer, blocks.reshape(-1).float(), xs, messages, ys


def pack_f32(buf, tensor):
    data = tensor.detach().float().reshape(-1).numpy().astype("float32").tobytes()
    buf.append(struct.pack("<I", len(data) // 4))
    buf.append(data)


def main():
    from safetensors.torch import save_file

    ns = load(sys.argv[1])
    cases = [
        (False, 6, 11, "sliding."),
        (True, 11, 29, "full."),
    ]
    tensors = {}
    blob = [struct.pack("<III", 0x5259414C, 1, len(cases))]
    for full, layer_idx, seed, prefix in cases:
        layer, blocks, xs, messages, ys = run_case(ns, full, layer_idx, seed)
        for name, value in layer.state_dict().items():
            tensors[prefix + name] = value.detach().float().contiguous()
        cfg = config(full)
        header = struct.pack(
            "<IIIIIIIIIIIIIIIIII",
            int(full),
            layer_idx,
            int(layer_idx % 6 == 0),
            1,
            cfg.hidden_size,
            4,
            2,
            16,
            cfg.d_rel,
            cfg.rel_extent if full else cfg.sliding_window_size,
            cfg.sconv_kernel_size,
            cfg.conv_kernel_size,
            cfg.num_local_experts,
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            cfg.loop_gate_rank,
            0 if full else cfg.sliding_window_size,
            len(xs),
        )
        blob.append(header)
        scalars = struct.pack(
            "<8f",
            cfg.qk_l2_clip_threshold,
            cfg.rel_l2_clip_threshold,
            cfg.rms_norm_eps,
            cfg.routed_scaling_factor,
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
            cfg.log_scaling_alpha if full else 0.0,
            cfg.log_scaling_n_floor,
        )
        blob.append(scalars)
        blob.append(struct.pack("<I", 2))
        pack_f32(blob, blocks)
        for x, msg, y in zip(xs, messages, ys):
            pack_f32(blob, x)
            pack_f32(blob, msg)
            pack_f32(blob, y)
        print(prefix, "params", len(layer.state_dict()), "y0", float(ys[0][0]))
    save_file(tensors, "tests/data/decoder_layer.safetensors")
    open("tests/data/layer_goldens.bin", "wb").write(b"".join(blob))
    print("wrote safetensors and layer_goldens.bin")


if __name__ == "__main__":
    main()
