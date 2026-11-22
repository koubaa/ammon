#!/usr/bin/env python3
"""Emit decode-attention goldens from the pinned transformers attention classes.

Source: transformers commit eeee11099135592f1d03c2871fbd3d8132e5a4d7
`src/transformers/models/inkling2/modeling_inkling2.py`.

    python3 tests/gen_inkling_attention_goldens.py /path/to/modeling_inkling2.py
"""

import math
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

PIN = "eeee11099135592f1d03c2871fbd3d8132e5a4d7"


def load_attention(path: str):
    lines = open(path).read().splitlines()
    chunk = []
    for line in lines[100:730]:
        if line.lstrip().startswith("@"):
            continue
        chunk.append(line)
    ns = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Cache": object,
        "Inkling2TextConfig": object,
        "Unpack": dict,
        "TransformersKwargs": dict,
        "Callable": object,
        "ACT2FN": {},
    }
    exec("\n".join(chunk), ns)

    class _Attn:
        @staticmethod
        def get_interface(_name, default):
            return default

    ns["ALL_ATTENTION_FUNCTIONS"] = _Attn
    return ns


class StepCache:
    """Conv state plus the HF sliding/full KV update used by one attention layer."""

    def __init__(self, window):
        self.window = window
        self.conv_states = {}
        self.ready = set()
        self.seen = set()
        self.kernel = {}
        self.keys = None
        self.values = None
        self.cumulative = 0
        self.is_initialized = False

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
            self.seen.discard(key)
        if key not in self.seen:
            full = conv_states
            self.seen.add(key)
            if full.shape[-1] < self.kernel[key]:
                full = F.pad(full, (self.kernel[key] - full.shape[-1], 0))
        else:
            full = torch.cat([self.conv_states[key], conv_states], dim=-1)
        self.conv_states[key] = full[..., -self.kernel[key] :].clone()
        return full

    def update(self, key_states, value_states, *_args, **_kwargs):
        if not self.is_initialized:
            self.keys = torch.tensor([], dtype=key_states.dtype)
            self.values = torch.tensor([], dtype=value_states.dtype)
            self.is_initialized = True
        self.cumulative += key_states.shape[-2]
        full_k = torch.cat([self.keys, key_states], dim=-2)
        full_v = torch.cat([self.values, value_states], dim=-2)
        if self.window is None:
            self.keys, self.values = full_k, full_v
        else:
            keep = self.window - 1
            self.keys = full_k[:, :, -keep:, :]
            self.values = full_v[:, :, -keep:, :]
        return full_k, full_v


def f32s(t):
    return [float(v) for v in t.detach().float().reshape(-1).tolist()]


def rust_array(name, values):
    body = ", ".join(f"{v:.9e}" for v in values)
    return f"pub static {name}: [f32; {len(values)}] = [{body}];\n"


def config(**kwargs):
    base = dict(
        hidden_size=16,
        sconv_kernel_size=3,
        rms_norm_eps=1e-5,
        qk_l2_clip_threshold=1.25,
        rel_l2_clip_threshold=0.9,
        log_scaling_alpha=0.0,
        log_scaling_n_floor=1.0,
        d_rel=4,
        sliding_window_size=4,
        rel_extent=4,
        head_dim=8,
        swa_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        attention_dropout=0.0,
        _attn_implementation="eager",
        layer_types=["hybrid_sliding"],
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def fill(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.empty_like(p).normal_(0.0, 0.35))


def relative_distance(past, q_len, window):
    if window is None:
        kv_offset = 0
        kv_length = past + q_len
    else:
        kv_offset = max(past - window + 1, 0)
        kv_length = window - 1 + q_len if past >= window else past + q_len
    q = torch.arange(q_len) + past
    kv = torch.arange(kv_length) + kv_offset
    return q[:, None] - kv[None, :]


def run_case(ns, cls, cfg, steps, window, layer_idx):
    torch.manual_seed(cfg.seed)
    module = cls(cfg, layer_idx)
    fill(module)
    cache = StepCache(window)
    hidden = cfg.hidden_size
    rows = []
    for past in range(steps):
        q = torch.randn(1, 1, hidden)
        k = q if window is not None else torch.randn(1, 1, hidden)
        v = q if window is not None else torch.randn(1, 1, hidden)
        dist = relative_distance(past, 1, window)
        pos = torch.tensor([[past]])
        y, _ = module(
            q,
            attention_mask=None,
            relative_distance=dist,
            cache=cache,
            key_hidden_states=k,
            value_hidden_states=v,
            position_ids=pos,
        )
        rows.append((f32s(q), f32s(k), f32s(v), f32s(y)))
    return module, rows


def dump_module(prefix, module, dynamic):
    parts = []
    p = module
    def arr(name, tensor):
        parts.append(rust_array(f"{prefix}_{name}", f32s(tensor)))

    arr("WQ", p.q_proj.linear.weight)
    arr("WK", p.k_proj.linear.weight)
    arr("WV", p.v_proj.linear.weight)
    arr("WO", p.o_proj.linear.weight)
    arr("WGATE", p.o_proj.gate_proj.weight)
    arr("WSINK", p.sink_proj.weight)
    arr("WREL", p.relative_position_bias.linear.weight)
    arr("REL_PROJ", p.relative_position_bias.proj)
    arr("Q_SCALE", p.q_proj.norm.scale)
    arr("REL_SCALE", p.relative_position_bias.norm.scale)
    if dynamic:
        arr("K_CONV", p.k_proj.sconv.static_weight)
        arr("V_CONV", p.v_proj.sconv.static_weight)
        arr("K_CONV_PROJ", p.k_proj.sconv.weight_proj.weight)
        arr("V_CONV_PROJ", p.v_proj.sconv.weight_proj.weight)
    else:
        arr("K_CONV", p.k_proj.sconv.conv1d.weight.squeeze(1))
        arr("V_CONV", p.v_proj.sconv.conv1d.weight.squeeze(1))
    return parts


def main():
    ns = load_attention(sys.argv[1])
    sliding_cfg = config(seed=11, layer_types=["hybrid_sliding"])
    wide_cfg = config(
        seed=19,
        hidden_size=8,
        head_dim=128,
        swa_head_dim=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        d_rel=4,
        sliding_window_size=4,
        rel_extent=4,
        sconv_kernel_size=2,
        layer_types=["hybrid_sliding"],
    )
    full_cfg = config(
        seed=23,
        layer_types=["hybrid"],
        log_scaling_alpha=0.3,
        log_scaling_n_floor=2.0,
        rel_extent=3,
        d_rel=4,
        sliding_window_size=8,
        head_dim=8,
        swa_head_dim=8,
    )
    cases = [
        ("SLIDING", ns["Inkling2SlidingAttention"], sliding_cfg, 8, 4, False),
        ("WIDE", ns["Inkling2SlidingAttention"], wide_cfg, 6, 4, False),
        ("FULL", ns["Inkling2FullAttention"], full_cfg, 6, None, True),
    ]
    parts = [
        f"// Generated by tests/gen_inkling_attention_goldens.py from transformers {PIN}.\n",
        "// Do not edit by hand. Each step is one forward of the pinned attention class.\n",
    ]
    for prefix, cls, cfg, steps, window, dynamic in cases:
        module, rows = run_case(ns, cls, cfg, steps, window, 0)
        parts.extend(dump_module(prefix, module, dynamic))
        for i, (q, k, v, y) in enumerate(rows):
            parts.append(rust_array(f"{prefix}_Q_{i}", q))
            parts.append(rust_array(f"{prefix}_K_{i}", k))
            parts.append(rust_array(f"{prefix}_V_{i}", v))
            parts.append(rust_array(f"{prefix}_Y_{i}", y))
        print(prefix, "steps", steps, "out0", rows[0][3][:4])
    out = "tests/inkling_attention_goldens.rs"
    open(out, "w").write("".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main()
