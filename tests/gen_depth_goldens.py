#!/usr/bin/env python3
"""Compact depth-mix goldens from the pinned loop attention modules.

One binary, not a Rust array dump:

    python3 tests/gen_depth_goldens.py /path/to/modeling_inkling2_loop.py
"""

import struct
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

PIN = "eeee11099135592f1d03c2871fbd3d8132e5a4d7"


def load(path):
    lines = open(path).read().splitlines()
    ranges = [(67, 76), (99, 142), (145, 174), (828, 854)]
    chunk = []
    for start, end in ranges:
        for line in lines[start - 1 : end]:
            if line.lstrip().startswith("@"):
                continue
            chunk.append(line)
    ns = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "Callable": object,
        "Inkling2LoopTextConfig": object,
    }
    exec("\n".join(chunk), ns)

    class Attn:
        @staticmethod
        def get_interface(_name, default):
            return default

    ns["ALL_ATTENTION_FUNCTIONS"] = Attn
    return ns


class Plan:
    def __init__(self, layers, block, loop_start, loop_end, passes):
        self.layers = layers
        self.block = block
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.passes = passes

    def n_loop(self):
        return self.loop_end - self.loop_start

    def visits(self):
        pre = self.loop_start
        post = self.layers - self.loop_end
        return pre + self.passes * self.n_loop() + post

    def is_full(self, layer):
        return (layer + 1) % self.block == 0

    def is_boundary(self, layer):
        return layer % self.block == 0

    def n_full(self):
        return sum(self.is_full(i) for i in range(self.layers))

    def full_index(self, layer):
        return sum(self.is_full(i) for i in range(layer))

    def visit(self, layer, pass_):
        n_loop = self.n_loop()
        if layer < self.loop_start:
            return layer
        if layer < self.loop_end:
            return self.loop_start + pass_ * n_loop + (layer - self.loop_start)
        return self.loop_start + self.passes * n_loop + (layer - self.loop_end)

    def message_visits(self):
        return max(self.passes - 1, 0) * self.n_loop()


def mix(mod, blocks, partial, query):
    h = partial.shape[-1]
    if blocks.numel() == 0:
        blocks = blocks.reshape(0, h)
    with torch.no_grad():
        mod.query.copy_(query.reshape(mod.query.shape))
    out = mod(blocks.reshape(1, 1, -1, h), partial.reshape(1, 1, h))
    return torch.stack([t.reshape(h) for t in out], 0)


def message(mod, blocks, partial, query_row, layer_idx):
    h = partial.shape[-1]
    if blocks.numel() == 0:
        blocks = blocks.reshape(0, h)
    with torch.no_grad():
        mod.query[layer_idx].copy_(query_row)
    out = mod(blocks.reshape(1, 1, -1, h), partial.reshape(1, 1, h), layer_idx)
    return out.reshape(4, h)


def run_token(ns, plan, hidden, token, weights):
    cfg = SimpleNamespace(
        hidden_size=hidden,
        _attn_implementation="eager",
        loop_layer_index_start=plan.loop_start,
        loop_layer_index_end=plan.loop_end,
    )
    one = ns["Inkling2LoopBlockAttentionResidual"](cfg, 1)
    three = ns["Inkling2LoopBlockAttentionResidual"](cfg, 3)
    msg = ns["Inkling2LoopMessageAttentionResidual"](cfg)
    q_single, q_full, q_mlp, q_msg, q_final = weights
    entries, mlp_partial, pass_exit, final_partial = token
    visits = plan.visits()
    attn = torch.zeros(visits, 3, hidden)
    mlp = torch.zeros(visits, hidden)
    n_msg = plan.message_visits()
    messages = torch.zeros(max(n_msg, 1), 4, hidden)
    tape = torch.zeros(0, hidden)
    length = 0
    snap = None
    snap_len = 0

    def layer(pass_, layer_i):
        nonlocal tape, length
        visit = plan.visit(layer_i, pass_)
        restarted = pass_ > 0 and layer_i == plan.loop_start
        entry_visit = plan.visit(plan.loop_start, 0) if restarted else visit
        entry = entries[entry_visit]
        if layer_i > 0:
            if plan.is_full(layer_i):
                q = q_full[plan.full_index(layer_i)]
                y = mix(three, tape[:length], entry, q)
                attn[visit, :3] = y
            else:
                y = mix(one, tape[:length], entry, q_single[layer_i])
                attn[visit, :1] = y
        if plan.is_boundary(layer_i):
            tape = torch.cat([tape[:length], entry.reshape(1, hidden)], 0)
            length += 1
        y = mix(one, tape[:length], mlp_partial[visit], q_mlp[layer_i])
        mlp[visit] = y[0]

    for layer_i in range(plan.loop_start):
        layer(0, layer_i)
    snap = tape[:length].clone()
    snap_len = length
    for pass_ in range(plan.passes):
        if pass_ > 0:
            msg_n = max(length - (snap_len + 1), 0)
            msg_blocks = tape[snap_len + 1 : snap_len + 1 + msg_n]
            tape = snap.clone()
            length = snap_len
            prev = pass_exit[pass_ - 1]
            for loop_layer in range(plan.n_loop()):
                y = message(msg, msg_blocks, prev, q_msg[loop_layer], loop_layer)
                messages[(pass_ - 1) * plan.n_loop() + loop_layer] = y
        for layer_i in range(plan.loop_start, plan.loop_end):
            layer(pass_, layer_i)
    for layer_i in range(plan.loop_end, plan.layers):
        layer(0, layer_i)
    final = mix(one, tape[:length], final_partial, q_final)[0]
    return attn, mlp, messages[:n_msg], final, tape[:length]


def pack_f32(buf, tensor):
    data = tensor.detach().float().reshape(-1).numpy().astype("float32").tobytes()
    buf.append(struct.pack("<I", len(data) // 4))
    buf.append(data)


def main():
    ns = load(sys.argv[1])
    plans = [
        Plan(8, 4, 4, 8, 2),
        Plan(24, 6, 6, 18, 3),
    ]
    blob = [struct.pack("<II", 0x48545044, 1), struct.pack("<I", len(plans))]
    for plan in plans:
        torch.manual_seed(plan.layers * 100 + plan.passes)
        hidden = 8
        n_tokens = 2
        visits = plan.visits()
        q_single = torch.randn(plan.layers, hidden)
        q_full = torch.randn(plan.n_full(), 3, hidden)
        q_mlp = torch.randn(plan.layers, hidden)
        q_msg = torch.randn(plan.n_loop(), 4, hidden)
        q_final = torch.randn(hidden)
        weights = (q_single, q_full, q_mlp, q_msg, q_final)
        tokens = []
        outs = []
        for _ in range(n_tokens):
            token = (
                torch.randn(visits, hidden),
                torch.randn(visits, hidden),
                torch.randn(plan.passes, hidden),
                torch.randn(hidden),
            )
            tokens.append(token)
            outs.append(run_token(ns, plan, hidden, token, weights))
        header = struct.pack(
            "<IIIIIII",
            plan.layers,
            plan.block,
            plan.loop_start,
            plan.loop_end,
            plan.passes,
            hidden,
            n_tokens,
        )
        blob.append(header)
        for tensor in weights:
            pack_f32(blob, tensor)
        for token, out in zip(tokens, outs):
            for tensor in token:
                pack_f32(blob, tensor)
            for tensor in out:
                pack_f32(blob, tensor)
        print(
            "plan",
            plan.layers,
            "visits",
            visits,
            "tape",
            tuple(outs[0][4].shape),
            "final",
            tuple(outs[0][3].shape),
        )
    open("tests/data/depth_goldens.bin", "wb").write(b"".join(blob))
    print("wrote tests/data/depth_goldens.bin", sum(len(p) for p in blob))


if __name__ == "__main__":
    main()
