# Ammon

Inference operators, a generic tokenizer, and a greedy decode loop on [Goldy](https://github.com/koubaa/goldy) tensors.

Ammon is not a training framework and not a specific model. It sits between Goldy (parcels, schemes, tensors) and architecture crates such as `llama3.goldy`.

## What it owns

- GPU kernels: embedding, RMSNorm, RoPE, attention, SwiGLU, pos-strided cache GEMV
- Recordable modules that own decode scratch where needed (`CausalAttentionBlock`, `SwiGluBlock`, `KvCache`) and kernel-only modules (`Embedding`, `RmsNorm`, `Linear`)
- Weight view structs (`CausalAttentionBlockWeights`, `SwiGluBlockWeights`); packed checkpoints stay in architecture crates
- Prepared kernels cached per Goldy runtime (not part of the architecture-crate API)
- Re-exported Goldy tensor kernels: residual `add` and semantic `matmul` (`TensorKernels`)
- `DecodeStep { token, position }` control parcel (`DecodeStep::parcel`, `DecodeStep::deposit_target`)
- `AutoregressiveModel` / `Tokenizer` traits
- llama2.c-compatible BPE (`BpeTokenizer`)
- Greedy `generate_tokens` / `generate`

## What it does not own

- Packed checkpoint layouts
- The submitting worker scheme or its memory exchange
- Host compatibility patches (dream-prompt rewrite, CJK `safe_printf`)

## Tests

```bash
cargo test --offline
cargo test --features cuda --test kernels
```

On macOS, use `--features metal` in place of `cuda`.
