# Ammon

Inference operators, a generic tokenizer, and a greedy decode loop on [Goldy](https://github.com/koubaa/goldy) tensors.

Ammon is not a training framework and not a specific model. It sits between Goldy (parcels, schemes, tensors) and architecture crates such as `llama3.goldy`.

## What it owns

- GPU kernels: embedding, RMSNorm, RoPE, attention, SwiGLU, pos-strided cache GEMV
- Re-exported Goldy tensor kernels: residual `add` and semantic `matmul` (`TensorKernels`)
- `DecodeStep { token, position }` control parcel
- `AutoregressiveModel` / `Tokenizer` traits
- llama2.c-compatible BPE (`BpeTokenizer`)
- Greedy `generate_tokens` / `generate`

## What it does not own

- Packed checkpoint layouts
- Llama layer graphs
- Host compatibility patches (dream-prompt rewrite, CJK `safe_printf`)

## Tests

```bash
cargo test --offline
cargo test --features cuda --test kernels
```

On macOS, use `--features metal` in place of `cuda`.
