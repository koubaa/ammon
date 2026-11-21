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

On macOS, use `--features metal` in place of `cuda`. On Linux, `--features vulkan` runs the same checks on Goldy's Vulkan backend (Vulkan 1.4+).

Headless machines can use Mesa lavapipe, the llvmpipe software Vulkan driver. Mesa 25 or newer advertises Vulkan 1.4, which Goldy requires:

```bash
sudo apt-get install mesa-vulkan-drivers vulkan-tools
# Pin the ICD when other Vulkan drivers are installed. DISPLAY may be unset.
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
vulkaninfo --summary   # deviceName llvmpipe, apiVersion 1.4.*
cargo test --features vulkan --lib --test kernels --test semantic_norm
```

Kernel and block results are compared to an f64 CPU reference at atol 1e-5 and rtol 1e-4. `semantic_norm` turns on `AMMON_SEMANTIC_NORM=1` with automatic fusion so the tensor-op RMSNorm path is what runs inside the pre-norm blocks. The first run compiles shaders through Slang; later runs reuse Goldy's on-disk shader cache. Lavapipe is CPU-bound, so `-- --test-threads=1` is the reliable way to run the suite.
