# Ammon

Inference operators, a generic tokenizer, and a greedy decode loop on [Goldy](https://github.com/koubaa/goldy) tensors.

Ammon is not a training framework and not a specific model. It sits between Goldy (parcels, schemes, tensors) and architecture crates such as `llama3.goldy`.

## What it owns

- GPU kernels: embedding, RMSNorm, RoPE, attention, SwiGLU, pos-strided cache GEMV, per-head L2 clip, softcap, sigmoid gate, softcapped SwiGLU, one-token causal depthwise convolution
- Recordable modules that own decode scratch where needed (`CausalAttentionBlock`, `SwiGluBlock`, `KvCache`, `ShortConv`, `DynamicShortConv`) and kernel-only modules (`Embedding`, `RmsNorm`, `Linear`, `HeadL2Clip`, `Softcap`, `SigmoidGate`, `SoftcappedSwiGlu`)
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

## API notes

`RmsnormKernel::record` and `RmsnormInplaceKernel::record` take a trailing `eps: f32`. That is a source break for anyone calling those kernels directly. `RmsNorm::new`, `RmsNorm::record`, `RmsNorm::record_inplace`, and the pre-norm block `new` constructors are unchanged and still use `DEFAULT_RMS_EPS` (`1e-5`). Pass another epsilon with `with_eps`.

## CI goldy pin

`.github/workflows/ci.yml` sets `GOLDY_REF` (currently `19a383f30e8571800d83c23a5df30d65274c0b81`). Ammon depends on Goldy by path (`../goldy`), so CI checks that commit out beside this repo. To move the pin, change only `GOLDY_REF` and push. This pin is the merge of [koubaa/goldy#332](https://github.com/koubaa/goldy/pull/332) into goldy `main`: `goldy::gpu::tanh` and `goldy::gpu::sigmoid` use the odd Taylor series through `x^11` for `|x| < 0.5`.

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
cargo test --features vulkan --lib --test kernels --test semantic_norm --test decode_ops
```

Kernel and block results are compared to an f64 CPU reference at atol 1e-5 and rtol 1e-4. `semantic_norm` turns on `AMMON_SEMANTIC_NORM=1` with automatic fusion so the tensor-op RMSNorm path is what runs inside the pre-norm blocks. `decode_ops` checks per-head L2 clip (optional scale and log-position scale), softcap, the sigmoid gate, softcapped SwiGLU, and static and dynamic short convolutions including reset. The first run compiles shaders through Slang; later runs reuse Goldy's on-disk shader cache. Lavapipe is CPU-bound, so `-- --test-threads=1` is the reliable way to run the suite.
