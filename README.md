# Ammon

Inference operators, a generic tokenizer, and a greedy decode loop on [Goldy](https://github.com/koubaa/goldy) tensors.

Ammon is not a training framework and not a specific model. It sits between Goldy (parcels, schemes, tensors) and architecture crates such as `llama3.goldy`.

## What it owns

- GPU kernels: embedding, RMSNorm, RoPE, attention, SwiGLU, pos-strided cache GEMV, per-head L2 clip, softcap, sigmoid gate, softcapped SwiGLU, one-token causal depthwise convolution, ring or full KV cache, relative-bias decode attention, routed experts (top-k on device), depth mix over the block tape
- Recordable modules that own the scratch their dispatches bind (`CausalAttentionBlock`, `SwiGluBlock`, `KvCache`, `RmsNorm`, `Softcap`, `SigmoidGate`, `SoftcappedSwiGlu`, `ShortConv`, `DynamicShortConv`, `DecodeAttention`, `DecodeLayerState`, `DecodeMoe`, `DepthRunner`) and kernel-only modules (`Embedding`, `Linear`, `HeadL2Clip`, `DepthMix`)
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

Ammon inference accepts normal floating-point noise. Configure each retained worker
scheme with `ammon::configure_inference_scheme(&mut scheme)` to select Goldy's relaxed
generated-kernel and fusion variants. Validation or reproducibility checks can leave the
default `ArithmeticPolicy::MatchRecorded` in place.

`RmsnormKernel::record` and `RmsnormInplaceKernel::record` take a trailing `eps: f32`. That is a source break for anyone calling those kernels directly. `RmsNorm::new` and the pre-norm block `new` constructors use `DEFAULT_RMS_EPS` (`1e-5`). Pass another epsilon with `with_eps`.

A module owns every buffer its recorded dispatches bind, allocated once when it is built, and must outlive the schemes it records into. Schemes do not keep scratch alive: dropping a module and submitting a scheme it recorded into fails with `StaleResource`. Modules whose scratch depends on the input width take it at construction: `RmsNorm::new(runtime, dim)`, `RmsNorm::with_eps(runtime, dim, eps)`, `Softcap::new(runtime, len, cap)`, `SigmoidGate::new(runtime, len)`, and `SoftcappedSwiGlu::new(runtime, len, gate_cap, up_cap)`. Each record on one module binds the same scratch, so give sites that should not share it their own module.

## CI goldy pin

`.github/workflows/ci.yml` sets `GOLDY_REF` to goldy `main`, or to a dependency branch while a goldy pull request is open. Ammon depends on Goldy by path (`../goldy`), so CI checks that branch out beside this repo. `goldy::gpu::tanh` and `goldy::gpu::sigmoid` still use the odd Taylor series through `x^11` for `|x| < 0.5`.

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
cargo test --features vulkan --lib --test kernels --test fused_blocks --test decode_ops
```

Kernel and block results are compared to an f64 CPU reference at atol 1e-5 and rtol 1e-4. `fused_blocks` runs the pre-norm blocks under automatic fusion against the reference, and checks that the fused SwiGLU block stores the same bits as the unfused one under `MatchRecorded`. `decode_ops` checks per-head L2 clip (optional scale and log-position scale), softcap, the sigmoid gate, softcapped SwiGLU, and static and dynamic short convolutions including reset. Softcap, the gate, and softcapped SwiGLU call `goldy::gpu::tanh` and `goldy::gpu::sigmoid`. The first run compiles shaders through Slang; later runs reuse Goldy's on-disk shader cache. Lavapipe is CPU-bound, so `-- --test-threads=1` is the reliable way to run the suite.
