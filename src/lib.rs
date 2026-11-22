//! Inference operators and decode control on Goldy. Not a training runtime.

pub mod generate;
pub mod model;
pub mod tokenizer;

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod blocks;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod gpu;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod kernels;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod modules;

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub use modules::{
    CausalAttentionBlock, CausalAttentionBlockWeights, Embedding, KvCache, KvLayer, Linear,
    RmsNorm, SwiGluBlock, SwiGluBlockWeights, DEFAULT_RMS_EPS,
};

pub use generate::{
    generate, generate_tokens, GenerateOptions, GenerateOutput, GenerateTokenOutput,
};
pub use model::AutoregressiveModel;
pub use tokenizer::{sample_argmax, BpeTokenizer, Tokenizer};
