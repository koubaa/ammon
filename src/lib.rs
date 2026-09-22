//! Inference operators and decode control on Goldy. Not a training runtime.

pub mod generate;
pub mod model;
pub mod tokenizer;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod blocks;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod kernels;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub use blocks::{AttentionSites, Blocks, FfnSites};

pub use generate::{
    generate, generate_tokens, GenerateOptions, GenerateOutput, GenerateTokenOutput,
};
pub use model::AutoregressiveModel;
pub use tokenizer::{sample_argmax, BpeTokenizer, Tokenizer};
