//! Inference operators and decode control on Goldy. Not a training runtime.

/// Configure a Goldy scheme for inference arithmetic.
///
/// This permits ordinary floating-point transformations in generated kernels and
/// their fused replacements. Call it before or after recording; exact validation
/// schemes can keep Goldy's default [`goldy::ArithmeticPolicy::MatchRecorded`].
pub fn configure_inference_scheme(scheme: &mut goldy::Scheme) {
    scheme.set_arithmetic_policy(goldy::ArithmeticPolicy::Relaxed);
}

pub mod checkpoint;
pub mod generate;
pub mod model;
pub mod tokenizer;

#[cfg(feature = "hf-tokenizer")]
mod hf_tokenizer;
#[cfg(feature = "hf-tokenizer")]
pub use hf_tokenizer::HfTokenizer;

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod blocks;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod decode_attn;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod decode_ops;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod depth;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod gpu;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod kernels;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod modules;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
mod moe;

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub use modules::{
    CausalAttentionBlock, CausalAttentionBlockWeights, DecodeAttention, DecodeAttentionKind,
    DecodeAttentionWeights, DecodeKvCache, DecodeLayerState, DecodeMoe, DepthMix, DynamicShortConv,
    Embedding, ExpertWeight, HeadL2Clip, KvCache, KvLayer, Linear, LogPositionScale, PackedExperts,
    RmsNorm, RouterWeight, ShortConv, SigmoidGate, Softcap, SoftcappedSwiGlu, SwiGluBlock,
    SwiGluBlockWeights, DEFAULT_RMS_EPS,
};

pub use generate::{
    generate, generate_tokens, greedy_decode, GenerateOptions, GenerateOutput, GenerateTokenOutput,
    GreedyReport,
};
pub use model::AutoregressiveModel;
pub use tokenizer::{sample_argmax, BpeTokenizer, Tokenizer};
