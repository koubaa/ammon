//! Inference operators and decode control on Goldy. Not a training runtime.

/// Process-wide override for inference arithmetic.
///
/// Accepted values are `relaxed` (the default), `match-recorded`, and `strict`.
pub const ARITHMETIC_POLICY_ENV: &str = "AMMON_ARITHMETIC_POLICY";

/// Arithmetic policy selected for newly recorded inference schemes.
///
/// The default is [`goldy::ArithmeticPolicy::Relaxed`]. Set
/// [`ARITHMETIC_POLICY_ENV`] to `match-recorded` or `strict` before recording
/// to preserve every rounding boundary represented by the recorded dispatches.
pub fn inference_arithmetic_policy() -> Result<goldy::ArithmeticPolicy, String> {
    let value = match std::env::var(ARITHMETIC_POLICY_ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => String::new(),
        Err(std::env::VarError::NotUnicode(value)) => {
            return Err(format!("{ARITHMETIC_POLICY_ENV} is not Unicode: {value:?}"));
        }
    };
    parse_inference_arithmetic_policy(&value)
}

fn parse_inference_arithmetic_policy(value: &str) -> Result<goldy::ArithmeticPolicy, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "relaxed" => Ok(goldy::ArithmeticPolicy::Relaxed),
        "match-recorded" | "match_recorded" | "strict" => {
            Ok(goldy::ArithmeticPolicy::MatchRecorded)
        }
        value => Err(format!(
            "invalid {ARITHMETIC_POLICY_ENV}={value:?}; expected relaxed, match-recorded, or strict"
        )),
    }
}

/// Configure a Goldy scheme for inference arithmetic.
///
/// This permits ordinary floating-point transformations in generated kernels and
/// their fused replacements. Call it before or after recording; exact validation
/// schemes can keep Goldy's default [`goldy::ArithmeticPolicy::MatchRecorded`].
pub fn configure_inference_scheme(scheme: &mut goldy::Scheme) {
    let policy = inference_arithmetic_policy().unwrap_or_else(|error| panic!("{error}"));
    scheme.set_arithmetic_policy(policy);
}

#[cfg(test)]
mod arithmetic_policy_tests {
    use super::parse_inference_arithmetic_policy;
    use goldy::ArithmeticPolicy;

    #[test]
    fn parses_inference_arithmetic_policy() {
        assert_eq!(
            parse_inference_arithmetic_policy("").unwrap(),
            ArithmeticPolicy::Relaxed
        );
        assert_eq!(
            parse_inference_arithmetic_policy(" relaxed ").unwrap(),
            ArithmeticPolicy::Relaxed
        );
        assert_eq!(
            parse_inference_arithmetic_policy("match-recorded").unwrap(),
            ArithmeticPolicy::MatchRecorded
        );
        assert_eq!(
            parse_inference_arithmetic_policy("STRICT").unwrap(),
            ArithmeticPolicy::MatchRecorded
        );
        assert!(parse_inference_arithmetic_policy("fast").is_err());
    }
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
    DecodeAttentionWeights, DecodeKvCache, DecodeLayerState, DecodeLayerStateSpec, DecodeMoe,
    DepthMix, DynamicShortConv, Embedding, ExpertWeight, HeadL2Clip, KvCache, KvLayer, Linear,
    LogPositionScale, PackedExperts, RmsNorm, RouterWeight, ShortConv, SigmoidGate, Softcap,
    SoftcappedSwiGlu, SwiGluBlock, SwiGluBlockWeights, DEFAULT_RMS_EPS, DEPTH_RMS_EPS,
};

pub use generate::{
    generate, generate_tokens, greedy_decode, GenerateOptions, GenerateOutput, GenerateTokenOutput,
    GreedyReport,
};
pub use model::{AutoregressiveModel, LogitsView};
pub use tokenizer::{sample_argmax, BpeTokenizer, Tokenizer};
