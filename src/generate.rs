//! Greedy decode over [`crate::AutoregressiveModel`].

use crate::model::AutoregressiveModel;
use crate::tokenizer::{sample_argmax, Tokenizer};
use anyhow::Result;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: u32,
    pub bos: bool,
    pub eos: bool,
    /// Stop when this token is chosen after the prompt (llama2.c uses BOS=1).
    pub stop_on: Option<i32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            bos: true,
            eos: false,
            stop_on: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateTokenOutput {
    pub tokens: Vec<i32>,
    pub prompt_tokens: usize,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens: Vec<i32>,
    pub prompt_tokens: usize,
    pub tokens_per_second: f64,
}

pub fn generate_tokens<M: AutoregressiveModel>(
    model: &mut M,
    prompt_tokens: &[i32],
    max_new_tokens: u32,
    stop_on: Option<i32>,
) -> Result<GenerateTokenOutput> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "expected at least one prompt token"
    );

    let max_new_tokens = max_new_tokens.min(model.max_seq_len());
    let mut tokens = Vec::new();
    let mut token = prompt_tokens[0] as u32;
    let mut pos = 0u32;
    let mut start = None;

    while pos < max_new_tokens.saturating_sub(1) {
        let logits = model.step(token, pos)?;
        let next = if (pos as usize) < prompt_tokens.len() - 1 {
            prompt_tokens[pos as usize + 1]
        } else {
            sample_argmax(&logits)
        };
        pos += 1;
        if stop_on == Some(next) {
            break;
        }
        tokens.push(next);
        token = next as u32;
        if start.is_none() {
            start = Some(Instant::now());
        }
    }

    let elapsed = start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
    let gen_tokens = pos.saturating_sub(1) as f64;
    let tokens_per_second = if elapsed > 0.0 {
        gen_tokens / elapsed
    } else {
        0.0
    };

    Ok(GenerateTokenOutput {
        tokens,
        prompt_tokens: prompt_tokens.len(),
        tokens_per_second,
    })
}

pub fn generate<M: AutoregressiveModel, T: Tokenizer>(
    model: &mut M,
    tokenizer: &T,
    prompt: &str,
    options: GenerateOptions,
) -> Result<GenerateOutput> {
    let prompt_tokens = tokenizer.encode(prompt, options.bos, options.eos);
    let out = generate_tokens(
        model,
        &prompt_tokens,
        options.max_new_tokens,
        options.stop_on,
    )?;
    let mut text = String::new();
    let mut prev = prompt_tokens[0];
    for &tok in &out.tokens {
        text.push_str(&tokenizer.decode(prev, tok));
        prev = tok;
    }
    Ok(GenerateOutput {
        text,
        tokens: out.tokens,
        prompt_tokens: out.prompt_tokens,
        tokens_per_second: out.tokens_per_second,
    })
}
