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

#[derive(Debug, Clone)]
pub struct GreedyReport {
    pub new_tokens: Vec<u32>,
    /// Wall time of the prompt forwards.
    pub prefill_secs: f64,
    /// Forwards after the prompt. The first new token comes from the last
    /// prompt logits, so this covers `new_tokens.len().saturating_sub(1)` steps.
    pub decode_secs: f64,
    pub decode_steps: u32,
    /// One entry per forward, prompt first. The first entry includes compile.
    pub step_secs: Vec<f64>,
    /// `top1 - top2` for each new token. A small margin means the token is not
    /// evidence by itself; compare logits.
    pub margins: Vec<f32>,
    /// Logits of the last prompt position, the distribution the first new
    /// token was drawn from.
    pub last_logits: Vec<f32>,
}

fn logit_margin(logits: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &value in logits {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    if second == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        best - second
    }
}

/// Teacher-force `prompt`, then append `max_new` greedy tokens.
///
/// The model view is dropped before the next step so the device buffer can
/// be written again.
pub fn greedy_decode<M: AutoregressiveModel>(
    model: &mut M,
    prompt: &[u32],
    max_new: u32,
) -> Result<GreedyReport> {
    anyhow::ensure!(!prompt.is_empty(), "greedy decode needs a prompt token");
    let mut step_secs = Vec::with_capacity(prompt.len() + max_new as usize);
    let started = Instant::now();
    let step_started = Instant::now();
    let mut logits = model.step(prompt[0], 0)?;
    step_secs.push(step_started.elapsed().as_secs_f64());
    for (index, token) in prompt.iter().enumerate().skip(1) {
        drop(logits);
        let step_started = Instant::now();
        logits = model.step(*token, index as u32)?;
        step_secs.push(step_started.elapsed().as_secs_f64());
    }
    let prefill_secs = started.elapsed().as_secs_f64();
    let last_logits = logits.to_vec();
    let mut new_tokens = Vec::with_capacity(max_new as usize);
    let mut margins = Vec::with_capacity(max_new as usize);
    let mut pos = prompt.len() as u32;
    let decode_started = Instant::now();
    let mut decode_steps = 0u32;
    for step in 0..max_new {
        let next = sample_argmax(&logits);
        margins.push(logit_margin(&logits));
        drop(logits);
        anyhow::ensure!(next >= 0, "argmax returned {next}");
        new_tokens.push(next as u32);
        if step + 1 == max_new {
            break;
        }
        let step_started = Instant::now();
        logits = model.step(next as u32, pos)?;
        step_secs.push(step_started.elapsed().as_secs_f64());
        pos += 1;
        decode_steps += 1;
    }
    Ok(GreedyReport {
        new_tokens,
        prefill_secs,
        decode_secs: decode_started.elapsed().as_secs_f64(),
        decode_steps,
        step_secs,
        margins,
        last_logits,
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
