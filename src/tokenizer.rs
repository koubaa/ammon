//! Generic tokenizer trait and llama2.c-compatible BPE.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const BOS_ID: i32 = 1;
const EOS_ID: i32 = 2;

/// String ↔ token ids. Architecture crates may wrap this for host compatibility.
pub trait Tokenizer {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i32>;
    fn decode(&self, prev_token: i32, token: i32) -> String;
}

/// SentencePiece-style BPE from llama2.c `tokenizer.bin`.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: u32,
    sorted: Vec<(String, i32)>,
}

impl BpeTokenizer {
    pub fn from_path(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            File::open(path).with_context(|| format!("open tokenizer {}", path.display()))?;
        let mut max_token_length = [0u8; 4];
        file.read_exact(&mut max_token_length)
            .context("read tokenizer max_token_length")?;
        let max_token_length = u32::from_le_bytes(max_token_length);

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let mut score_bytes = [0u8; 4];
            file.read_exact(&mut score_bytes)
                .with_context(|| format!("read tokenizer score {i}"))?;
            vocab_scores.push(f32::from_le_bytes(score_bytes));
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)
                .with_context(|| format!("read tokenizer length {i}"))?;
            let len = i32::from_le_bytes(len_bytes);
            if len < 0 {
                bail!("tokenizer piece {i} has negative length {len}");
            }
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf)
                .with_context(|| format!("read tokenizer piece {i}"))?;
            vocab.push(String::from_utf8_lossy(&buf).into_owned());
        }

        Ok(Self::from_parts(vocab, vocab_scores, max_token_length))
    }

    pub fn from_parts(vocab: Vec<String>, vocab_scores: Vec<f32>, max_token_length: u32) -> Self {
        assert_eq!(vocab.len(), vocab_scores.len());
        let mut sorted: Vec<(String, i32)> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as i32))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            vocab,
            vocab_scores,
            max_token_length,
            sorted,
        }
    }

    pub fn max_token_length(&self) -> u32 {
        self.max_token_length
    }

    pub fn bos_id() -> i32 {
        BOS_ID
    }

    pub fn eos_id() -> i32 {
        EOS_ID
    }

    fn lookup(&self, s: &str) -> Option<i32> {
        self.sorted
            .binary_search_by(|(k, _)| k.as_str().cmp(s))
            .ok()
            .map(|i| self.sorted[i].1)
    }
}

impl Tokenizer for BpeTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i32> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(BOS_ID);
        }
        if !text.is_empty() {
            if let Some(dummy) = self.lookup(" ") {
                tokens.push(dummy);
            }
        }

        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 && i - start < 4 {
                i += 1;
            }
            let piece = &bytes[start..i];
            if let Ok(s) = std::str::from_utf8(piece) {
                if let Some(id) = self.lookup(s) {
                    tokens.push(id);
                    continue;
                }
            }
            for &b in piece {
                tokens.push(i32::from(b) + 3);
            }
        }

        loop {
            let mut best_score = -1e10f32;
            let mut best_id = -1;
            let mut best_idx = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = format!(
                    "{}{}",
                    self.vocab[tokens[i] as usize],
                    self.vocab[tokens[i + 1] as usize]
                );
                if let Some(id) = self.lookup(&merged) {
                    let score = self.vocab_scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_id = id;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else { break };
            tokens[idx] = best_id;
            tokens.remove(idx + 1);
        }

        if eos {
            tokens.push(EOS_ID);
        }
        tokens
    }

    fn decode(&self, prev_token: i32, token: i32) -> String {
        let mut piece = self.vocab.get(token as usize).cloned().unwrap_or_default();
        if prev_token == BOS_ID && piece.starts_with(' ') {
            piece.remove(0);
        }
        if let Some(byte) = parse_byte_token(&piece) {
            return (byte as char).to_string();
        }
        piece
    }
}

fn parse_byte_token(piece: &str) -> Option<u8> {
    let rest = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(rest, 16).ok()
}

pub fn sample_argmax(logits: &[f32]) -> i32 {
    let mut max_i = 0;
    let mut max_p = logits[0];
    for (i, &p) in logits.iter().enumerate().skip(1) {
        if p > max_p {
            max_i = i;
            max_p = p;
        }
    }
    max_i as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> BpeTokenizer {
        let vocab = vec![
            "<unk>".into(),
            "<s>".into(),
            "</s>".into(),
            "a".into(),
            "b".into(),
            "ab".into(),
            " ".into(),
        ];
        let scores = vec![0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0];
        BpeTokenizer::from_parts(vocab, scores, 8)
    }

    #[test]
    fn encodes_with_bos_dummy_prefix_and_merge() {
        let tok = toy_tokenizer();
        let ids = tok.encode("ab", true, false);
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 6, "dummy prefix is the space token");
        assert_eq!(ids[2], 5, "a+b should merge to ab");
    }

    #[test]
    fn decode_strips_space_after_bos() {
        let tok = toy_tokenizer();
        assert_eq!(tok.decode(1, 6), "");
        assert_eq!(tok.decode(0, 6), " ");
    }

    #[test]
    fn argmax_picks_first_on_ties() {
        assert_eq!(sample_argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn byte_token_decode() {
        let vocab = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "<0x41>".into()];
        let tok = BpeTokenizer::from_parts(vocab, vec![0.0; 4], 8);
        assert_eq!(tok.decode(0, 3), "A");
    }
}
