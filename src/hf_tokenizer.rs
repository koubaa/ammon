//! Hugging Face `tokenizer.json` wrapper. The file and its checksum stay with
//! the model crate. This type only speaks the generic [`Tokenizer`] contract
//! plus the fast-tokenizer encode/decode the file already defines.

use super::Tokenizer;
use anyhow::{Context, Result};
use std::path::Path;

/// A `tokenizers` tokenizer loaded from `tokenizer.json`.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("load tokenizer {}", path.display()))?;
        Ok(Self { inner })
    }

    /// Token ids for `text`, without adding special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("encode: {err}"))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|err| anyhow::anyhow!("decode: {err}"))
    }
}

impl Tokenizer for HfTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i32> {
        let mut ids = HfTokenizer::encode(self, text)
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        if bos {
            if let Some(id) = self
                .inner
                .token_to_id("<s>")
                .or_else(|| self.inner.token_to_id("<|begin_of_text|>"))
            {
                ids.insert(0, id as i32);
            }
        }
        if eos {
            if let Some(id) = self
                .inner
                .token_to_id("</s>")
                .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
            {
                ids.push(id as i32);
            }
        }
        ids
    }

    fn decode(&self, _prev_token: i32, token: i32) -> String {
        HfTokenizer::decode(self, &[token as u32]).unwrap_or_default()
    }
}
