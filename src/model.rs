//! Decode-step model contract. Architecture crates supply the graph.

use anyhow::Result;

/// One-token-at-a-time generator. `step` returns a borrowed or owned view of
/// vocabulary logits for `(token, pos)`.
pub trait AutoregressiveModel {
    fn vocab_size(&self) -> usize;
    fn max_seq_len(&self) -> u32;
    fn step(&mut self, token: u32, pos: u32) -> Result<goldy::HostView<f32>>;
}
