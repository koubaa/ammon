//! Decode-step model contract. Architecture crates supply the graph.

use anyhow::Result;
use std::ops::Deref;

/// Borrowed mapped [`goldy::HostView`] or owned `Vec<f32>` of vocabulary logits for one step.
///
/// Dropping a [`Self::Host`] view releases its [`goldy::HostSink`] claim so the next decode
/// submission may write into the sink again without copying the vocabulary vector onto the heap.
pub enum LogitsView {
    Host(goldy::HostView<f32>),
    Owned(Vec<f32>),
}

impl LogitsView {
    pub fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Host(view) => view.to_vec(),
            Self::Owned(vec) => vec,
        }
    }
}

impl Deref for LogitsView {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match self {
            Self::Host(view) => view,
            Self::Owned(vec) => vec,
        }
    }
}

impl AsRef<[f32]> for LogitsView {
    fn as_ref(&self) -> &[f32] {
        self
    }
}

/// One-token-at-a-time generator.
pub trait AutoregressiveModel {
    fn vocab_size(&self) -> usize;
    fn max_seq_len(&self) -> u32;

    /// Logits for `token` at absolute position `pos` as an owned vector.
    fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>>;

    /// Logits for `token` at absolute position `pos` without forcing a heap copy when backed by
    /// a mapped [`goldy::HostSink`].
    ///
    /// Callers must drop the returned [`LogitsView`] before invoking the next [`Self::step`],
    /// [`Self::step_view`], or [`Self::advance`].
    fn step_view(&mut self, token: u32, pos: u32) -> Result<LogitsView> {
        Ok(LogitsView::Owned(self.step(token, pos)?))
    }

    /// Advance persistent model state for `(token, pos)` when the caller does not consume logits
    /// (for example, all prompt tokens before the final prefill position).
    ///
    /// Implementations backed by a GPU [`goldy::Scheme`] can submit the step without claiming or
    /// blocking on the vocabulary [`goldy::HostSink`], pipelining prompt steps on the device.
    fn advance(&mut self, token: u32, pos: u32) -> Result<()> {
        let _ = self.step_view(token, pos)?;
        Ok(())
    }
}
