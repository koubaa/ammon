//! Goldy device bootstrap for CUDA, Metal, and Vulkan builds.

use anyhow::{Context, Result};
use goldy::{ComputePartitionSplit, Instance, RequestAdapterOptions, Runtime, RuntimeDescriptor};

pub fn create_runtime() -> Result<Runtime> {
    let instance = Instance::new().context("create Goldy instance")?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .context("request GPU adapter")?
        .request_runtime(&RuntimeDescriptor {
            // A serial token is one command buffer. Overlapping partitions race
            // on the hidden state, and one buffer stays inside the retained-row
            // budget. `GOLDY_PARTITION_SPLIT` overrides this.
            compute_partition_split: Some(ComputePartitionSplit::None),
            ..RuntimeDescriptor::default()
        })
        .context("request Goldy runtime")
}
