//! Goldy device bootstrap for CUDA, Metal, and Vulkan builds.

use anyhow::{Context, Result};
use goldy::{ComputePartitionSplit, Instance, RequestAdapterOptions, Runtime, RuntimeDescriptor};

pub fn create_runtime() -> Result<Runtime> {
    runtime_with(ComputePartitionSplit::ENCODE_OVERLAP_DEFAULT)
}

/// One command buffer per scheme.
///
/// A serial token cannot overlap its own partitions: the next chunk reads the
/// hidden state the previous chunk writes. One buffer also stays inside the
/// retained frame-table rows. `GOLDY_PARTITION_SPLIT` still overrides this.
pub fn create_serial_runtime() -> Result<Runtime> {
    runtime_with(ComputePartitionSplit::None)
}

fn runtime_with(split: ComputePartitionSplit) -> Result<Runtime> {
    let instance = Instance::new().context("create Goldy instance")?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .context("request GPU adapter")?
        .request_runtime(&RuntimeDescriptor {
            compute_partition_split: Some(split),
            ..RuntimeDescriptor::default()
        })
        .context("request Goldy runtime")
}
