//! Goldy device bootstrap for CUDA / Metal builds.

use anyhow::{Context, Result};
use goldy::{ComputePartitionSplit, Instance, RequestAdapterOptions, Runtime, RuntimeDescriptor};

pub fn create_runtime() -> Result<Runtime> {
    let instance = Instance::new().context("create Goldy instance")?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .context("request GPU adapter")?
        .request_runtime(&RuntimeDescriptor {
            // Each token waits on the previous one, so commit a short head while the CPU
            // encodes the rest. `GOLDY_PARTITION_SPLIT` overrides this.
            compute_partition_split: Some(ComputePartitionSplit::ENCODE_OVERLAP_DEFAULT),
            ..RuntimeDescriptor::default()
        })
        .context("request Goldy runtime")
}
