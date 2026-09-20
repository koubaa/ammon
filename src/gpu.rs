//! Goldy device bootstrap for CUDA / Metal builds.

use anyhow::{Context, Result};
use goldy::{Instance, RequestAdapterOptions, Runtime, RuntimeDescriptor};

pub fn create_runtime() -> Result<Runtime> {
    let instance = Instance::new().context("create Goldy instance")?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .context("request GPU adapter")?
        .request_runtime(&RuntimeDescriptor::default())
        .context("request Goldy runtime")
}
