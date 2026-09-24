//! Prepared kernels shared per [`goldy::Runtime`].
//!
//! Modules record dispatches into a [`goldy::Scheme`]. Exchanges stay on the
//! submitting root; these functions only bind retained parcels.

use crate::kernels::{
    AttentionKernel, CacheStoreKernel, EmbedKernel, RmsnormInplaceKernel, RmsnormKernel,
    RopeStoreKernel, SwigluKernel, TensorKernels, DEFAULT_ROPE_THETA,
};
use anyhow::Context;
use goldy::{Buffer, GoldyError, Runtime, Scheme, TensorView};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Prepared kernels for one runtime. Used while recording; schemes intern the pipelines.
pub(crate) struct Blocks {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    rope_store: RopeStoreKernel,
    cache_store: CacheStoreKernel,
    attention: AttentionKernel,
    swiglu: SwigluKernel,
    tensors: TensorKernels,
}

/// Activation and weight views for one attention block.
pub(crate) struct AttentionSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub xb2: TensorView<'a>,
    pub q: TensorView<'a>,
    pub att: TensorView<'a>,
    pub key: TensorView<'a>,
    pub value: TensorView<'a>,
    pub step: &'a Buffer,
    pub rms: TensorView<'a>,
    pub wq: TensorView<'a>,
    pub wk: TensorView<'a>,
    pub wv: TensorView<'a>,
    pub wo: TensorView<'a>,
}

/// Activation and weight views for one SwiGLU feed-forward block.
pub(crate) struct FfnSites<'a> {
    pub x: TensorView<'a>,
    pub xb: TensorView<'a>,
    pub hb: TensorView<'a>,
    pub hb2: TensorView<'a>,
    pub rms: TensorView<'a>,
    pub w1: TensorView<'a>,
    pub w2: TensorView<'a>,
    pub w3: TensorView<'a>,
}

impl Blocks {
    pub(crate) fn shared(runtime: &Runtime) -> anyhow::Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let mut cache = cache.lock().expect("blocks cache");
        let key = runtime.substrate_ptr() as usize;
        cache.retain(|(_, weak)| weak.strong_count() > 0);
        if let Some((_, weak)) = cache.iter().find(|(k, _)| *k == key) {
            if let Some(blocks) = weak.upgrade() {
                return Ok(blocks);
            }
        }
        let blocks = Arc::new(Self::prepare(runtime)?);
        cache.push((key, Arc::downgrade(&blocks)));
        Ok(blocks)
    }

    fn prepare(runtime: &Runtime) -> anyhow::Result<Self> {
        Ok(Self {
            embed: EmbedKernel::prepare(runtime).context("prepare embed kernel")?,
            rmsnorm: RmsnormKernel::prepare(runtime).context("prepare rmsnorm kernel")?,
            rmsnorm_inplace: RmsnormInplaceKernel::prepare(runtime)
                .context("prepare rmsnorm_inplace kernel")?,
            rope_store: RopeStoreKernel::prepare(runtime).context("prepare rope_store kernel")?,
            cache_store: CacheStoreKernel::prepare(runtime)
                .context("prepare cache_store kernel")?,
            attention: AttentionKernel::prepare(runtime).context("prepare attention kernel")?,
            swiglu: SwigluKernel::prepare(runtime).context("prepare swiglu kernel")?,
            tensors: TensorKernels::prepare(runtime).context("prepare tensor kernels")?,
        })
    }

    /// Token-row gather into `x`.
    pub(crate) fn record_embed(
        &self,
        scheme: &mut Scheme,
        table: TensorView<'_>,
        step: &Buffer,
        x: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.embed
            .record(scheme, "embed", table, step, x)?
            .over_tensor(&x);
        Ok(())
    }

    /// RMSNorm, Q/K/V, RoPE, attention, output projection, residual into `x`.
    ///
    /// `wk` and `wv` are semantic GEMVs into the prefix of `xb2`. [`RopeStoreKernel`]
    /// rotates Q in place and writes RoPE(K) into the cache. [`CacheStoreKernel`]
    /// copies V into the cache. `wo` reuses all of `xb2` after those stores.
    pub(crate) fn record_attention(
        &self,
        scheme: &mut Scheme,
        sites: AttentionSites<'_>,
    ) -> Result<(), GoldyError> {
        let head = sites.key.shape().dim(2)?;
        let q_heads = sites.q.shape().dim(0)?;
        let q = sites.q;
        let q_flat = q.reshape(&[q.numel_u32()])?;
        let xb_heads = sites.xb.reshape(&[q_heads, head])?;
        let kv_dim = sites.wk.shape().dim(0)?;
        if sites.wv.shape().dim(0)? != kv_dim {
            return Err(GoldyError::Validation(
                "attention: wk and wv row counts differ".into(),
            ));
        }
        if kv_dim > q.numel_u32() || kv_dim % 2 != 0 {
            return Err(GoldyError::Validation(
                "attention: kv width must be even and no wider than q".into(),
            ));
        }
        if sites.xb2.numel_u32() < kv_dim {
            return Err(GoldyError::Validation(
                "attention: xb2 is shorter than the kv projection".into(),
            ));
        }
        let proj = sites.xb2.narrow(0, 0, kv_dim)?;

        self.rmsnorm
            .record(scheme, "rmsnorm", sites.x, sites.rms, sites.xb)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wq", sites.wq, sites.xb, q_flat)?;
        self.tensors
            .matmul_into(scheme, "wk", sites.wk, sites.xb, proj)?;
        self.rope_store
            .record(
                scheme,
                "rope",
                q,
                proj,
                sites.key,
                sites.step,
                DEFAULT_ROPE_THETA,
            )?
            .over_1d((q.numel_u32() / 2).max(1));
        self.tensors
            .matmul_into(scheme, "wv", sites.wv, sites.xb, proj)?;
        self.cache_store
            .record(scheme, "wv_store", proj, sites.value, sites.step)?
            .over_1d(kv_dim);
        self.attention
            .record(
                scheme,
                "attn",
                q,
                sites.att,
                xb_heads,
                sites.key,
                sites.value,
                sites.step,
            )?
            .groups([q_heads, 1, 1]);
        self.tensors
            .matmul_into(scheme, "wo", sites.wo, sites.xb, sites.xb2)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb2, sites.x)?;
        Ok(())
    }

    /// RMSNorm, SwiGLU, down projection, residual into `x`.
    pub(crate) fn record_ffn(
        &self,
        scheme: &mut Scheme,
        sites: FfnSites<'_>,
    ) -> Result<(), GoldyError> {
        self.rmsnorm
            .record(scheme, "rmsnorm", sites.x, sites.rms, sites.xb)?
            .groups([1, 1, 1]);
        self.tensors
            .matmul_into(scheme, "w1", sites.w1, sites.xb, sites.hb)?;
        self.tensors
            .matmul_into(scheme, "w3", sites.w3, sites.xb, sites.hb2)?;
        self.swiglu
            .record(scheme, "swiglu", sites.hb, sites.hb2)?
            .over_tensor(&sites.hb);
        self.tensors
            .matmul_into(scheme, "w2", sites.w2, sites.hb, sites.xb)?;
        self.tensors
            .add_into(scheme, "residual", sites.x, sites.xb, sites.x)?;
        Ok(())
    }

    pub(crate) fn record_rmsnorm(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.rmsnorm
            .record(scheme, "rmsnorm", x, weight, out)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_rmsnorm_inplace(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.rmsnorm_inplace
            .record(scheme, "rmsnorm", x, weight)?
            .groups([1, 1, 1]);
        Ok(())
    }

    pub(crate) fn record_linear(
        &self,
        scheme: &mut Scheme,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.tensors.matmul_into(scheme, label, weight, x, out)
    }
}

static CACHE: OnceLock<Mutex<Vec<(usize, Weak<Blocks>)>>> = OnceLock::new();
