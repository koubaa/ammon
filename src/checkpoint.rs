//! Name-free safetensors reader.
//!
//! Names, dtypes, and shapes are returned as stored. Callers decide which names
//! they understand. [`TensorEntry::upload_f32`] copies an F32 payload.
//! [`MappedCheckpoint`] memory-maps the file and uploads packed dtypes
//! (BF16, U8, F8_E4M3, F32) without expanding them to FP32.

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Result};
use goldy::{Runtime, Tensor, TensorDType, TensorShape};
use safetensors::Dtype;
use safetensors::SafeTensors;

/// One tensor from a safetensors blob.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorEntry {
    pub fn is_f32(&self) -> bool {
        self.dtype == "F32"
    }

    /// Upload an F32 tensor. The shape must match the payload length.
    pub fn upload_f32(&self, runtime: &Runtime) -> Result<Tensor> {
        if !self.is_f32() {
            bail!("`{}` is {}, expected F32", self.name, self.dtype);
        }
        if self.data.len() != self.shape.iter().product::<usize>() * 4 {
            bail!("`{}` F32 payload does not match its shape", self.name);
        }
        let dims: Vec<u32> = self
            .shape
            .iter()
            .map(|d| u32::try_from(*d))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| anyhow::anyhow!("`{}` shape does not fit u32", self.name))?;
        let values: Vec<f32> = self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shape = TensorShape::from_dims(&dims)?;
        Ok(Tensor::from_f32(runtime, shape, &values)?)
    }
}

/// Read every tensor. The file is not interpreted as any model's checkpoint.
pub fn read_safetensors(bytes: &[u8]) -> Result<Vec<TensorEntry>> {
    let file =
        SafeTensors::deserialize(bytes).map_err(|err| anyhow::anyhow!("safetensors: {err}"))?;
    let mut out = Vec::new();
    for (name, view) in file.iter() {
        out.push(TensorEntry {
            name: name.to_string(),
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Dtype name used by [`read_safetensors`] for an F32 tensor.
pub fn f32_dtype_name() -> &'static str {
    let _ = Dtype::F32;
    "F32"
}

/// One tensor inside a [`MappedCheckpoint`]. The payload stays in the mapping.
#[derive(Clone, Debug)]
pub struct MappedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A safetensors file mapped once. Tensor payloads are borrowed from the map
/// and uploaded in their stored dtype. BF16, U8, and F8_E4M3 are not expanded.
pub struct MappedCheckpoint {
    mmap: memmap2::Mmap,
    tensors: Vec<MappedTensor>,
}

impl MappedCheckpoint {
    /// Map `path` and read the header. The file is not copied.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // Read-only map. Callers do not write the file while it is open.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let parsed =
            SafeTensors::deserialize(&mmap).map_err(|err| anyhow::anyhow!("safetensors: {err}"))?;
        let base = mmap.as_ptr() as usize;
        let mut tensors = Vec::new();
        for (name, view) in parsed.iter() {
            let data = view.data();
            let start = data.as_ptr() as usize - base;
            let end = start + data.len();
            if end > mmap.len() {
                bail!("`{name}` extent is outside the mapped file");
            }
            tensors.push(MappedTensor {
                name: name.to_string(),
                dtype: dtype_name(view.dtype()),
                shape: view.shape().to_vec(),
                start,
                end,
            });
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { mmap, tensors })
    }

    pub fn tensors(&self) -> &[MappedTensor] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Result<&MappedTensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor `{name}`"))
    }

    /// Packed bytes of one tensor. The length is `numel * dtype size`.
    pub fn bytes(&self, tensor: &MappedTensor) -> &[u8] {
        &self.mmap[tensor.start..tensor.end]
    }

    pub fn bytes_named(&self, name: &str) -> Result<&[u8]> {
        let tensor = self.get(name)?;
        Ok(self.bytes(tensor))
    }

    /// Upload one tensor in its stored dtype. F32 stays 4 bytes per element,
    /// BF16 stays 2, and U8 / F8_E4M3 stay 1.
    pub fn upload(&self, runtime: &Runtime, name: &str) -> Result<Tensor> {
        let tensor = self.get(name)?;
        let dtype = match tensor.dtype.as_str() {
            "F32" => TensorDType::F32,
            "BF16" => TensorDType::BF16,
            "U8" => TensorDType::U8,
            "F8_E4M3" => TensorDType::F8E4M3,
            "U32" => TensorDType::U32,
            "I32" => TensorDType::I32,
            other => bail!("`{name}` dtype {other} has no packed upload"),
        };
        let dims = tensor
            .shape
            .iter()
            .map(|dim| {
                u32::try_from(*dim).map_err(|_| anyhow::anyhow!("`{name}` shape does not fit u32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = TensorShape::from_dims(&dims)?;
        Ok(Tensor::from_bytes(
            runtime,
            shape,
            dtype,
            self.bytes(tensor),
        )?)
    }
}

fn dtype_name(dtype: Dtype) -> String {
    match dtype {
        Dtype::F32 => "F32".to_string(),
        Dtype::BF16 => "BF16".to_string(),
        Dtype::U8 => "U8".to_string(),
        Dtype::F8_E4M3 => "F8_E4M3".to_string(),
        Dtype::U32 => "U32".to_string(),
        Dtype::I32 => "I32".to_string(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};

    #[test]
    fn reads_names_shapes_and_f32_bytes_without_a_model_map() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let view = TensorView::new(Dtype::F32, vec![2, 2], bytemuck::cast_slice(&values)).unwrap();
        let bytes = serialize([("any.name", view)], None).unwrap();
        let entries = read_safetensors(&bytes).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "any.name");
        assert_eq!(entries[0].shape, vec![2, 2]);
        assert!(entries[0].is_f32());
        assert_eq!(entries[0].data.len(), 16);
    }

    #[test]
    fn mapped_file_keeps_packed_dtypes_at_their_stored_size() {
        let bf16 = [0x80u8, 0x3f, 0x00, 0x40];
        let codes = [0x24u8, 0x10];
        let scales = [0x38u8, 0x38];
        let f32s = [1.0f32, -2.0];
        let views = [
            (
                "linear.weight",
                TensorView::new(Dtype::BF16, vec![2, 1], &bf16).unwrap(),
            ),
            (
                "experts.codes",
                TensorView::new(Dtype::U8, vec![1, 2, 1], &codes).unwrap(),
            ),
            (
                "experts.scale",
                TensorView::new(Dtype::F8_E4M3, vec![1, 2, 1], &scales).unwrap(),
            ),
            (
                "bias",
                TensorView::new(Dtype::F32, vec![2], bytemuck::cast_slice(&f32s)).unwrap(),
            ),
        ];
        let bytes = serialize(views, None).unwrap();
        let path = std::env::temp_dir().join("ammon-packed-checkpoint-test.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        let mapped = MappedCheckpoint::open(&path).unwrap();
        let linear = mapped.get("linear.weight").unwrap();
        assert_eq!(linear.dtype, "BF16");
        assert_eq!(linear.shape, vec![2, 1]);
        assert_eq!(mapped.bytes(linear), &bf16);
        assert_eq!(mapped.bytes_named("experts.codes").unwrap(), &codes);
        assert_eq!(mapped.bytes_named("experts.scale").unwrap(), &scales);
        assert_eq!(mapped.get("experts.scale").unwrap().dtype, "F8_E4M3");
        assert_eq!(mapped.bytes_named("bias").unwrap().len(), 8);
        let _ = std::fs::remove_file(&path);
    }
}
