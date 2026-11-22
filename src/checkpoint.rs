//! Name-free safetensors reader.
//!
//! Names, dtypes, and shapes are returned as stored. Callers decide which names
//! they understand. FP32 payloads can be uploaded with [`TensorEntry::upload_f32`].

use anyhow::{bail, Result};
use goldy::{Runtime, Tensor, TensorShape};
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
}
