use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use ndarray::{Array1, Array3, Axis};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynValue, Tensor},
};

use crate::tokenizer::TokenDecoder;

pub struct OrtEncoder {
    session: Session,
    input_names: Vec<String>,
}

impl OrtEncoder {
    pub fn new<P: AsRef<Path>>(model_path: P, device: i32, intra_threads: usize) -> Result<Self> {
        let builder = Session::builder()
            .map_err(|e| anyhow!("ORT session builder error: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("ORT optimization level error: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow!("ORT intra threads error: {e}"))?;

        if device >= 0 {
            tracing::warn!("CUDA execution provider not configured; falling back to CPU");
        }

        let model_bytes = std::fs::read(model_path.as_ref())
            .with_context(|| format!("read encoder model {}", model_path.as_ref().display()))?;
        let session = builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| anyhow!("ORT load model error: {e}"))?;
        let input_names = session
            .inputs
            .iter()
            .map(|info| info.name.clone())
            .collect();
        Ok(Self {
            session,
            input_names,
        })
    }

    pub fn run_and_decode(
        &mut self,
        decoder: &TokenDecoder,
        feats: ndarray::ArrayView3<'_, f32>, // [B=1, T, D]
        language_id: i32,
        use_itn: bool,
    ) -> Result<String> {
        let b = feats.len_of(Axis(0));
        ensure!(b == 1, "batch=1 only");
        let t = feats.len_of(Axis(1));
        let d = feats.len_of(Axis(2));
        ensure!(d == 560, "expect feature dim 560 but got {}", d);
        ensure!(
            language_id >= 0 && language_id < 16,
            "invalid language id {language_id}"
        );

        let text_norm_idx = if use_itn { 14 } else { 15 };

        let input_tensor =
            Tensor::from_array(feats.to_owned()).map_err(|e| anyhow!("ORT tensor error: {e}"))?;
        let len_tensor = Tensor::from_array(Array1::from_vec(vec![t as i32]))
            .map_err(|e| anyhow!("ORT tensor error: {e}"))?;
        let lang_tensor = Tensor::from_array(Array1::from_vec(vec![language_id]))
            .map_err(|e| anyhow!("ORT tensor error: {e}"))?;
        let tn_tensor = Tensor::from_array(Array1::from_vec(vec![text_norm_idx]))
            .map_err(|e| anyhow!("ORT tensor error: {e}"))?;

        let mut x_val = Some(input_tensor.into_dyn());
        let mut len_val = Some(len_tensor.into_dyn());
        let mut lang_val = Some(lang_tensor.into_dyn());
        let mut tn_val = Some(tn_tensor.into_dyn());

        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let value = match name.as_str() {
                "x" => x_val
                    .take()
                    .ok_or_else(|| anyhow!("duplicate tensor binding for input 'x'"))?,
                "x_length" => len_val
                    .take()
                    .ok_or_else(|| anyhow!("duplicate tensor binding for input 'x_length'"))?,
                "language" => lang_val
                    .take()
                    .ok_or_else(|| anyhow!("duplicate tensor binding for input 'language'"))?,
                "text_norm" => tn_val
                    .take()
                    .ok_or_else(|| anyhow!("duplicate tensor binding for input 'text_norm'"))?,
                other => anyhow::bail!("unexpected encoder input '{other}'"),
            };
            inputs.push((name.clone(), value));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow!("ORT run error: {e}"))?;
        let logits_value = &outputs[0];
        let (shape, data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("ORT extract tensor error: {e}"))?;
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        ensure!(dims.len() == 3, "unexpected logits rank: {:?}", dims);
        ensure!(dims[0] == 1, "expect batch=1 but got {}", dims[0]);
        let logits = Array3::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec())?;
        let ids = argmax_and_unique(logits.index_axis(Axis(0), 0));
        Ok(decoder.decode_ids(&ids))
    }
}

fn argmax_and_unique(logits: ndarray::ArrayView2<'_, f32>) -> Vec<i32> {
    let blank_id = 0i32;
    let mut prev: Option<i32> = None;
    let mut out = Vec::new();
    for t in 0..logits.len_of(Axis(0)) {
        let row = logits.index_axis(Axis(0), t);
        let mut maxv = f32::MIN;
        let mut arg = 0i32;
        for (i, v) in row.iter().enumerate() {
            if *v > maxv {
                maxv = *v;
                arg = i as i32;
            }
        }
        if Some(arg) != prev {
            if arg != blank_id {
                out.push(arg);
            }
            prev = Some(arg);
        }
    }
    out
}
