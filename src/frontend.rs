use anyhow::Result;
use ndarray::{s, Array1, Array2};
use std::ffi::c_float;

#[derive(Clone, Copy, Debug)]
pub struct FrontendConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub lfr_m: usize,
    pub lfr_n: usize,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            lfr_m: 7,
            lfr_n: 6,
        }
    }
}

pub struct FeaturePipeline {
    cfg: FrontendConfig,
}

impl FeaturePipeline {
    pub fn new(cfg: FrontendConfig) -> Self {
        Self { cfg }
    }

    pub fn compute_features(&mut self, pcm: &Vec<f32>, sample_rate: u32) -> Result<Array2<f32>> {
        anyhow::ensure!(
            sample_rate as usize == self.cfg.sample_rate,
            "expect sample rate {} but got {}",
            self.cfg.sample_rate,
            sample_rate
        );
        if pcm.is_empty() {
            anyhow::bail!("audio length too short for feature extraction");
        }

        // Keep config knobs alive for potential future customization of the backend extractor.
        let _ = (self.cfg.frame_length_ms, self.cfg.frame_shift_ms);

        let scaled: Vec<f32> = pcm.iter().map(|x| *x * (1 << 15) as f32).collect();

        let mut result = unsafe {
            knf_rs_sys::ComputeFbank(scaled.as_ptr() as *const c_float, scaled.len() as i32)
        };

        anyhow::ensure!(
            result.num_bins > 0 && result.num_frames > 0,
            "fbank extraction failed"
        );
        let frame_count = result.num_frames as usize;
        let mel_bins = result.num_bins as usize;
        anyhow::ensure!(
            mel_bins == self.cfg.n_mels,
            "expected {} mel bins but got {}",
            self.cfg.n_mels,
            mel_bins
        );

        let fbank_vec =
            unsafe { std::slice::from_raw_parts(result.frames, frame_count * mel_bins).to_vec() };
        unsafe {
            knf_rs_sys::DestroyFbankResult(&mut result as *mut _);
        }
        let fbank = Array2::<f32>::from_shape_vec((frame_count, mel_bins), fbank_vec)?;

        let lfr = apply_lfr(&fbank, self.cfg.lfr_m, self.cfg.lfr_n);
        Ok(lfr)
    }
}

pub(crate) fn apply_lfr(fbank: &Array2<f32>, lfr_m: usize, lfr_n: usize) -> Array2<f32> {
    if lfr_m == 1 && lfr_n == 1 {
        return fbank.to_owned();
    }
    let t = fbank.len_of(ndarray::Axis(0));
    let d = fbank.len_of(ndarray::Axis(1));
    if t == 0 {
        return Array2::<f32>::zeros((0, d * lfr_m));
    }
    let pad = (lfr_m - 1) / 2;
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(t + pad);
    let first = fbank.slice(s![0, ..]).to_owned().to_vec();
    for _ in 0..pad {
        padded.push(first.clone());
    }
    for row in fbank.rows() {
        padded.push(row.to_vec());
    }
    let t_lfr = ((t as f32) / (lfr_n as f32)).ceil() as usize;
    let mut out = Array2::<f32>::zeros((t_lfr, d * lfr_m));

    for i in 0..t_lfr {
        let start = i * lfr_n;
        let mut frame_vec = Vec::with_capacity(d * lfr_m);
        for m in 0..lfr_m {
            let idx = start + m;
            if let Some(row) = padded.get(idx) {
                frame_vec.extend_from_slice(row);
            } else {
                let last = padded.last().unwrap();
                frame_vec.extend_from_slice(last);
            }
        }
        let row = Array1::from_vec(frame_vec);
        out.slice_mut(s![i, ..]).assign(&row.view());
    }

    out
}
