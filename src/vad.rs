use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_silence_ms: f32,
    pub min_speech_ms: f32,
    pub speech_pad_ms: f32,
    pub merge_gap_ms: f32,
}

impl VadConfig {
    pub fn new(
        threshold: f32,
        min_silence_ms: f32,
        min_speech_ms: f32,
        speech_pad_ms: f32,
        merge_gap_ms: f32,
    ) -> Self {
        Self {
            threshold,
            min_silence_ms,
            min_speech_ms,
            speech_pad_ms,
            merge_gap_ms,
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_ms: 200.0,
            min_speech_ms: 400.0,
            speech_pad_ms: 120.0,
            merge_gap_ms: 200.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadSegment {
    pub start: usize,
    pub end: usize,
}

pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    sample_rate: usize,
    window_size: usize,
    threshold: f32,
    min_silence_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    merge_gap_samples: usize,
    config: VadConfig,
}

impl SileroVad {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        sample_rate: usize,
        intra_threads: usize,
        config: VadConfig,
    ) -> Result<Self> {
        let window_size = match sample_rate {
            8000 => 256,
            16000 => 512,
            32000 => 1024,
            44100 | 48000 => 1536,
            other => {
                return Err(anyhow!(
                    "unsupported sample rate {} for Silero VAD (expected 8k/16k/32k/44.1k/48k)",
                    other
                ))
            }
        };

        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("ORT session builder error: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ORT optimization level error: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow::anyhow!("ORT intra threads error: {e}"))?;
        let model_bytes = std::fs::read(model_path.as_ref())
            .with_context(|| format!("read Silero VAD model {}", model_path.as_ref().display()))?;
        let session = builder.commit_from_memory(&model_bytes).map_err(|e| {
            anyhow::anyhow!(
                "ORT load model error for {}: {e}",
                model_path.as_ref().display()
            )
        })?;

        let state = Array3::<f32>::zeros((2, 1, 128));
        let sanitized_config = sanitize_config(config);
        let threshold = sanitized_config.threshold;
        let min_silence_samples =
            ms_to_samples(sanitized_config.min_silence_ms, sample_rate).max(1);
        let min_speech_samples = ms_to_samples(sanitized_config.min_speech_ms, sample_rate).max(1);
        let speech_pad_samples = ms_to_samples(sanitized_config.speech_pad_ms, sample_rate);
        let merge_gap_samples = ms_to_samples(sanitized_config.merge_gap_ms, sample_rate);

        Ok(Self {
            session,
            state,
            sample_rate,
            window_size,
            threshold,
            min_silence_samples,
            min_speech_samples,
            speech_pad_samples,
            merge_gap_samples,
            config: sanitized_config,
        })
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn config(&self) -> VadConfig {
        self.config
    }

    pub fn collect_segments(&mut self, pcm: &[f32]) -> Result<Vec<VadSegment>> {
        self.reset();
        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        let mut segments = Vec::new();
        let mut triggered = false;
        let mut speech_start = 0usize;
        let mut silence_acc = 0usize;

        let mut frame = vec![0.0f32; self.window_size];
        let mut offset = 0usize;
        while offset < pcm.len() {
            let frame_end = (offset + self.window_size).min(pcm.len());
            frame.iter_mut().for_each(|v| *v = 0.0);
            let slice = &pcm[offset..frame_end];
            frame[..slice.len()].copy_from_slice(slice);

            let prob = self.predict_frame(&frame)?;
            if prob >= self.threshold {
                if !triggered {
                    triggered = true;
                    speech_start = offset.saturating_sub(self.speech_pad_samples);
                    silence_acc = 0;
                } else {
                    silence_acc = 0;
                }
            } else if triggered {
                silence_acc += frame_end - offset;
                if silence_acc >= self.min_silence_samples {
                    let mut end = frame_end + self.speech_pad_samples;
                    if end > pcm.len() {
                        end = pcm.len();
                    }
                    if end > speech_start && (end - speech_start) >= self.min_speech_samples {
                        segments.push(VadSegment {
                            start: speech_start,
                            end,
                        });
                    }
                    triggered = false;
                    silence_acc = 0;
                }
            }

            offset += self.window_size;
        }

        if triggered {
            let end = pcm.len();
            if end > speech_start && (end - speech_start) >= self.min_speech_samples {
                segments.push(VadSegment {
                    start: speech_start,
                    end,
                });
            }
        }

        if segments.len() > 1 {
            segments.sort_by_key(|seg| seg.start);
            let mut merged: Vec<VadSegment> = Vec::with_capacity(segments.len());
            for seg in segments.into_iter() {
                if let Some(last) = merged.last_mut() {
                    if seg.start <= last.end {
                        if seg.end > last.end {
                            last.end = seg.end;
                        }
                        continue;
                    }
                    let gap = seg.start.saturating_sub(last.end);
                    if gap <= self.merge_gap_samples {
                        if seg.end > last.end {
                            last.end = seg.end;
                        }
                        continue;
                    }
                }
                merged.push(seg);
            }
            self.reset();
            Ok(merged)
        } else {
            self.reset();
            Ok(segments)
        }
    }

    fn predict_frame(&mut self, samples: &[f32]) -> Result<f32> {
        let input = Array2::<f32>::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let sample_rate = Array1::<i64>::from(vec![self.sample_rate as i64]);

        let input_value =
            Value::from_array(input).map_err(|e| anyhow::anyhow!("ORT tensor error: {e}"))?;
        let sr_value =
            Value::from_array(sample_rate).map_err(|e| anyhow::anyhow!("ORT tensor error: {e}"))?;
        let state_value = Value::from_array(self.state.clone())
            .map_err(|e| anyhow::anyhow!("ORT tensor error: {e}"))?;

        let inputs = ort::inputs![
            "input" => input_value,
            "sr" => sr_value,
            "state" => state_value,
        ];
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("ORT run error: {e}"))?;

        let (_prob_shape, prob_data) = outputs
            .get("output")
            .ok_or_else(|| anyhow!("Output 'output' not found"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("ORT extract tensor error: {e}"))?;
        let probability = prob_data[0];

        let (state_shape, state_data) = outputs
            .get("stateN")
            .ok_or_else(|| anyhow!("Output 'stateN' not found"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("ORT extract tensor error: {e}"))?;

        let state_array = Array3::<f32>::from_shape_vec(
            (
                state_shape[0] as usize,
                state_shape[1] as usize,
                state_shape[2] as usize,
            ),
            state_data.to_vec(),
        )
        .map_err(|e| anyhow::anyhow!("reshape state array error: {e}"))?;
        self.state.assign(&state_array);

        Ok(probability)
    }
}

fn sanitize_config(config: VadConfig) -> VadConfig {
    let defaults = VadConfig::default();
    VadConfig {
        threshold: sanitize_threshold(config.threshold, defaults.threshold),
        min_silence_ms: sanitize_duration(config.min_silence_ms, defaults.min_silence_ms),
        min_speech_ms: sanitize_duration(config.min_speech_ms, defaults.min_speech_ms),
        speech_pad_ms: sanitize_duration(config.speech_pad_ms, defaults.speech_pad_ms),
        merge_gap_ms: sanitize_duration(config.merge_gap_ms, defaults.merge_gap_ms),
    }
}

fn sanitize_threshold(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

fn sanitize_duration(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

fn ms_to_samples(ms: f32, sample_rate: usize) -> usize {
    if ms <= 0.0 {
        return 0;
    }
    ((sample_rate as f32) * (ms / 1000.0)).round() as usize
}
