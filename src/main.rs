use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use ndarray::Axis;
use serde::Serialize;
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{debug, error, warn, Level};
use tracing_subscriber::filter::LevelFilter;

mod audio;
mod frontend;
mod sensevoice;
mod tokenizer;
mod vad;

use crate::audio::{decode_audio_multi, resample_channels};
use crate::frontend::{FeaturePipeline, FrontendConfig};

use crate::sensevoice::SensevoiceEncoder;
use crate::tokenizer::TokenDecoder;
use crate::vad::{SileroVad, VadConfig, VadSegment};

fn language_id_from_code(code: &str) -> i32 {
    // Python mapping: {"auto":0,"zh":3,"en":4,"yue":7,"ja":11,"ko":12,"nospeech":13}
    match code.to_lowercase().as_str() {
        "auto" => 0,
        "zh" => 3,
        "en" => 4,
        "yue" => 7,
        "ja" => 11,
        "ko" => 12,
        "nospeech" => 13,
        _ => 0,
    }
}

fn user_home_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .or_else(|| {
                let drive = env::var("HOMEDRIVE").ok()?;
                let path = env::var("HOMEPATH").ok()?;
                Some(PathBuf::from(format!("{drive}{path}")))
            })
            .or_else(|| env::var_os("HOME").map(PathBuf::from))
    } else {
        env::var_os("HOME").map(PathBuf::from)
    }
}

fn default_models_dir() -> PathBuf {
    user_home_dir()
        .map(|mut home| {
            home.push(".sensevoice-models");
            home
        })
        .unwrap_or_else(|| PathBuf::from(".sensevoice-models"))
}

fn resolve_download_path(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~") {
        if let Some(home) = user_home_dir() {
            if stripped.as_os_str().is_empty() {
                return home;
            }
            return home.join(stripped);
        }
    }
    path.to_path_buf()
}

#[derive(Parser, Debug)]
#[command(
    name = "sensevoice",
    author,
    version,
    about = "SenseVoice Rust CLI (ORT + Symphonia + HF Hub)"
)]
struct Cli {
    /// Download/cache directory for models and resources
    #[arg(long = "models-path", default_value_os_t = default_models_dir())]
    models_path: PathBuf,

    /// Intra-op threads for ONNX Runtime
    #[arg(short = 't', long = "threads", default_value_t = 1)]
    num_threads: usize,

    /// Language code: auto, zh, en, yue, ja, ko, nospeech
    #[arg(short = 'l', long = "language", default_value = "auto")]
    language: String,

    /// Use ITN post-processing
    #[arg(long = "use-itn", action = ArgAction::SetTrue)]
    use_itn: bool,

    /// Use int8 Silero VAD model
    #[arg(long = "vad-int8", action = ArgAction::SetTrue)]
    vad_int8: bool,

    /// Disable Silero VAD and transcribe full audio without segmentation
    #[arg(long = "no-vad", action = ArgAction::SetTrue)]
    no_vad: bool,

    /// VAD probability threshold (0.0-1.0)
    #[arg(long = "vad-threshold", default_value_t = 0.5)]
    vad_threshold: f32,

    /// Minimum speech duration in milliseconds for a valid segment
    #[arg(long = "vad-min-speech-ms", default_value_t = 250.0)]
    vad_min_speech_ms: f32,

    /// Minimum silence duration in milliseconds before closing a segment
    #[arg(long = "vad-min-silence-ms", default_value_t = 100.0)]
    vad_min_silence_ms: f32,

    /// Extra padding in milliseconds added to segment boundaries
    #[arg(long = "vad-speech-pad-ms", default_value_t = 30.0)]
    vad_speech_pad_ms: f32,

    /// Optional HF endpoint/mirror (overrides env HF_ENDPOINT/HF_MIRROR)
    #[arg(long = "hf-endpoint")]
    hf_endpoint: Option<String>,

    /// Log level
    #[arg(long = "log")]
    log: Option<String>,

    /// Output JSON file path
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Maximum number of audio channels to transcribe (0 = all)
    #[arg(short = 'c', long = "channels", default_value_t = 1)]
    channels: usize,

    #[arg(long = "download-only", action = ArgAction::SetTrue, default_value_t = false)]
    /// Download models only and exit
    download_only: bool,

    /// Input audio file (wav/mp3/ogg/flac)
    #[arg(value_name = "AUDIO")]
    audio: Option<PathBuf>,
}

#[derive(Serialize)]
struct ChannelResult {
    channel: usize,
    duration_sec: f32,
    rtf: f32,
    segments: Vec<Segment>,
}

#[derive(Serialize)]
struct Segment {
    start_sec: f32,
    end_sec: f32,
    text: String,
    tags: Vec<String>,
}

fn extract_tags(text: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    let mut tags = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let mut j = i + 1;
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '>' {
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed {
                let tag_content: String = chars[i + 1..j].iter().collect();
                let tag = tag_content.trim();
                if !tag.is_empty() {
                    tags.push(tag.to_string());
                }
                i = j + 1;
                continue;
            }
        }
        clean.push(chars[i]);
        i += 1;
    }
    let cleaned = clean.trim().to_string();
    (cleaned, tags)
}

fn build_api(hf_endpoint_cli: &Option<String>) -> Result<Api> {
    if let Some(ep) = hf_endpoint_cli
        .clone()
        .or_else(|| env::var("HF_ENDPOINT").ok())
        .or_else(|| env::var("HF_MIRROR").ok())
    {
        // Fallback approach: set env for hf-hub to pick up endpoint
        env::set_var("HF_ENDPOINT", ep);
    }
    let api = ApiBuilder::from_env().build()?;
    Ok(api)
}

fn ensure_repo_files(
    api: &Api,
    dest_dir: &Path,
    repo_id: &str,
    revision: &str,
    repo_type: RepoType,
    files: &[&str],
    label: &str,
) -> Result<()> {
    fs::create_dir_all(dest_dir).with_context(|| {
        format!(
            "create or access destination directory {}",
            dest_dir.display()
        )
    })?;

    let repo = Repo::with_revision(repo_id.to_string(), repo_type, revision.to_string());
    let repo = api.repo(repo);

    for f in files.iter() {
        let dest = dest_dir.join(f);
        if dest.exists() {
            continue;
        }
        warn!(file = %f, resource = %label, "resource missing, downloading...");
        match repo.get(f) {
            Ok(src_path) => {
                if let Err(e) = copy_into_dir(&src_path, &dest) {
                    warn!(file = %f, resource = %label, error = %e, "failed to copy cached resource, retrying via direct download");
                    let url = repo.url(f);
                    if let Err(dl_err) = download_without_range(&url, &dest) {
                        error!(file = %f, resource = %label, error = %dl_err, "failed to fetch resource from mirror");
                    }
                }
            }
            Err(err) => {
                warn!(file = %f, resource = %label, error = ?err, "hf-hub get failed, retrying via direct download");
                let url = repo.url(f);
                if let Err(dl_err) = download_without_range(&url, &dest) {
                    error!(file = %f, resource = %label, error = %dl_err, "failed to fetch resource from mirror");
                }
            }
        }
    }

    Ok(())
}

fn ensure_models(api: &Api, download_path: &PathBuf) -> Result<PathBuf> {
    ensure_repo_files(
        api,
        download_path,
        "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        "main",
        RepoType::Model,
        &["model.int8.onnx", "tokens.txt"],
        "ASR model/resource file",
    )?;
    Ok(download_path.clone())
}

fn copy_into_dir(src: &Path, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("ensure parent directory {}", parent.display()))?;
    }
    if src == dest {
        return Ok(());
    }
    fs::copy(src, dest).with_context(|| format!("copy {} to {}", src.display(), dest.display()))?;
    Ok(())
}

fn download_without_range(url: &str, dest: &Path) -> Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("ensure parent directory {}", parent.display()))?;
    }

    let temp_path = dest.with_extension("download");
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;

    if !(200..=299).contains(&response.status()) {
        return Err(anyhow::anyhow!(
            "unexpected status {} while downloading {}",
            response.status(),
            url
        ));
    }

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&temp_path)
        .with_context(|| format!("create temporary file {}", temp_path.display()))?;
    io::copy(&mut reader, &mut file)
        .with_context(|| format!("write bytes for {}", dest.display()))?;
    file.sync_all()
        .with_context(|| format!("flush file {}", temp_path.display()))?;

    if dest.exists() {
        fs::remove_file(dest)
            .with_context(|| format!("remove existing file {}", dest.display()))?;
    }
    fs::rename(&temp_path, dest)
        .with_context(|| format!("finalize download to {}", dest.display()))?;

    Ok(dest.to_path_buf())
}

fn ensure_vad_model(api: &Api, download_path: &PathBuf, use_int8: bool) -> Result<PathBuf> {
    let vad_dir = download_path.join("silero-vad");
    ensure_repo_files(
        api,
        &vad_dir,
        "onnx-community/silero-vad",
        "main",
        RepoType::Model,
        &["onnx/model.onnx", "onnx/model_int8.onnx"],
        "Silero VAD model file",
    )?;

    let selected = if use_int8 {
        vad_dir.join("onnx/model_int8.onnx")
    } else {
        vad_dir.join("onnx/model.onnx")
    };
    if !selected.exists() {
        anyhow::bail!(
            "Silero VAD model {:?} missing after download attempt",
            selected
        );
    }
    Ok(selected)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let download_path = resolve_download_path(&cli.models_path);
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::from_level(
            cli.log
                .as_ref()
                .map(|l| l.parse().ok())
                .flatten()
                .unwrap_or(Level::WARN),
        ))
        .with_target(false)
        .init();

    // 1) HF Hub models
    let api = build_api(&cli.hf_endpoint)?;
    debug!(
        endpoint = cli.hf_endpoint,
        "checking/downloading models from HF Hub (mirror-aware)"
    );
    let snapshot_dir = ensure_models(&api, &download_path)?;
    let vad_model_path = if cli.no_vad {
        None
    } else {
        Some(ensure_vad_model(&api, &download_path, cli.vad_int8)?)
    };

    let encoder_path = snapshot_dir.join("model.int8.onnx");
    let tokens_path = snapshot_dir.join("tokens.txt");

    if !encoder_path.exists() || !tokens_path.exists() {
        error!("model/resource files missing in snapshot. Please check repository contents.");
    }
    if cli.download_only {
        debug!("download-only flag set, exiting after model download");
        return Ok(());
    }

    let fe_cfg = FrontendConfig::default();
    let target_sample_rate = fe_cfg.sample_rate as u32;
    let vad_config = VadConfig::new(
        cli.vad_threshold,
        cli.vad_min_silence_ms,
        cli.vad_min_speech_ms,
        cli.vad_speech_pad_ms,
    );
    let mut vad = if let Some(vad_model_path) = vad_model_path {
        match SileroVad::new(
            &vad_model_path,
            fe_cfg.sample_rate,
            cli.num_threads,
            vad_config,
        ) {
            Ok(v) => {
                let config = v.config();
                debug!(
                    use_int8 = cli.vad_int8,
                    vad_model = %vad_model_path.display(),
                    threshold = format!("{:.3}", config.threshold),
                    min_speech_ms = format!("{:.1}", config.min_speech_ms),
                    min_silence_ms = format!("{:.1}", config.min_silence_ms),
                    speech_pad_ms = format!("{:.1}", config.speech_pad_ms),
                    "Silero VAD model initialized"
                );
                Some(v)
            }
            Err(err) => {
                if cli.vad_int8 {
                    warn!(
                        error = %err,
                        vad_model = %vad_model_path.display(),
                        "failed to initialize int8 Silero VAD, retrying with float32 model"
                    );
                    let fallback_path = ensure_vad_model(&api, &download_path, false)?;
                    let fallback_vad = SileroVad::new(
                        &fallback_path,
                        fe_cfg.sample_rate,
                        cli.num_threads,
                        vad_config,
                    )
                    .with_context(|| {
                        format!("load Silero VAD fallback from {}", fallback_path.display())
                    })?;
                    let config = fallback_vad.config();
                    debug!(
                        use_int8 = false,
                        vad_model = %fallback_path.display(),
                        threshold = format!("{:.3}", config.threshold),
                        min_speech_ms = format!("{:.1}", config.min_speech_ms),
                        min_silence_ms = format!("{:.1}", config.min_silence_ms),
                        speech_pad_ms = format!("{:.1}", config.speech_pad_ms),
                        "Silero VAD model initialized after fallback"
                    );
                    Some(fallback_vad)
                } else {
                    return Err(err);
                }
            }
        }
    } else {
        None
    };

    // 4) ORT Session for encoder + tokenizer
    let mut encoder = SensevoiceEncoder::new(&encoder_path, cli.num_threads)?;
    let decoder = TokenDecoder::new(&tokens_path)?;
    let lang_id = language_id_from_code(&cli.language);
    let audio = match &cli.audio {
        Some(p) => p,
        None => {
            anyhow::bail!("no input audio file specified. Please provide an audio file path.");
        }
    };
    // 2) Audio decode (multi-channel)
    let t0 = Instant::now();
    let (decoded_sample_rate, total_channels, mut samples_per_channel) = decode_audio_multi(audio)?;
    let requested_channels = if cli.channels == 0 {
        samples_per_channel.len()
    } else {
        cli.channels.min(samples_per_channel.len())
    };
    if samples_per_channel.len() > requested_channels {
        samples_per_channel.truncate(requested_channels);
    }
    if samples_per_channel.is_empty() {
        anyhow::bail!("no audio channels available for transcription");
    }
    let processed_channels = samples_per_channel.len();
    let durations: Vec<f32> = samples_per_channel
        .iter()
        .map(|ch| ch.len() as f32 / decoded_sample_rate as f32)
        .collect();
    let audio_duration_sec = durations.first().copied().unwrap_or(0.0);
    if decoded_sample_rate != target_sample_rate {
        debug!(
            "resampling audio from {} Hz to {} Hz",
            decoded_sample_rate, target_sample_rate
        );
        samples_per_channel =
            resample_channels(samples_per_channel, decoded_sample_rate, target_sample_rate)?;
    }
    debug!(
        "decoded audio: {} Hz, {} ch (processing {}), duration ~{:.2}s",
        decoded_sample_rate, total_channels, processed_channels, audio_duration_sec
    );

    // 3) Frontend: fbank + LFR + CMVN (Kaldi-like defaults)
    let mut fe = FeaturePipeline::new(fe_cfg);

    let mut results: Vec<ChannelResult> = Vec::new();

    for (ch_idx, ch) in samples_per_channel.iter().enumerate() {
        let detected_segments = if let Some(vad_ref) = vad.as_mut() {
            let mut segments = vad_ref.collect_segments(ch)?;
            if segments.is_empty() {
                warn!(
                    channel = ch_idx,
                    "no speech detected by VAD, falling back to full channel"
                );
                segments.push(VadSegment {
                    start: 0,
                    end: ch.len(),
                });
            }
            segments
        } else {
            vec![VadSegment {
                start: 0,
                end: ch.len(),
            }]
        };

        let mut channel_segments: Vec<Segment> = Vec::new();

        for seg in detected_segments {
            let start = seg.start.min(ch.len());
            let end = seg.end.min(ch.len());
            if end <= start {
                continue;
            }
            let segment_samples = &ch[start..end];
            if segment_samples.is_empty() {
                continue;
            }

            let feats = match fe.compute_features(segment_samples, target_sample_rate) {
                Ok(f) => f,
                Err(e) => {
                    warn!(channel = ch_idx, start = start, end = end, error = %e, "feature extraction failed for segment");
                    continue;
                }
            };
            if feats.is_empty() {
                warn!(
                    channel = ch_idx,
                    start = start,
                    end = end,
                    "empty feature matrix for segment, skipping"
                );
                continue;
            }
            let feats = feats.insert_axis(Axis(0));
            let (frames, dims) = (feats.len_of(Axis(1)), feats.len_of(Axis(2)));
            let elem = (frames * dims).max(1) as f32;
            let sum: f32 = feats.iter().copied().sum();
            let mean = sum / elem;
            let var = feats
                .iter()
                .map(|v| {
                    let diff = *v - mean;
                    diff * diff
                })
                .sum::<f32>()
                / elem;
            let std = var.sqrt();
            debug!(
                channel = ch_idx,
                start = start,
                end = end,
                frames,
                dims,
                mean = format!("{:.3}", mean),
                std = format!("{:.3}", std),
                "segment features computed"
            );

            let raw_text = encoder.run_and_decode(&decoder, feats.view(), lang_id, cli.use_itn)?;
            let (clean_text, tags) = extract_tags(&raw_text);
            channel_segments.push(Segment {
                start_sec: start as f32 / target_sample_rate as f32,
                end_sec: end as f32 / target_sample_rate as f32,
                text: clean_text,
                tags,
            });
        }

        if channel_segments.is_empty() {
            if vad.is_some() {
                warn!(
                    channel = ch_idx,
                    "no valid segments produced after VAD, falling back to full channel transcription"
                );
            } else {
                warn!(
                    channel = ch_idx,
                    "no valid segments produced, falling back to full channel transcription"
                );
            }
            let feats = fe.compute_features(ch, target_sample_rate)?;
            if !feats.is_empty() {
                let feats = feats.insert_axis(Axis(0));
                let raw_text =
                    encoder.run_and_decode(&decoder, feats.view(), lang_id, cli.use_itn)?;
                let (clean_text, tags) = extract_tags(&raw_text);
                channel_segments.push(Segment {
                    start_sec: 0.0,
                    end_sec: ch.len() as f32 / target_sample_rate as f32,
                    text: clean_text,
                    tags,
                });
            }
        }

        results.push(ChannelResult {
            channel: ch_idx,
            duration_sec: durations
                .get(ch_idx)
                .copied()
                .unwrap_or_else(|| ch.len() as f32 / target_sample_rate as f32),
            rtf: 0.0,
            segments: channel_segments,
        });
    }

    let elapsed = t0.elapsed();
    let denom_channels = processed_channels.max(1);
    let rtf = elapsed.as_secs_f32() / (denom_channels as f32 * audio_duration_sec.max(1e-6));
    for result in &mut results {
        result.rtf = rtf;
    }

    let json = serde_json::to_string_pretty(&results)?;
    if let Some(output_path) = &cli.output {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        fs::write(output_path, json.as_bytes())
            .with_context(|| format!("write JSON output to {}", output_path.display()))?;
        debug!(
            path = %output_path.display(),
            "transcription JSON written to file"
        );
    } else {
        println!("{}", json);
    }
    debug!("time: {:?}, rtf: {:.3}", elapsed, rtf);
    Ok(())
}
