use anyhow::{ensure, Context, Result};
use std::path::Path;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL, CODEC_TYPE_OPUS};
use symphonia::core::errors::Error as SymphErr;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub fn decode_audio_multi(path: &Path) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {:?}", path))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("missing sample rate"))?;
    let channels = track
        .codec_params
        .channels
        .ok_or_else(|| anyhow::anyhow!("missing channels"))?
        .count();

    if track.codec_params.codec == CODEC_TYPE_OPUS {
        #[cfg(feature = "opus")]
        {
            return decode_opus_track(format.as_mut(), track, sample_rate, channels);
        }
        #[cfg(not(feature = "opus"))]
        {
            anyhow::bail!("Opus decoding not enabled (enable the 'opus' feature)");
        }
    } else {
        decode_with_builtin_decoder(format.as_mut(), track, sample_rate, channels)
    }
}

fn decode_with_builtin_decoder(
    format: &mut dyn symphonia::core::formats::FormatReader,
    track: symphonia::core::formats::Track,
    sample_rate: u32,
    channels: usize,
) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(s) => s,
            Err(SymphErr::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec_val = *decoded.spec();

        match decoded {
            symphonia::core::audio::AudioBufferRef::F32(buf) => {
                let chans = buf.spec().channels.count();
                for ch in 0..chans {
                    per_channel[ch].extend(buf.chan(ch));
                }
            }
            other => {
                // Convert to f32 when decoder provided non-f32 samples.
                let mut buf = symphonia::core::audio::AudioBuffer::<f32>::new(
                    other.capacity() as u64,
                    spec_val,
                );
                other.convert(&mut buf);
                let chans = buf.spec().channels.count();
                for ch in 0..chans {
                    per_channel[ch].extend(buf.chan(ch));
                }
            }
        }
    }

    Ok((sample_rate, channels, per_channel))
}

#[cfg(feature = "opus")]
fn decode_opus_track(
    format: &mut dyn symphonia::core::formats::FormatReader,
    track: symphonia::core::formats::Track,
    sample_rate: u32,
    channels: usize,
) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    use opus::{Channels as OpusChannels, Decoder as OpusDecoder};
    /// Maximum number of samples per Opus frame at 48 kHz (120 ms).
    const OPUS_MAX_FRAME_SAMPLES: usize = 5760;

    let opus_channels = match channels {
        1 => OpusChannels::Mono,
        2 => OpusChannels::Stereo,
        _ => anyhow::bail!("Opus decoder currently supports only mono or stereo (got {channels})"),
    };

    let mut decoder = OpusDecoder::new(sample_rate, opus_channels)
        .map_err(|e| anyhow::anyhow!("create Opus decoder: {e}"))?;

    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut decode_buf = vec![0.0_f32; OPUS_MAX_FRAME_SAMPLES * channels.max(1)];

    // Skip encoder priming samples if present.
    let mut skip_samples = track.codec_params.delay.unwrap_or(0) as usize;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }

        let data = packet.buf();
        if data.is_empty() {
            continue;
        }

        let required_frames = decoder
            .get_nb_samples(data)
            .unwrap_or(OPUS_MAX_FRAME_SAMPLES);
        if required_frames * channels > decode_buf.len() {
            decode_buf.resize(required_frames * channels, 0.0);
        }

        let frames = decoder
            .decode_float(data, &mut decode_buf, false)
            .map_err(|e| anyhow::anyhow!("decode Opus packet: {e}"))?;

        if frames == 0 {
            continue;
        }

        let mut start = packet.trim_start as usize;
        let end = frames.saturating_sub(packet.trim_end as usize);
        if start >= end {
            continue;
        }

        if skip_samples > 0 {
            let to_skip = skip_samples.min(end - start);
            start += to_skip;
            skip_samples -= to_skip;
            if start >= end {
                continue;
            }
        }

        let available_frames = end - start;
        if available_frames == 0 {
            continue;
        }

        for frame_idx in start..end {
            let base = frame_idx * channels;
            for ch in 0..channels {
                per_channel[ch].push(decode_buf[base + ch]);
            }
        }
    }

    // Align number of samples across channels in case of unexpected discrepancies.
    if let Some(min_len) = per_channel.iter().map(|c| c.len()).min() {
        for ch in per_channel.iter_mut() {
            ch.truncate(min_len);
        }
    }

    Ok((sample_rate, channels, per_channel))
}

pub fn resample_channels(
    samples_per_channel: Vec<Vec<f32>>,
    src_rate: u32,
    dst_rate: u32,
) -> Result<Vec<Vec<f32>>> {
    if src_rate == dst_rate {
        return Ok(samples_per_channel);
    }
    ensure!(src_rate > 0 && dst_rate > 0, "sample rate must be positive");
    let ratio = dst_rate as f64 / src_rate as f64;
    let mut out = Vec::with_capacity(samples_per_channel.len());
    for channel in samples_per_channel.iter() {
        out.push(resample_linear(channel, ratio));
    }
    Ok(out)
}

fn resample_linear(input: &[f32], ratio: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let output_len = ((input.len() as f64) * ratio).ceil().max(1.0) as usize;
    let mut out = Vec::with_capacity(output_len);
    let last = input.len() - 1;
    for n in 0..output_len {
        let pos = (n as f64) / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let i0 = idx.min(last);
        let i1 = (idx + 1).min(last);
        let s0 = input[i0];
        let s1 = input[i1];
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}
