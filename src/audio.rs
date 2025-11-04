use anyhow::{ensure, Context, Result};
use std::path::Path;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::codecs::CODEC_TYPE_NULL;
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
                // Convert to f32
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
