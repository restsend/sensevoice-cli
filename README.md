# SenseVoice CLI

A lightweight command-line front end for the SenseVoice multilingual speech recognition model.

## Installation

### Prerequisites
- Rust 1.75 or later
- Cargo package manager
- pkg-config
- cmake
- opus

Linux: 
```bash
apt-get install -y cmake pkg-config
```

Mac:
```bash
brew install cmake
```

```
cargo install sensevoice-cli

# or without opus(.ogg) format
cargo install sensevoice-cli --no-default-features
```

## Usage

```
SenseVoice Rust CLI (ORT + Symphonia + HF Hub)

Usage: sensevoice-cli [OPTIONS] [AUDIO]

Arguments:
  [AUDIO]  Input audio file (wav/mp3/ogg/flac/opus/vorbis)

Options:
    --models-path <MODELS_PATH>     Download/cache directory for models and resources [default: ~/.sensevoice-models]
  -t, --threads <NUM_THREADS>         Intra-op threads for ONNX Runtime [default: 1]
  -l, --language <LANGUAGE>           Language code: auto, zh, en, yue, ja, ko, nospeech [default: auto]
    --use-itn                       Use ITN post-processing
    --vad-int8                      Use int8 Silero VAD model
    --no-vad                        Disable Silero VAD segmentation
    --vad-threshold <VAD_THRESHOLD> VAD probability threshold (0.0-1.0) [default: 0.5]
    --vad-min-speech-ms <VAD_MIN_SPEECH_MS>
                     Minimum speech duration in milliseconds [default: 400]
    --vad-min-silence-ms <VAD_MIN_SILENCE_MS>
                     Minimum silence duration in milliseconds [default: 200]
    --vad-speech-pad-ms <VAD_SPEECH_PAD_MS>
                     Additional padding in milliseconds around segments [default: 120]
  --vad-merge-gap-ms <VAD_MERGE_GAP_MS>
               Merge adjacent segments separated by <= gap milliseconds [default: 1200]
    --hf-endpoint <HF_ENDPOINT>     Optional HF endpoint/mirror (overrides env HF_ENDPOINT/HF_MIRROR)
    --log <LOG>                     Log level
  -o, --output <OUTPUT>               Output JSON file path
  -c, --channels <CHANNELS>           Maximum number of audio channels to transcribe (0 = all) [default: 1]
    --download-only                 Download models only and exit
  -h, --help                          Print help
  -V, --version                       Print version
```

### Quick start

```
sensevoice-cli path/to/audio.wav
sensevoice-cli -o transcript.json path/to/audio.wav
```

Output:
```json
[
  {
    "channel": 0,
    "duration_sec": 7.152,
    "rtf": 0.019359846,
    "segments": [
      {
        "start_sec": 1.09,
        "end_sec": 3.614,
        "text": "THE DRIBL TEETHIN CALLD FOR THE BOY",
        "tags": []
      },
      {
        "start_sec": 3.842,
        "end_sec": 6.59,
        "text": "AND PRESENTED HIM WITH FIFTY PIECES OF COATD",
        "tags": []
      }
    ]
  }
]
```

- Input formats: WAV, MP3, OGG, and FLAC.
- Default output: JSON written to stdout with per-channel segments.
- Models download into `~/.sensevoice-models` on first run (override with `--models-path`).

### Handy flags

```
sensevoice-cli -l zh --use-itn -c 2 samples/demo.wav
```

- `-l/--language`: explicit language hint (`auto`, `zh`, `en`, `yue`, `ja`, `ko`, `nospeech`).
- `--use-itn`: enable inverse text normalization for cleaner numbers and dates.
- `-c/--channels`: limit the number of channels to transcribe (default 1, set 0 for all).
- `-o/--output`: write JSON to a file instead of stdout.
- `--log`: set log verbosity (e.g. `info`, `debug`).
- `--download-only`: prefetch model assets without running inference.
- `--no-vad`: bypass voice activity detection and transcribe each channel as a whole.
- `--vad-*`: tune Silero VAD behaviour (threshold, speech/silence durations, padding, merge gap) without editing code.

## Advanced tips

- Mirror-friendly downloads: add `--hf-endpoint https://hf-mirror.com` (or set `HF_ENDPOINT/HF_MIRROR`) to speed up model fetches from mainland China.
- Multi-channel aware: every audio channel is decoded separately; VAD segments are merged into a single JSON array with channel metadata.
- VAD precision: append `--vad-int8` to prefer the quantized Silero VAD model when CPU resources are limited.
- VAD controls: fine-tune segmentation with the `--vad-*` flags (threshold, speech/silence durations, padding, merge gap).
- Performance tuning: adjust `-t/--threads` to match available CPU cores. GPU execution currently requires rebuilding with CUDA-enabled ONNX Runtime.
- Session warm-up: the first run saves optimized `.ort` graphs next to the downloaded models; later runs reuse them to avoid ONNX Runtime re-optimization costs.
