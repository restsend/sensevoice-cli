# SenseVoice CLI

A lightweight command-line front end for the SenseVoice multilingual speech recognition model.

## Installation
```
cargo install sensevoice-cli
```

## Basic Usage

```
sensevoice-cli path/to/audio.wav
sensevoice-cli -o transcript.json path/to/audio.wav
```

Outout:
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
- Models download into `~/.sensevoice-models` on first run (override with `-d/--models_path`).

### Handy Flags

```
sensevoice-cli -l zh -i -c 2 samples/demo.wav
```

- `-l/--language`: explicit language hint (`auto`, `zh`, `en`, `yue`, `ja`, `ko`, `nospeech`).
- `--use_itn`: enable inverse text normalization for cleaner numbers and dates.
- `-c/--channels`: limit the number of channels to transcribe (default 1, set 0 for all).
- `-o/--output`: write JSON to a file instead of stdout.

## Advanced Tips

- Mirror friendly downloads: add `--hf-endpoint https://hf-mirror.com` (or set `HF_ENDPOINT/HF_MIRROR`) to speed up model fetches from mainland China.
- Multi-channel aware: every audio channel is decoded separately; VAD segments are merged into a single JSON array with channel metadata.
- VAD precision: append `--vad-int8` to prefer the quantized Silero VAD model when CPU resources are limited.
- Performance tuning: adjust `-t/--num_threads` to match available CPU cores; set `--device` to a CUDA ID (`0`, `1`, ...) for GPU inference when ONNX Runtime is built with CUDA.
