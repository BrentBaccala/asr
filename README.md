# asr — streaming speech recognition on pony

Live-mic ASR pipeline running on pony's RTX 3090. Part of a larger
phone-call transcription project (see
`~/project/docs/phone-asr-pipeline-design.md` on samsung for the
broader design — Bluetooth antenna for the BCM20702 is the
hardware blocker for the actual phone-audio-routing half).

## What's here

Three sibling scripts, each pairing one speech model with a Silero
VAD chunker. Choose by trade-off, not by absolute quality — all
three transcribe English well enough to be useful.

| Script | Model | Engine | Languages | When to pick |
|---|---|---|---|---|
| `asr-stream-whisper.py` | `Systran/faster-whisper-large-v3` | CTranslate2 | 99 (auto-detect) | Best multilingual coverage; lightest runtime |
| `asr-stream-parakeet.py` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo (PyTorch) | 25 (auto-detect, en/es/fr/de/+) | Fastest inference (~1 s on 6 s audio); current best multilingual NeMo |
| `asr-stream-canary.py` | `nvidia/canary-1b-flash` | NeMo (PyTorch) | en/es/fr/de | Single-pass ASR + translation in one model (`--task ast --source-lang es --target-lang en`) |

Common shape: read raw 16-bit-LE mono PCM from stdin at 16 kHz,
chunk on Silero VAD utterance boundaries (500 ms silence
trigger), run each utterance through the model, print one line
per utterance with timing tags. ^C to stop.

## How to run

```bash
ssh claude@pony 'arecord -q -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 2>/dev/null \
  | ~/asr/asr-stream-parakeet.py'
```

Substitute any of the three scripts. Each script's shebang
absolute-paths to its venv; `chmod +x` is already set, so no
explicit interpreter on the command line.

The pony-side mic is a Blue Snowball USB condenser plugged into
`hw:Snowball,0`. claude is in the `audio` group and goes through
ALSA directly — does not contend with cosine's pipewire (which
is the active audio session on pony for everything else).

## Two venvs because of Python 3.14

| Venv | Python | Purpose |
|---|---|---|
| `asr-env/` | 3.14.4 (system) | faster-whisper stack: ctranslate2, silero-vad, onnxruntime |
| `asr-env-canary/` | 3.12.13 (uv-installed) | NeMo stack: torch 2.9.1+cu128, nemo_toolkit[asr], silero-vad, onnxruntime, torchaudio |

Pony runs Ubuntu 26.04 (Resolute Raccoon) which ships only
Python 3.14. NeMo's released wheels target 3.10–3.12; on 3.14
several of NeMo's deps fail to resolve. Workaround: `uv python
install 3.12` pulls a standalone CPython 3.12 build (lives at
`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/`) and
the canary/parakeet venv points at it.

faster-whisper installs cleanly on 3.14 — no need to consolidate.
The two venvs total ~7 GB on disk and don't conflict.

## System-wide deps (apt-installed)

`libcublas12` and `nvidia-cudnn` from Ubuntu multiverse — needed
because **ctranslate2 looks up CUDA libraries on the system loader
path** (it doesn't bundle them, unlike PyTorch / NeMo / ollama
which all ship their own bundled CUDA libs in their package trees).
The install dragged in the full CUDA 12.4 dev toolkit (~5 GB) as
a transitive dep of the `nvidia-cudnn` install-script package, but
that's also useful for any future from-source GPU builds (llama.cpp,
custom CUDA kernels, etc.).

NVIDIA's `ubuntu2604` apt repo is added (via `cuda-keyring`) but
empty for our needs — it ships only CUDA 13.x packages, and
ctranslate2 is built against CUDA 12. Not currently used; can be
removed if it ever causes friction.

## Model storage

All model weights live in the shared HuggingFace cache at
`~/.cache/huggingface/hub/`. Currently ~23 GB across:

- `Systran/faster-whisper-large-v3` (~3 GB)
- `Helsinki-NLP/opus-mt-es-en` (~300 MB) — translator, currently
  unused but downloaded for a possible Whisper+Marian two-hop path
- `nvidia/canary-1b-flash` (~3 GB)
- `nvidia/canary-1b-v2` (~3 GB)
- `nvidia/parakeet-tdt-0.6b-v3` (~6 GB — ships both `.nemo` and
  HF safetensors)
- `nvidia/multitalker-parakeet-streaming-0.6b-v1` (~5 GB —
  downloaded for streaming research; not yet wired up)

Both venvs see the same cache (HF default location).

## TODO — `asr-stream-multitalker.py`

The multitalker streaming model is downloaded but no script yet
because the integration is significantly heavier than a drop-in
swap:

- Requires a *second* model (`nvidia/diar_streaming_sortformer_4spk-v2.1`,
  not yet downloaded) running in parallel for streaming diarization
- Per-speaker model instances: 1 ASR copy per speaker, so 2-speaker
  call = 2× the 0.6B model in VRAM
- Streaming inference uses `CacheAwareStreamingAudioBuffer` +
  `SpeakerTaggedASR` orchestrator from NeMo, designed for
  pre-recorded files; adapting to a live arecord stream + extracting
  transcripts mid-stream (rather than at end-of-stream) is non-trivial
- Reference implementation:
  https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py

For 1:1 phone calls where the speaker identity is known by routing,
diarization adds little. Lighter alternatives if the goal is just
sub-second latency:

- `nvidia/parakeet_realtime_eou_120m-v1` — streaming-native, 80–160 ms
  latency, self-detects end-of-utterance via `<EOU>` token. English
  only. Drops VAD entirely. Smallest engineering lift among the
  streaming-native options.
- Tighter VAD threshold (`min_silence_duration_ms=200`) on any of the
  existing scripts gives snappier perceived response without
  architectural changes.

## Tested

- Smoke-tested all three scripts post-move (`script < /dev/null`):
  models load, silero-vad initializes, ready-to-listen banner prints.
- Live mic input has been confirmed working on `asr-stream-whisper.py`
  earlier in development (pre-rename / pre-move). The other two
  haven't been tested with real speech yet.
- `/tmp/out.wav` (a 5.8-second 8 kHz BBB IVR clip) transcribes
  identically (ignoring brand-name capitalization differences) on
  all four models tried so far: faster-whisper-large-v3,
  canary-1b-flash, parakeet-tdt-0.6b-v3, and
  multitalker-parakeet-streaming-0.6b-v1 (via the offline transcribe
  interface, not its streaming API).
