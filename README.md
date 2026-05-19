# asr — live phone-call speech recognition + translation

Streaming ASR tooling that runs on **pony** (RTX 3090), built for live
phone-call transcription. samsung taps a Bluetooth call's two audio
directions and ships them as RTP to pony; pony transcribes them with
speaker labels and (for Spanish calls) an inline English translation.

The primary tool is **`asr-tui.py`**. The other `stream-*.py` scripts
are alternative ASR approaches that were evaluated along the way and
kept for reference; Voxtral was ultimately chosen for the live pipeline.

## `asr-tui.py` — the live TUI (primary tool)

A terminal UI showing live Spanish transcription with an in-place,
continuously-refining English translation, freezing finished sentences
into a scrolling speaker-tagged history.

Pipeline:

```
audio → Voxtral-Mini-4B-Realtime-2602   (vLLM /v1/realtime WS, GPU, Spanish ASR)
      → NLLB-200-distilled-600M int8     (CTranslate2, CPU, ES→EN)
```

Cascade (not direct speech-to-text-translation) because the 3090 only
has ~2.3 GB free after Voxtral; NLLB on CPU adds no GPU contention and
keeps both the Spanish transcript and the English.

Run it:

```bash
# single stream (remote party only) — audio piped in on stdin
pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
          --channels=1 - | ~/asr/asr-tui.py

# dual stream — script owns both taps, [Remote] + [Me] labelled
~/asr/asr-tui.py --dual

# headless (no TUI; prints finalized ES/EN lines) — add --plain
~/asr/asr-tui.py --dual --plain
```

In `--dual` the script spawns its own two `pw-record` subprocesses on
the canonical PipeWire sources `rtp_call_remote_source` (Remote) and
`rtp_call_me_source` (Me), and runs two independent Voxtral WS sessions
concurrently. Both speakers' in-progress blocks are shown stacked and
always present (no active-speaker switching — a silent channel's
sporadic deltas can't flip the panel). Finalized pairs interleave in
one chronological, speaker-tagged history.

A live line is finalized into history by any of four triggers:
sentence punctuation, width pressure (`clause_flush`), a speech pause
(`--pause-ms`, default 800 ms; empty/whitespace deltas do not refresh
the pause clock, so a trailed-off line still promotes when the source
goes silent), or end-of-stream.

Quit with Ctrl-C (the terminal is restored cleanly).

Requires the Voxtral vLLM server running locally on `:8000`
(`/v1/realtime`). The latency knob is `transcription_delay_ms` in the
model snapshot's `tekken.json` (read only at server start). Note: in
`--dual` two growing Voxtral sequences share one `--max-model-len`
KV-cache budget, roughly halving the per-stream context ceiling — a
long bidirectional call can approach it.

## `asr-call-transcribe` — pre-Voxtral dual-stream transcriber

The earlier faster-whisper dual-stream tool: two `pw-record` taps on
the same two sources, one thread per source, a shared Whisper model
serialized by a lock, interleaved timestamped `[Remote]`/`[Me]` lines.
Superseded by `asr-tui.py` for live use, kept as the reference design.

```bash
~/asr/asr-call-transcribe [model] [lang]      # default small.en
~/asr/asr-call-transcribe large-v3 es         # Spanish, GPU
```

## Alternative streaming scripts (evaluated, reference)

Each reads 16-bit-LE mono 16 kHz PCM on stdin and prints transcripts;
they explore different latency/quality trade-offs.

| Script | Model / engine | Note |
|---|---|---|
| `stream-voxtral.py` | Voxtral-Realtime (vLLM WS) | headless ES stream — the cascade's ASR half |
| `stream-voxtral-translate.py` | Voxtral + NLLB | headless live ES + inline EN (pre-TUI form) |
| `stream-vosk.py` | Vosk/Kaldi | true-streaming Spanish, ≈Whisper accuracy on clean audio |
| `stream-sherpa-ipa.py` | sherpa-onnx zipformer (bookbot) | true-streaming ES, IPA phonemes (no word boundaries) |
| `stream-parakeet-live.py` | parakeet-tdt-0.6b-v3 | immediate-emit, low latency |
| `stream-parakeet.py` | parakeet-tdt-0.6b-v3 + Silero VAD | VAD-chunked; `--max-sec` force-flush for pauseless speech |
| `stream-parakeet-buffered.py` | parakeet-tdt (offline) | LocalAgreement-2 sliding window |
| `stream-whisper.py` | faster-whisper-large-v3 | 99-language auto-detect |
| `stream-whisper-buffered.py` | faster-whisper | LocalAgreement-2, language-pinned |
| `stream-canary.py` | canary-1b-flash (NeMo) | single-pass ASR + translation |
| `stream-cacheaware.py` | NeMo fastconformer | true cache-aware English streaming |

## Runtime deps (not in git)

Model weights and the Python venvs are large and **gitignored**
(`models/`, `*-env/`). On pony they live in `~/asr.bak/` (the
pre-GitHub-migration working tree) and are symlinked into `~/asr/`;
script shebangs are absolute and resolve through the symlinks. A fresh
clone has no models/venvs until those symlinks (or real dirs) are in
place. The Voxtral vLLM server is launched separately, outside this
repo.

## Workflow

This repo is canonical on GitHub. Development and pushing happen on
**samsung** (`/home/claude/asr`); **pony** pulls read-only over https
(`git -C ~/asr pull`) and is the only host that *runs* the pipeline
(it has the GPU, the audio session, the venvs/models). pony has no
GitHub push credentials by design. Loop: edit on samsung → push →
pull on pony → run on pony.

---

*This repository is developed with the assistance of an AI agent
(Claude, via Claude Code) on behalf of Brent Baccala
(cosine@freesoft.org). The `asr-tui.py` Voxtral dual-stream pipeline
and the more recent streaming scripts were AI-authored across
interactive sessions and task-runner tasks; per-commit authorship and
co-authorship trailers record the provenance.*
