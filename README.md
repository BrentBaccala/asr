# asr — live streaming speech recognition with inline translation

A terminal UI (`asr-tui.py`) that transcribes audio in real time using
**Voxtral-Mini-4B-Realtime** (served by vLLM, GPU) and translates the
transcription into **Spanish and English** with
**NLLB-200-distilled-600M** (CTranslate2 int8, CPU). Designed for live
two-channel phone-call transcription with separately-labelled
`[Remote]` and `[Me]` streams, but the single-stream path works with
any local or piped audio source.

Three views per channel render in the live panel:

```
[Remote] Live ▸  raw Voxtral output (any language; possibly code-switched)
         ES   ▸  NLLB-cleaned Spanish translation (masked-tail live preview)
         EN   ▸  NLLB English translation         (masked-tail live preview)
```

Finalized sentences interleave into a scrolling speaker-tagged history
below. Sentence-level *marker-MT* (NLLB receives the whole accumulated
Spanish with `[1] [2] [3]` markers between visual chunks) keeps the
per-chunk translations coherent and aligned, rather than fragmented
into context-free pieces.

## Pipeline

```
audio (S16LE 16 kHz mono)
   → Voxtral-Mini-4B-Realtime-2602    (vLLM /v1/realtime WS, GPU)   ─┐
                                                                      │ raw text deltas
   ← cur_live (raw, possibly code-switched) ← ← ← ← ← ← ← ← ← ← ← ← ←┘
   → NLLB-200-distilled-600M int8     (CTranslate2, CPU, batched)
                                       target_prefix=[[spa],[eng]]
   → live preview: cur_es, cur_en (masked tail)
   → marker-MT at sentence boundaries → chunk-aligned ES/EN backfill
                                       into the scrolling history pane
```

The cascade (rather than direct speech-to-text-translation) is
deliberate: it keeps NLLB entirely on CPU so all the GPU goes to
Voxtral, and the raw `Live` text stays visible alongside the cleaned
ES/EN.

## Features

- **Three-line live region per stream**; interleaved speaker-tagged
  history.
- **Whole-sentence marker-MT** — chunks become visible as they're
  spoken (`⋯` placeholder), then backfill with proper chunk-aligned
  translation when the sentence completes. No more `elocuencia` →
  `I'm not a good speaker` from context-free fragments.
- **Dual-stream concurrent transcription** (`--dual`) — two
  independent vLLM sessions, separately auto-recycled per stream.
- **Scrollback** — mouse wheel + ↑/↓/PgUp/PgDn/Home/End on the
  history pane. Live region keeps updating regardless.
- **Ctrl-L** clears the history and recycles the WS sessions (frees
  the vLLM KV cache for both streams immediately).
- **Ctrl-W** writes the entire running transcript to
  `DDMonYYYY-HHMM.txt` in the current directory (a same-minute repeat
  gets a `-2`/`-3` suffix) and drops a dim "transcript written …"
  marker into the history at that point.
- **VAD-driven auto session recycle** (silero-vad) — each stream's
  WS closes/reopens during natural silences, bounding its session
  length to speaker-continuous-talk-time rather than the call's
  duration. Avoids the vLLM `--max-model-len` ceiling that otherwise
  crashes the EngineCore around 22 minutes of continuous audio per
  session.
- **`--plain`** headless mode for logging and validation.

## Running

### Single-stream (any audio source piped to stdin)

```bash
your_audio_source | ./asr-tui.py
```

The audio must be **S16LE mono at 16 kHz**. With PipeWire:

```bash
pw-record --target <your-source> --format=s16 --rate=16000 \
          --channels=1 - | ./asr-tui.py
```

With ALSA / arecord:

```bash
arecord -f S16_LE -r 16000 -c 1 -D <your-device> | ./asr-tui.py
```

From a file:

```bash
sox input.wav -t raw -r 16000 -c 1 -b 16 -e signed-integer - | ./asr-tui.py
```

### Dual-stream

```bash
./asr-tui.py --dual
```

This spawns its own `pw-record` subprocesses targeting the PipeWire
sources **`rtp_call_remote_source`** and **`rtp_call_me_source`**, and
opens two independent Voxtral WS sessions concurrently. The source
names are hard-wired in the script's `DUAL_SOURCES` constant (near
the top); you can either configure PipeWire to expose your sources
under those names (see below), or edit the constant for your setup.

### Headless / `--plain`

```bash
./asr-tui.py [--dual] --plain
```

Skips the alt-screen TUI and prints `[spk] Live/ES/EN` triples on
stdout per finalized chunk. Useful for piping into a logger.

### CLI knobs

Run `./asr-tui.py --help` for the full list with current defaults.
Highlights:

- `--pause-ms` / `--sentence-close-ms` — two-tier pause behaviour
  (short pause flushes a visual chunk; long pause closes the sentence
  and triggers marker-MT).
- `--mask` — words to hide off the live preview tail (model tail
  tokens are unstable until more context arrives).
- `--beam` — NLLB beam-search width (1 = greedy, default).
- `--auto-recycle-silence-ms` / `--auto-recycle-min-s` /
  `--auto-recycle-backstop-s` — when the WS auto-recycle fires.
- `--no-line-flush` — disable the last-resort `clause_flush` width
  break entirely.

## Requirements

- **vLLM serving Voxtral-Mini-4B-Realtime** on
  `127.0.0.1:8000/v1/realtime` (host/port configurable at the top of
  `asr-tui.py`).
- An **NLLB-200-distilled-600M int8 CTranslate2** model at
  `./models/nllb-600m-ct2/`.
- A **CUDA GPU** with ≥ 16 GB VRAM (Voxtral weights + KV cache at
  `--max-model-len 16384`).
- **PipeWire** (Linux) for the `--dual` audio capture; single-stream
  mode works with any audio source piped to stdin (PipeWire, ALSA,
  sox, ffmpeg, etc.).

End-to-end setup — venvs, model conversion, the optional vLLM systemd
unit, and recreating the alternative venvs for the reference scripts
— is in **[INSTALL.md](INSTALL.md)**.

## PipeWire setup for the dual-stream RTP path

The dual-stream design assumes the two audio channels arrive as
**RTP streams** from a sender host — e.g. a desktop with a phone
paired over Bluetooth that taps the analog sink monitor (remote-party
voice) and the analog mic capture (local voice), shipping each as a
UDP RTP stream to the receiver running `asr-tui.py`. This makes the
transcribing machine independent of where the audio source actually
lives (and lets you run the GPU on a separate, more powerful host).

### On the receiver (the machine running `asr-tui.py --dual`)

Drop this into
`~/.config/pipewire/pipewire.conf.d/91-rtp-call-stream-source.conf`:

```pipewire
context.modules = [
    {   name = libpipewire-module-rtp-source
        args = {
            source.ip         = "0.0.0.0"
            source.port       = 46000
            sess.latency.msec = 300
            stream.props = {
                node.name        = "rtp_call_remote_source"
                node.description = "RTP <- remote party"
                media.class      = "Audio/Source"
                audio.format     = "S16LE"
                audio.rate       = 48000
                audio.channels   = 2
                audio.position   = [ FL FR ]
            }
        }
    }

    {   name = libpipewire-module-rtp-source
        args = {
            source.ip         = "0.0.0.0"
            source.port       = 46002
            sess.latency.msec = 300
            stream.props = {
                node.name        = "rtp_call_me_source"
                node.description = "RTP <- local mic"
                media.class      = "Audio/Source"
                audio.format     = "S16LE"
                audio.rate       = 48000
                audio.channels   = 2
                audio.position   = [ FL FR ]
            }
        }
    }
]
```

Then restart PipeWire (`systemctl --user restart pipewire wireplumber
pipewire-pulse`) or log out and back in. Verify with `pw-cli ls Node |
grep rtp_call_` — both sources should appear. `asr-tui.py --dual`
will then find them by name and start streaming.

`pw-record`'s own resampler converts 48 kHz stereo down to 16 kHz
mono on the consumer side, so the RTP wire format stays at the
loopback's native 48 kHz stereo.

### On the sender (the machine producing the audio)

The sender combines `libpipewire-module-rtp-sink` (the UDP egress)
with `libpipewire-module-loopback` (routing specific PipeWire devices
into each sink). Sketch — for a setup with a Bluetooth-paired phone
playing audio through the sender's analog sink:

```pipewire
context.modules = [
    # ---- Remote party: capture the analog sink's monitor ----
    {   name = libpipewire-module-rtp-sink
        args = {
            destination.ip   = "<receiver-IP>"
            destination.port = 46000
            stream.props = {
                node.name      = "rtp_call_remote_sink"
                media.class    = "Audio/Sink"
                audio.format   = "S16LE"
                audio.rate     = 48000
                audio.channels = 2
                audio.position = [ FL FR ]
            }
        }
    }
    {   name = libpipewire-module-loopback
        args = {
            capture.props = {
                stream.capture.sink = true
                target.object       = "<your-analog-sink-node-name>"
                audio.rate          = 48000
                audio.channels      = 2
            }
            playback.props = {
                target.object = "rtp_call_remote_sink"
                audio.rate    = 48000
                audio.channels = 2
            }
        }
    }

    # ---- Local mic: capture the mic device ----
    {   name = libpipewire-module-rtp-sink
        args = {
            destination.ip   = "<receiver-IP>"
            destination.port = 46002
            stream.props = { node.name = "rtp_call_me_sink"; ... }
        }
    }
    {   name = libpipewire-module-loopback
        args = {
            capture.props  = {
                target.object = "<your-analog-mic-node-name>"
                audio.rate = 48000; audio.channels = 2
            }
            playback.props = {
                target.object = "rtp_call_me_sink"
                audio.rate = 48000; audio.channels = 2
            }
        }
    }
]
```

Two gotchas learned the hard way:

- **Loopback identity rule.** `module-loopback` must use the **same**
  `audio.rate` and `audio.channels` on both `capture.props` and
  `playback.props`. Any mismatch silently triggers an
  auto-resample/downmix path that attenuates the signal by ~25 dB —
  the loopback shows as `[active]` in `wpctl status` but the audio
  on the playback side is 15× too quiet. Keep loopbacks as identity
  passthrough; do rate/channel conversion at the consumer.
- **`.monitor` suffix gotcha.** `target.object =
  "<sink-name>.monitor"` does **not** match a real node in PipeWire
  1.0.5 — the session manager treats it as "target not found" and
  falls back to the default source (usually the mic). To tap a
  sink's monitor, set `stream.capture.sink = true` on `capture.props`
  instead.

The RTP overlay is *fire-and-forget UDP*. If the receiver is down,
the sender's audio path is unaffected and the packets are discarded.
When the receiver comes back up, transcription resumes without any
sender-side action.

## Same-host / local-audio variants

If your audio is already on the same machine as Voxtral (no RTP
needed), you have two options:

- **Single-stream mode** — pipe your audio in on stdin:
  ```bash
  pw-record --target <your-source-or-monitor> --format=s16 \
            --rate=16000 --channels=1 - | ./asr-tui.py
  ```
  The 3-line `Live`/`ES`/`EN` TUI works the same; you just have one
  stream instead of two.
- **Dual-stream mode with local sources** — either configure PipeWire
  to expose your two sources under the canonical names
  (`rtp_call_remote_source` and `rtp_call_me_source`), or edit
  `DUAL_SOURCES` in `asr-tui.py` to point at your local source names.

## Alternative streaming scripts (in this repo for reference)

The repo also contains several standalone `stream-*.py` scripts that
explore different ASR engines and latency/quality trade-offs. They
were evaluated during the design of `asr-tui.py`; Voxtral was chosen
for the final TUI. Each is self-contained — reads raw 16-bit-LE mono
16 kHz PCM on stdin and prints transcripts on stdout.

| Script | Model / engine | Notes |
|---|---|---|
| `stream-voxtral.py` | Voxtral-Realtime (vLLM WS) | headless ES stream (the ASR half of the cascade) |
| `stream-voxtral-translate.py` | Voxtral + NLLB | headless live ES + inline EN (pre-TUI form) |
| `stream-vosk.py` | Vosk/Kaldi | true-streaming Spanish, ≈Whisper accuracy on clean audio |
| `stream-sherpa-ipa.py` | sherpa-onnx zipformer | streaming ES in IPA phonemes (no word boundaries) |
| `stream-parakeet-live.py` | parakeet-tdt-0.6b-v3 | immediate-emit, low latency |
| `stream-parakeet.py` | parakeet-tdt-0.6b-v3 + Silero VAD | VAD-chunked + `--max-sec` force-flush for pauseless speech |
| `stream-parakeet-buffered.py` | parakeet-tdt (offline) | LocalAgreement-2 sliding window |
| `stream-whisper.py` | faster-whisper-large-v3 | 99-language auto-detect |
| `stream-whisper-buffered.py` | faster-whisper | LocalAgreement-2, language-pinned |
| `stream-canary.py` | canary-1b-flash (NeMo) | single-pass ASR + translation |
| `stream-cacheaware.py` | NeMo fastconformer | true cache-aware English streaming |

`asr-call-transcribe` is a faster-whisper dual-stream transcriber
that predates `asr-tui.py --dual`. Same `rtp_call_remote_source` /
`rtp_call_me_source` audio-input architecture but with whisper
instead of Voxtral + NLLB. Kept as the reference design for the
two-source plumbing.

Setup commands for recreating any of these venvs and downloading the
models they need are in **[INSTALL.md](INSTALL.md#recreating-the-alternative-venvs-and-models)**.

---

*This repository is developed with the assistance of an AI agent
(Claude, via Claude Code) on behalf of Brent Baccala
(cosine@freesoft.org). Per-commit authorship and co-authorship
trailers record the provenance.*
