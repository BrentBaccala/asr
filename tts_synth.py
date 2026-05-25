#!/home/cosine/asr/tts-env/bin/python
"""tts_synth.py — long-lived Kyutai Pocket TTS synthesis sidecar.

One sidecar process owns ONE Pocket TTS model on ONE device (model load
is expensive — seconds — so the freesoft-asr tts_worker keeps a single
sidecar alive per active speak-language and pipes utterances to it). The
sidecar reads newline-delimited JSON requests on stdin and writes raw
PCM (s16le, mono, at the model's sample rate) framed on stdout, with a
JSON status line on stderr per request.

Protocol
--------
Request  (stdin, one JSON object per line):
    {"id": <int>, "text": "<utterance>"}
    {"id": <int>, "quit": true}          # shut down cleanly

Response (stdout, binary framing, one per non-empty synthesis):
    a header line  "AUDIO <id> <nbytes> <sample_rate>\n"  (ASCII)
    followed immediately by exactly <nbytes> of raw little-endian
    signed-16-bit mono PCM at <sample_rate> Hz.
On error / empty text:
    a header line  "ERR <id> <message>\n"  (no PCM follows).

Status (stderr, one JSON line per request — diagnostic, never parsed
for audio):
    {"id":..., "rtf":..., "audio_ms":..., "gen_ms":..., "samples":...}

Why this design
---------------
Pocket TTS (`generate_audio_stream`) is **block-input, streaming-output**:
its public API takes a complete `str` (beartype-guarded — a text
generator is rejected) and yields audio chunks. It does NOT expose a
drivable incremental-text frontier, and feeding successive clauses on a
shared `copy_state=False` model_state does NOT cleanly continue (each
call appends its own EOS and truncates the next). So the sidecar
synthesizes one *complete clause/sentence per request* (block input) and
streams the resulting audio out — the v0 "per-clause-block" eager-speak
granularity. The worker decides clause vs sentence granularity.

The model is loaded fresh for each request's state from the language's
default predefined voice (cached once at startup), with copy_state=True
so each utterance is independent (no cross-utterance EOS bleed).
"""
import argparse
import json
import sys

import numpy as np
import torch

from pocket_tts import TTSModel
from pocket_tts.default_parameters import get_default_voice_for_language


def _log(obj):
    sys.stderr.write(json.dumps(obj) + "\n")
    sys.stderr.flush()


# pocket-tts emits audio at 12.5 frames/s (80 ms/frame). Once the EOS
# token fires, generate_audio appends only a few trailing frames by
# default (spanish_24l sets no model_recommended_frames_after_eos, so it
# falls to a guess of ~3-5), which clips the final syllable of the spoken
# output. Force a generous tail so the last syllable fully renders. 8 is
# french_24l's own recommended value (= 640 ms). Lower toward 4-5 if a
# trailing silence/breath becomes noticeable; raise if it still clips.
FRAMES_AFTER_EOS = 8


def main():
    ap = argparse.ArgumentParser(description="Pocket TTS synthesis sidecar")
    ap.add_argument("--language", required=True,
                    help="Pocket TTS language/model id (e.g. english, "
                         "spanish_24l, french_24l, german_24l, "
                         "portuguese_24l, italian_24l)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="torch device to run the model on")
    ap.add_argument("--voice", default=None,
                    help="predefined voice name (default: the language's "
                         "built-in default voice, e.g. 'lola' for spanish)")
    ap.add_argument("--quantize", action="store_true",
                    help="apply int8 quantization (CPU speed/memory; "
                         "no effect worth it on GPU)")
    args = ap.parse_args()

    # Load the model on the requested device. load_model() returns a
    # model on CPU; .to(device) moves it (the CLI does exactly this).
    model = TTSModel.load_model(language=args.language, quantize=args.quantize)
    model.to(args.device)
    sr = int(model.sample_rate)

    voice_name = args.voice or get_default_voice_for_language(args.language)
    # Predefined-voice state: extracted once, reused (copy_state=True per
    # utterance keeps each synthesis independent).
    voice_state = model.get_state_for_audio_prompt(voice_name)

    _log({"event": "ready", "language": args.language, "device": args.device,
          "voice": voice_name, "sample_rate": sr})

    out = sys.stdout.buffer
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _log({"event": "bad_request", "error": str(e)})
            continue
        rid = req.get("id", 0)
        if req.get("quit"):
            _log({"event": "quit", "id": rid})
            break
        text = (req.get("text") or "").strip()
        if not text:
            out.write(f"ERR {rid} empty-text\n".encode())
            out.flush()
            continue
        try:
            import time
            t0 = time.monotonic()
            # generate_audio collects all chunks; we want the whole
            # utterance as one PCM blob to hand to pw-play. copy_state
            # True => each utterance independent (no EOS bleed).
            audio = model.generate_audio(
                model_state=voice_state, text_to_generate=text,
                frames_after_eos=FRAMES_AFTER_EOS, copy_state=True)
            gen_ms = (time.monotonic() - t0) * 1000.0
            # audio is a float tensor in [-1, 1], shape [samples] (the
            # short-text path yields [samples]; generate_audio cats on
            # dim=0). Convert to s16le.
            wav = audio.detach().to("cpu").float().numpy().reshape(-1)
            pcm = np.clip(wav, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
            out.write(f"AUDIO {rid} {len(pcm16)} {sr}\n".encode())
            out.write(pcm16)
            out.flush()
            audio_ms = len(wav) * 1000.0 / sr
            _log({"event": "synth", "id": rid, "samples": len(wav),
                  "audio_ms": round(audio_ms, 1), "gen_ms": round(gen_ms, 1),
                  "rtf": round(audio_ms / gen_ms, 2) if gen_ms else None})
        except Exception as e:
            out.write(f"ERR {rid} {str(e)[:120]}\n".encode())
            out.flush()
            _log({"event": "synth_error", "id": rid, "error": str(e)[:200]})


if __name__ == "__main__":
    main()
