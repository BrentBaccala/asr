#!/home/cosine/asr/piper-env/bin/python
"""tts_synth_piper.py — Piper TTS synthesis sidecar.

Drop-in alternative to tts_synth.py (the pocket-tts sidecar), speaking the
SAME stdin/stdout protocol so freesoft-asr's tts_worker can drive either:

Request  (stdin, one JSON object per line):
    {"id": <int>, "text": "<utterance>"}
    {"id": <int>, "quit": true}

Response (stdout):
    "AUDIO <id> <nbytes> <sample_rate>\n" + <nbytes> raw s16le mono PCM
    "ERR <id> <message>\n"               on error / empty text

Why Piper: it's a deterministic VITS model — reliable onsets, no first-word
dropouts. pocket-tts's Spanish previews (spanish / spanish_24l) drop or
mangle the first word ~50% of the time on short utterances, temperature-
independent (measured). Piper is block-input and fast on CPU, which is fine
for the interpreter's per-clause-block granularity.

The voice is an .onnx model id (looked up under ~/asr/piper-voices) or an
explicit .onnx path. Sample rate comes from the voice config (22050 for the
es_ES medium voices); it is reported in the AUDIO header so freesoft-asr's
pw-play uses the right rate.
"""
import argparse
import json
import os
import sys
import time

from piper import PiperVoice

VOICES_DIR = os.path.expanduser("~/asr/piper-voices")


def _log(obj):
    sys.stderr.write(json.dumps(obj) + "\n")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description="Piper TTS synthesis sidecar")
    ap.add_argument("--voice", required=True,
                    help="Piper voice id (e.g. es_ES-sharvard-medium, looked "
                         "up as <id>.onnx under ~/asr/piper-voices) or an "
                         "explicit path to a .onnx model.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda needs onnxruntime-gpu; CPU is fast + the "
                         "default (Piper is real-time+ on CPU).")
    args = ap.parse_args()

    model_path = (args.voice if args.voice.endswith(".onnx")
                  else os.path.join(VOICES_DIR, args.voice + ".onnx"))
    voice = PiperVoice.load(model_path, use_cuda=(args.device == "cuda"))
    sr = int(voice.config.sample_rate)
    _log({"event": "ready", "voice": args.voice, "device": args.device,
          "sample_rate": sr})

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
            t0 = time.monotonic()
            # synthesize() yields AudioChunk objects; concatenate their
            # raw s16le PCM into one blob for pw-play.
            pcm = b"".join(ch.audio_int16_bytes for ch in voice.synthesize(text))
            gen_ms = (time.monotonic() - t0) * 1000.0
            out.write(f"AUDIO {rid} {len(pcm)} {sr}\n".encode())
            out.write(pcm)
            out.flush()
            audio_ms = (len(pcm) / 2) * 1000.0 / sr
            _log({"event": "synth", "id": rid, "audio_ms": round(audio_ms, 1),
                  "gen_ms": round(gen_ms, 1),
                  "rtf": round(audio_ms / gen_ms, 2) if gen_ms else None})
        except Exception as e:
            out.write(f"ERR {rid} {str(e)[:120]}\n".encode())
            out.flush()
            _log({"event": "synth_error", "id": rid, "error": str(e)[:200]})


if __name__ == "__main__":
    main()
