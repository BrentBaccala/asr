#!/home/cosine/asr/vosk-env/bin/python
"""
True frame-synchronous streaming Spanish ASR with Vosk (Kaldi).

Pipe raw PCM in:
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-vosk.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

Unlike the sliding-window parakeet/whisper scripts, Vosk is a genuinely
streaming online recognizer: it emits a growing partial hypothesis as
audio arrives (no recompute, no confirmation window), so latency is the
model's intrinsic decoding lag (~0.1-0.3 s). Accuracy is below
parakeet-tdt/whisper, especially on telephone-band audio, but it
outputs Spanish words directly and is the lowest-latency local option.

Emits, on every partial change, a line tagged with the audio start time
of the first not-yet-final word (so latency tooling can compare wall vs
audio time, same format as stream-parakeet-live.py). Final segments are
newline-committed.

Options:
  --model PATH    vosk model dir (default ~/asr/models/vosk-model-es-0.42)
  --read-ms MS    stdin read granularity (default 120)
"""
import sys
import os
import json
import argparse

from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)
SAMPLE_RATE = 16000

ap = argparse.ArgumentParser()
ap.add_argument("--model",
                default=os.path.expanduser("~/asr/models/vosk-model-es-0.42"))
ap.add_argument("--read-ms", type=int, default=120)
args = ap.parse_args()

print(f"loading {args.model}...", file=sys.stderr, flush=True)
model = Model(args.model)
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)
rec.SetPartialWords(True)
print("ready, listening...", file=sys.stderr, flush=True)

read_bytes = int(SAMPLE_RATE * args.read_ms / 1000) * 2
shown = ""


def show(ts, text):
    global shown
    if text and text != shown:
        line = f"[{ts:6.1f}s] {text}"
        sys.stdout.write("\r" + line + " " * max(0, len(shown) - len(line)))
        sys.stdout.flush()
        shown = line


def commit(ts, text):
    global shown
    if text:
        sys.stdout.write("\r" + f"[{ts:6.1f}s] {text}" + "\n")
        sys.stdout.flush()
    shown = ""


while True:
    data = sys.stdin.buffer.read(read_bytes)
    if not data:
        break
    if rec.AcceptWaveform(bytes(data)):
        r = json.loads(rec.Result())
        words = r.get("result", [])
        if r.get("text"):
            ts = words[-1]["end"] if words else 0.0
            commit(ts, r["text"])
    else:
        pr = json.loads(rec.PartialResult())
        pw = pr.get("partial_result", [])
        if pr.get("partial"):
            ts = pw[-1]["end"] if pw else 0.0
            show(ts, pr["partial"])

fr = json.loads(rec.FinalResult())
fw = fr.get("result", [])
if fr.get("text"):
    commit(fw[0]["start"] if fw else 0.0, fr["text"])
print("[stream-vosk] bye.", file=sys.stderr)
