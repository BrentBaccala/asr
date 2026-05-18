#!/home/cosine/asr/vosk-env/bin/python
"""
True frame-synchronous streaming Spanish ASR with Vosk (Kaldi).

Pipe raw PCM in:
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-vosk.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

Vosk is a genuinely streaming online recognizer: it emits a growing
partial hypothesis as audio arrives (no recompute, no confirmation
window). Output is delta-printed -- only newly appended text is
written, and a finalized segment ends the line with a newline. This is
terminal-wrap safe (no carriage-return overwrite, which breaks once a
line wraps) and matches the cache-aware English script's UX. On the
rare non-prefix tail revision the line is reprinted fresh.

Accuracy is below parakeet-tdt/whisper on telephone-band audio but on
clean speech is close to whisper-large-v3; outputs Spanish words
directly and is the lowest-latency *stable* local option (CPU-only;
Vosk has no GPU path).

Options:
  --model PATH    vosk model dir (default ~/asr/models/vosk-model-es-0.42;
                  pass ~/asr/models/vosk-model-small-es-0.42 for lower
                  latency at some accuracy cost)
  --read-ms MS    stdin read granularity (default 120)
  --timestamps    append [start-end s] to each finalized line
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
ap.add_argument("--timestamps", action="store_true",
                help="append [start-end s] to each finalized line")
args = ap.parse_args()

print(f"loading {args.model}...", file=sys.stderr, flush=True)
model = Model(args.model)
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)
rec.SetPartialWords(True)
print("ready, listening...", file=sys.stderr, flush=True)

read_bytes = int(SAMPLE_RATE * args.read_ms / 1000) * 2
shown = ""  # text already written on the current (unterminated) line


def stream(text):
    """Print `text` as the current line incrementally: just the new
    suffix when it extends what's shown, else a newline + full reprint.
    Never uses carriage return, so terminal line-wrap is harmless."""
    global shown
    if text == shown:
        return
    if text.startswith(shown):
        sys.stdout.write(text[len(shown):])
    else:
        sys.stdout.write("\n" + text)
    sys.stdout.flush()
    shown = text


def endline(text, span):
    """Finalize the current segment: flush any remaining suffix, then a
    newline (optionally a timestamp), and reset the line."""
    global shown
    if not text:
        return
    if text != shown:
        if text.startswith(shown):
            sys.stdout.write(text[len(shown):])
        else:
            sys.stdout.write(("\n" if shown else "") + text)
    if args.timestamps and span is not None:
        sys.stdout.write(f"   [{span[0]:.1f}-{span[1]:.1f}s]")
    sys.stdout.write("\n")
    sys.stdout.flush()
    shown = ""


while True:
    data = sys.stdin.buffer.read(read_bytes)
    if not data:
        break
    if rec.AcceptWaveform(bytes(data)):
        r = json.loads(rec.Result())
        w = r.get("result", [])
        if r.get("text"):
            endline(r["text"], (w[0]["start"], w[-1]["end"]) if w else None)
    else:
        pr = json.loads(rec.PartialResult())
        if pr.get("partial"):
            stream(pr["partial"])

fr = json.loads(rec.FinalResult())
fw = fr.get("result", [])
if fr.get("text"):
    endline(fr["text"], (fw[0]["start"], fw[-1]["end"]) if fw else None)
print("[stream-vosk] bye.", file=sys.stderr)
