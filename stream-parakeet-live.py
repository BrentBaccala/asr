#!/home/cosine/venv-3.12-torch/bin/python
"""
Low-latency streaming transcription with Parakeet TDT v3 (multilingual
incl. Spanish), clean append-only output.

Pipe raw PCM in:
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-parakeet-live.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

parakeet-tdt-0.6b-v3 has no cache-aware streaming mode, and no
cache-aware Spanish/multilingual model exists (May 2026). This script
runs the offline model on a rolling window every --chunk-sec and emits
only words whose end is older than the --commit-sec horizon: those are
stable, so they are printed once, append-only, and never revised. The
unstable trailing edge is deliberately NOT shown -- displaying it (an
earlier design) both required carriage-return overwrite (which corrupts
once a line wraps in the terminal) and surfaced parakeet's worst
sliding-window artifacts (duplicated / dropped words). Output is
therefore wrap-safe and clean; latency is ~commit-sec.

Spanish is auto-detected (parakeet-tdt-0.6b-v3 has no language input and
cannot be hard-locked); on short / ambiguous windows it can briefly flip
to English. For stable Spanish at lower latency consider stream-vosk.py
(true streaming, ~whisper accuracy on clean audio); for zero-flip
Spanish use stream-whisper-buffered.py.

Options:
  --model REPO         HF repo id (default nvidia/parakeet-tdt-0.6b-v3)
  --chunk-sec S        re-transcribe cadence (default 0.5)
  --commit-sec S       words whose end is older than (window_end -
                        commit-sec) are emitted as final (default 1.5).
                        This is the latency floor; lower = snappier but
                        more end-of-window errors.
  --max-context-sec S  rolling window hard cap (default 8.0)
  --silence-rms R      whole-window RMS below this = silence; the line is
                        ended and the window reset (default 0.004)
"""
import sys
import argparse
import warnings
import logging
import numpy as np

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.utils import logging as _nemo_logging

_nemo_logging.setLevel(logging.ERROR)

SAMPLE_RATE = 16000

p = argparse.ArgumentParser()
p.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
p.add_argument("--chunk-sec", type=float, default=0.5)
p.add_argument("--commit-sec", type=float, default=1.5)
p.add_argument("--max-context-sec", type=float, default=8.0)
p.add_argument("--silence-rms", type=float, default=0.004)
args = p.parse_args()

CHUNK_BYTES = int(args.chunk_sec * SAMPLE_RATE) * 2
MAX_SAMPLES = int(args.max_context_sec * SAMPLE_RATE)

print(f"loading {args.model}...", file=sys.stderr, flush=True)
asr = nemo_asr.models.ASRModel.from_pretrained(args.model)
asr = asr.eval().to("cuda")
print(f"ready, listening... (chunk-sec={args.chunk_sec}, "
      f"commit-sec={args.commit_sec})", file=sys.stderr, flush=True)


def transcribe(buf):
    out = asr.transcribe([buf], batch_size=1, timestamps=True, verbose=False)
    if not out:
        return []
    h = out[0]
    return [(w["word"], float(w["start"]), float(w["end"]))
            for w in (h.timestamp or {}).get("word", [])]


def read_chunk():
    b = b""
    while len(b) < CHUNK_BYTES:
        piece = sys.stdin.buffer.read(CHUNK_BYTES - len(b))
        if not piece:
            return b
        b += piece
    return b


audio_buf = np.zeros(0, dtype=np.float32)
buf_start_t = 0.0          # absolute time of audio_buf[0]
total_consumed = 0
committed_end_t = 0.0       # abs end time of last emitted word
at_bol = True               # at beginning of an output line


def emit(words):
    """Append finalized words to the current line (space-separated, no
    carriage return, so terminal line-wrap is harmless)."""
    global at_bol
    if not words:
        return
    txt = " ".join(w[0] for w in words)
    sys.stdout.write(txt if at_bol else " " + txt)
    sys.stdout.flush()
    at_bol = False


def newline():
    global at_bol
    if not at_bol:
        sys.stdout.write("\n")
        sys.stdout.flush()
        at_bol = True


eof = False
while not eof:
    raw = read_chunk()
    if len(raw) < CHUNK_BYTES:
        eof = True
        if not raw:
            break
    chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total_consumed += len(chunk)
    audio_buf = np.concatenate([audio_buf, chunk])
    window_end_t = total_consumed / SAMPLE_RATE

    rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size else 0.0
    if rms < args.silence_rms and not eof:
        newline()
        audio_buf = np.zeros(0, dtype=np.float32)
        buf_start_t = window_end_t
        committed_end_t = max(committed_end_t, window_end_t)
        continue

    words = transcribe(audio_buf)
    words = [(w, s + buf_start_t, e + buf_start_t) for (w, s, e) in words]

    horizon = window_end_t if eof else (window_end_t - args.commit_sec)
    new_final = [w for w in words
                 if w[2] <= horizon and w[2] > committed_end_t + 1e-3]
    if new_final:
        emit(new_final)
        committed_end_t = new_final[-1][2]

    if eof:
        break

    # Trim the buffer up to the last finalized word; hard-cap the window.
    drop = int((committed_end_t - buf_start_t) * SAMPLE_RATE)
    if len(audio_buf) - drop > MAX_SAMPLES:
        drop = len(audio_buf) - MAX_SAMPLES
    if drop > 0:
        audio_buf = audio_buf[drop:]
        buf_start_t += drop / SAMPLE_RATE

newline()
print("[stream-parakeet-live] bye.", file=sys.stderr)
