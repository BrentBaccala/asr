#!/home/cosine/venv-3.12-torch/bin/python
"""
Low-latency buffered streaming transcription with Parakeet TDT v3.

Pipe raw PCM in:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | stream-parakeet-buffered.py
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-parakeet-buffered.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

parakeet-tdt-0.6b-v3 is a *full-context offline* model (it has no
cache-aware streaming mode). To get low latency from it on continuous
speech we run it on an overlapping sliding window every --chunk-sec
seconds and commit words with a LocalAgreement-2 policy: a word is
printed only once two successive overlapping transcriptions agree on
it, so the unstable trailing edge never reaches the screen. The audio
buffer is trimmed at the last committed word (using parakeet's word
timestamps), bounding both latency and recompute.

This is the live-stream form of NeMo's buffered RNNT inference
(NeMo's own speech_to_text_buffered_infer_rnnt is file-oriented).
It keeps Spanish (auto-detected, 25 languages) and emits text roughly
every --chunk-sec instead of waiting for a pause. Not "true" streaming
— there is a ~chunk-sec latency floor and redundant compute — but a
large improvement over pause-triggered VAD chunking on a monologue.

Options:
  --model REPO            HF repo id (default nvidia/parakeet-tdt-0.6b-v3)
  --chunk-sec S           new audio per step / emit cadence (default 1.0).
                          Lower = snappier, more recompute.
  --max-context-sec S     cap on the rolling window (default 12.0)
  --trim-margin S         audio kept before last committed word (default 0.2)
  --silence-rms R         whole-buffer RMS below this = silence; buffer is
                          reset so dead air / music can't grow it (default 0.004)
"""
import sys
import time
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
p.add_argument("--chunk-sec", type=float, default=1.0)
p.add_argument("--max-context-sec", type=float, default=12.0)
p.add_argument("--trim-margin", type=float, default=0.2)
p.add_argument("--silence-rms", type=float, default=0.004)
args = p.parse_args()

CHUNK_BYTES = int(args.chunk_sec * SAMPLE_RATE) * 2
MAX_SAMPLES = int(args.max_context_sec * SAMPLE_RATE)

print(f"loading {args.model}...", file=sys.stderr, flush=True)
asr = nemo_asr.models.ASRModel.from_pretrained(args.model)
asr = asr.eval().to("cuda")
print(f"ready, listening... (chunk-sec={args.chunk_sec}, "
      f"max-context-sec={args.max_context_sec})", file=sys.stderr, flush=True)


def transcribe(buf):
    """Return [(word_lower, word_orig, abs_start, abs_end), ...]."""
    out = asr.transcribe([buf], batch_size=1, timestamps=True, verbose=False)
    if not out:
        return "", []
    h = out[0]
    words = []
    for w in (h.timestamp or {}).get("word", []):
        words.append((w["word"].lower(), w["word"],
                      float(w["start"]), float(w["end"])))
    return (h.text or "").strip(), words


def common_prefix(a, b):
    """Length of the longest common prefix by (lowercased) word text."""
    n = 0
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        n += 1
    return n


audio_buf = np.zeros(0, dtype=np.float32)
buf_start_t = 0.0          # absolute time (s) of audio_buf[0]
total_consumed = 0          # samples read from stdin so far
committed_end_t = 0.0       # absolute time up to which words are printed
prev_unc = []               # previous run's uncommitted word list


def read_chunk():
    b = b""
    while len(b) < CHUNK_BYTES:
        piece = sys.stdin.buffer.read(CHUNK_BYTES - len(b))
        if not piece:
            return b
        b += piece
    return b


def emit(words):
    if not words:
        return
    ts = words[0][2]
    line = " ".join(w[1] for w in words)
    print(f"[{ts:6.1f}s] {line}", flush=True)


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

    # Drop dead air / music so it can't grow the buffer unbounded.
    rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size else 0.0
    if rms < args.silence_rms and not eof:
        now_t = total_consumed / SAMPLE_RATE
        audio_buf = np.zeros(0, dtype=np.float32)
        buf_start_t = now_t
        committed_end_t = max(committed_end_t, now_t)
        prev_unc = []
        continue

    text, words = transcribe(audio_buf)
    # buffer-relative -> absolute
    words = [(wl, wo, s + buf_start_t, e + buf_start_t)
             for (wl, wo, s, e) in words]
    unc = [w for w in words if w[3] > committed_end_t + 1e-3]

    if eof:
        # stream ended: nothing more will arrive, commit the tail as-is
        emit(unc)
        break

    # LocalAgreement-2: commit the agreeing prefix of the last two runs
    k = common_prefix(unc, prev_unc)
    if k:
        emit(unc[:k])
        committed_end_t = unc[k - 1][3]
    prev_unc = unc

    # Trim the buffer at the last committed word (keep a small margin),
    # and hard-cap the window length.
    target_start = max(buf_start_t, committed_end_t - args.trim_margin)
    drop = int((target_start - buf_start_t) * SAMPLE_RATE)
    if len(audio_buf) - drop > MAX_SAMPLES:
        drop = len(audio_buf) - MAX_SAMPLES
    if drop > 0:
        audio_buf = audio_buf[drop:]
        buf_start_t += drop / SAMPLE_RATE

print("\n[stream-parakeet-buffered] bye.", file=sys.stderr)
