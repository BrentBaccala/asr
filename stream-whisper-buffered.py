#!/home/cosine/asr/asr-env/bin/python3
"""
Low-latency buffered streaming transcription with faster-whisper,
language hard-pinned (default Spanish).

Pipe raw PCM in:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | stream-whisper-buffered.py
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-whisper-buffered.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

Unlike parakeet-tdt (which auto-detects language per window and flips
to English on ambiguous chunks), Whisper takes an explicit `language`
and decodes the whole session under that one language token — so with
--lang es the output is exclusively Spanish, no per-chunk re-detection.

Same low-latency scheme as stream-parakeet-buffered.py: run the offline
model on an overlapping sliding window every --chunk-sec and commit
words with a LocalAgreement-2 policy (a word prints only once two
successive overlapping transcriptions agree on it). The buffer is
trimmed at the last committed word using Whisper word timestamps,
bounding latency and recompute. Not "true" streaming — there is a
~chunk-sec latency floor plus redundant compute — but continuous
output instead of waiting for a pause.

large-v3 is heavier than parakeet-0.6b; if the GPU can't keep up at
--chunk-sec 1.0, raise it (e.g. 1.5) or use a lighter --model.

Options:
  --model NAME          faster-whisper model (default large-v3)
  --lang CODE           language, hard-pinned (default es). "auto" lets
                        Whisper detect once per window (NOT recommended —
                        defeats the point; included for debugging).
  --chunk-sec S         new audio per step / emit cadence (default 1.0)
  --max-context-sec S   cap on the rolling window (default 12.0)
  --trim-margin S       audio kept before last committed word (default 0.2)
  --silence-rms R       whole-buffer RMS below this = silence; buffer reset
                        so dead air / music can't grow it (default 0.004)
  --beam-size N         decoding beam (default 1 = greedy, fastest)
  --compute-type T      ctranslate2 compute type (default auto)
"""
import sys
import argparse
import warnings
import logging
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

p = argparse.ArgumentParser()
p.add_argument("--model", default="large-v3")
p.add_argument("--lang", default="es")
p.add_argument("--chunk-sec", type=float, default=1.0)
p.add_argument("--max-context-sec", type=float, default=12.0)
p.add_argument("--trim-margin", type=float, default=0.2)
p.add_argument("--silence-rms", type=float, default=0.004)
p.add_argument("--beam-size", type=int, default=1)
p.add_argument("--compute-type", default="auto")
args = p.parse_args()

CHUNK_BYTES = int(args.chunk_sec * SAMPLE_RATE) * 2
MAX_SAMPLES = int(args.max_context_sec * SAMPLE_RATE)
LANG = None if args.lang.lower() in ("auto", "none", "detect") else args.lang

print(f"loading faster-whisper {args.model}...", file=sys.stderr, flush=True)
model = WhisperModel(args.model, device="auto", compute_type=args.compute_type)
print(f"ready, listening... (lang={LANG or 'AUTO-detect'}, "
      f"chunk-sec={args.chunk_sec}, max-context-sec={args.max_context_sec})",
      file=sys.stderr, flush=True)


def transcribe(buf):
    """Return [(word_lower, word_orig, abs_start, abs_end), ...] (buffer-rel)."""
    segs, _ = model.transcribe(
        buf, language=LANG, word_timestamps=True,
        beam_size=args.beam_size, vad_filter=False,
        condition_on_previous_text=False,
    )
    words = []
    for s in segs:
        for w in (s.words or []):
            t = w.word.strip()
            if t:
                words.append((t.lower(), t, float(w.start), float(w.end)))
    return words


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        n += 1
    return n


audio_buf = np.zeros(0, dtype=np.float32)
buf_start_t = 0.0
total_consumed = 0
committed_end_t = 0.0
prev_unc = []


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
    print(f"[{ts:6.1f}s] " + " ".join(w[1] for w in words), flush=True)


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

    rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size else 0.0
    if rms < args.silence_rms and not eof:
        now_t = total_consumed / SAMPLE_RATE
        audio_buf = np.zeros(0, dtype=np.float32)
        buf_start_t = now_t
        committed_end_t = max(committed_end_t, now_t)
        prev_unc = []
        continue

    words = [(wl, wo, s + buf_start_t, e + buf_start_t)
             for (wl, wo, s, e) in transcribe(audio_buf)]
    unc = [w for w in words if w[3] > committed_end_t + 1e-3]

    if eof:
        emit(unc)
        break

    k = common_prefix(unc, prev_unc)
    if k:
        emit(unc[:k])
        committed_end_t = unc[k - 1][3]
    prev_unc = unc

    target_start = max(buf_start_t, committed_end_t - args.trim_margin)
    drop = int((target_start - buf_start_t) * SAMPLE_RATE)
    if len(audio_buf) - drop > MAX_SAMPLES:
        drop = len(audio_buf) - MAX_SAMPLES
    if drop > 0:
        audio_buf = audio_buf[drop:]
        buf_start_t += drop / SAMPLE_RATE

print("\n[stream-whisper-buffered] bye.", file=sys.stderr)
