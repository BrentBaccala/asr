#!/home/cosine/venv-3.12-torch/bin/python
"""
Low-latency *immediate-emit* streaming transcription with Parakeet
TDT v3 (multilingual incl. Spanish).

Pipe raw PCM in:
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-parakeet-live.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

parakeet-tdt-0.6b-v3 has no cache-aware streaming mode, and no
cache-aware Spanish/multilingual model exists (May 2026). The buffered
LocalAgreement-2 scripts (stream-{whisper,parakeet}-buffered.py) add a
two-window confirmation delay on top of --chunk-sec, so their latency
floor is ~2x chunk-sec plus model time -> several seconds.

This script drops the confirmation wait entirely: every --chunk-sec it
re-transcribes the rolling window and reprints the current hypothesis
(delta-printed, like the cache-aware English script). Latency is then
~chunk-sec + one fast parakeet-0.6b forward (~0.1-0.2 s on a 3090), so
sub-second is reachable. The cost is the same as the cache-aware
script's: the trailing words can be revised as more context arrives.

Spanish is auto-detected (parakeet-tdt-0.6b-v3 has no language input and
cannot be hard-locked); on short / ambiguous windows it can briefly flip
to English. For exclusively-Spanish output with no flipping, use
stream-whisper-buffered.py instead and accept its higher latency.

Options:
  --model REPO         HF repo id (default nvidia/parakeet-tdt-0.6b-v3)
  --chunk-sec S        re-transcribe + emit cadence (default 0.5)
  --max-context-sec S  rolling window cap; older audio is dropped after
                        it has been emitted (default 8.0)
  --commit-sec S       words whose end is older than (window_end -
                        commit-sec) are treated as final: printed with a
                        newline and never revised again (default 2.0)
  --silence-rms R      whole-window RMS below this = silence; window is
                        reset so dead air / music can't grow it
                        (default 0.004)
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
p.add_argument("--max-context-sec", type=float, default=8.0)
p.add_argument("--commit-sec", type=float, default=2.0)
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
buf_start_t = 0.0       # absolute time of audio_buf[0]
total_consumed = 0
committed_line = ""     # text already finalized (newline-printed)
shown = ""              # current provisional tail on screen


def flush_line(text):
    """Finalize a line: overwrite the provisional tail, newline."""
    global committed_line, shown
    sys.stdout.write("\r" + text + "\n")
    sys.stdout.flush()
    committed_line = ""
    shown = ""


def show(text):
    """Reprint the provisional tail in place."""
    global shown
    if text != shown:
        sys.stdout.write("\r" + text + " " * max(0, len(shown) - len(text)))
        sys.stdout.flush()
        shown = text


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
        if shown:
            flush_line(shown)
        audio_buf = np.zeros(0, dtype=np.float32)
        buf_start_t = window_end_t
        continue

    words = transcribe(audio_buf)
    words = [(w, s + buf_start_t, e + buf_start_t) for (w, s, e) in words]

    # Split at the commit horizon: anything ending before it is final.
    horizon = window_end_t - args.commit_sec
    final = [w for w in words if w[2] <= horizon]
    tail = [w for w in words if w[2] > horizon]
    final_txt = " ".join(w[0] for w in final).strip()
    tail_txt = " ".join(w[0] for w in tail).strip()

    if eof:
        show((final_txt + " " + tail_txt).strip())
        break

    if final_txt:
        flush_line(("[%6.1fs] " % (final[0][1])) + final_txt)
        # Trim the buffer up to the last finalized word.
        cut = final[-1][2]
        drop = int((cut - buf_start_t) * SAMPLE_RATE)
        if drop > 0:
            audio_buf = audio_buf[drop:]
            buf_start_t += drop / SAMPLE_RATE
    if len(audio_buf) > MAX_SAMPLES:                 # hard safety cap
        d = len(audio_buf) - MAX_SAMPLES
        audio_buf = audio_buf[d:]
        buf_start_t += d / SAMPLE_RATE
    if tail_txt:
        show(("[%6.1fs] " % (tail[0][1])) + tail_txt)

if shown:
    sys.stdout.write("\n")
print("[stream-parakeet-live] bye.", file=sys.stderr)
