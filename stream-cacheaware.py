#!/home/cosine/venv-3.12-torch/bin/python
"""
Cache-aware streaming English ASR from stdin PCM audio.

Pipe 16-bit signed little-endian mono PCM at 16 kHz in:

  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
    --channels=1 - | stream-cacheaware.py

Drives nvidia/stt_en_fastconformer_hybrid_large_streaming_multi, a true
cache-aware FastConformer hybrid RNNT/CTC model (English-only) via
NeMo's CacheAwareStreamingAudioBuffer + conformer_stream_step. Unlike
the buffered-sliding-window scripts, this is genuine low-latency
streaming: the encoder carries cached state across chunks, so latency
is the model's structural lookahead, not a recompute window.

--lookahead picks the latency/accuracy trade (encoder frames are 80 ms):
  0  -> [70, 0]   ~0   ms lookahead, lowest accuracy
  1  -> [70, 1]   ~80  ms
  6  -> [70, 6]   ~480 ms   (default; good balance)
  13 -> [70, 13]  ~1040 ms, highest accuracy

Audio is fed to the buffer in exact shift_size-aligned blocks (leftover
carried here), so the streaming iterator never emits a partial chunk
mid-stream — that partial-chunk-then-skip is what corrupts the larger
lookahead settings if you drain naively.

English-only by construction (NVIDIA stt_en_*). For Spanish use the
buffered whisper script instead.
"""
import os
import sys
import argparse

os.environ.setdefault("NEMO_TESTING", "0")

import numpy as np
import torch

# NeMo uses its own logger; the stdlib root logger level does not gag it.
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel("ERROR")

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)

SAMPLE_RATE = 16000
LOOKAHEAD = {0: [70, 0], 1: [70, 1], 6: [70, 6], 13: [70, 13]}

ap = argparse.ArgumentParser()
ap.add_argument(
    "--model",
    default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
)
ap.add_argument(
    "--lookahead", type=int, default=6, choices=[0, 1, 6, 13],
    help="att lookahead frames: 0=~0ms 1=~80ms 6=~480ms 13=~1040ms",
)
args = ap.parse_args()

print(
    f"loading {args.model} (lookahead [70,{args.lookahead}])...",
    file=sys.stderr, flush=True,
)
asr = nemo_asr.models.ASRModel.from_pretrained(args.model)
asr.encoder.set_default_att_context_size(LOOKAHEAD[args.lookahead])
if hasattr(asr.encoder, "setup_streaming_params"):
    asr.encoder.setup_streaming_params()
asr.eval()
asr = asr.to("cuda")

buf = CacheAwareStreamingAudioBuffer(
    model=asr, online_normalization=False, pad_and_drop_preencoded=True
)
(
    cache_last_channel,
    cache_last_time,
    cache_last_channel_len,
) = asr.encoder.get_initial_cache_state(batch_size=1)
prev_hyp = None
pred_out = None
printed = ""

sc = asr.encoder.streaming_cfg


def _cfg(v):
    return v[1] if isinstance(v, (list, tuple)) else v


CHUNK_FRAMES = int(_cfg(sc.chunk_size))  # mel frames per streaming step
hop = int(round(asr.cfg.preprocessor.window_stride * SAMPLE_RATE))
print(
    f"chunk={CHUNK_FRAMES} mel-frames (~{CHUNK_FRAMES * hop / SAMPLE_RATE * 1000:.0f} ms) per step",
    file=sys.stderr, flush=True,
)


def _text(texts):
    if not texts:
        return ""
    t = texts[0]
    while isinstance(t, (list, tuple)) and t:
        t = t[0]
    if hasattr(t, "text"):
        t = t.text
    return t if isinstance(t, str) else ""


def _emit(texts):
    global printed
    cur = _text(texts)
    if cur and cur != printed:
        if cur.startswith(printed):
            sys.stdout.write(cur[len(printed):])
        else:
            sys.stdout.write("\n" + cur)
        sys.stdout.flush()
        printed = cur


def drain(flush=False):
    """Step the streaming iterator one chunk at a time, but only while a
    full chunk of *real* audio is buffered ahead of buffer_idx. The
    iterator otherwise emits a short trailing chunk and still advances
    buffer_idx by a full step, skipping audio (worse at larger
    lookahead). On flush (EOF) we consume whatever remains."""
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global prev_hyp, pred_out
    while buf.buffer is not None:
        avail = buf.buffer.size(-1) - buf.buffer_idx
        if avail <= 0:
            break
        if not flush and avail < CHUNK_FRAMES:
            break
        try:
            chunk, clen = next(iter(buf))  # exactly one chunk; advances buffer_idx
        except StopIteration:
            break
        with torch.inference_mode():
            (
                pred_out,
                texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                prev_hyp,
            ) = asr.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=clen,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=buf.is_buffer_empty(),
                previous_hypotheses=prev_hyp,
                previous_pred_out=pred_out,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        _emit(texts)


print("ready, streaming...", file=sys.stderr, flush=True)
first = True
read_bytes = CHUNK_FRAMES * hop * 2  # ~one streaming step of audio per read

while True:
    raw = sys.stdin.buffer.read(read_bytes)
    if not raw:
        break
    s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    buf.append_audio(s, stream_id=-1 if first else 0)
    first = False
    drain()

if not first:
    drain(flush=True)
sys.stdout.write("\n")
sys.stdout.flush()
sys.stdout.write("\n")
sys.stdout.flush()
