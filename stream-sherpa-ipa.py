#!/home/cosine/asr/sherpa-env/bin/python
"""
True streaming Spanish *IPA-phoneme* ASR (sherpa-onnx Zipformer).

Pipe raw PCM in:
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | stream-sherpa-ipa.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

Model: bookbot/sherpa-onnx-zipformer-streaming-robust-es-v0 -- a
genuinely frame-synchronous streaming Zipformer transducer
(chunk-16-left-128) whose output vocabulary is IPA phonemes (gruut),
NOT Spanish orthography. So the transcript is a phoneme stream
(/g u s t a β o .../), not words. "robust" = noise-augmented training.
Output is delta-printed (wrap-safe); a detected endpoint ends the line.
"""
import sys
import os
import argparse
import numpy as np
import sherpa_onnx

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir",
                default=os.path.expanduser("~/asr/models/sherpa-es-ipa"))
ap.add_argument("--read-ms", type=int, default=120)
args = ap.parse_args()

m = args.model_dir
S = "epoch-80-avg-3-chunk-16-left-128.int8.onnx"
print("loading sherpa-onnx zipformer (es IPA)...", file=sys.stderr, flush=True)
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{m}/tokens.txt",
    encoder=f"{m}/encoder-{S}",
    decoder=f"{m}/decoder-{S}",
    joiner=f"{m}/joiner-{S}",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300,
)
print("ready, listening...", file=sys.stderr, flush=True)

stream = rec.create_stream()
read_bytes = int(16000 * args.read_ms / 1000) * 2
shown = ""


def emit(t):
    global shown
    if t == shown:
        return
    if t.startswith(shown):
        sys.stdout.write(t[len(shown):])
    else:
        sys.stdout.write("\n" + t)
    sys.stdout.flush()
    shown = t


while True:
    data = sys.stdin.buffer.read(read_bytes)
    if not data:
        break
    s = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(16000, s)
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    r = rec.get_result(stream)
    if rec.is_endpoint(stream):
        if r:
            emit(r)
        sys.stdout.write("\n")
        sys.stdout.flush()
        shown = ""
        rec.reset(stream)
    elif r:
        emit(r)

stream.input_finished()
while rec.is_ready(stream):
    rec.decode_stream(stream)
r = rec.get_result(stream)
if r:
    emit(r)
sys.stdout.write("\n")
print("[stream-sherpa-ipa] bye.", file=sys.stderr)
