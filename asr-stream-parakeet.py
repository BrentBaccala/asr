#!/home/claude/asr/asr-env-canary/bin/python
"""
Streaming Parakeet TDT v3 transcription from stdin PCM audio.

Pipe arecord into this:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | asr-stream-parakeet.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.
Uses Silero VAD to chunk speech utterances; each utterance is sent to
nvidia/parakeet-tdt-0.6b-v3 on the GPU when 500 ms of silence is detected.

Parakeet TDT v3 is multilingual (25 languages incl. en/es/fr/de) and
auto-detects language per utterance — no language flag needed.

Options:
  --model REPO    HF repo id   (default: nvidia/parakeet-tdt-0.6b-v3)
"""
import sys
import time
import argparse
import warnings
import logging
import numpy as np

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from silero_vad import load_silero_vad, VADIterator
import nemo.collections.asr as nemo_asr

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512
BYTES_PER_FRAME = VAD_FRAME_SAMPLES * 2

p = argparse.ArgumentParser()
p.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
args = p.parse_args()

print("loading silero-vad...", file=sys.stderr, flush=True)
vad_model = load_silero_vad(onnx=True)
vad_iter = VADIterator(
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)

print(f"loading {args.model}...", file=sys.stderr, flush=True)
asr = nemo_asr.models.ASRModel.from_pretrained(args.model)
asr = asr.eval().to("cuda")

print("ready, listening...", file=sys.stderr, flush=True)

speech_chunks = []
in_speech = False

while True:
    raw = sys.stdin.buffer.read(BYTES_PER_FRAME)
    if not raw or len(raw) < BYTES_PER_FRAME:
        break
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    event = vad_iter(audio)

    if in_speech:
        speech_chunks.append(audio)

    if event is not None:
        if "start" in event:
            in_speech = True
            speech_chunks = [audio]
        elif "end" in event:
            in_speech = False
            full = np.concatenate(speech_chunks)
            dur = len(full) / SAMPLE_RATE
            t0 = time.time()
            out = asr.transcribe([full], batch_size=1)
            text = out[0].text.strip() if out else ""
            asr_t = time.time() - t0
            print(f"[{dur:.1f}s in / {asr_t:.2f}s asr] {text}", flush=True)
            speech_chunks = []
