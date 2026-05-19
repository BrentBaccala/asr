#!/home/cosine/venv-3.12-torch/bin/python
"""
Streaming Canary ASR/AST transcription from stdin PCM audio.

Pipe arecord into this:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | stream-canary.py [opts]

Reads 16-bit signed little-endian mono PCM at 16 kHz.
Uses Silero VAD to chunk speech utterances; each utterance is sent to
the chosen Canary model on the GPU when 500 ms of silence is detected.

Canary needs explicit source-lang per utterance (no auto-detect).
For code-switching audio you'd need a separate language ID step;
for now this script transcribes in a fixed language per invocation.

Options:
  --task {asr,ast}        asr = transcribe in source-lang (default)
                          ast = translate from source-lang to target-lang
  --source-lang LANG      en, es, fr, de        (default: en)
  --target-lang LANG      en, es, fr, de        (default: en; AST only)
  --model REPO            HF repo id            (default: nvidia/canary-1b-flash)
  --pnc {yes,no}          punctuation/caps      (default: yes)
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
p.add_argument("--task", choices=["asr", "ast"], default="asr")
p.add_argument("--source-lang", default="en")
p.add_argument("--target-lang", default="en")
p.add_argument("--model", default="nvidia/canary-1b-flash")
p.add_argument("--pnc", choices=["yes", "no"], default="yes")
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
asr = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(args.model)
asr = asr.eval().to("cuda")

print(
    f"ready, listening (task={args.task} {args.source_lang}->{args.target_lang})...",
    file=sys.stderr, flush=True,
)

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
            out = asr.transcribe(
                audio=[full],
                batch_size=1,
                task=args.task,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                pnc=args.pnc,
            )
            text = out[0].text.strip() if out else ""
            asr_t = time.time() - t0
            tag = f"{args.source_lang}->{args.target_lang}" if args.task == "ast" else args.source_lang
            print(f"[{tag} | {dur:.1f}s in / {asr_t:.2f}s asr] {text}", flush=True)
            speech_chunks = []
