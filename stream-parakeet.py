#!/home/cosine/venv-3.12-torch/bin/python
"""
Streaming Parakeet TDT v3 transcription from stdin PCM audio.

Pipe arecord into this:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | stream-parakeet.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.

Silero VAD chunks speech into utterances; an utterance is transcribed
when 500 ms of silence is detected (a natural pause). To bound latency
on *continuous* speech with no pauses (e.g. a news broadcast), an
utterance is also force-flushed once it reaches --max-sec seconds, so
text appears at least that often regardless of speaking style.

Parakeet TDT v3 is multilingual (25 languages incl. en/es/fr/de) and
auto-detects language per utterance — no language flag needed.

Options:
  --model REPO         HF repo id        (default: nvidia/parakeet-tdt-0.6b-v3)
  --max-sec SECONDS     force-flush a no-pause utterance after this long
                        (default: 6.0). Lower = snappier, but more
                        word-boundary splits and slightly less context.
  --min-silence-ms MS   silence needed to end an utterance at a natural
                        pause (default: 500). Lower = snappier turn-taking.
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
p.add_argument("--max-sec", type=float, default=6.0)
p.add_argument("--min-silence-ms", type=int, default=500)
args = p.parse_args()

MAX_SAMPLES = int(args.max_sec * SAMPLE_RATE)

print("loading silero-vad...", file=sys.stderr, flush=True)
vad_model = load_silero_vad(onnx=True)
vad_iter = VADIterator(
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,
    min_silence_duration_ms=args.min_silence_ms,
    speech_pad_ms=200,
)

print(f"loading {args.model}...", file=sys.stderr, flush=True)
asr = nemo_asr.models.ASRModel.from_pretrained(args.model)
asr = asr.eval().to("cuda")

print(f"ready, listening... (max-sec={args.max_sec}, "
      f"min-silence-ms={args.min_silence_ms})", file=sys.stderr, flush=True)

speech_chunks = []
n_samples = 0
in_speech = False


def flush(reason: str) -> None:
    """Transcribe whatever speech has accumulated and print one line."""
    global speech_chunks, n_samples
    if not speech_chunks:
        return
    full = np.concatenate(speech_chunks)
    dur = len(full) / SAMPLE_RATE
    t0 = time.time()
    out = asr.transcribe([full], batch_size=1)
    text = out[0].text.strip() if out else ""
    asr_t = time.time() - t0
    if text:
        print(f"[{dur:.1f}s in / {asr_t:.2f}s asr / {reason}] {text}",
              flush=True)
    speech_chunks = []
    n_samples = 0


while True:
    raw = sys.stdin.buffer.read(BYTES_PER_FRAME)
    if not raw or len(raw) < BYTES_PER_FRAME:
        break
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    event = vad_iter(audio)

    if in_speech:
        speech_chunks.append(audio)
        n_samples += len(audio)

    if event is not None:
        if "start" in event:
            in_speech = True
            speech_chunks = [audio]
            n_samples = len(audio)
        elif "end" in event:
            in_speech = False
            flush("pause")

    # Continuous speech with no pause (news anchor): don't wait forever
    # for a VAD endpoint — force a flush and keep transcribing the
    # ongoing monologue in ~max-sec slices.
    if in_speech and n_samples >= MAX_SAMPLES:
        flush("cap")

# stream ended (EOF) — emit any speech still buffered
flush("eof")
