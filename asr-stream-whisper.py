#!/home/claude/asr/asr-env/bin/python
"""
Streaming faster-whisper transcription from stdin PCM audio.

Pipe arecord into this:
  arecord -D plughw:Snowball,0 -f S16_LE -r 16000 -c 1 | asr-stream-whisper.py

Reads 16-bit signed little-endian mono PCM at 16 kHz.
Uses Silero VAD to chunk speech utterances; each utterance is sent to
faster-whisper-large-v3 on the GPU when 500 ms of silence is detected.
Transcribed text goes to stdout, status to stderr.
"""
import sys
import time
import numpy as np
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512        # silero-vad requires exactly 512 @ 16 kHz
BYTES_PER_FRAME = VAD_FRAME_SAMPLES * 2  # int16 = 2 bytes

print("loading silero-vad...", file=sys.stderr, flush=True)
vad_model = load_silero_vad(onnx=True)
vad_iter = VADIterator(
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)

print("loading faster-whisper-large-v3...", file=sys.stderr, flush=True)
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

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
            segs, info = asr.transcribe(full, language=None, beam_size=5)
            text = " ".join(s.text.strip() for s in segs).strip()
            asr_t = time.time() - t0
            print(f"[{info.language} | {dur:.1f}s in / {asr_t:.2f}s asr] {text}", flush=True)
            speech_chunks = []
