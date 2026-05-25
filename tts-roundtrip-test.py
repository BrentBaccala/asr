#!/home/cosine/asr/mt-env/bin/python
"""tts-roundtrip-test.py — offline end-to-end validation of the
freesoft-interpret cascade (ASR -> MT -> TTS), both directions, against
the Spanish audiobook file. No RTP / no live program needed.

Runs ENTIRELY on pony. Exercises:
  Forward:  audiobook slice (es) -> Voxtral ASR -> Spanish text
            -> NLLB es->en -> English text
            -> Pocket TTS `english` (CPU) -> English WAV
  Reverse:  the English WAV     -> Voxtral ASR -> English text
            -> NLLB en->es -> Spanish text
            -> Pocket TTS `spanish_24l` (GPU) -> Spanish WAV

Emits a side-by-side table (original ES | forward EN | round-trip ES) and
saves every intermediate WAV under --outdir so audio existence/length is
checkable.

Venvs: this script runs under mt-env (websockets + ctranslate2 +
transformers). Pocket TTS lives in a separate venv (tts-env) and is
invoked as a subprocess via its CLI (`python -m pocket_tts generate`).
Voxtral must be serving on 127.0.0.1:8000.

Usage (on pony):
  ~/asr/tts-roundtrip-test.py \
      --audiobook ~/muerteenvenecia_02_mann_64kb.mp3 \
      --start 60 --dur 75 --outdir /tmp/tts-roundtrip
"""
import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import threading
import time

import websockets

HOST, PORT = "127.0.0.1", 8000
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SAMPLE_RATE = 16000
CHUNK = 2048 * 2
TTS_PY = os.path.expanduser("~/asr/tts-env/bin/python")
NLLB_DIR = os.path.expanduser("~/asr/models/nllb-600m-ct2")


def decode_pcm(path, start, dur):
    """ffmpeg-decode a slice of an audio file to 16 kHz mono s16le PCM."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "s16le", "-"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


def wav_to_pcm(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


async def asr_pcm(pcm: bytes, pace: bool = True) -> str:
    """Transcribe raw s16le mono 16 kHz PCM via Voxtral /v1/realtime.
    Returns the full transcript. Paces the audio at ~1x so the realtime
    model behaves like the live pipeline."""
    uri = f"ws://{HOST}:{PORT}/v1/realtime"
    text_parts = []
    async with websockets.connect(uri, max_size=None) as ws:
        r = json.loads(await ws.recv())
        if r.get("type") != "session.created":
            raise RuntimeError(f"unexpected handshake: {r}")
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        done = asyncio.Event()

        async def receiver():
            while True:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                except (asyncio.TimeoutError, Exception):
                    return
                t = m.get("type")
                if t == "transcription.delta":
                    text_parts.append(m["delta"])
                elif t == "transcription.done":
                    done.set()
                    return
                elif t == "error":
                    raise RuntimeError(f"server error: {m}")

        rx = asyncio.create_task(receiver())
        # ~128 ms per CHUNK; pace at 1x real-time when requested.
        per_chunk_s = (CHUNK / 2) / SAMPLE_RATE
        for i in range(0, len(pcm), CHUNK):
            chunk = pcm[i:i + CHUNK]
            await ws.send(json.dumps(
                {"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(chunk).decode()}))
            if pace:
                await asyncio.sleep(per_chunk_s * 0.5)   # 2x faster than RT
        await ws.send(json.dumps(
            {"type": "input_audio_buffer.commit", "final": True}))
        try:
            await asyncio.wait_for(done.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.wait_for(rx, timeout=5)
        except asyncio.TimeoutError:
            rx.cancel()
    return "".join(text_parts).strip()


_tok = None
_tr = None


def load_nllb():
    global _tok, _tr
    if _tr is not None:
        return
    import ctranslate2
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(NLLB_DIR)
    _tr = ctranslate2.Translator(NLLB_DIR, device="cpu", compute_type="int8",
                                 inter_threads=1, intra_threads=8)


def translate(text, src, tgt):
    load_nllb()
    _tok.src_lang = src
    src_tokens = _tok.convert_ids_to_tokens(_tok.encode(text))
    res = _tr.translate_batch([src_tokens], target_prefix=[[tgt]],
                              beam_size=1, max_decoding_length=512)
    hyp = res[0].hypotheses[0]
    if hyp and hyp[0] == tgt:
        hyp = hyp[1:]
    return _tok.decode(_tok.convert_tokens_to_ids(hyp)).strip()


def tts(text, language, device, out_wav):
    """Synthesize via the pocket-tts CLI in the tts-env. Returns the WAV
    path on success (raises on failure)."""
    cmd = [TTS_PY, "-m", "pocket_tts", "generate", "--language", language,
           "--device", device, "--text", text, "--output-path", out_wav,
           "--quiet"]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def wav_seconds(path):
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audiobook",
                    default=os.path.expanduser(
                        "~/muerteenvenecia_02_mann_64kb.mp3"))
    ap.add_argument("--start", type=float, default=60.0,
                    help="start offset (s) into the audiobook")
    ap.add_argument("--dur", type=float, default=75.0,
                    help="slice duration (s)")
    ap.add_argument("--outdir", default="/tmp/tts-roundtrip")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1] decoding {args.dur}s @ {args.start}s of {args.audiobook} ...",
          flush=True)
    pcm = decode_pcm(args.audiobook, args.start, args.dur)
    print(f"    {len(pcm)} bytes = {(len(pcm)/2)/SAMPLE_RATE:.1f}s PCM",
          flush=True)

    print("[2] FORWARD: Voxtral ASR (es) ...", flush=True)
    es_text = asyncio.run(asr_pcm(pcm))
    print(f"    ES: {es_text!r}", flush=True)

    print("[3] FORWARD: NLLB es->en ...", flush=True)
    en_text = translate(es_text, "spa_Latn", "eng_Latn")
    print(f"    EN: {en_text!r}", flush=True)

    print("[4] FORWARD: Pocket TTS english (CPU) -> WAV ...", flush=True)
    en_wav = os.path.join(args.outdir, "forward_en.wav")
    t0 = time.monotonic()
    tts(en_text, "english", "cpu", en_wav)
    en_secs = wav_seconds(en_wav)
    print(f"    {en_wav}  {en_secs:.1f}s  (synth {time.monotonic()-t0:.1f}s)",
          flush=True)

    print("[5] REVERSE: Voxtral ASR (en) on the synthesized WAV ...",
          flush=True)
    en_pcm = wav_to_pcm(en_wav)
    en_text2 = asyncio.run(asr_pcm(en_pcm))
    print(f"    EN(rt): {en_text2!r}", flush=True)

    print("[6] REVERSE: NLLB en->es ...", flush=True)
    es_text2 = translate(en_text2, "eng_Latn", "spa_Latn")
    print(f"    ES(rt): {es_text2!r}", flush=True)

    print("[7] REVERSE: Pocket TTS spanish_24l (GPU) -> WAV ...", flush=True)
    es_wav = os.path.join(args.outdir, "roundtrip_es.wav")
    t0 = time.monotonic()
    tts(es_text2, "spanish_24l", "cuda", es_wav)
    es_secs = wav_seconds(es_wav)
    print(f"    {es_wav}  {es_secs:.1f}s  (synth {time.monotonic()-t0:.1f}s)",
          flush=True)

    print("\n" + "=" * 78)
    print("ROUND-TRIP TABLE")
    print("=" * 78)
    print(f"\n[ORIGINAL ES]\n{es_text}\n")
    print(f"[FORWARD EN]\n{en_text}\n")
    print(f"[ROUND-TRIP ES]\n{es_text2}\n")
    print("-" * 78)
    print(f"WAVs: {en_wav} ({en_secs:.1f}s), {es_wav} ({es_secs:.1f}s)")
    print(f"forward EN non-empty: {bool(en_text.strip())}; "
          f"round-trip ES non-empty: {bool(es_text2.strip())}")
    print(f"english WAV >0s: {en_secs > 0}; spanish WAV >0s: {es_secs > 0}")


if __name__ == "__main__":
    main()
