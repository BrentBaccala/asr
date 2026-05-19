#!/home/cosine/asr/mt-env/bin/python
"""
Live Spanish transcription (Voxtral realtime, vLLM) + running English
translation (NLLB-200-distilled-600M int8, CTranslate2, CPU).

Pipe the samsung->pony RTP tap in, exactly like stream-voxtral.py:

  ssh cosine@pony
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | ~/asr/stream-voxtral-translate.py

Design (the research report's v1 = COMMIT-ONLY translation, zero churn):
Voxtral streams committed Spanish; we accumulate it into sentences and,
on each sentence-final boundary (.?! / utterance end), print a paired
block:

  ES  <spanish sentence>
  EN  <english translation>

No live re-translating partial sentences (no flicker). MT runs on the
CPU in a worker thread (NLLB int8, ~0.3-0.8 s/sentence) so it never
blocks the audio WebSocket. The Spanish side and the GPU vLLM server
are untouched.

Prereqs: vLLM Voxtral server on 127.0.0.1:8000; NLLB CT2 model at
~/asr/models/nllb-600m-ct2.
"""
import asyncio
import base64
import json
import os
import queue
import re
import sys
import threading

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import websockets
import ctranslate2
from transformers import AutoTokenizer

HOST, PORT = "127.0.0.1", 8000
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
NLLB_DIR = os.path.expanduser("~/asr/models/nllb-600m-ct2")
CHUNK = 2048 * 2                       # 128 ms s16le @ 16 kHz
SENT_RE = re.compile(r'(.+?[.!?…]+["»”\'\)\]]*)(?:\s+|$)', re.S)
SOFT_CAP = 240                         # flush a run-on as a clause past this

print("[stream-voxtral-translate] loading NLLB...", file=sys.stderr,
      flush=True)
_tok = AutoTokenizer.from_pretrained(NLLB_DIR)
_tok.src_lang = "spa_Latn"
_tr = ctranslate2.Translator(NLLB_DIR, device="cpu", compute_type="int8",
                             inter_threads=2, intra_threads=6)


def translate(es: str) -> str:
    src = _tok.convert_ids_to_tokens(_tok.encode(es))
    r = _tr.translate_batch([src], target_prefix=[["eng_Latn"]],
                            beam_size=2, max_decoding_length=512)
    h = r[0].hypotheses[0]
    if h and h[0] == "eng_Latn":
        h = h[1:]
    return _tok.decode(_tok.convert_tokens_to_ids(h)).strip()


# MT worker: completed Spanish sentences in -> paired ES/EN block out.
_sent_q: "queue.Queue[str|None]" = queue.Queue()


def mt_worker():
    while True:
        es = _sent_q.get()
        if es is None:
            return
        es = es.strip()
        if not es:
            continue
        try:
            en = translate(es)
        except Exception as e:                       # never kill the stream
            en = f"[translate error: {e}]"
        sys.stdout.write(f"ES  {es}\nEN  {en}\n\n")
        sys.stdout.flush()


def stdin_reader(q, loop):
    while True:
        b = sys.stdin.buffer.read(CHUNK)
        if not b:
            loop.call_soon_threadsafe(q.put_nowait, None)
            return
        loop.call_soon_threadsafe(q.put_nowait, b)


def split_sentences(buf: str):
    """Return (complete_sentences, remaining_tail) from buf."""
    out, pos = [], 0
    for m in SENT_RE.finditer(buf):
        out.append(m.group(1).strip())
        pos = m.end()
    tail = buf[pos:]
    # Run-on with no terminal punctuation: flush at last space as a clause.
    if len(tail) > SOFT_CAP:
        cut = tail.rfind(" ", 0, SOFT_CAP)
        if cut > 0:
            out.append(tail[:cut].strip())
            tail = tail[cut + 1:]
    return out, tail


async def main():
    uri = f"ws://{HOST}:{PORT}/v1/realtime"
    try:
        ws = await websockets.connect(uri, max_size=None)
    except Exception as e:
        print(f"[stream-voxtral-translate] cannot reach vLLM at {uri}: {e}",
              file=sys.stderr)
        return
    async with ws:
        r = json.loads(await ws.recv())
        if r.get("type") != "session.created":
            print(f"[stream-voxtral-translate] unexpected: {r}",
                  file=sys.stderr)
            return
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("[stream-voxtral-translate] connected, streaming "
              "(ES + running EN)...", file=sys.stderr, flush=True)

        threading.Thread(target=mt_worker, daemon=True).start()
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        threading.Thread(target=stdin_reader, args=(q, loop),
                         daemon=True).start()

        buf = [""]

        async def receiver():
            while True:
                m = json.loads(await ws.recv())
                t = m.get("type")
                if t == "transcription.delta":
                    buf[0] += m["delta"]
                    sents, buf[0] = split_sentences(buf[0])
                    for s in sents:
                        _sent_q.put(s)
                elif t == "transcription.done":
                    if buf[0].strip():
                        _sent_q.put(buf[0].strip())
                        buf[0] = ""
                elif t == "error":
                    print(f"\n[stream-voxtral-translate] server error: {m}",
                          file=sys.stderr)
                    return

        rx = asyncio.create_task(receiver())
        while True:
            chunk = await q.get()
            if chunk is None:
                await ws.send(json.dumps(
                    {"type": "input_audio_buffer.commit", "final": True}))
                break
            await ws.send(json.dumps(
                {"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(chunk).decode()}))
        try:
            await asyncio.wait_for(rx, timeout=15)
        except asyncio.TimeoutError:
            pass
        if buf[0].strip():
            _sent_q.put(buf[0].strip())
    _sent_q.put(None)
    print("[stream-voxtral-translate] bye.", file=sys.stderr)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
