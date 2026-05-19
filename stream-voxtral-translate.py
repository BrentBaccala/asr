#!/home/cosine/asr/mt-env/bin/python
"""
Live Spanish transcription (Voxtral realtime, vLLM) with an English
translation injected inline whenever it is ready.

Pipe the samsung->pony RTP tap in, like stream-voxtral.py:

  ssh cosine@pony
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | ~/asr/stream-voxtral-translate.py

UX: the Spanish flows in near-real-time as Voxtral emits it. Each
completed sentence is translated off-thread on the CPU (NLLB-200
distilled-600M int8, CTranslate2); when a translation is ready it is
printed on its own marked line, then the live Spanish continues. The
EN line therefore appears *interleaved* with the ongoing Spanish (one
sentence behind, wherever the ~0.3-0.8 s MT finished) — that lag is
inherent to keeping the Spanish real-time and is the explicit design.

All stdout goes through ONE writer thread fed by a single queue, so
the live ES deltas and the async EN blocks never race; their order is
exactly the order things became ready. Append-only (no carriage
return), so terminal line-wrap is harmless. Voxtral and the GPU vLLM
server are untouched; MT is CPU-only (zero GPU contention).

Prereqs: vLLM Voxtral server on 127.0.0.1:8000; NLLB CT2 model at
~/asr/models/nllb-600m-ct2.
"""
import argparse
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
EN_PREFIX = "\n      \033[36m[EN]\033[0m "   # cyan tag, own line
EN_SUFFIX = "\n"

ap = argparse.ArgumentParser()
ap.add_argument("--beam", type=int, default=1,
                help="MT beam size (1=greedy, fastest; 2=slightly better, ~2x slower)")
ap.add_argument("--clause", action=argparse.BooleanOptionalAction, default=False,
                help="also flush at clause boundaries , ; : — measured: "
                     "negligible latency gain on real prose AND mistranslates "
                     "context-dependent fragments. Off by default; "
                     "real low-latency path is masked re-translation in a TUI")
ap.add_argument("--min-clause", type=int, default=45,
                help="only clause-flush once the pending clause is >= this many chars")
ap.add_argument("--soft-cap", type=int, default=140,
                help="hard flush a run-on (no punctuation) past this many chars")
args = ap.parse_args()
CLAUSE_RE = re.compile(r'(.+?[,;:]["»”\'\)\]]*)(?:\s+|$)', re.S)

print("[stream-voxtral-translate] loading NLLB...", file=sys.stderr,
      flush=True)
_tok = AutoTokenizer.from_pretrained(NLLB_DIR)
_tok.src_lang = "spa_Latn"
_tr = ctranslate2.Translator(NLLB_DIR, device="cpu", compute_type="int8",
                             inter_threads=2, intra_threads=6)


def translate(es: str) -> str:
    src = _tok.convert_ids_to_tokens(_tok.encode(es))
    r = _tr.translate_batch([src], target_prefix=[["eng_Latn"]],
                            beam_size=args.beam, max_decoding_length=512)
    h = r[0].hypotheses[0]
    if h and h[0] == "eng_Latn":
        h = h[1:]
    return _tok.decode(_tok.convert_tokens_to_ids(h)).strip()


# ---- single stdout owner: drains out_q, renders ES inline + EN blocks ----
# items: ("es", text)  live Spanish delta
#        ("en", text)  finished translation, inject on its own line
out_q: "queue.Queue[tuple[str,str]|None]" = queue.Queue()


def writer():
    at_bol = True                      # at beginning of an output line?
    while True:
        item = out_q.get()
        if item is None:
            if not at_bol:
                sys.stdout.write("\n")
            sys.stdout.flush()
            return
        kind, text = item
        if kind == "es":
            sys.stdout.write(text)
            if text:
                at_bol = text.endswith("\n")
        else:  # "en" — break the Spanish line, print marked, resume after
            if not at_bol:
                sys.stdout.write("\n")
            sys.stdout.write(EN_PREFIX + text + EN_SUFFIX)
            at_bol = True
        sys.stdout.flush()


# ---- MT worker: completed Spanish sentences -> ("en", translation) ----
sent_q: "queue.Queue[str|None]" = queue.Queue()


def mt_worker():
    while True:
        es = sent_q.get()
        if es is None:
            return
        es = es.strip()
        if not es:
            continue
        try:
            en = translate(es)
        except Exception as e:
            en = f"[translate error: {e}]"
        out_q.put(("en", en))


def stdin_reader(q, loop):
    while True:
        b = sys.stdin.buffer.read(CHUNK)
        if not b:
            loop.call_soon_threadsafe(q.put_nowait, None)
            return
        loop.call_soon_threadsafe(q.put_nowait, b)


def split_sentences(buf: str):
    """Pull translatable units off the front of buf.

    Always flush on sentence end (.?!). With --clause, also flush at
    , ; : once the pending clause is long enough (--min-clause) so EN
    appears mid-sentence instead of waiting for the period. Run-ons with
    no punctuation flush at --soft-cap. Returns (units, remaining_tail).
    """
    out, pos = [], 0
    for m in SENT_RE.finditer(buf):
        out.append(m.group(1).strip())
        pos = m.end()
    tail = buf[pos:]
    if args.clause:
        cpos = 0
        for m in CLAUSE_RE.finditer(tail):
            if m.end() - cpos >= args.min_clause:
                out.append(m.group(1).strip())
                cpos = m.end()
        tail = tail[cpos:]
    if len(tail) > args.soft_cap:
        cut = tail.rfind(" ", 0, args.soft_cap)
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
        print("[stream-voxtral-translate] connected — live ES, inline EN...",
              file=sys.stderr, flush=True)

        threading.Thread(target=writer, daemon=True).start()
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
                    d = m["delta"]
                    out_q.put(("es", d))            # live Spanish, now
                    buf[0] += d
                    sents, buf[0] = split_sentences(buf[0])
                    for s in sents:
                        sent_q.put(s)               # translate off-thread
                elif t == "transcription.done":
                    if buf[0].strip():
                        sent_q.put(buf[0].strip())
                        buf[0] = ""
                    out_q.put(("es", "\n"))          # break at utterance end
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
            sent_q.put(buf[0].strip())
    sent_q.put(None)
    out_q.put(None)
    print("[stream-voxtral-translate] bye.", file=sys.stderr)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
