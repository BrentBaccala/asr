#!/home/cosine/asr/mt-env/bin/python
"""
asr-tui — live Spanish transcription with an in-place refining English
translation, in a terminal UI.

  ssh cosine@pony
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | ~/asr/asr-tui.py          # the TUI
  ... | ~/asr/asr-tui.py --plain                        # headless log

Pipeline (unchanged from the headless scripts): audio on stdin ->
Voxtral-Mini-4B-Realtime via the local vLLM /v1/realtime WS (Spanish);
NLLB-200-distilled-600M int8 (CTranslate2, CPU) for English.

What's new vs stream-voxtral-translate.py: the English for the
*in-progress* sentence is re-translated continuously from the full
Spanish-so-far (full context -> no fragment garbage) and shown with the
last few words masked (the unstable tail). It refines in place in a
fixed pane instead of scrolling, so low-latency EN is possible without
churn. On sentence end the pair freezes into scrolling history.

Audio is on stdin, so the TUI renders via ANSI to the terminal (stdout)
and never reads stdin for keys; quit with Ctrl-C (terminal is restored).
"""
import argparse
import asyncio
import base64
import json
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import websockets
import ctranslate2
from transformers import AutoTokenizer

HOST, PORT = "127.0.0.1", 8000
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
NLLB_DIR = os.path.expanduser("~/asr/models/nllb-600m-ct2")
CHUNK = 2048 * 2
SENT_RE = re.compile(r'(.+?[.!?…]+["»”\'\)\]]*)(?:\s+|$)', re.S)
MASK_K = 4              # words of the live EN tail to hide (unstable)
FROZEN_CAP = 300

ap = argparse.ArgumentParser()
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--mask", type=int, default=MASK_K,
                help="words of the in-progress EN tail to hide (default 4)")
ap.add_argument("--line-flush", action=argparse.BooleanOptionalAction,
                default=True,
                help="when the live line fills the screen width, commit the "
                     "leading clause at a comma (keeps live to ~1 line; "
                     "trades some full-sentence MT coherence). On by default")
ap.add_argument("--min-clause", type=int, default=25,
                help="never line-flush a leading clause shorter than this "
                     "many chars (avoids tiny-fragment mistranslation)")
ap.add_argument("--plain", action="store_true",
                help="no TUI; print state transitions (headless validation)")
args = ap.parse_args()
CLAUSE_DELIM = re.compile(r'[,;:]')

# ---------------- shared state ----------------
_lock = threading.Lock()
_state = {
    "frozen": [],          # list[(es, en)]
    "cur_es": "",          # in-progress Spanish (no sentence end yet)
    "cur_en": "",          # masked live English of cur_es
    "status": "starting",
    "audio_t": 0.0,        # wall time of last audio chunk (activity)
    "cols": 100,           # live-line width budget, kept current by render
    "stop": False,
}


def st_get():
    with _lock:
        return (list(_state["frozen"]), _state["cur_es"], _state["cur_en"],
                _state["status"], _state["audio_t"])


# ---------------- NLLB ----------------
print("loading NLLB...", file=sys.stderr, flush=True)
_tok = AutoTokenizer.from_pretrained(NLLB_DIR)
_tok.src_lang = "spa_Latn"
_tr = ctranslate2.Translator(NLLB_DIR, device="cpu", compute_type="int8",
                             inter_threads=1, intra_threads=8)


def translate(es: str) -> str:
    src = _tok.convert_ids_to_tokens(_tok.encode(es))
    r = _tr.translate_batch([src], target_prefix=[["eng_Latn"]],
                            beam_size=args.beam, max_decoding_length=512)
    h = r[0].hypotheses[0]
    if h and h[0] == "eng_Latn":
        h = h[1:]
    return _tok.decode(_tok.convert_tokens_to_ids(h)).strip()


def mask_tail(en: str, k: int) -> str:
    w = en.split()
    if len(w) <= k:
        return ""           # too short to show anything stable yet
    return " ".join(w[:-k])


# finalize jobs (sentence text) get priority over live re-translation
_final_q: "queue.Queue[str]" = queue.Queue()


def mt_worker():
    last_src = None
    while True:
        with _lock:
            if _state["stop"]:
                return
        # 1. priority: finalize completed sentences
        try:
            s = _final_q.get_nowait()
        except queue.Empty:
            s = None
        if s is not None:
            s = s.strip()
            if s:
                try:
                    en = translate(s)
                except Exception as e:
                    en = f"[mt error: {e}]"
                with _lock:
                    _state["frozen"].append((s, en))
                    _state["frozen"][:] = _state["frozen"][-FROZEN_CAP:]
                if args.plain:
                    sys.stdout.write(f"ES  {s}\nEN  {en}\n\n")
                    sys.stdout.flush()
            continue
        # 2. else: re-translate the current in-progress sentence
        with _lock:
            cur = _state["cur_es"].strip()
        if cur and cur != last_src:
            last_src = cur
            try:
                en = translate(cur)
            except Exception as e:
                en = f"[mt error: {e}]"
            with _lock:
                _state["cur_en"] = mask_tail(en, args.mask)
        else:
            time.sleep(0.05)


# ---------------- network (asyncio in a thread) ----------------
def stdin_reader(q, loop):
    while True:
        b = sys.stdin.buffer.read(CHUNK)
        if not b:
            loop.call_soon_threadsafe(q.put_nowait, None)
            return
        loop.call_soon_threadsafe(q.put_nowait, b)


def take_sentences():
    """Move any completed sentences out of cur_es into _final_q."""
    with _lock:
        buf = _state["cur_es"]
        out, pos = [], 0
        for m in SENT_RE.finditer(buf):
            out.append(m.group(1).strip())
            pos = m.end()
        if out:
            _state["cur_es"] = buf[pos:]
            _state["cur_en"] = ""        # reset live EN for the new sentence
    for s in out:
        _final_q.put(s)


def clause_flush():
    """Pressure-based: only when the in-progress Spanish would exceed one
    live line, commit a leading clause at the *rightmost* comma that still
    fits the line (so the flushed clause is near-full -> good context,
    short tail restarts). No comma within a line and badly overflowing ->
    hard word-cut so it can't run forever; mildly over with no comma ->
    leave it (let it wrap until a comma/sentence end arrives). Each
    flushed clause goes to history like a sentence."""
    if not args.line_flush:
        return
    with _lock:
        budget = max(40, _state["cols"] - 4)     # one ES▸ line of chars
        buf = _state["cur_es"]
        minc = max(15, args.min_clause)
        flushed, changed = [], False
        while len(buf) > budget:
            cut = -1
            for mt in CLAUSE_DELIM.finditer(buf[:budget + 1]):
                if mt.end() >= minc:
                    cut = mt.end()               # rightmost within budget
            if cut == -1:
                if len(buf) > 2 * budget:        # comma-less run-on: cut
                    sp = buf.rfind(" ", minc, budget)
                    cut = sp if sp > minc else budget
                else:
                    break                        # let it wrap for now
            flushed.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
            changed = True
        if changed:
            _state["cur_es"] = buf
            _state["cur_en"] = ""
    for s in flushed:
        if s:
            _final_q.put(s)


async def net_main():
    uri = f"ws://{HOST}:{PORT}/v1/realtime"
    try:
        ws = await websockets.connect(uri, max_size=None)
    except Exception as e:
        with _lock:
            _state["status"] = f"vLLM unreachable: {e}"
        return
    async with ws:
        r = json.loads(await ws.recv())
        if r.get("type") != "session.created":
            with _lock:
                _state["status"] = f"unexpected: {r}"
            return
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        with _lock:
            _state["status"] = "streaming"

        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        threading.Thread(target=stdin_reader, args=(q, loop),
                         daemon=True).start()

        async def receiver():
            while True:
                m = json.loads(await ws.recv())
                t = m.get("type")
                if t == "transcription.delta":
                    with _lock:
                        _state["cur_es"] += m["delta"]
                    take_sentences()
                    clause_flush()
                elif t == "transcription.done":
                    take_sentences()
                    with _lock:
                        rem = _state["cur_es"].strip()
                        if rem:
                            _state["cur_es"] = ""
                            _state["cur_en"] = ""
                    if rem:
                        _final_q.put(rem)
                elif t == "error":
                    with _lock:
                        _state["status"] = f"server error: {m}"
                    return

        rx = asyncio.create_task(receiver())
        while True:
            chunk = await q.get()
            if chunk is None:
                with _lock:
                    _state["status"] = "audio stream ended"
                await ws.send(json.dumps(
                    {"type": "input_audio_buffer.commit", "final": True}))
                break
            with _lock:
                _state["audio_t"] = time.time()
            await ws.send(json.dumps(
                {"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(chunk).decode()}))
        try:
            await asyncio.wait_for(rx, timeout=10)
        except asyncio.TimeoutError:
            pass


def net_thread():
    try:
        asyncio.run(net_main())
    except Exception as e:
        with _lock:
            _state["status"] = f"net error: {e}"


# ---------------- ANSI TUI ----------------
CSI = "\033["
ALT_ON, ALT_OFF = CSI + "?1049h", CSI + "?1049l"
CUR_HIDE, CUR_SHOW = CSI + "?25l", CSI + "?25h"


def wrap(s, w):
    s = s.replace("\n", " ")
    out, line = [], ""
    for word in s.split(" "):
        if not word:
            continue
        if len(line) + len(word) + 1 > w and line:
            out.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return out or [""]


def render():
    w = sys.stdout.write
    w(ALT_ON + CUR_HIDE)
    sys.stdout.flush()
    try:
        while True:
            with _lock:
                if _state["stop"]:
                    return
            cols, rows = shutil.get_terminal_size((100, 30))
            with _lock:
                _state["cols"] = cols
            frozen, ces, cen, status, at = st_get()
            live = "●" if (time.time() - at) < 1.5 else "○"
            head = (f" asr-tui  │  {status}  │  audio {live}  │  "
                    f"{len(frozen)} done  │  Ctrl-C quit")
            lines = [CSI + "7m" + head[:cols].ljust(cols) + CSI + "0m"]
            # --- live block: full text, multi-line, expands/contracts ---
            def wrap_pref(text, prefix):
                indent = " " * len(prefix)
                body = wrap(text, max(8, cols - len(prefix)))
                return [(prefix if i == 0 else indent) + ln
                        for i, ln in enumerate(body)]

            es_lines = wrap_pref(ces, "ES▸ ") if ces else ["ES▸"]
            en_lines = (wrap_pref(cen, "EN▸ ") if cen
                        else ["EN▸ (translating…)"])
            # never overflow the screen (that would scroll the terminal
            # and corrupt the layout): header + separator + live must fit.
            # If the live block is huge, keep its most recent lines —
            # the refining EN tail — and prioritise EN over ES.
            live_budget = max(2, rows - 2)
            if len(es_lines) + len(en_lines) > live_budget:
                en_keep = min(len(en_lines), max(1, live_budget // 2))
                es_keep = max(1, live_budget - en_keep)
                es_lines = es_lines[-es_keep:]
                en_lines = en_lines[-en_keep:]
            live_h = 1 + len(es_lines) + len(en_lines)   # +1 separator
            body_h = max(0, rows - 1 - live_h)
            # history (most recent at the bottom), fills remaining space
            hist = []
            for es, en in frozen:
                for i, ln in enumerate(wrap(es, cols - 4)):
                    hist.append(("  " if i else "ES ") + ln)
                for i, ln in enumerate(wrap(en, cols - 4)):
                    hist.append((CSI + "36m" + ("  " if i else "EN ")
                                 + ln + CSI + "0m"))
                hist.append("")
            hist = hist[-body_h:] if body_h > 0 else []
            while len(hist) < body_h:
                hist.insert(0, "")
            lines += hist
            lines.append(CSI + "2m" + ("─" * cols) + CSI + "0m")
            for ln in es_lines:
                lines.append(CSI + "1m" + ln + CSI + "0m")
            for ln in en_lines:
                lines.append(CSI + "1;36m" + ln + CSI + "0m")
            frame = CSI + "H"
            for ln in lines[:rows]:
                frame += ln + CSI + "K\r\n"
            frame += CSI + "J"
            w(frame)
            sys.stdout.flush()
            time.sleep(0.1)
    finally:
        w(CUR_SHOW + ALT_OFF)
        sys.stdout.flush()


def plain_loop():
    # mt_worker prints ES/EN on finalize; just keep alive + show live EN
    last = ""
    while True:
        with _lock:
            if _state["stop"]:
                return
        _, _, cen, _, _ = st_get()
        if cen and cen != last:
            sys.stderr.write(f"\r[live EN] {cen[:120]}\033[K")
            sys.stderr.flush()
            last = cen
        time.sleep(0.2)


def main():
    def _stop(*_):
        with _lock:
            _state["stop"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    threading.Thread(target=mt_worker, daemon=True).start()
    threading.Thread(target=net_thread, daemon=True).start()
    if args.plain:
        plain_loop()
    else:
        render()
    print("[asr-tui] bye.", file=sys.stderr)


if __name__ == "__main__":
    main()
