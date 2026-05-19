#!/home/cosine/asr/mt-env/bin/python
"""
asr-tui — live Spanish transcription with an in-place refining English
translation, in a terminal UI.

  ssh cosine@pony
  export XDG_RUNTIME_DIR=/run/user/$(id -u)

  # single-stream (stdin pipe — unchanged behaviour):
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | ~/asr/asr-tui.py          # the TUI
  ... | ~/asr/asr-tui.py --plain                        # headless log

  # dual-stream (script owns both taps — [Remote]/[Me] labelled):
  ~/asr/asr-tui.py --dual                                # the TUI
  ~/asr/asr-tui.py --dual --plain                        # headless log

Pipeline (unchanged from the headless scripts): audio -> Voxtral-Mini-
4B-Realtime via the local vLLM /v1/realtime WS (Spanish); NLLB-200-
distilled-600M int8 (CTranslate2, CPU) for English.

In --dual mode the script spawns two pw-record subprocesses on the two
canonical PipeWire source names (rtp_call_remote_source = Remote,
rtp_call_me_source = Me, as in asr-call-transcribe) and runs two
independent Voxtral /v1/realtime WS sessions concurrently. Each stream
keeps its own in-progress ES/EN; finalized pairs interleave in one
speaker-tagged scrolling history. The live region shows BOTH streams
stacked ([Remote] then [Me]), always present and refining
independently — no active-speaker switching (a silent channel's
sporadic deltas must not flip the panel).

What's new vs stream-voxtral-translate.py: the English for the
*in-progress* sentence is re-translated continuously from the full
Spanish-so-far (full context -> no fragment garbage) and shown with the
last few words masked (the unstable tail). It refines in place in a
fixed pane instead of scrolling, so low-latency EN is possible without
churn. On sentence end the pair freezes into scrolling history.

Audio is on stdin (single-stream) or owned subprocesses (dual), so the
TUI renders via ANSI to the terminal (stdout) and never reads stdin for
keys; quit with Ctrl-C (terminal is restored).
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
import subprocess
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
SAMPLE_RATE = 16000

# Dual-stream sources: (PipeWire node name, label, label ANSI accent).
# Matches asr-call-transcribe (Remote cyan / Me green).
DUAL_SOURCES = [
    ("rtp_call_remote_source", "Remote", "1;36"),  # cyan
    ("rtp_call_me_source",     "Me",     "1;32"),  # green
]
# Single-stream sentinel label (stdin path).
SOLO = "(solo)"

ap = argparse.ArgumentParser()
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--dual", action="store_true",
                help="dual-stream: the script owns both taps "
                     "(rtp_call_remote_source=Remote, "
                     "rtp_call_me_source=Me) via two pw-record "
                     "subprocesses + two Voxtral WS sessions. Without "
                     "this, audio is read from stdin (single stream).")
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
ap.add_argument("--pause-ms", type=int, default=800,
                help="end the working line after this many ms with no new "
                     "transcription delta (speaker paused). 0 disables. "
                     "Keep comfortably above transcription_delay_ms (240)")
ap.add_argument("--plain", action="store_true",
                help="no TUI; print state transitions (headless validation)")
args = ap.parse_args()
CLAUSE_DELIM = re.compile(r'[,;:]')

# Active stream labels (order = render/iteration order).
STREAMS = ([lbl for _, lbl, _ in DUAL_SOURCES] if args.dual else [SOLO])
# label -> ANSI accent for the [label] tag in history (solo: plain).
ACCENT = ({lbl: c for _, lbl, c in DUAL_SOURCES} if args.dual
          else {SOLO: "0"})

# ---------------- shared state ----------------
_lock = threading.Lock()
# Set by the signal handler ONLY. Never acquire _lock in the handler:
# the handler runs in the main thread and would deadlock if interrupted
# while the main thread already holds _lock (this hung a run for 15 min).
_STOP = threading.Event()


def _new_stream_state():
    return {
        "cur_es": "",      # in-progress Spanish (no sentence end yet)
        "cur_en": "",      # masked live English of cur_es
        "delta_t": 0.0,    # wall time of last transcription delta (speech)
    }


_state = {
    "frozen": [],          # list[(speaker, es, en)] — interleaved history
    "streams": {lbl: _new_stream_state() for lbl in STREAMS},
    "active": STREAMS[0],  # label of the last stream to get a delta
    "status": "starting",
    "audio_t": 0.0,        # wall time of last audio chunk, any stream
    "cols": 100,           # live-line width budget, kept current by render
}


def st_get():
    with _lock:
        act = _state["active"]
        s = _state["streams"][act]
        return (list(_state["frozen"]), act, s["cur_es"], s["cur_en"],
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


# finalize jobs ((speaker, sentence)) get priority over live re-translation
_final_q: "queue.Queue[tuple]" = queue.Queue()


def mt_worker():
    # per-stream last-source so a quiet stream's stale cur_es isn't
    # re-translated, and one stream's text never seeds another's.
    last_src = {lbl: None for lbl in STREAMS}
    while True:
        if _STOP.is_set():
            return
        # 1. priority: finalize completed sentences (speaker-tagged)
        try:
            spk, s = _final_q.get_nowait()
        except queue.Empty:
            spk, s = None, None
        if s is not None:
            s = s.strip()
            if s:
                try:
                    en = translate(s)
                except Exception as e:
                    en = f"[mt error: {e}]"
                with _lock:
                    _state["frozen"].append((spk, s, en))
                    _state["frozen"][:] = _state["frozen"][-FROZEN_CAP:]
                if args.plain:
                    tag = "" if spk == SOLO else f"[{spk}] "
                    sys.stdout.write(f"{tag}ES  {s}\n{tag}EN  {en}\n\n")
                    sys.stdout.flush()
            continue
        # 2. else: re-translate every stream's in-progress sentence
        did = False
        for lbl in STREAMS:
            with _lock:
                cur = _state["streams"][lbl]["cur_es"].strip()
            if cur and cur != last_src[lbl]:
                last_src[lbl] = cur
                try:
                    en = translate(cur)
                except Exception as e:
                    en = f"[mt error: {e}]"
                with _lock:
                    # only write back if still the current text (the
                    # stream may have flushed while we translated)
                    if _state["streams"][lbl]["cur_es"].strip() == cur:
                        _state["streams"][lbl]["cur_en"] = \
                            mask_tail(en, args.mask)
                did = True
        if not did:
            time.sleep(0.05)


# ---------------- network (asyncio in a thread) ----------------
def stdin_reader(q, loop):
    while True:
        b = sys.stdin.buffer.read(CHUNK)
        if not b:
            loop.call_soon_threadsafe(q.put_nowait, None)
            return
        loop.call_soon_threadsafe(q.put_nowait, b)


def pw_reader(source_name, q, loop):
    """Dual-stream feed: one pw-record subprocess -> queue (asr-call-
    transcribe's capture shape, but raw byte chunks for the WS)."""
    cmd = ["pw-record", "--target", source_name,
           "--format=s16", f"--rate={SAMPLE_RATE}", "--channels=1", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    try:
        while not _STOP.is_set():
            b = proc.stdout.read(CHUNK)
            if not b:
                break
            loop.call_soon_threadsafe(q.put_nowait, b)
    finally:
        loop.call_soon_threadsafe(q.put_nowait, None)
        try:
            proc.terminate()
        except Exception:
            pass


def take_sentences(spk):
    """Move any completed sentences out of this stream's cur_es into
    _final_q (speaker-tagged)."""
    with _lock:
        s = _state["streams"][spk]
        buf = s["cur_es"]
        out, pos = [], 0
        for m in SENT_RE.finditer(buf):
            out.append(m.group(1).strip())
            pos = m.end()
        if out:
            s["cur_es"] = buf[pos:]
            s["cur_en"] = ""             # reset live EN for the new sentence
    for sent in out:
        _final_q.put((spk, sent))


def clause_flush(spk):
    """Last-resort guard against unbounded growth — NOT a per-line
    chopper. Normal sentences end via punctuation (take_sentences) or a
    speech pause (pause_watcher), which keep whole clauses together so
    NLLB has context. Only a genuine run-on with no punctuation for many
    lines is force-broken here, at the rightmost comma that leaves a
    substantial chunk (>= minc). Breaking mid-sentence into short
    fragments was translating them out of context (e.g. a lone
    'elocuencia' -> garbage); hence the generous multi-line budget and
    large minimum."""
    if not args.line_flush:
        return
    with _lock:
        s = _state["streams"][spk]
        line = max(40, _state["cols"] - 4)       # one ES▸ row of chars
        # allow several wrapped rows before any forced break, so
        # ordinary sentences finalize whole via punctuation/pause
        budget = line * 6
        buf = s["cur_es"]
        minc = max(80, args.min_clause)          # never flush a short bit
        flushed, changed = [], False
        while len(buf) > budget:
            cut = -1
            for mt in CLAUSE_DELIM.finditer(buf[:budget + 1]):
                if mt.end() >= minc:
                    cut = mt.end()               # rightmost within budget
            if cut == -1:
                # No comma within the line: word-cut at the last space
                # that fits, so the live region ALWAYS stays ~1 line and
                # the chunk promotes into history.
                sp = buf.rfind(" ", minc, budget)
                cut = sp if sp > minc else budget
            flushed.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
            changed = True
        if changed:
            s["cur_es"] = buf
            s["cur_en"] = ""
    for sent in flushed:
        if sent:
            _final_q.put((spk, sent))


def pause_watcher():
    """Fourth flush trigger: a speech pause, evaluated PER STREAM.
    take_sentences() ends a line on punctuation, clause_flush() on width
    pressure; this ends it when the speaker just trails off -> no new
    transcription delta for --pause-ms on that stream. Pauses are gaps in
    DELTAS, not audio: pw-record streams silence frames, so each stream
    has its own delta_t. delta_t == 0.0 until the first delta on that
    stream ever arrives: don't flush during a silent direction's startup
    (a silent direction must not flush a phantom line)."""
    if args.pause_ms <= 0:
        return
    gap_s = args.pause_ms / 1000.0
    while not _STOP.is_set():
        time.sleep(0.1)
        pending = []
        with _lock:
            now = time.time()
            for lbl in STREAMS:
                s = _state["streams"][lbl]
                seen = s["delta_t"] > 0.0
                gap = now - s["delta_t"]
                rem = s["cur_es"].strip()
                if seen and gap >= gap_s and rem:
                    # Pause reached: ALWAYS clear the live line so a
                    # short tail can't stay stuck (the "Por eso," that
                    # never promoted nor cleared). Promote to history
                    # only if substantive; a tiny scrap would just make
                    # NLLB hallucinate, so drop it rather than freeze it.
                    s["cur_es"] = ""
                    s["cur_en"] = ""
                    if len(rem) >= max(12, args.min_clause // 2):
                        pending.append((lbl, rem))
        for lbl, rem in pending:
            _final_q.put((lbl, rem))


async def net_main(source_name, label, q, loop):
    """One Voxtral /v1/realtime WS session + one audio feed for `label`.
    source_name is the pw-record target in --dual, or None for the
    stdin (single-stream) path. `q` is this session's audio queue."""
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

        if source_name is None:
            threading.Thread(target=stdin_reader, args=(q, loop),
                             daemon=True).start()
        else:
            threading.Thread(target=pw_reader,
                             args=(source_name, q, loop),
                             daemon=True).start()

        async def receiver():
            while True:
                m = json.loads(await ws.recv())
                t = m.get("type")
                if t == "transcription.delta":
                    d = m["delta"]
                    with _lock:
                        s = _state["streams"][label]
                        s["cur_es"] += d          # keep text fidelity
                        # Only a delta with non-whitespace content counts
                        # as speech activity. An always-on channel keeps
                        # streaming silence; Voxtral answers with empty/
                        # whitespace deltas, which must NOT refresh the
                        # pause clock (else pause_watcher never fires and
                        # a trailed-off line stays stuck — observed when
                        # the source audio is stopped but RTP keeps
                        # sending zeros).
                        if d.strip():
                            s["delta_t"] = time.time()
                            _state["active"] = label
                    take_sentences(label)
                    clause_flush(label)
                elif t == "transcription.done":
                    take_sentences(label)
                    with _lock:
                        s = _state["streams"][label]
                        rem = s["cur_es"].strip()
                        if rem:
                            s["cur_es"] = ""
                            s["cur_en"] = ""
                    if rem:
                        _final_q.put((label, rem))
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
        # Promote whatever is still in this stream's live buffer (the
        # speaker/audio ended mid-sentence, so no period and maybe no
        # transcription.done ever arrives).
        with _lock:
            s = _state["streams"][label]
            rem = s["cur_es"].strip()
            # only promote a substantive tail; a 1-3 char scrap ("se")
            # just makes NLLB hallucinate boilerplate — drop it.
            if rem and len(rem) >= max(12, args.min_clause // 2):
                s["cur_es"] = ""
                s["cur_en"] = ""
            else:
                rem = ""
        if rem:
            _final_q.put((label, rem))
            time.sleep(2.0)            # let the MT worker finalize it


def net_thread():
    async def _run():
        loop = asyncio.get_running_loop()
        if args.dual:
            tasks = []
            for src, lbl, _ in DUAL_SOURCES:
                q = asyncio.Queue()
                tasks.append(asyncio.create_task(
                    net_main(src, lbl, q, loop)))
            await asyncio.gather(*tasks)
        else:
            q = asyncio.Queue()
            await net_main(None, SOLO, q, loop)
    try:
        asyncio.run(_run())
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
            if _STOP.is_set():
                return
            cols, rows = shutil.get_terminal_size((100, 30))
            with _lock:
                _state["cols"] = cols
                frozen = list(_state["frozen"])
                status = _state["status"]
                at = _state["audio_t"]
                snap = [(lbl,
                         _state["streams"][lbl]["cur_es"],
                         _state["streams"][lbl]["cur_en"])
                        for lbl in STREAMS]
            live_mark = "●" if (time.time() - at) < 1.5 else "○"
            dual = STREAMS != [SOLO]
            spk_tag = "  │  Remote+Me" if dual else ""
            head = (f" asr-tui  │  {status}  │  audio {live_mark}{spk_tag}"
                    f"  │  {len(frozen)} done  │  Ctrl-C quit")
            lines = [CSI + "7m" + head[:cols].ljust(cols) + CSI + "0m"]

            def wrap_pref(text, prefix):
                indent = " " * len(prefix)
                body = wrap(text, max(8, cols - len(prefix)))
                return [(prefix if i == 0 else indent) + ln
                        for i, ln in enumerate(body)]

            # --- live block: every stream, always shown, stacked
            # (Remote then Me). No active-speaker selection, so a
            # silent channel's sporadic Voxtral deltas can't flip the
            # panel back and forth (the reported flicker).
            live = []   # list[(text, ansi)]
            for lbl, ces, cen in snap:
                if dual:
                    es_pref, en_pref = f"[{lbl}] ES▸ ", f"[{lbl}] EN▸ "
                else:
                    es_pref, en_pref = "ES▸ ", "EN▸ "
                es_l = (wrap_pref(ces, es_pref) if ces
                        else [es_pref.rstrip()])
                en_l = (wrap_pref(cen, en_pref) if cen
                        else [en_pref + "(translating…)"])
                for ln in es_l:
                    live.append((ln, "1m"))
                for ln in en_l:
                    live.append((ln, "1;36m"))
            # never overflow the screen (that would scroll the terminal
            # and corrupt the layout): keep the most recent live lines.
            live_budget = max(2, rows - 2)
            if len(live) > live_budget:
                live = live[-live_budget:]
            live_h = 1 + len(live)                       # +1 separator
            body_h = max(0, rows - 1 - live_h)
            # history (most recent at the bottom), fills remaining space,
            # interleaved + speaker-tagged.
            hist = []
            for spk, es, en in frozen:
                if spk == SOLO:
                    es_tag, en_tag = "ES ", "EN "
                else:
                    acc = ACCENT.get(spk, "0")
                    es_tag = (CSI + acc + "m" + f"[{spk}] " + CSI + "0m"
                              + "ES ")
                    en_tag = (CSI + acc + "m" + f"[{spk}] " + CSI + "0m"
                              + "EN ")
                for i, ln in enumerate(wrap(es, cols - 4)):
                    hist.append((es_tag if i == 0 else "  ") + ln)
                for i, ln in enumerate(wrap(en, cols - 4)):
                    hist.append((CSI + "36m"
                                 + (en_tag if i == 0 else "  ")
                                 + ln + CSI + "0m"))
                hist.append("")
            hist = hist[-body_h:] if body_h > 0 else []
            while len(hist) < body_h:
                hist.insert(0, "")
            lines += hist
            lines.append(CSI + "2m" + ("─" * cols) + CSI + "0m")
            for ln, ansi in live:
                lines.append(CSI + ansi + ln + CSI + "0m")
            # Newlines BETWEEN lines only — never after the last shown
            # line. A trailing \r\n on the rows-th line drops the cursor
            # one row past the bottom, scrolling the alt-screen up by one
            # every frame, which scrolled the header off the top ~10x/s
            # (the "flickering top line").
            shown = lines[:rows]
            frame = (CSI + "H" + (CSI + "K\r\n").join(shown)
                     + CSI + "K" + CSI + "J")
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
        if _STOP.is_set():
            return
        _, act, _, cen, _, _ = st_get()
        if cen and cen != last:
            tag = "" if act == SOLO else f"[{act}] "
            sys.stderr.write(f"\r[live EN] {tag}{cen[:120]}\033[K")
            sys.stderr.flush()
            last = cen
        time.sleep(0.2)


def main():
    def _stop(*_):
        _STOP.set()                 # lock-free: signal-handler safe
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    threading.Thread(target=mt_worker, daemon=True).start()
    threading.Thread(target=net_thread, daemon=True).start()
    threading.Thread(target=pause_watcher, daemon=True).start()
    if args.plain:
        plain_loop()
    else:
        render()
    print("[asr-tui] bye.", file=sys.stderr)


if __name__ == "__main__":
    main()
