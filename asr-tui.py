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
import atexit
import base64
import json
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

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
                help="short pause: flush the live region as a VISUAL "
                     "chunk after this many ms with no new delta. The "
                     "sentence stays open so MT runs on the whole "
                     "sentence (good context). 0 disables visual flush.")
ap.add_argument("--sentence-close-ms", type=int, default=3000,
                help="long pause: CLOSE the sentence after this many ms "
                     "with no new delta (speaker truly stopped). Marker-"
                     "MT runs and chunk ES/EN backfill. Must be > "
                     "--pause-ms; should be long enough to not fire "
                     "during a natural inter-clause pause.")
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
        # Three live "views" per stream: cur_live is the raw Voxtral
        # transcription (possibly code-switched); cur_es and cur_en are
        # NLLB translations of cur_live into Spanish and English
        # respectively. All three render in the live block as Live▸ /
        # ES▸ / EN▸; on sentence finalize the chunk's es/en fields in
        # `frozen` are filled with the marker-aligned segments.
        "cur_live": "",      # raw Voxtral text since last visual chunk
        "cur_es": "",        # masked live Spanish translation of cur_live
        "cur_en": "",        # masked live English translation of cur_live
        "delta_t": 0.0,      # wall time of last transcription delta (speech)
        # Visual chunking (clause_flush / take_sentences) decouples from
        # MT: each chunk lands in `frozen` immediately with es=None and
        # en=None (live raw flow), and its translations are filled in
        # once the WHOLE sentence is translated TWICE (target=spa_Latn,
        # target=eng_Latn). `sent_raw` joins emitted raw chunks with
        # [N] markers (which NLLB passes through cleanly — empirically
        # verified, 0 marker drops across 7 sentences, perfect order);
        # at sentence end each translation is split on the markers and
        # each open chunk's es/en is backfilled by its chunk_id.
        "sent_raw": "",      # marker-laden raw text for the in-progress sentence
        "open_ids": [],      # chunk_ids whose es/en are pending backfill
    }


_state = {
    "frozen": [],          # list[(speaker, raw, es_or_None, en_or_None, chunk_id)]
    "streams": {lbl: _new_stream_state() for lbl in STREAMS},
    "active": STREAMS[0],  # label of the last stream to get a delta
    "status": "starting",
    "audio_t": 0.0,        # wall time of last audio chunk, any stream
    "cols": 100,           # live-line width budget, kept current by render
    "next_chunk_id": 1,    # monotonic; survives FROZEN_CAP truncation
    # Scrollback: how many rendered rows the history region is scrolled
    # back from the bottom (0 = follow live). Live region keeps updating
    # regardless. Input thread (input_reader) bumps this on wheel/keys;
    # render() consumes + clamps + reads back the clamped value.
    "scroll_offset": 0,
    "last_body_h": 1,      # body height of the last render (PgUp/PgDn unit)
    "last_hist_total": 0,  # total rendered hist lines last frame (clamp)
}


def st_get():
    with _lock:
        act = _state["active"]
        s = _state["streams"][act]
        return (list(_state["frozen"]), act, s["cur_live"], s["cur_en"],
                _state["status"], _state["audio_t"])


# ---------------- NLLB ----------------
print("loading NLLB...", file=sys.stderr, flush=True)
_tok = AutoTokenizer.from_pretrained(NLLB_DIR)
_tok.src_lang = "spa_Latn"
_tr = ctranslate2.Translator(NLLB_DIR, device="cpu", compute_type="int8",
                             inter_threads=1, intra_threads=8)


def translate(text: str, tgt: str = "eng_Latn") -> str:
    """Translate `text` into target language `tgt` (a FLORES-200 code,
    e.g. 'eng_Latn' or 'spa_Latn'). Source language is the tokenizer's
    src_lang set at load time."""
    src = _tok.convert_ids_to_tokens(_tok.encode(text))
    r = _tr.translate_batch([src], target_prefix=[[tgt]],
                            beam_size=args.beam, max_decoding_length=512)
    h = r[0].hypotheses[0]
    if h and h[0] == tgt:
        h = h[1:]
    return _tok.decode(_tok.convert_tokens_to_ids(h)).strip()


def translate_dual(text: str) -> tuple:
    """Translate `text` into BOTH Spanish (spa_Latn) and English
    (eng_Latn) in ONE batched translate_batch call. Measured ~40 %
    faster than two sequential single-target calls on NLLB-600M int8
    CPU — the encoder pass is shared across the batch and only the
    decoder runs twice with different target prefixes. Returns
    (es, en)."""
    src = _tok.convert_ids_to_tokens(_tok.encode(text))
    r = _tr.translate_batch([src, src],
                            target_prefix=[["spa_Latn"], ["eng_Latn"]],
                            beam_size=args.beam, max_decoding_length=512)
    es_hyp = r[0].hypotheses[0]
    en_hyp = r[1].hypotheses[0]
    if es_hyp and es_hyp[0] == "spa_Latn":
        es_hyp = es_hyp[1:]
    if en_hyp and en_hyp[0] == "eng_Latn":
        en_hyp = en_hyp[1:]
    es = _tok.decode(_tok.convert_tokens_to_ids(es_hyp)).strip()
    en = _tok.decode(_tok.convert_tokens_to_ids(en_hyp)).strip()
    return es, en


def mask_tail(en: str, k: int) -> str:
    w = en.split()
    if len(w) <= k:
        return ""           # too short to show anything stable yet
    return " ".join(w[:-k])


# Marker pattern for whole-sentence MT with chunk-aligned EN backfill.
# Inserted as " [N] " between visual chunks in sent_raw; NLLB passes
# them through at semantically aligned English positions.
MARKER_RE = re.compile(r"\s*\[\d+\]\s*")


def _split_on_markers(en: str, expected_n: int) -> list:
    """Split EN on [N] markers; return exactly expected_n segments.
    Marker count mismatch (model dropped/added) is best-effort:
      - fewer markers than expected: pad trailing positions with "".
      - more markers than expected: collapse extras into the last seg.
    Empty segments are legal — they mean "this Spanish chunk merged
    into the next clause in English"; render shows them as `EN —`."""
    parts = [p.strip() for p in MARKER_RE.split(en.strip())]
    if len(parts) == expected_n:
        return parts
    if len(parts) < expected_n:
        return parts + [""] * (expected_n - len(parts))
    return parts[:expected_n - 1] + [" ".join(parts[expected_n - 1:])]


# Sentence-complete jobs: (speaker, sent_raw_with_markers, list_of_chunk_ids).
# One job per sentence (not per chunk).
_final_q: "queue.Queue[tuple]" = queue.Queue()


def _emit_chunk_locked(spk, es_text, is_final):
    """Caller holds _lock. Appends a visual chunk to `frozen` with
    en=None (live ES flow); marker-joins it into the stream's sent_raw.
    Returns a (spk, sent_raw, open_ids) sentence-finalize job iff
    is_final and there's a sentence to translate, else None — caller
    queues the job after releasing the lock."""
    es_text = es_text.strip()
    s = _state["streams"][spk]
    if es_text:
        cid = _state["next_chunk_id"]
        _state["next_chunk_id"] += 1
        # 5-tuple: (spk, raw, es_translation, en_translation, chunk_id);
        # both translations start as None pending sentence-level MT.
        _state["frozen"].append((spk, es_text, None, None, cid))
        _state["frozen"][:] = _state["frozen"][-FROZEN_CAP:]
        # Marker BEFORE this chunk if it isn't the first of the sentence.
        # Number the markers 1, 2, ... per the probe (which used [1][2][3]).
        if s["open_ids"]:
            s["sent_raw"] += f" [{len(s['open_ids'])}] " + es_text
        else:
            s["sent_raw"] = es_text
        s["open_ids"].append(cid)
    if is_final and s["sent_raw"]:
        job = (spk, s["sent_raw"], list(s["open_ids"]))
        s["sent_raw"] = ""
        s["open_ids"] = []
        return job
    return None


def _backfill_chunk_translations_locked(chunk_id: int,
                                        es: str, en: str) -> None:
    """Caller holds _lock. Find the frozen row with this chunk_id and
    set BOTH its es and en fields. Silent no-op if the row was
    truncated by FROZEN_CAP (very-long-call edge case)."""
    fr = _state["frozen"]
    for i in range(len(fr) - 1, -1, -1):    # search from the back (recent)
        if fr[i][4] == chunk_id:            # chunk_id is now at index 4
            spk, raw, _old_es, _old_en, cid = fr[i]
            fr[i] = (spk, raw, es, en, cid)
            return


def mt_worker():
    # per-stream last-source so a quiet stream's stale cur_live isn't
    # re-translated, and one stream's text never seeds another's.
    last_src = {lbl: None for lbl in STREAMS}
    while True:
        if _STOP.is_set():
            return
        # 1. priority: a sentence has completed — translate marker-laden
        # sent_raw, split EN on markers, backfill the open chunks' EN.
        try:
            job = _final_q.get_nowait()
        except queue.Empty:
            job = None
        if job is not None:
            spk, sent_raw, open_ids = job
            try:
                full_es, full_en = translate_dual(sent_raw)
            except Exception as e:
                full_es = full_en = f"[mt error: {e}]"
            es_segs = _split_on_markers(full_es, len(open_ids))
            en_segs = _split_on_markers(full_en, len(open_ids))
            with _lock:
                for cid, es, en in zip(open_ids, es_segs, en_segs):
                    _backfill_chunk_translations_locked(cid, es, en)
            if args.plain:
                tag = "" if spk == SOLO else f"[{spk}] "
                # Per-chunk Live/ES/EN so --plain mirrors `frozen`.
                with _lock:
                    fr = {row[4]: row for row in _state["frozen"]}
                for cid, es, en in zip(open_ids, es_segs, en_segs):
                    row = fr.get(cid)
                    if row is None:
                        continue
                    sys.stdout.write(f"{tag}Live  {row[1]}\n"
                                     f"{tag}ES    {es}\n"
                                     f"{tag}EN    {en}\n")
                sys.stdout.write("\n")
                sys.stdout.flush()
            continue
        # 2. else: re-translate every stream's in-progress raw tail for
        # the live ES + EN previews (cur_live is the tail since the last
        # chunk emit; no markers — preview only, not used for history).
        # One batched translate_batch per stream (~40% faster on this
        # model than two sequential single-target calls).
        did = False
        for lbl in STREAMS:
            with _lock:
                cur = _state["streams"][lbl]["cur_live"].strip()
            if cur and cur != last_src[lbl]:
                last_src[lbl] = cur
                try:
                    es, en = translate_dual(cur)
                except Exception as e:
                    es = en = f"[mt error: {e}]"
                with _lock:
                    # only write back if still the current text (the
                    # stream may have flushed while we translated)
                    if _state["streams"][lbl]["cur_live"].strip() == cur:
                        _state["streams"][lbl]["cur_es"] = \
                            mask_tail(es, args.mask)
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
    """Move any completed sentences out of this stream's cur_live into
    `frozen` as is_final visual chunks (closing each sentence and
    queueing it for whole-sentence MT)."""
    jobs = []
    with _lock:
        s = _state["streams"][spk]
        buf = s["cur_live"]
        out, pos = [], 0
        for m in SENT_RE.finditer(buf):
            out.append(m.group(1).strip())
            pos = m.end()
        if out:
            s["cur_live"] = buf[pos:]
            s["cur_es"] = s["cur_en"] = ""             # reset live EN for the new sentence
        for sent in out:
            job = _emit_chunk_locked(spk, sent, is_final=True)
            if job is not None:
                jobs.append(job)
    for job in jobs:
        _final_q.put(job)


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
        budget = line * 6                        # allow several rows
        buf = s["cur_live"]
        minc = max(80, args.min_clause)
        flushed, changed = [], False
        while len(buf) > budget:
            cut = -1
            for mt in CLAUSE_DELIM.finditer(buf[:budget + 1]):
                if mt.end() >= minc:
                    cut = mt.end()
            if cut == -1:
                sp = buf.rfind(" ", minc, budget)
                cut = sp if sp > minc else budget
            flushed.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
            changed = True
        if changed:
            s["cur_live"] = buf
            s["cur_es"] = s["cur_en"] = ""
        # Visual chunks: NOT is_final. The chunk lands in `frozen` with
        # en=None and joins this sentence's marker-laden sent_raw; the EN
        # will backfill when the sentence completes (punctuation/pause).
        for piece in flushed:
            if piece:
                _emit_chunk_locked(spk, piece, is_final=False)


def pause_watcher():
    """Two-tier pause behaviour, evaluated PER STREAM. Pauses are gaps
    in transcription DELTAS, not audio (pw-record streams silence
    frames; each stream has its own delta_t, refreshed only by deltas
    with non-whitespace content).

      • short pause (--pause-ms, default 800ms) -> flush the live
        region as a VISUAL chunk (is_final=False). The chunk lands in
        history with ES/EN = "⋯"; sent_raw keeps accumulating with the
        next [N] marker. Marker-MT does NOT run yet. This gives the
        comfortable visual flow between clauses without fragmenting MT.
      • long  pause (--sentence-close-ms, default 3000ms) -> CLOSE the
        sentence (is_final=True). Marker-MT translates the whole
        sent_raw and all open chunks get their ES/EN backfilled.

    delta_t == 0.0 until the first delta on a stream ever arrives: a
    silent direction must not flush a phantom line during startup."""
    if args.pause_ms <= 0 and args.sentence_close_ms <= 0:
        return
    visual_s = args.pause_ms / 1000.0 if args.pause_ms > 0 else None
    close_s = (args.sentence_close_ms / 1000.0
               if args.sentence_close_ms > 0 else None)
    min_sub = max(12, args.min_clause // 2)
    while not _STOP.is_set():
        time.sleep(0.1)
        jobs = []
        with _lock:
            now = time.time()
            for lbl in STREAMS:
                s = _state["streams"][lbl]
                if s["delta_t"] <= 0.0:
                    continue
                gap = now - s["delta_t"]
                rem = s["cur_live"].strip()
                # Long pause first (higher threshold) — covers the case
                # where the speaker truly stopped and the sentence must
                # close so open chunks' ES/EN can backfill.
                if (close_s is not None and gap >= close_s
                        and (rem or s["open_ids"])):
                    s["cur_live"] = ""
                    s["cur_es"] = s["cur_en"] = ""
                    tail = rem if (rem and len(rem) >= min_sub) else ""
                    job = _emit_chunk_locked(lbl, tail, is_final=True)
                    if job is not None:
                        jobs.append(job)
                # Short pause: flush a visual chunk, keep the sentence
                # open so MT still translates the whole utterance.
                elif (visual_s is not None and gap >= visual_s
                      and rem and len(rem) >= min_sub):
                    s["cur_live"] = ""
                    s["cur_es"] = s["cur_en"] = ""
                    _emit_chunk_locked(lbl, rem, is_final=False)
        for job in jobs:
            _final_q.put(job)


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
                        s["cur_live"] += d          # keep text fidelity
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
                    job = None
                    with _lock:
                        s = _state["streams"][label]
                        rem = s["cur_live"].strip()
                        if rem or s["open_ids"]:
                            s["cur_live"] = ""
                            s["cur_es"] = s["cur_en"] = ""
                            # Always close the sentence; if rem is empty
                            # we still need to flush any open chunks.
                            job = _emit_chunk_locked(label, rem,
                                                    is_final=True)
                    if job is not None:
                        _final_q.put(job)
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
        # Close the in-progress sentence: emit any substantive tail as
        # the final chunk, then translate sent_raw so open chunks get
        # backfilled. If only a 1-3 char scrap remains we discard the
        # tail (NLLB hallucinates on scraps) but still close so any
        # earlier-flushed chunks finish.
        job = None
        with _lock:
            s = _state["streams"][label]
            rem = s["cur_live"].strip()
            if rem or s["open_ids"]:
                s["cur_live"] = ""
                s["cur_es"] = s["cur_en"] = ""
                tail = rem if (rem and len(rem)
                               >= max(12, args.min_clause // 2)) else ""
                job = _emit_chunk_locked(label, tail, is_final=True)
        if job is not None:
            _final_q.put(job)
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
# Mouse tracking on the alt-screen: basic button events (?1000) + SGR
# extended encoding (?1006). The terminal then sends wheel-up/down as
# CSI < 64;col;row M / CSI < 65;col;row M on /dev/tty.
MOUSE_ON  = CSI + "?1000h" + CSI + "?1006h"
MOUSE_OFF = CSI + "?1006l" + CSI + "?1000l"


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


# ---------------- scrollback input ----------------
# Wheel/keys arrive on /dev/tty (NOT stdin — stdin is the audio pipe in
# single-stream mode). We open /dev/tty separately, put it in cbreak so
# we get bytes per keystroke without echo, parse CSI sequences, and
# update _state["scroll_offset"]. ISIG stays on so Ctrl-C still
# delivers SIGINT to the main thread.

_tty_fd = None
_tty_old_attrs = None
_WHEEL_LINES = 3            # rows per wheel click


def _tty_setup():
    """Open /dev/tty in cbreak mode and remember its prior attrs.
    Returns the fd on success, None if /dev/tty isn't available (no
    controlling terminal — scrollback then silently disabled)."""
    global _tty_fd, _tty_old_attrs
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        _tty_old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, OSError):
        os.close(fd)
        return None
    _tty_fd = fd
    atexit.register(_tty_restore)
    return fd


def _tty_restore():
    """Restore /dev/tty to its prior attrs. Idempotent + crash-safe
    (atexit-registered too) so a hard exit can't leave the user's
    shell in cbreak."""
    global _tty_fd, _tty_old_attrs
    if _tty_fd is not None and _tty_old_attrs is not None:
        try:
            termios.tcsetattr(_tty_fd, termios.TCSANOW, _tty_old_attrs)
        except Exception:
            pass
        try:
            os.close(_tty_fd)
        except Exception:
            pass
    _tty_fd, _tty_old_attrs = None, None


def _scroll(delta):
    """Adjust scroll_offset by delta (positive = back/up, negative =
    down/toward live). Clamping happens in render() — it knows the
    true upper bound (total hist lines - body_h)."""
    with _lock:
        _state["scroll_offset"] = max(0, _state["scroll_offset"] + delta)


# SGR mouse event:  CSI <  B ; X ; Y  M/m   (M=press, m=release)
_MOUSE_RE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


def input_reader(fd):
    """Daemon: read /dev/tty, parse wheel + arrow/PgUp/PgDn/Home/End/q,
    drive scroll_offset and the quit signal."""
    buf = b""
    while not _STOP.is_set():
        try:
            r, _, _ = select.select([fd], [], [], 0.1)
        except (OSError, ValueError):
            return
        if not r:
            continue
        try:
            chunk = os.read(fd, 256)
        except (BlockingIOError, OSError):
            continue
        if not chunk:
            continue
        buf += chunk
        # Try to consume one event at a time from the front of buf.
        while buf:
            # SGR mouse first (longest specific match)
            m = _MOUSE_RE.match(buf)
            if m:
                btn, _x, _y, _kind = m.groups()
                btn = int(btn)
                if btn == 64:                    # wheel up
                    _scroll(+_WHEEL_LINES)
                elif btn == 65:                  # wheel down
                    _scroll(-_WHEEL_LINES)
                buf = buf[m.end():]
                continue
            b0 = buf[:1]
            if b0 in (b"q", b"Q"):
                _STOP.set()
                buf = buf[1:]
                continue
            if b0 != b"\x1b":
                # ignore stray printable chars
                buf = buf[1:]
                continue
            # ESC-prefixed sequences
            if buf.startswith(b"\x1b[A"):    _scroll(+1);              buf = buf[3:]; continue
            if buf.startswith(b"\x1b[B"):    _scroll(-1);              buf = buf[3:]; continue
            if buf.startswith(b"\x1b[5~"):   _scroll(+_state["last_body_h"]); buf = buf[4:]; continue
            if buf.startswith(b"\x1b[6~"):   _scroll(-_state["last_body_h"]); buf = buf[4:]; continue
            if buf.startswith(b"\x1b[H") or buf.startswith(b"\x1b[1~"):
                with _lock:
                    _state["scroll_offset"] = 10**9   # clamped down to max in render
                buf = buf[3 if buf[2:3] == b"H" else 4:]; continue
            if buf.startswith(b"\x1b[F") or buf.startswith(b"\x1b[4~"):
                with _lock:
                    _state["scroll_offset"] = 0
                buf = buf[3 if buf[2:3] == b"F" else 4:]; continue
            # Unknown ESC sequence: try to skip past terminator, else
            # break to wait for more bytes.
            if len(buf) < 2:
                break
            if buf[1:2] not in (b"[", b"O"):
                buf = buf[1:]
                continue
            term = -1
            for i in range(2, len(buf)):
                if 0x40 <= buf[i] <= 0x7e:
                    term = i; break
            if term == -1:
                break       # incomplete; wait
            buf = buf[term + 1:]


def render():
    w = sys.stdout.write
    w(ALT_ON + CUR_HIDE + MOUSE_ON)
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
                         _state["streams"][lbl]["cur_live"],
                         _state["streams"][lbl]["cur_es"],
                         _state["streams"][lbl]["cur_en"])
                        for lbl in STREAMS]
            live_mark = "●" if (time.time() - at) < 1.5 else "○"
            dual = STREAMS != [SOLO]
            spk_tag = "  │  Remote+Me" if dual else ""
            with _lock:
                so = _state["scroll_offset"]
            scroll_tag = f"  │  ↑ scrolled +{so}" if so > 0 else ""
            head = (f" asr-tui  │  {status}  │  audio {live_mark}"
                    f"{spk_tag}{scroll_tag}  │  {len(frozen)} done  "
                    f"│  Ctrl-C quit")
            lines = [CSI + "7m" + head[:cols].ljust(cols) + CSI + "0m"]

            def wrap_pref(text, prefix):
                indent = " " * len(prefix)
                body = wrap(text, max(8, cols - len(prefix)))
                return [(prefix if i == 0 else indent) + ln
                        for i, ln in enumerate(body)]

            # --- live block: three lines per stream — Live (raw
            # Voxtral transcription), ES (NLLB Spanish translation,
            # masked-tail preview), EN (NLLB English translation,
            # masked-tail preview). Always shown, both streams
            # stacked; no active-speaker selection.
            live = []   # list[(text, ansi)]
            for lbl, ces_live, ces_trans, cen_trans in snap:
                if dual:
                    live_pref = f"[{lbl}] Live▸ "
                    es_pref   = f"[{lbl}] ES▸   "
                    en_pref   = f"[{lbl}] EN▸   "
                else:
                    live_pref = "Live▸ "
                    es_pref   = "ES▸   "
                    en_pref   = "EN▸   "
                live_l = (wrap_pref(ces_live, live_pref) if ces_live
                          else [live_pref.rstrip()])
                # "(translating…)" only when there's raw text awaiting a
                # translation — not when the channel is idle (cur_live
                # empty), where all three lines should sit empty.
                if not ces_live:
                    es_l = [es_pref.rstrip()]
                    en_l = [en_pref.rstrip()]
                else:
                    es_l = (wrap_pref(ces_trans, es_pref) if ces_trans
                            else [es_pref + "(translating…)"])
                    en_l = (wrap_pref(cen_trans, en_pref) if cen_trans
                            else [en_pref + "(translating…)"])
                for ln in live_l:
                    live.append((ln, "1m"))           # raw: bold
                for ln in es_l:
                    live.append((ln, "1;33m"))        # ES: bold yellow
                for ln in en_l:
                    live.append((ln, "1;36m"))        # EN: bold cyan
            # never overflow the screen (that would scroll the terminal
            # and corrupt the layout): keep the most recent live lines.
            live_budget = max(2, rows - 2)
            if len(live) > live_budget:
                live = live[-live_budget:]
            live_h = 1 + len(live)                       # +1 separator
            body_h = max(0, rows - 1 - live_h)
            # history (most recent at the bottom). Each chunk renders
            # three lines — Live (raw transcription), ES (NLLB Spanish
            # translation), EN (NLLB English translation), plus a
            # separator blank. es/en are None until the whole sentence
            # is translated, "" when the marker-split placed that
            # field's content on a neighboring chunk (rendered "—").
            def _trans_line(en_tag, value, ansi):
                if value is None:
                    return CSI + "2;" + ansi + "m" + en_tag + "⋯" + CSI + "0m"
                if value == "":
                    return CSI + "2;" + ansi + "m" + en_tag + "—" + CSI + "0m"
                wrapped = wrap(value, cols - 4)
                return [(CSI + ansi + "m"
                         + (en_tag if i == 0 else "  ")
                         + ln + CSI + "0m")
                        for i, ln in enumerate(wrapped)]
            hist = []
            for spk, raw, es, en, _cid in frozen:
                if spk == SOLO:
                    live_tag = "Live "
                    es_tag = "ES   "
                    en_tag = "EN   "
                else:
                    acc = ACCENT.get(spk, "0")
                    lead = CSI + acc + "m" + f"[{spk}] " + CSI + "0m"
                    live_tag = lead + "Live "
                    es_tag   = lead + "ES   "
                    en_tag   = lead + "EN   "
                # Live (raw) — never None/empty: it's what got emitted.
                for i, ln in enumerate(wrap(raw, cols - 4)):
                    hist.append((live_tag if i == 0 else "  ") + ln)
                for fn, tag, ansi in (
                        (es, es_tag, "33"),   # yellow for ES translation
                        (en, en_tag, "36")):  # cyan for EN translation
                    out = _trans_line(tag, fn, ansi)
                    if isinstance(out, list):
                        hist.extend(out)
                    else:
                        hist.append(out)
                hist.append("")
            # Apply scrollback: scroll_offset is rows back from the
            # bottom (the live-following position). Clamp it to a sane
            # range and write the clamped value back so the indicator
            # never advertises a position you can't reach.
            total = len(hist)
            max_off = max(0, total - body_h)
            so = min(max(0, so), max_off)
            with _lock:
                _state["scroll_offset"] = so
                _state["last_body_h"] = max(1, body_h)
                _state["last_hist_total"] = total
            if body_h > 0:
                end = total - so
                start = max(0, end - body_h)
                hist = hist[start:end]
            else:
                hist = []
            while len(hist) < body_h:
                hist.insert(0, "")
            lines += hist
            lines.append(CSI + "2m" + ("─" * cols) + CSI + "0m")
            for ln, ansi in live:
                lines.append(CSI + ansi + ln + CSI + "0m")
            # Position each row ABSOLUTELY with CSI <r>;1H rather than
            # advancing with \r\n. The header is exactly `cols` chars
            # wide; on eager-wrap terminals the cols-th char auto-wraps
            # the cursor, so a between-lines \r\n then over-advances by
            # one row, line N lands at row 2N-1, and the late rows
            # scroll the header off. Absolute positioning avoids any
            # dependence on auto-wrap or newline-induced scroll.
            shown = lines[:rows]
            parts = [f"{CSI}{i+1};1H{CSI}K{ln}"
                     for i, ln in enumerate(shown)]
            if len(shown) < rows:
                parts.append(f"{CSI}{len(shown)+1};1H{CSI}J")
            frame = "".join(parts)
            w(frame)
            sys.stdout.flush()
            time.sleep(0.1)
    finally:
        w(MOUSE_OFF + CUR_SHOW + ALT_OFF)
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
        # Open /dev/tty for keyboard/mouse input (scrollback). Mouse
        # tracking is enabled inside render() (MOUSE_ON in the alt-
        # screen setup) so terminal mode is restored even if render
        # crashes. tty restoration is also atexit-registered.
        fd = _tty_setup()
        try:
            if fd is not None:
                threading.Thread(target=input_reader, args=(fd,),
                                 daemon=True).start()
            render()
        finally:
            _tty_restore()
    print("[asr-tui] bye.", file=sys.stderr)


if __name__ == "__main__":
    main()
