#!/usr/bin/env python3
"""wx_caption.py — diarized, word-timed captioning pipeline (pony, whisperx venv).

Faithful generalization of the 2026-07-23 asr-eval bake-off winner:

    plain faster-whisper (NO VAD)  ->  whisperx forced alignment
      ->  pyannote diarization  ->  snap speaker turns to pauses
      ->  JSON / SRT / ASS / txt

Why this shape (see ~/project/reports/recording-captioning-pipeline-study.md):
  * Whisper beat the audio-LLMs (Voxtral, Qwen2.5-Omni, Phi-4) on NATIVE word
    timestamps + verbatim completeness — the two things captions need.
  * WhisperX is used ONLY for align + diarize. Its VAD-batched transcriber
    silently drops overlapping speech, so transcription is plain faster-whisper
    with vad_filter=False.
  * pyannote turn boundaries can land mid-phrase; snap_turns() moves each
    speaker change to the largest nearby inter-word pause.

Run with the whisperx venv python:
  /mnt/models/venvs/whisperx/bin/python wx_caption.py INPUT --out PREFIX [opts]

INPUT may be any audio/video ffmpeg can read; it is resampled to 16 kHz mono
internally. Emits PREFIX.json, PREFIX.srt, PREFIX.ass, PREFIX.txt.
"""
import os, sys, json, gc, argparse, subprocess, tempfile
from collections import Counter

os.environ.setdefault("HF_HOME", "/mnt/models/hf")

HF_HUB = "/mnt/models/hf/hub"
TOKEN_FILE = "/home/claude/.hf_token"
# ASS speaker colours (&HAABBGGRR, i.e. BGR); cycled if more speakers than colours
ASS_COLORS = ["&H00FFFF00", "&H0000FFFF", "&H0000FF00", "&H00FF66CC",
              "&H000066FF", "&H00FFCC66", "&H006699FF", "&H00CC99FF"]
MAX_CHARS = 84          # ~2 lines of ~42 chars once libass WrapStyle=0 wraps
SNAP_WINDOW = 3         # words either side of a detected speaker change
SNAP_MIN_GAP = 0.25     # only snap to an inter-word gap at least this long


def log(*a):
    print("[wx_caption]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- timestamps
def ts_srt(t):
    ms = int(round(float(t) * 1000)); h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_ass(t):
    cs = int(round(float(t) * 100)); h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000); s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------- audio prep
def to_wav16k(src, dst):
    """Resample any ffmpeg-readable input to 16 kHz mono s16le wav."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src,
         "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", dst],
        check=True, capture_output=True)


# ---------------------------------------------------------------- snap turns
def snap_turns(segs, window=SNAP_WINDOW, min_gap=SNAP_MIN_GAP):
    """Move pyannote speaker-change boundaries to the nearest real pause.
    Mutates `segs` in place; recomputes each segment's dominant speaker."""
    flat = []
    for si, s in enumerate(segs):
        for wi, w in enumerate(s.get("words", [])):
            if w.get("start") is None or w.get("end") is None:
                continue
            flat.append({"si": si, "wi": wi,
                         "start": float(w["start"]), "end": float(w["end"]),
                         "spk": w.get("speaker") or s.get("speaker")})

    def gap_before(i):
        return 0.0 if i <= 0 else flat[i]["start"] - flat[i - 1]["end"]

    moves = 0
    i = 1
    while i < len(flat):
        if (flat[i]["spk"] and flat[i - 1]["spk"]
                and flat[i]["spk"] != flat[i - 1]["spk"]):
            lo = max(1, i - window)
            hi = min(len(flat), i + window + 1)
            best = max(range(lo, hi), key=gap_before)
            if (best != i and gap_before(best) >= min_gap
                    and gap_before(best) > gap_before(i)):
                old, new = flat[i - 1]["spk"], flat[i]["spk"]
                if best < i:
                    for k in range(best, i):
                        flat[k]["spk"] = new
                else:
                    for k in range(i, best):
                        flat[k]["spk"] = old
                moves += 1
                i = best
        i += 1

    for f in flat:
        segs[f["si"]]["words"][f["wi"]]["speaker"] = f["spk"]
    for s in segs:
        c = Counter(w.get("speaker") for w in s.get("words", [])
                    if w.get("speaker"))
        if c:
            s["speaker"] = c.most_common(1)[0][0]
    return moves


# ---------------------------------------------------------------- naming
def speaker_names(segs, override=None):
    """Map raw pyannote labels (SPEAKER_00...) to display names in order of
    first appearance. `override` is an optional list like ['Bruce','Brent']."""
    order = []
    for s in segs:
        sp = s.get("speaker")
        if sp and sp not in order:
            order.append(sp)
    names = {}
    for idx, sp in enumerate(order):
        if override and idx < len(override):
            names[sp] = override[idx]
        else:
            names[sp] = f"Speaker {idx + 1}"
    return names, order


# ---------------------------------------------------------------- writers
def write_txt(segs, names, path):
    with open(path, "w") as f:
        for s in segs:
            who = names.get(s.get("speaker"), "")
            txt = s.get("text", "").strip()
            if not txt:
                continue
            f.write((f"{who}: " if who else "") + txt + "\n")


def write_srt(segs, names, path):
    with open(path, "w") as f:
        n = 0
        for s in segs:
            txt = s.get("text", "").strip()
            if not txt:
                continue
            who = names.get(s.get("speaker"), "")
            n += 1
            f.write(f"{n}\n{ts_srt(s['start'])} --> {ts_srt(s['end'])}\n")
            f.write((f"[{who}] " if who else "") + txt + "\n\n")


def _split_cues(seg):
    """Yield (start, end, text) cues <= MAX_CHARS, timed by word alignment."""
    text = seg.get("text", "").strip()
    words = seg.get("words", [])
    if len(text) <= MAX_CHARS or not words:
        yield seg["start"], seg["end"], text
        return
    cur, cur_len, cstart, last_en = [], 0, None, None
    for wd in words:
        tok = str(wd.get("word", "")).strip()
        if not tok:
            continue
        st, en = wd.get("start"), wd.get("end")
        if cur and cur_len + 1 + len(tok) > MAX_CHARS:
            yield (cstart if cstart is not None else seg["start"],
                   last_en if last_en is not None else seg["end"],
                   " ".join(cur))
            cur, cur_len, cstart = [], 0, None
        if cstart is None and st is not None:
            cstart = st
        cur.append(tok); cur_len += 1 + len(tok)
        if en is not None:
            last_en = en
    if cur:
        yield (cstart if cstart is not None else seg["start"],
               last_en if last_en is not None else seg["end"],
               " ".join(cur))


def write_ass(segs, names, order, path, playres=(1280, 720)):
    W, H = playres
    font = max(24, round(H / 22))
    styles = []
    for idx, sp in enumerate(order):
        col = ASS_COLORS[idx % len(ASS_COLORS)]
        styles.append(
            f"Style: {names[sp]},Arial,{font},{col},{col},&H00202020,"
            f"&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,30,1")
    default_col = ASS_COLORS[0]
    styles.append(
        f"Style: Default,Arial,{font},{default_col},{default_col},&H00202020,"
        f"&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,30,1")
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
        "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
        "MarginR,MarginV,Encoding\n" + "\n".join(styles) +
        "\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,"
        "MarginV,Effect,Text\n")
    ev = []
    for s in segs:
        style = names.get(s.get("speaker"), "Default")
        for st, en, txt in _split_cues(s):
            txt = txt.strip().replace("\n", " ")
            if txt:
                ev.append(f"Dialogue: 0,{ts_ass(st)},{ts_ass(en)},{style},,"
                          f"0,0,0,,{txt}")
    with open(path, "w") as f:
        f.write(head + "\n".join(ev) + "\n")
    return len(ev)


# ---------------------------------------------------------------- pipeline
def run(args):
    import torch, whisperx
    from faster_whisper import WhisperModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctype = "float16" if dev == "cuda" else "int8"
    log(f"device={dev} compute_type={ctype} model={args.model}")

    tmp = tempfile.mkdtemp(prefix="wxcap-")
    wav = os.path.join(tmp, "audio16k.wav")
    log("resampling input -> 16 kHz mono wav")
    to_wav16k(args.input, wav)

    def free(m):
        del m; gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    # 1) transcribe (plain faster-whisper, no VAD batching)
    log("[1/3] transcribe (plain faster-whisper, vad_filter=False)")
    fw = WhisperModel(args.model, device=dev, compute_type=ctype,
                      download_root=HF_HUB)
    segs_it, info = fw.transcribe(
        wav, beam_size=5, language=args.language,
        word_timestamps=True, vad_filter=False,
        initial_prompt=args.prompt or None)
    lang = info.language
    fw_segs = [{"start": s.start, "end": s.end, "text": s.text.strip()}
               for s in segs_it if s.text.strip()]
    free(fw)
    log(f"language={lang} segments={len(fw_segs)}")
    if not fw_segs:
        log("no speech transcribed; writing empty outputs")

    res = {"segments": fw_segs, "language": lang}

    # 2) forced alignment (best effort — falls back to fw word timings)
    if fw_segs:
        try:
            log("[2/3] align (whisperx)")
            audio = whisperx.load_audio(wav)
            am, meta = whisperx.load_align_model(language_code=lang, device=dev)
            res = whisperx.align(fw_segs, am, meta, audio, dev,
                                 return_char_alignments=False)
            free(am)
        except Exception as e:
            log(f"align skipped ({type(e).__name__}: {str(e)[:100]}); "
                "using faster-whisper word timings")
            audio = whisperx.load_audio(wav)

        # 3) diarize (pyannote via whisperx)
        if not args.no_diarize:
            try:
                log("[3/3] diarize (pyannote)")
                from whisperx.diarize import DiarizationPipeline
                tok = open(TOKEN_FILE).read().strip()
                dia = DiarizationPipeline(token=tok, device=dev)
                kw = {}
                if args.min_speakers:
                    kw["min_speakers"] = args.min_speakers
                if args.max_speakers:
                    kw["max_speakers"] = args.max_speakers
                res = whisperx.assign_word_speakers(dia(audio, **kw), res)
                moves = snap_turns(res["segments"])
                log(f"diarization ok; snapped {moves} turn boundaries")
            except Exception as e:
                log(f"DIARIZATION SKIPPED: {type(e).__name__}: {str(e)[:160]}")

    segs = res["segments"]
    override = args.names.split(",") if args.names else None
    names, order = speaker_names(segs, override)
    log(f"speakers={ {names[o]: o for o in order} } segments={len(segs)}")

    prefix = args.out
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    written = []
    if "json" in formats:
        p = f"{prefix}.json"
        json.dump({"language": res.get("language", lang),
                   "speakers": {names[o]: o for o in order},
                   "segments": segs},
                  open(p, "w"), indent=1, default=str)
        written.append(p)
    if "txt" in formats:
        p = f"{prefix}.txt"; write_txt(segs, names, p); written.append(p)
    if "srt" in formats:
        p = f"{prefix}.srt"; write_srt(segs, names, p); written.append(p)
    if "ass" in formats:
        pr = (args.width, args.height) if args.width and args.height else (1280, 720)
        p = f"{prefix}.ass"; n = write_ass(segs, names, order, p, pr)
        written.append(p); log(f"ass: {n} cues at {pr[0]}x{pr[1]}")

    log("wrote: " + ", ".join(written))
    # machine-readable summary on stdout (last line) for the service to parse
    print(json.dumps({"ok": True, "language": res.get("language", lang),
                      "segments": len(segs),
                      "speakers": [names[o] for o in order],
                      "outputs": written}))


def main():
    ap = argparse.ArgumentParser(description="Diarized word-timed captioning")
    ap.add_argument("input", help="audio/video file (any ffmpeg format)")
    ap.add_argument("--out", help="output path prefix (default: input stem)")
    ap.add_argument("--language", default=None,
                    help="ISO code (default: auto-detect)")
    ap.add_argument("--prompt", default=None,
                    help="initial_prompt vocabulary bias (names, jargon)")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--names", default=None,
                    help="comma list overriding Speaker N (in order heard)")
    ap.add_argument("--formats", default="json,srt,ass,txt")
    ap.add_argument("--width", type=int, default=None, help="ASS PlayResX")
    ap.add_argument("--height", type=int, default=None, help="ASS PlayResY")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.splitext(args.input)[0]
    run(args)


if __name__ == "__main__":
    main()
