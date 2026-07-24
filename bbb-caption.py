#!/usr/bin/env python3
"""
bbb-caption.py — diarized, word-timed captions for a recording via pony.

The batch cousin of bbb-transcribe.py. Where bbb-transcribe.py hits the Voxtral
realtime endpoint (fast, but no speakers and only ~chunk-level timing), this
uploads a recording to pony's caption service, which runs the Whisper +
forced-alignment + pyannote-diarization pipeline (the 2026-07-23 bake-off
winner) and returns SRT / ASS / JSON / txt with real word timings and speaker
labels.

Flow: extract 16 kHz mono wav locally (small upload) -> POST to
/caption/submit -> poll /caption/status/<id> -> download artifacts beside the
input file. The pipeline shares pony's GPU with the live Voxtral ASR via
gpu-lease, so a job may queue briefly while a live session finishes.

Dependencies: Python 3 stdlib + ffmpeg on PATH. No pip installs.

Examples:
  # env token, write .srt + .ass + .json beside the input
  VOXTRAL_TOKEN=$(cat /etc/bbb-transcribe.token) ./bbb-caption.py class.mp4

  # two known speakers, vocabulary bias, only SRT
  ./bbb-caption.py talk.mp4 --token-file /etc/bbb-transcribe.token \
      --min-speakers 2 --max-speakers 2 --names "Bruce,Brent" \
      --prompt "ITPIE demo by Bruce Caslow and Brent Baccala" --formats srt

  # course glossary biasing the whole recording (not just the opening)
  ./bbb-caption.py class.mp4 \
      --hotwords "vSphere, vCenter, NSX-T, Tanzu, ESXi, vMotion, vSAN"
"""
import argparse
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

DEFAULT_BASE = "https://osito.freesoft.org:8443"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def to_wav16k(src, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    "-vn", "-f", "wav", dst],
                   check=True, capture_output=True, text=True)


def probe_wh(src):
    """(width, height) of the first video stream, or (None, None)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", src],
            capture_output=True, text=True).stdout.strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception:
        return None, None


def req(url, token, data=None, method=None, ctx=None, length=None):
    r = urllib.request.Request(url, data=data, method=method)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        r.add_header("Content-Type", "application/octet-stream")
        # Set Content-Length explicitly so a file-object body is sent with a
        # fixed length, NOT chunked Transfer-Encoding (the stdlib server reads
        # Content-Length only). This also streams large files without buffering.
        if length is None and hasattr(data, "fileno"):
            try:
                length = os.fstat(data.fileno()).st_size
            except OSError:
                pass
        if length is not None:
            r.add_header("Content-Length", str(length))
    return urllib.request.urlopen(r, context=ctx)


def main():
    ap = argparse.ArgumentParser(description="Diarized captions via pony")
    ap.add_argument("input", help="audio or video file")
    ap.add_argument("--base", default=os.environ.get("CAPTION_BASE", DEFAULT_BASE),
                    help="service base URL (default %(default)s)")
    ap.add_argument("--token", default=os.environ.get("VOXTRAL_TOKEN"))
    ap.add_argument("--token-file")
    ap.add_argument("--formats", default="srt,ass,json,txt",
                    help="comma list: json,srt,ass,txt")
    ap.add_argument("--language", help="ISO code (default: auto-detect)")
    ap.add_argument("--prompt", help="vocabulary bias (names, jargon); "
                                     "conditions the opening window only")
    ap.add_argument("--hotwords", help="glossary applied to every window; "
                                       "ignored if --prompt is also given")
    ap.add_argument("--names", help="comma list overriding Speaker N")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--model", help="faster-whisper model (default large-v3)")
    ap.add_argument("--out-prefix", help="output path prefix (default: input stem)")
    ap.add_argument("--poll", type=float, default=5.0, help="status poll seconds")
    ap.add_argument("--timeout", type=float, default=7200, help="max wait seconds")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification")
    ap.add_argument("--keep-wav", action="store_true")
    args = ap.parse_args()

    token = args.token
    if not token and args.token_file:
        token = open(args.token_file).read().strip()
    if not token:
        log("ERROR: no token (set VOXTRAL_TOKEN, --token, or --token-file)")
        sys.exit(2)

    ctx = None
    if args.insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    out_prefix = args.out_prefix or os.path.splitext(args.input)[0]

    # 1) extract 16 kHz mono wav locally
    wav = out_prefix + ".caption-upload.wav"
    log(f"[1/4] extracting 16 kHz mono wav from {args.input}")
    to_wav16k(args.input, wav)

    # build submit query
    q = {"filename": os.path.basename(args.input), "formats": args.formats}
    for k, v in (("language", args.language), ("prompt", args.prompt),
                 ("hotwords", args.hotwords),
                 ("names", args.names), ("model", args.model),
                 ("min_speakers", args.min_speakers),
                 ("max_speakers", args.max_speakers)):
        if v not in (None, ""):
            q[k] = v
    w, h = probe_wh(args.input)
    if w and h:
        q["width"], q["height"] = w, h

    submit_url = f"{args.base}/caption/submit?" + urllib.parse.urlencode(q)
    size = os.path.getsize(wav)
    log(f"[2/4] uploading {size/1e6:.1f} MB -> {args.base}/caption/submit")
    try:
        with open(wav, "rb") as f, req(submit_url, token, data=f, method="POST",
                                       ctx=ctx, length=size) as resp:
            sub = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        log(f"ERROR submit: HTTP {e.code} {e.read()[:300]!r}")
        sys.exit(1)
    finally:
        if not args.keep_wav:
            try:
                os.remove(wav)
            except OSError:
                pass
    jid = sub["job_id"]
    log(f"      job {jid} queued")

    # 3) poll
    log("[3/4] waiting for pipeline (Whisper + align + pyannote)...")
    t0 = time.time()
    last = None
    while True:
        if time.time() - t0 > args.timeout:
            log("ERROR: timed out waiting for job"); sys.exit(1)
        time.sleep(args.poll)
        try:
            with req(f"{args.base}/caption/status/{jid}", token, ctx=ctx) as resp:
                st = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            log(f"      status HTTP {e.code}; retrying"); continue
        s = st.get("status")
        if s != last:
            d = st.get("detail")
            log(f"      status: {s}" + (f" — {d}" if d else ""))
            last = s
        if s == "done":
            log(f"      done: {st.get('segments')} segments, "
                f"speakers={st.get('speakers')}, "
                f"lang={st.get('language')}, {st.get('duration_s')}s")
            break
        if s == "error":
            log(f"ERROR: {st.get('detail') or 'job failed'}")
            sys.exit(1)

    # 4) download artifacts
    log("[4/4] downloading artifacts")
    got = []
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        try:
            with req(f"{args.base}/caption/result/{jid}/{fmt}", token, ctx=ctx) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            log(f"      {fmt}: HTTP {e.code} (skipped)"); continue
        path = f"{out_prefix}.{fmt}"
        with open(path, "wb") as f:
            f.write(data)
        got.append(path)
    log("wrote: " + ", ".join(got))
    print("\n".join(got))


if __name__ == "__main__":
    main()
