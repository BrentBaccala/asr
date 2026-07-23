#!/usr/bin/env python3
"""
bbb-transcribe.py — batch-transcribe a recording via the pony Voxtral endpoint.

Takes any audio or video file, resamples to 16 kHz mono, splits it into
short chunks at natural silence boundaries (so words aren't cut), POSTs each
chunk to an OpenAI-compatible /v1/audio/transcriptions endpoint (the vLLM
Voxtral server fronted by osito.freesoft.org:8443), and reassembles the
transcript as plain text and/or SRT.

Why chunk: the Voxtral realtime model runs with --max-model-len 16384, so a
single request only handles a few minutes of audio (and overrunning it has
crashed the vLLM engine). Class-length recordings must be split client-side.

Dependencies: Python 3 stdlib + ffmpeg/ffprobe on PATH. No pip installs.

Examples:
  # simplest — env token, plain text to stdout
  VOXTRAL_TOKEN=... ./bbb-transcribe.py recording.mp4

  # write both a .txt and a timestamped .srt
  ./bbb-transcribe.py class.wav --token-file token.txt \
      --txt class.txt --srt class.srt

  # against a local server (e.g. ssh tunnel to pony :8000, no auth)
  ./bbb-transcribe.py clip.wav --endpoint http://127.0.0.1:8000/v1/audio/transcriptions
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import uuid

DEFAULT_ENDPOINT = "https://osito.freesoft.org:8443/v1/audio/transcriptions"
DEFAULT_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ffprobe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path]).stdout.strip()
    return float(out)


def to_wav16k(src, dst):
    """Resample any input to 16 kHz mono s16le WAV."""
    run(["ffmpeg", "-nostdin", "-y", "-i", src,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", dst])


def detect_silences(wav, noise_db, min_sil):
    """Return list of (start, end) silence intervals via ffmpeg silencedetect."""
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", wav, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    sils, cur = [], None
    for line in p.stderr.splitlines():
        if "silence_start:" in line:
            cur = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and cur is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            sils.append((cur, end))
            cur = None
    return sils


def plan_chunks(duration, silences, max_chunk, min_chunk):
    """Greedily pack [start,end] chunks up to max_chunk s, cutting at the
    midpoint of a silence when one is available, else a hard cut."""
    # candidate cut points = silence midpoints, sorted
    cuts = sorted((s + e) / 2.0 for s, e in silences)
    chunks = []
    t = 0.0
    while t < duration - 1e-3:
        target = t + max_chunk
        if target >= duration:
            chunks.append((t, duration))
            break
        # best silence cut in (t+min_chunk, target]
        best = None
        for c in cuts:
            if c <= t + min_chunk:
                continue
            if c > target:
                break
            best = c
        end = best if best is not None else target
        chunks.append((t, end))
        t = end
    return chunks


def extract(wav, start, end, dst):
    run(["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", wav, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         "-f", "wav", dst])


def post_chunk(endpoint, token, model, wav_path, timeout, retries):
    with open(wav_path, "rb") as f:
        audio = f.read()
    boundary = uuid.uuid4().hex
    parts = []
    for name, val in (("model", model), ("response_format", "json")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{name}"\r\n\r\n{val}\r\n'.encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="file"; filename="chunk.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n".encode())
    body = b"".join(parts) + audio + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(endpoint, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
                return data.get("text", "").strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = " " + e.read().decode()[:200]
                except Exception:
                    pass
            log(f"    attempt {attempt}/{retries} failed: {e}{detail}")
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"chunk failed after {retries} attempts: {last}")


def fmt_ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60)
    s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="audio or video file to transcribe")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--token", default=os.environ.get("VOXTRAL_TOKEN"),
                    help="Bearer token (or set VOXTRAL_TOKEN)")
    ap.add_argument("--token-file", help="file whose first line is the token")
    ap.add_argument("--max-chunk", type=float, default=30.0,
                    help="max chunk length in seconds (default 30)")
    ap.add_argument("--min-chunk", type=float, default=5.0,
                    help="don't cut earlier than this into a chunk (default 5)")
    ap.add_argument("--noise-db", type=float, default=-30.0,
                    help="silencedetect noise floor in dB (default -30)")
    ap.add_argument("--min-silence", type=float, default=0.4,
                    help="min silence duration to be a cut point (default 0.4s)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--txt", help="write plain transcript here (default: stdout)")
    ap.add_argument("--srt", help="also write timestamped SRT here")
    ap.add_argument("--json", dest="json_out",
                    help="write structured chunk list (start/end/text) here")
    args = ap.parse_args()

    token = args.token
    if args.token_file:
        with open(args.token_file) as f:
            token = f.readline().strip().removeprefix("Bearer ").strip()

    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True).returncode:
            log(f"ERROR: {tool} not found on PATH"); sys.exit(2)

    with tempfile.TemporaryDirectory(prefix="bbbtx-") as tmp:
        wav = os.path.join(tmp, "full.wav")
        log(f"[1/4] resampling {args.input} -> 16 kHz mono ...")
        to_wav16k(args.input, wav)
        duration = ffprobe_duration(wav)

        log(f"[2/4] detecting silences (noise={args.noise_db}dB, "
            f"d={args.min_silence}s) over {duration:.1f}s ...")
        sils = detect_silences(wav, args.noise_db, args.min_silence)
        chunks = plan_chunks(duration, sils, args.max_chunk, args.min_chunk)
        log(f"      {len(sils)} silences -> {len(chunks)} chunks")

        log(f"[3/4] transcribing {len(chunks)} chunks via {args.endpoint} ...")
        results = []
        for i, (s, e) in enumerate(chunks, 1):
            cw = os.path.join(tmp, f"c{i:04d}.wav")
            extract(wav, s, e, cw)
            t0 = time.time()
            text = post_chunk(args.endpoint, token, args.model, cw,
                              args.timeout, args.retries)
            log(f"      chunk {i}/{len(chunks)} [{fmt_ts(s)}-{fmt_ts(e)}] "
                f"{time.time()-t0:.1f}s: {text[:60]!r}")
            results.append({"start": s, "end": e, "text": text})

    log("[4/4] assembling output ...")
    full_text = " ".join(r["text"] for r in results if r["text"]).strip()

    if args.txt:
        with open(args.txt, "w") as f:
            f.write(full_text + "\n")
        log(f"      wrote {args.txt}")
    else:
        print(full_text)

    if args.srt:
        with open(args.srt, "w") as f:
            n = 0
            for r in results:
                if not r["text"]:
                    continue
                n += 1
                f.write(f"{n}\n{fmt_ts(r['start'])} --> {fmt_ts(r['end'])}\n"
                        f"{r['text']}\n\n")
        log(f"      wrote {args.srt}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        log(f"      wrote {args.json_out}")


if __name__ == "__main__":
    main()
