#!/usr/bin/env python3
"""caption_service.py — diarized captioning job service on pony.

A dependency-free (stdlib only) HTTP job queue that fronts wx_caption.py and
arbitrates the GPU with the live Voxtral ASR via the `gpu-lease` tool.

Design
------
* Listens on 127.0.0.1:8001. It is NOT exposed directly — pony's haproxy
  `voxtral8443` frontend path-routes `/caption*` here behind the same Bearer
  token + TLS, so this service needs no auth of its own.
* Jobs are async: submit returns a job_id immediately; the client polls status
  and downloads artifacts when done. Individual HTTP requests stay short, so no
  long-lived connections through haproxy.
* A single worker thread serializes GPU use. It wraps a *burst* of jobs in one
  `gpu-lease claim ... / release` cycle (draining the queue with a short idle
  linger) so a batch of recordings costs only one voxtral bounce, not one per
  job. Each job is marked `done` the instant its artifacts are written —
  BEFORE the (slow) voxtral restart — so clients never wait on the release.
* The GPU pipeline runs as a fresh subprocess in the whisperx venv, so all CUDA
  memory is released on exit before voxtral is restarted.

Endpoints
---------
  POST /caption/submit?<opts>   body = 16 kHz mono wav (or any audio) -> {job_id}
  GET  /caption/status/<id>     -> {status, ...}
  GET  /caption/result/<id>/<fmt>   fmt in json|srt|ass|txt -> file
  GET  /caption/health          -> "ok"

Submit query options (all optional): language, min_speakers, max_speakers,
prompt, names, formats, width, height, filename.
"""
import os, sys, json, uuid, time, queue, threading, subprocess, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WHISPERX_PY = "/mnt/models/venvs/whisperx/bin/python"
WX_CAPTION = os.path.join(HERE, "wx_caption.py")
GPU_LEASE = "/usr/local/bin/gpu-lease"
JOBS_DIR = os.environ.get("CAPTION_JOBS_DIR", "/mnt/models/caption-jobs")
BIND = os.environ.get("CAPTION_BIND", "127.0.0.1")
PORT = int(os.environ.get("CAPTION_PORT", "8001"))
LEASE_LABEL = os.environ.get("CAPTION_LEASE_LABEL", "caption")
IDLE_LINGER = float(os.environ.get("CAPTION_IDLE_LINGER", "20"))  # s to hold lease waiting for more work
MAX_UPLOAD = int(os.environ.get("CAPTION_MAX_UPLOAD", str(2 * 1024**3)))  # 2 GiB
FORMATS_OK = {"json", "srt", "ass", "txt"}

os.makedirs(JOBS_DIR, exist_ok=True)

_jobs = {}                 # id -> dict (in-memory view)
_jobs_lock = threading.Lock()
_q = queue.Queue()


def log(*a):
    print("[caption]", *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ job state
def _job_path(jid):
    return os.path.join(JOBS_DIR, jid)


def _save(job):
    with _jobs_lock:
        _jobs[job["id"]] = job
    try:
        with open(os.path.join(_job_path(job["id"]), "job.json"), "w") as f:
            json.dump(job, f, indent=1)
    except Exception as e:
        log("save failed:", e)


def _get(jid):
    with _jobs_lock:
        j = _jobs.get(jid)
    if j:
        return j
    # fall back to disk (survives a service restart)
    p = os.path.join(_job_path(jid), "job.json")
    if os.path.exists(p):
        try:
            j = json.load(open(p))
            with _jobs_lock:
                _jobs[jid] = j
            return j
        except Exception:
            pass
    return None


def _load_existing():
    if not os.path.isdir(JOBS_DIR):
        return
    for jid in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, jid, "job.json")
        if os.path.exists(p):
            try:
                j = json.load(open(p))
                # anything left 'queued'/'running' after a restart is stale
                if j.get("status") in ("queued", "running"):
                    j["status"] = "error"
                    j["detail"] = "service restarted while pending"
                _jobs[jid] = j
            except Exception:
                pass


# ------------------------------------------------------------------ worker
def _process(job):
    jid = job["id"]
    jd = _job_path(jid)
    job["status"] = "running"
    job["started_at"] = time.time()
    _save(job)
    log(f"job {jid}: running ({job.get('filename','audio')})")

    prefix = os.path.join(jd, "out")
    cmd = [WHISPERX_PY, WX_CAPTION, job["input"], "--out", prefix,
           "--formats", job["opts"].get("formats", "json,srt,ass,txt")]
    o = job["opts"]
    for flag, key in (("--language", "language"), ("--prompt", "prompt"),
                      ("--names", "names"), ("--model", "model"),
                      ("--min-speakers", "min_speakers"),
                      ("--max-speakers", "max_speakers"),
                      ("--width", "width"), ("--height", "height")):
        v = o.get(key)
        if v not in (None, ""):
            cmd += [flag, str(v)]

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=int(os.environ.get("CAPTION_JOB_TIMEOUT", "7200")))
    except subprocess.TimeoutExpired:
        job["status"] = "error"; job["detail"] = "pipeline timed out"
        _save(job); log(f"job {jid}: TIMEOUT"); return
    dur = time.time() - t0

    if r.returncode != 0:
        job["status"] = "error"
        job["detail"] = (r.stderr or "")[-1500:]
        _save(job); log(f"job {jid}: FAILED rc={r.returncode}"); return

    # last stdout line is the wx_caption json summary
    summary = {}
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line); break
            except Exception:
                pass
    avail = [f for f in FORMATS_OK if os.path.exists(f"{prefix}.{f}")]
    job["status"] = "done"
    job["finished_at"] = time.time()
    job["duration_s"] = round(dur, 1)
    job["language"] = summary.get("language")
    job["speakers"] = summary.get("speakers")
    job["segments"] = summary.get("segments")
    job["formats"] = avail
    job["log_tail"] = (r.stderr or "").strip().splitlines()[-6:]
    _save(job)
    log(f"job {jid}: DONE in {dur:.0f}s "
        f"({job.get('segments')} segs, speakers={job.get('speakers')})")


def _lease(cmd_args):
    try:
        r = subprocess.run([GPU_LEASE] + cmd_args, capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            log(f"gpu-lease {' '.join(cmd_args)} rc={r.returncode}: "
                f"{(r.stderr or '').strip()[-300:]}")
        return r.returncode == 0
    except Exception as e:
        log(f"gpu-lease {' '.join(cmd_args)} raised: {e}")
        return False


def _worker():
    log("worker started")
    while True:
        job = _q.get()                        # block for first job of a burst
        log(f"claiming GPU for burst (job {job['id']})")
        if not _lease(["claim", "--wait", LEASE_LABEL]):
            job["status"] = "error"
            job["detail"] = "could not claim GPU (gpu-lease claim failed)"
            _save(job)
            continue
        try:
            _process(job)
            # drain any further queued jobs under the same lease
            while True:
                try:
                    job = _q.get(timeout=IDLE_LINGER)
                except queue.Empty:
                    break
                _process(job)
        finally:
            log("releasing GPU (restarting voxtral)")
            _lease(["release"])               # best-effort; jobs already 'done'


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body=b"", ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def log_message(self, fmt, *args):
        log("http", self.address_string(), fmt % args)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        parts = [p for p in u.path.split("/") if p]
        # /caption/health
        if parts == ["caption", "health"]:
            return self._send(200, "ok", "text/plain")
        # /caption/status/<id>
        if len(parts) == 3 and parts[:2] == ["caption", "status"]:
            job = _get(parts[2])
            if not job:
                return self._json(404, {"error": "unknown job"})
            view = {k: job.get(k) for k in
                    ("id", "status", "detail", "filename", "language",
                     "speakers", "segments", "duration_s", "formats",
                     "submitted_at", "finished_at", "log_tail")}
            return self._json(200, view)
        # /caption/result/<id>/<fmt>
        if len(parts) == 4 and parts[:2] == ["caption", "result"]:
            jid, fmt = parts[2], parts[3]
            if fmt not in FORMATS_OK:
                return self._json(400, {"error": "bad format"})
            job = _get(jid)
            if not job:
                return self._json(404, {"error": "unknown job"})
            if job.get("status") != "done":
                return self._json(409, {"error": "not ready",
                                        "status": job.get("status")})
            path = os.path.join(_job_path(jid), f"out.{fmt}")
            if not os.path.exists(path):
                return self._json(404, {"error": "format not produced"})
            data = open(path, "rb").read()
            ct = {"json": "application/json", "srt": "text/plain",
                  "ass": "text/plain", "txt": "text/plain"}[fmt]
            return self._send(200, data, ct)
        return self._json(404, {"error": "not found"})

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        parts = [p for p in u.path.split("/") if p]
        if parts != ["caption", "submit"]:
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return self._json(400, {"error": "empty body (send audio bytes)"})
        if length > MAX_UPLOAD:
            return self._json(413, {"error": "upload too large"})

        q = urllib.parse.parse_qs(u.query)
        opts = {}
        for k in ("language", "prompt", "names", "model", "formats"):
            if k in q:
                opts[k] = q[k][0]
        for k in ("min_speakers", "max_speakers", "width", "height"):
            if k in q:
                try:
                    opts[k] = int(q[k][0])
                except ValueError:
                    pass
        fmts = opts.get("formats")
        if fmts:
            bad = set(f.strip() for f in fmts.split(",")) - FORMATS_OK
            if bad:
                return self._json(400, {"error": f"bad formats: {sorted(bad)}"})
        filename = q.get("filename", ["audio"])[0]

        jid = uuid.uuid4().hex[:16]
        jd = _job_path(jid)
        os.makedirs(jd, exist_ok=True)
        inp = os.path.join(jd, "input")
        # stream body to disk
        remaining = length
        with open(inp, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            job = {"id": jid, "status": "error", "detail": "truncated upload"}
            _save(job)
            return self._json(400, {"error": "truncated upload"})

        job = {"id": jid, "status": "queued", "filename": filename,
               "input": inp, "opts": opts, "submitted_at": time.time(),
               "bytes": length}
        _save(job)
        _q.put(job)
        log(f"job {jid}: queued ({filename}, {length} bytes, opts={opts})")
        return self._json(202, {"job_id": jid, "status": "queued"})


def main():
    _load_existing()
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer((BIND, PORT), Handler)
    log(f"listening on {BIND}:{PORT}  jobs_dir={JOBS_DIR}  lease='{LEASE_LABEL}'")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
