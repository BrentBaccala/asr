"""TTS sidecar manager — long-lived Pocket-TTS / Piper synthesis
subprocesses, shared per (engine, model, device) across sessions.

Each sidecar (tts_synth.py / tts_synth_piper.py) owns ONE model and
speaks the newline-JSON -> framed-PCM protocol documented in
tts_synth.py:

    request : {"id": <int>, "text": "<utterance>"}\\n   on stdin
    reply   : "AUDIO <id> <nbytes> <sr>\\n" + <nbytes> s16le mono PCM
              or "ERR <id> <msg>\\n"                     on stdout

Synthesis is blocking and a single sidecar serializes requests (one
model, one process), so `synth()` holds a per-sidecar lock. Run it in a
thread executor from asyncio code.
"""
import json
import os
import select
import subprocess
import threading
import time

# A wedged sidecar (e.g. a model whose load stalled while the disk was
# full) stays alive but never answers, so a blocking read would hang the
# calling thread forever and silently break that language. Every read is
# capped with a deadline; on timeout the proc is killed so TtsManager
# respawns a fresh one. Generous, because the AUDIO header only arrives
# after the whole utterance is synthesized (slow engines + long
# sentences); a truly wedged sidecar never answers, so even a large value
# cleanly separates the two. Override via env for tuning.
_SYNTH_TIMEOUT = float(os.environ.get("ASRPIPE_TTS_SYNTH_TIMEOUT", "60"))


class _Sidecar:
    def __init__(self, cmd):
        self.cmd = cmd
        self.lock = threading.Lock()
        self._id = 0
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0)

    def _kill(self):
        """Kill the sidecar so TtsManager._sidecar_for (which checks
        proc.poll()) respawns a fresh one on the next request."""
        try:
            self.proc.kill()
        except Exception:
            pass

    def _read_until(self, n: int, deadline: float):
        """Read exactly n bytes from stdout before `deadline` (monotonic),
        or return None on timeout/EOF — the caller treats None as a
        dead/wedged sidecar. Safe because stdout is raw (bufsize=0), so
        select() on the fd reflects real readability with no hidden
        Python-level buffering."""
        fd = self.proc.stdout
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                return None
            chunk = fd.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _readline_until(self, deadline: float):
        fd = self.proc.stdout
        buf = bytearray()
        while not buf.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                return None
            c = fd.read(1)
            if not c:
                return None
            buf.extend(c)
        return bytes(buf)

    def synth(self, text: str):
        """Return (pcm_bytes, sample_rate) or (b'', 0) on error/empty.

        Every read is bounded by _SYNTH_TIMEOUT; if the sidecar fails to
        answer in time it is presumed wedged (alive but stuck — the
        full-disk failure mode), killed, and respawned on the next call,
        rather than reused forever."""
        text = (text or "").strip()
        if not text:
            return b"", 0
        with self.lock:
            if self.proc.poll() is not None:
                return b"", 0
            self._id += 1
            rid = self._id
            try:
                self.proc.stdin.write(
                    (json.dumps({"id": rid, "text": text}) + "\n").encode())
                self.proc.stdin.flush()
            except Exception:
                self._kill()
                return b"", 0
            # The AUDIO header arrives only once the utterance is fully
            # synthesized, so this deadline must cover synth latency.
            hdr = self._readline_until(time.monotonic() + _SYNTH_TIMEOUT)
            if hdr is None:
                self._kill()
                return b"", 0
            parts = hdr.decode(errors="replace").split()
            if not parts or parts[0] != "AUDIO":
                # e.g. "ERR <id> <msg>" — a per-request failure, not a dead
                # sidecar; leave it running.
                return b"", 0
            try:
                nbytes, sr = int(parts[2]), int(parts[3])
            except (IndexError, ValueError):
                return b"", 0
            # PCM is already buffered in the sidecar once the header is out,
            # so a fresh deadline keeps the body read bounded too.
            pcm = self._read_until(nbytes, time.monotonic() + _SYNTH_TIMEOUT)
            if pcm is None:
                self._kill()
                return b"", 0
            return pcm, sr

    def close(self):
        try:
            self.proc.stdin.write((json.dumps({"id": 0, "quit": True}) + "\n").encode())
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class TtsManager:
    """Process-wide registry of TTS sidecars, keyed by (engine, model,
    device) so two sessions speaking the same language share one
    sidecar."""
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls, tts_lang: dict, synth_py: str, piper_py: str,
               melo_py: str = None):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(tts_lang, synth_py, piper_py, melo_py)
            return cls._instance

    def __init__(self, tts_lang: dict, synth_py: str, piper_py: str,
                 melo_py: str = None):
        self.tts_lang = tts_lang
        self.synth_py = synth_py
        self.piper_py = piper_py
        self.melo_py = melo_py
        self._sidecars = {}
        self._reg_lock = threading.Lock()

    def _cmd_for(self, code: str):
        cfg = self.tts_lang.get(code)
        if not cfg:
            return None, None
        engine = str(cfg.get("engine", "pocket-tts"))
        device = str(cfg.get("device", "cpu"))
        model = str(cfg.get("model", "english"))
        if engine == "piper":
            cmd = [self.piper_py, "--voice", model, "--device", device]
        elif engine == "melo":
            cmd = [self.melo_py, "--language", model, "--device", device]
        else:
            cmd = [self.synth_py, "--language", model, "--device", device]
            voice = cfg.get("voice")
            if voice:
                cmd += ["--voice", str(voice)]
        return (engine, model, device), cmd

    def _sidecar_for(self, code: str):
        key, cmd = self._cmd_for(code)
        if key is None:
            return None
        with self._reg_lock:
            sc = self._sidecars.get(key)
            if sc is None or sc.proc.poll() is not None:
                sc = _Sidecar(cmd)
                self._sidecars[key] = sc
            return sc

    def synth(self, code: str, text: str):
        """Synthesize `text` for FLORES code `code`. Returns (pcm, sr)."""
        sc = self._sidecar_for(code)
        if sc is None:
            return b"", 0
        return sc.synth(text)

    def prewarm(self, codes):
        """Spawn (but don't synthesize through) the sidecars for `codes`
        so the first real utterance doesn't pay model-load latency."""
        for c in codes:
            self._sidecar_for(c)

    def close(self):
        with self._reg_lock:
            for sc in self._sidecars.values():
                sc.close()
            self._sidecars.clear()
