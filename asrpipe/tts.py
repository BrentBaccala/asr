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
import subprocess
import threading


def _read_exact(f, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            break
        buf.extend(c)
    return bytes(buf)


class _Sidecar:
    def __init__(self, cmd):
        self.cmd = cmd
        self.lock = threading.Lock()
        self._id = 0
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0)

    def synth(self, text: str):
        """Return (pcm_bytes, sample_rate) or (b'', 0) on error/empty."""
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
                return b"", 0
            # read header line
            hdr = b""
            while not hdr.endswith(b"\n"):
                c = self.proc.stdout.read(1)
                if not c:
                    return b"", 0
                hdr += c
            parts = hdr.decode(errors="replace").split()
            if not parts or parts[0] != "AUDIO":
                return b"", 0
            nbytes, sr = int(parts[2]), int(parts[3])
            pcm = _read_exact(self.proc.stdout, nbytes)
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
    def shared(cls, tts_lang: dict, synth_py: str, piper_py: str):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(tts_lang, synth_py, piper_py)
            return cls._instance

    def __init__(self, tts_lang: dict, synth_py: str, piper_py: str):
        self.tts_lang = tts_lang
        self.synth_py = synth_py
        self.piper_py = piper_py
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
