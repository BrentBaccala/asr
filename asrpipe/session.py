"""Session — one independent ASR+MT+TTS pipeline for a single audio
stream (one browser client, or one side of a Mode-2 room).

Lifecycle::

    sess = Session(config, src_lang="eng_Latn", targets=["spa_Latn"],
                   speak_targets=["spa_Latn"],
                   on_event=cb_event, on_audio=cb_audio, loop=loop)
    await sess.start()
    sess.feed_pcm(pcm_s16le_16k_mono)      # repeatedly, from the bridge
    sess.set_langs("eng_Latn", ["ita_Latn"], ["ita_Latn"])  # live re-target
    await sess.close()

Callbacks:
  * ``on_event(TranscriptEvent)`` — live partials, finalized source
    sentences, and per-target translations.
  * ``on_audio(code, pcm_bytes, sample_rate)`` — synthesized TTS audio
    for a spoken target (the bridge resamples + sends it back over RTC).

All heavy/blocking work (NLLB, TTS sidecars) runs in the default thread
executor so the event loop stays responsive; Voxtral WS I/O is native
asyncio.
"""
import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field

import websockets

from .config import (PipelineConfig, SAMPLE_RATE, TTS_SYNTH_PY,
                     TTS_SYNTH_PIPER_PY, TTS_SYNTH_MELO_PY)
from .nllb import NllbTranslator
from .tts import TtsManager

_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')


@dataclass
class TranscriptEvent:
    """One transcript/translation update for the UI."""
    kind: str                     # "live" | "final" | "translation" | "notice"
    speaker: str                  # session label (e.g. "A", "B", "me")
    text: str = ""                # source text (live/final) or notice
    lang: str = ""                # source FLORES code (final) or target (translation)
    translations: dict = field(default_factory=dict)  # final: code->text
    seq: int = 0                  # monotonic per-session sentence id

    def to_dict(self):
        return {
            "kind": self.kind, "speaker": self.speaker, "text": self.text,
            "lang": self.lang, "translations": self.translations,
            "seq": self.seq,
        }


class Session:
    def __init__(self, config: PipelineConfig, *, src_lang: str,
                 targets: list, speak_targets: list,
                 on_event=None, on_audio=None, label: str = "me",
                 loop=None):
        self.cfg = config
        self.label = label
        self._src_lang = src_lang
        self._targets = list(targets)
        self._speak = list(speak_targets)
        self.on_event = on_event or (lambda ev: None)
        self.on_audio = on_audio or (lambda code, pcm, sr: None)
        self.loop = loop or asyncio.get_event_loop()

        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._cur = ""                # un-finalized live transcript
        self._seq = 0
        self._lang_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._ws_task = None

        # Lazily-resolved shared singletons (constructed on start()).
        self._nllb = None
        self._tts = None

    # ---- public API ----------------------------------------------------
    async def start(self):
        # Build the shared NLLB + TTS managers off the event loop.
        self._nllb = await self.loop.run_in_executor(
            None, NllbTranslator.shared, self.cfg.nllb_dir, self.cfg.nllb_beam)
        self._tts = TtsManager.shared(self.cfg.tts_lang, TTS_SYNTH_PY,
                                      TTS_SYNTH_PIPER_PY, TTS_SYNTH_MELO_PY)
        # Prewarm the speak targets so first utterance isn't slow.
        if self._speak:
            self.loop.run_in_executor(None, self._tts.prewarm, list(self._speak))
        self._ws_task = asyncio.ensure_future(self._ws_loop())

    def feed_pcm(self, pcm: bytes):
        """Feed s16le 16 kHz mono PCM. Non-blocking; safe from the loop."""
        if pcm:
            self._audio_q.put_nowait(pcm)

    async def set_langs(self, src_lang: str, targets: list,
                        speak_targets: list):
        """Re-target mid-session without dropping the Voxtral WS."""
        async with self._lang_lock:
            self._src_lang = src_lang
            self._targets = list(targets)
            self._speak = list(speak_targets)
        if speak_targets:
            self.loop.run_in_executor(None, self._tts.prewarm, list(speak_targets))
        self._emit(TranscriptEvent(
            kind="notice", speaker=self.label,
            text=f"languages set: src={src_lang} targets={targets} speak={speak_targets}"))

    async def close(self):
        self._stop.set()
        self._audio_q.put_nowait(None)
        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=5)
            except Exception:
                self._ws_task.cancel()

    # ---- internals -----------------------------------------------------
    def _emit(self, ev: TranscriptEvent):
        try:
            self.on_event(ev)
        except Exception:
            pass

    async def _ws_loop(self):
        """Maintain a Voxtral /v1/realtime WS, reconnecting on drop
        (covers the ~20-min max-model-len session ceiling)."""
        while not self._stop.is_set():
            try:
                await self._ws_session()
            except Exception as e:
                self._emit(TranscriptEvent(kind="notice", speaker=self.label,
                                           text=f"voxtral ws error: {e}"))
            if self._stop.is_set():
                break
            await asyncio.sleep(1.0)   # brief backoff before reconnect

    async def _ws_session(self):
        uri = self.cfg.voxtral_uri
        async with websockets.connect(uri, max_size=None) as ws:
            r = json.loads(await ws.recv())
            if r.get("type") != "session.created":
                raise RuntimeError(f"unexpected handshake: {r}")
            await ws.send(json.dumps({"type": "session.update",
                                      "model": self.cfg.voxtral_model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            self._emit(TranscriptEvent(kind="notice", speaker=self.label,
                                       text="voxtral: streaming"))

            async def receiver():
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        return
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "transcription.delta":
                        self._cur += m.get("delta", "")
                        await self._drain_sentences()
                        self._emit(TranscriptEvent(
                            kind="live", speaker=self.label, text=self._cur))
                    elif t == "transcription.done":
                        await self._drain_sentences(force=True)
                    elif t == "error":
                        self._emit(TranscriptEvent(
                            kind="notice", speaker=self.label,
                            text=f"voxtral error: {m}"))
                        return

            rx = asyncio.ensure_future(receiver())
            try:
                while not self._stop.is_set():
                    try:
                        chunk = await asyncio.wait_for(self._audio_q.get(),
                                                       timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        break
                    await ws.send(json.dumps(
                        {"type": "input_audio_buffer.append",
                         "audio": base64.b64encode(chunk).decode()}))
            finally:
                rx.cancel()

    async def _drain_sentences(self, force=False):
        """Pull complete sentences out of self._cur and finalize each."""
        while True:
            parts = _SENT_SPLIT.split(self._cur, maxsplit=1)
            if len(parts) == 2:
                sent, self._cur = parts[0].strip(), parts[1]
                if sent:
                    await self._finalize(sent)
                continue
            break
        if force:
            sent = self._cur.strip()
            self._cur = ""
            if sent:
                await self._finalize(sent)

    async def _finalize(self, text: str):
        async with self._lang_lock:
            src = self._src_lang
            targets = list(self._targets)
            speak = [c for c in self._speak if c != src]
        self._seq += 1
        seq = self._seq
        self._emit(TranscriptEvent(kind="final", speaker=self.label,
                                   text=text, lang=src, seq=seq))
        if not targets:
            return
        tr = await self.loop.run_in_executor(
            None, self._nllb.translate_multi, text, targets, src)
        self._emit(TranscriptEvent(kind="translation", speaker=self.label,
                                   text=text, lang=src, translations=tr, seq=seq))
        for code in speak:
            txt = tr.get(code)
            if not txt:
                continue
            asyncio.ensure_future(self._speak_one(code, txt))

    async def _speak_one(self, code: str, txt: str):
        pcm, sr = await self.loop.run_in_executor(None, self._tts.synth, code, txt)
        if pcm and sr:
            try:
                self.on_audio(code, pcm, sr)
            except Exception:
                pass
