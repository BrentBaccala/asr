"""asrpipe — multi-instance streaming ASR + MT + TTS pipeline core.

Extracted from the `freesoft-asr` TUI so the same Voxtral (vLLM
`/v1/realtime`) + NLLB-200 (CTranslate2) + Kyutai Pocket TTS / Piper
stack can drive several *independent* concurrent streams (the web
interpreter needs one pipeline per browser client / per room side).

Unlike `freesoft-asr` (single-session, module globals, curses TUI),
this package is:

  * **multi-instance** — every `Session` owns its own Voxtral WS, its
    own segmentation state and its own target-language set; nothing is
    shared except the process-wide NLLB translator and the per-language
    TTS sidecars (both intentionally shared singletons — they are
    expensive to load and stateless across requests).
  * **transport-agnostic** — a `Session` consumes 16 kHz s16le mono PCM
    via `feed_pcm()` and emits transcript/translation *events* and TTS
    *PCM* through callbacks. The caller (the aiortc web bridge, or a
    test harness) owns the audio plumbing.
  * **re-targetable mid-session** — `set_langs()` swaps the source-hint
    and the translation/TTS targets without tearing down the Voxtral WS
    (Voxtral is multilingual / per-sentence auto-detected, so a source
    change is just a hint; the NLLB target + TTS sidecar are swapped).

The audio contract throughout is **s16le, 16 kHz, mono** PCM (Voxtral's
native rate). TTS PCM comes back at the sidecar's own rate (24 kHz for
the `_24l` pocket-tts models, 22.05 kHz for Piper) — the `on_audio`
callback is told the rate per chunk.
"""
from .config import (
    FT_TO_NLLB,
    BUILTIN_TTS_LANG,
    LANG_CHOICES,
    PipelineConfig,
)
from .nllb import NllbTranslator
from .tts import TtsManager
from .session import Session, TranscriptEvent

__all__ = [
    "Session",
    "TranscriptEvent",
    "PipelineConfig",
    "NllbTranslator",
    "TtsManager",
    "FT_TO_NLLB",
    "BUILTIN_TTS_LANG",
    "LANG_CHOICES",
]
