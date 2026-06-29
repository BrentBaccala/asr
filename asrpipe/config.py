"""asrpipe configuration — model locations, language maps, TTS engine
defaults. Values mirror `freesoft-asr`'s constants so the two stay in
lock-step; every path is overridable by env var so the web service
(running as `cosine` on pony) and a samsung dev box can share the code.
"""
import os
from dataclasses import dataclass, field

# Voxtral vLLM realtime endpoint + model.
VOXTRAL_HOST = os.environ.get("ASRPIPE_VOXTRAL_HOST", "127.0.0.1")
VOXTRAL_PORT = int(os.environ.get("ASRPIPE_VOXTRAL_PORT", "8000"))
VOXTRAL_MODEL = os.environ.get(
    "ASRPIPE_VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")

# NLLB-200 distilled-600M, CTranslate2 int8 (CPU).
NLLB_DIR = os.path.expanduser(
    os.environ.get("ASRPIPE_NLLB_DIR", "~/asr/models/nllb-600m-ct2"))
NLLB_BEAM = int(os.environ.get("ASRPIPE_NLLB_BEAM", "1"))

# Pocket-TTS / Piper sidecar scripts (ship next to this package's parent).
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_SYNTH_PY = os.path.join(_PKG_PARENT, "tts_synth.py")
TTS_SYNTH_PIPER_PY = os.path.join(_PKG_PARENT, "tts_synth_piper.py")
TTS_SYNTH_MELO_PY = os.path.join(_PKG_PARENT, "tts_synth_melo.py")

SAMPLE_RATE = 16000          # Voxtral input rate (s16le mono)

# fastText ISO-639-1 -> FLORES-200 (only used when a source lang is left
# to auto-detect; the web UI normally pins it from the dropdown).
FT_TO_NLLB = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ar": "arb_Arab", "nl": "nld_Latn",
    "pl": "pol_Latn", "tr": "tur_Latn", "vi": "vie_Latn",
    "hi": "hin_Deva", "uk": "ukr_Cyrl", "ca": "cat_Latn",
}

# FLORES-200 target -> TTS engine config (engine, model, device). Mirrors
# freesoft-asr's BUILTIN_TTS_LANG: English distils to CPU real-time;
# the European voices ship only as bigger `_24l` previews (real-time on
# the 3090, sub-real-time on CPU) so they default to cuda. Spanish uses
# Piper (es_ES-sharvard-medium) — pocket-tts's spanish drops the first
# word ~50% of the time on short utterances.
BUILTIN_TTS_LANG = {
    "eng_Latn": {"engine": "pocket-tts", "model": "english",
                 "device": "cpu"},
    "spa_Latn": {"engine": "piper", "model": "es_ES-sharvard-medium",
                 "device": "cpu"},
    "fra_Latn": {"engine": "pocket-tts", "model": "french_24l",
                 "device": "cuda"},
    "deu_Latn": {"engine": "pocket-tts", "model": "german_24l",
                 "device": "cuda"},
    "por_Latn": {"engine": "pocket-tts", "model": "portuguese_24l",
                 "device": "cuda"},
    "ita_Latn": {"engine": "pocket-tts", "model": "italian_24l",
                 "device": "cuda"},
    # --- Phase 2/3: the 7 languages Voxtral transcribes + NLLB translates
    # but had no speak-back. Five get intelligible CPU Piper voices (zero
    # GPU cost, real-time+); only Japanese/Korean lack a Piper voice and
    # are covered by the streaming GPU CJK model (see below).
    "nld_Latn": {"engine": "piper", "model": "nl_NL-mls-medium",
                 "device": "cpu"},
    "rus_Cyrl": {"engine": "piper", "model": "ru_RU-dmitri-medium",
                 "device": "cpu"},
    "arb_Arab": {"engine": "piper", "model": "ar_JO-kareem-medium",
                 "device": "cpu"},
    "hin_Deva": {"engine": "piper", "model": "hi_IN-pratham-medium",
                 "device": "cpu"},
    "zho_Hans": {"engine": "piper", "model": "zh_CN-huayan-medium",
                 "device": "cpu"},
    # Japanese/Korean have no Piper voice (espeak-ng can't phonemize kanji /
    # hangul). MeloTTS is a multilingual VITS model that does both, real-time+
    # on CPU (zero GPU). Runs from ~/asr/melo-env via tts_synth_melo.py.
    "jpn_Jpan": {"engine": "melo", "model": "JP", "device": "cpu"},
    "kor_Hang": {"engine": "melo", "model": "KR", "device": "cpu"},
}

# The languages the web UI offers in its dropdowns: (FLORES code, label).
LANG_CHOICES = [
    ("eng_Latn", "English"),
    ("spa_Latn", "Spanish"),
    ("ita_Latn", "Italian"),
    ("fra_Latn", "French"),
    ("deu_Latn", "German"),
    ("por_Latn", "Portuguese"),
    ("nld_Latn", "Dutch"),
    ("rus_Cyrl", "Russian"),
    ("arb_Arab", "Arabic"),
    ("hin_Deva", "Hindi"),
    ("zho_Hans", "Chinese"),
    ("jpn_Jpan", "Japanese"),
    ("kor_Hang", "Korean"),
]

LANG_LABEL = {c: l for c, l in LANG_CHOICES}


def lang_label(code: str) -> str:
    return LANG_LABEL.get(code, code.split("_")[0][:3].upper())


@dataclass
class PipelineConfig:
    """Per-process pipeline knobs (shared across sessions)."""
    voxtral_host: str = VOXTRAL_HOST
    voxtral_port: int = VOXTRAL_PORT
    voxtral_model: str = VOXTRAL_MODEL
    nllb_dir: str = NLLB_DIR
    nllb_beam: int = NLLB_BEAM
    tts_lang: dict = field(default_factory=lambda: dict(BUILTIN_TTS_LANG))

    @property
    def voxtral_uri(self) -> str:
        return f"ws://{self.voxtral_host}:{self.voxtral_port}/v1/realtime"
