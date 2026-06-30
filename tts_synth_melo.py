#!/home/cosine/asr/melo-env/bin/python
"""tts_synth_melo.py — MeloTTS synthesis sidecar.

Drop-in alternative to tts_synth.py / tts_synth_piper.py, speaking the SAME
newline-JSON stdin -> framed-PCM stdout protocol so asrpipe's TtsManager can
drive it:

Request  (stdin, one JSON object per line):
    {"id": <int>, "text": "<utterance>"}
    {"id": <int>, "quit": true}

Response (stdout):
    "AUDIO <id> <nbytes> <sample_rate>\n" + <nbytes> raw s16le mono PCM
    "ERR <id> <message>\n"               on error / empty text

Why MeloTTS: it is the speak-back engine for Japanese (JP) and Korean (KR),
the two of the interpreter's 13 languages that have no Piper voice (espeak-ng
cannot phonemize Japanese kanji / Korean hangul well). MeloTTS is a
multilingual VITS model that runs real-time+ on CPU, so — like the Piper
voices — it costs zero GPU. One sidecar owns ONE language model
(model load is multi-second). MeloTTS also covers EN/ES/FR/ZH if ever needed.

--language is a MeloTTS language id: EN, ES, FR, ZH, JP, KR. The model's
native sample rate (44100) is reported in the AUDIO header so the bridge
resamples correctly.
"""
import argparse
import json
import os
import sys
import time

# This sidecar only ever loads fixed, pre-cached models (the MeloTTS
# checkpoint + the language's BERT). Default to HuggingFace offline so model
# load is fast and deterministic — otherwise every load does slow HF/Xet
# round-trips to re-check cached files (observed ~minutes on the first synth).
# Pre-fetch the models once at install time; override by exporting
# HF_HUB_OFFLINE=0 if a model is genuinely missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np


def _log(obj):
    sys.stderr.write(json.dumps(obj) + "\n")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description="MeloTTS synthesis sidecar")
    ap.add_argument("--language", required=True,
                    help="MeloTTS language id (EN, ES, FR, ZH, JP, KR)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="torch device (CPU is real-time+ and the default)")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    from melo.api import TTS

    model = TTS(language=args.language.upper(), device=args.device)
    sr = int(model.hps.data.sampling_rate)
    # spk2id is an HParams object (dict-like but no .get): keys are speaker
    # names (e.g. "JP", "KR"), values the integer ids. Match the language,
    # else fall back to the first available speaker.
    spk2id = model.hps.data.spk2id
    lang_u = args.language.upper()
    keys = list(spk2id.keys())
    spk_id = spk2id[lang_u] if lang_u in keys else list(spk2id.values())[0]

    _log({"event": "ready", "language": args.language, "device": args.device,
          "sample_rate": sr, "speaker_id": spk_id})

    out = sys.stdout.buffer
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _log({"event": "bad_request", "error": str(e)})
            continue
        rid = req.get("id", 0)
        if req.get("quit"):
            _log({"event": "quit", "id": rid})
            break
        text = (req.get("text") or "").strip()
        if not text:
            out.write(f"ERR {rid} empty-text\n".encode())
            out.flush()
            continue
        try:
            t0 = time.monotonic()
            # output_path=None makes tts_to_file return the float32 audio array.
            audio = model.tts_to_file(text, spk_id, output_path=None,
                                      speed=args.speed, quiet=True)
            gen_ms = (time.monotonic() - t0) * 1000.0
            wav = np.asarray(audio, dtype=np.float32).reshape(-1)
            pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            out.write(f"AUDIO {rid} {len(pcm16)} {sr}\n".encode())
            out.write(pcm16)
            out.flush()
            audio_ms = len(wav) * 1000.0 / sr
            _log({"event": "synth", "id": rid, "audio_ms": round(audio_ms, 1),
                  "gen_ms": round(gen_ms, 1),
                  "rtf": round(audio_ms / gen_ms, 2) if gen_ms else None})
        except Exception as e:
            out.write(f"ERR {rid} {str(e)[:120]}\n".encode())
            out.flush()
            _log({"event": "synth_error", "id": rid, "error": str(e)[:200]})


if __name__ == "__main__":
    main()
