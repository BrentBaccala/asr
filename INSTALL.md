# Installation

Runtime setup for `freesoft-asr` on a Linux host with PipeWire and a
CUDA GPU. Assumes the repo is already cloned to `~/asr`. What follows
is the venvs and the model files — none of which are shipped in
the repo (too large to track).

> Translation models are only needed if you actually translate. A bare
> `freesoft-asr` (transcription-only of the default sink monitor) needs
> just the vLLM Voxtral endpoint — not NLLB or fastText.

## Prerequisites

- **Linux + PipeWire.** The `--dual` audio path uses `pw-record`;
  single-stream mode just needs anything that pipes 16-kHz mono
  S16LE PCM to stdin.
- **CUDA-capable GPU** with ≥ 16 GB VRAM (Voxtral weights + KV cache
  at `--max-model-len 16384`). Tested on an RTX 3090.
- **[`uv`](https://docs.astral.sh/uv/)** (default install location
  `~/.local/bin/uv`). All venv commands below go through `uv`.

## The two required venvs

`freesoft-asr` needs two Python venvs at fixed paths (the script
shebangs and the systemd unit's `ExecStart` reference them
absolutely):

| Venv | Role | Size |
|---|---|---|
| `~/asr/mt-env` | `freesoft-asr`'s interpreter — NLLB translation, WS client, TUI | ~5 GB |
| `~/asr/vllm-env` | vLLM serving Voxtral on `/v1/realtime` | ~7 GB |

### `mt-env`

```bash
uv venv --python 3.12 ~/asr/mt-env
~/.local/bin/uv pip install --python ~/asr/mt-env/bin/python \
    transformers ctranslate2 websockets silero-vad onnxruntime numpy torch \
    fasttext-langdetect
```

`fasttext-langdetect` (or `fasttext-wheel` + a local `lid.176.ftz`)
provides the per-sentence source-language detection. It's optional: if
absent, translation falls back to `default_src_lang` with a warning, and
transcription-only runs don't use it at all.

### `vllm-env`

```bash
uv venv --python 3.12 ~/asr/vllm-env
~/.local/bin/uv pip install --python ~/asr/vllm-env/bin/python vllm
```

vLLM brings ~200 transitive dependencies; install takes several
minutes on a fresh cache.

## The NLLB model (only if you translate)

`freesoft-asr` expects an int8-quantised CTranslate2 conversion of
NLLB-200-distilled-600M at `~/asr/models/nllb-600m-ct2/` (or wherever
`nllb_dir` / `--nllb-dir` points). Convert inside `mt-env` (which has
`ct2-transformers-converter`):

```bash
~/asr/mt-env/bin/ct2-transformers-converter \
    --model facebook/nllb-200-distilled-600M \
    --quantization int8 \
    --output_dir ~/asr/models/nllb-600m-ct2
```

Downloads ~1.2 GB from HuggingFace, produces a ~600 MB CT2 directory.
Takes ~5 minutes. Skip this entirely for transcription-only use.

## The fastText language-id model (optional)

Per-sentence source-language detection uses fastText `lid.176`. The
`fasttext-langdetect` wrapper downloads its own model (~126 MB) on first
`detect()` call. If you instead use raw `fasttext-wheel`, fetch the
~917 KB model and point `FASTTEXT_LID_MODEL` at it (or drop it at
`~/asr/models/lid.176.ftz`):

```bash
curl -L -o ~/asr/models/lid.176.ftz \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

> **NumPy 2.x caveat (important).** `fasttext-langdetect` pulls
> `fasttext 0.9.3`, whose `predict()` calls `np.array(obj, copy=False)`
> — which **raises** `ValueError: Unable to avoid copy…` under NumPy ≥ 2.0
> (the default in current environments, including this `mt-env`). The
> install *succeeds* but detection crashes on first use. Patch the three
> occurrences in the installed `fasttext` to the NumPy-2 idiom:
>
> ```bash
> F=$(~/asr/mt-env/bin/python -c 'import fasttext,os; print(os.path.join(os.path.dirname(fasttext.__file__),"FastText.py"))')
> sed -i 's/np\.array(\([A-Za-z0-9_]*\), copy=False)/np.asarray(\1)/g' "$F"
> ```
>
> Verify:
> `~/asr/mt-env/bin/python -c "from ftlangdetect import detect; print(detect(text='hola mundo')['lang'])"`
> should print `es`. The patch lives in the venv site-package, so
> **re-apply it after any `fasttext` reinstall** (or pin a
> NumPy-2-compatible fasttext fork). Without detection, translation
> still works — it just falls back to `default_src_lang`.

## vLLM as a systemd user service

Optional but recommended on a dedicated inference host. A tracked
unit file is provided at `~/asr/systemd/voxtral.service`; symlink it
into the user systemd directory:

```bash
mkdir -p ~/.config/systemd/user
ln -s ~/asr/systemd/voxtral.service ~/.config/systemd/user/voxtral.service
systemctl --user daemon-reload
systemctl --user start voxtral

# Poll for ready (~40-95s cold-cache; cudagraph capture + model load)
until [ "$(curl -s -m3 -o /dev/null -w %{http_code} \
           http://127.0.0.1:8000/health)" = "200" ]; do sleep 5; done
echo "vLLM ready"
```

Inspection and control:

```bash
systemctl --user status voxtral
journalctl --user -u voxtral -f       # live log
systemctl --user stop voxtral         # free the GPU for other tools
systemctl --user restart voxtral
systemctl --user enable voxtral       # also start at user-session boot
```

The unit auto-restarts on crash (handles the
`--max-model-len` assertion if `freesoft-asr`'s VAD-driven recycle ever
misses one), and exposes `PATH=~/asr/vllm-env/bin:...` so the
PIECEWISE cudagraph compile pass can find `ninja`.

If you don't want a service, just run the same `ExecStart` line by
hand in a terminal — `cat ~/asr/systemd/voxtral.service` has it.

## First run

```bash
~/asr/freesoft-asr                  # transcribe the default sink monitor
~/asr/freesoft-asr --dual           # if PipeWire RTP sources are set up
# or single-stream from stdin:
pw-record --target <your-source> --format=s16 --rate=16000 \
          --channels=1 - | ~/asr/freesoft-asr --source -
```

`freesoft-asr --help` documents the CLI knobs with current defaults, and
`freesoft-asr --write-config` emits a commented starter config. The
PipeWire setup for the `--dual` RTP sources is in
[README.md](README.md#pipewire-setup-for-the-dual-stream-rtp-path).

## Recreating the alternative venvs and models

The other `stream-*.py` scripts in the repo aren't required for
`freesoft-asr`. They're kept as reference implementations of different
ASR engines and latency/quality trade-offs (see the table in
[README.md](README.md#alternative-streaming-scripts-in-this-repo-for-reference)).
Their venvs aren't shipped either — recreate them only if you want
to run those specific scripts:

| Venv | Used by | Setup |
|---|---|---|
| `asr-env` | `stream-whisper.py`, `stream-whisper-buffered.py`, `asr-call-transcribe` | `uv venv ~/asr/asr-env && uv pip install --python ~/asr/asr-env/bin/python faster-whisper silero-vad onnxruntime` |
| `sherpa-env` | `stream-sherpa-ipa.py` | `uv venv ~/asr/sherpa-env && uv pip install --python ~/asr/sherpa-env/bin/python sherpa-onnx` |
| `vosk-env` | `stream-vosk.py` | `uv venv ~/asr/vosk-env && uv pip install --python ~/asr/vosk-env/bin/python vosk srt` |

The torch-based scripts (`stream-canary.py`, `stream-parakeet*.py`,
`stream-cacheaware.py`) expect a separate `~/venv-3.12-torch/` venv
shared with other torch projects — outside this repo by design, see
each script's docstring.

Model files those scripts load (also not shipped):

| Model dir | Used by | Source |
|---|---|---|
| `models/vosk-model-es-0.42` | `stream-vosk.py` | <https://alphacephei.com/vosk/models> |
| `models/vosk-model-small-es-0.42` | `stream-vosk.py --small` | <https://alphacephei.com/vosk/models> |
| `models/sherpa-es-ipa` | `stream-sherpa-ipa.py` | <https://huggingface.co/bookbot/sherpa-onnx-zipformer-streaming-robust-es-v0> |
