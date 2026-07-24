# caption — diarized, word-timed captioning service (pony)

The batch, high-quality counterpart to the Voxtral realtime endpoint. Where
`bbb-transcribe.py` hits Voxtral (fast, but no speakers and only ~chunk-level
timing), this runs the **2026-07-23 bake-off winner** pipeline:

```
plain faster-whisper (NO VAD) → whisperx forced alignment
  → pyannote diarization → snap turns to pauses → JSON / SRT / ASS / txt
```

Whisper won on **native word timestamps + verbatim completeness**; WhisperX is
used *only* for align + diarize (its VAD-batched transcriber drops overlapping
speech). See `~/project/reports/recording-captioning-pipeline-study.md`.

## Pieces

| File | Runs as | Role |
|------|---------|------|
| `wx_caption.py` | whisperx venv (`/mnt/models/venvs/whisperx/bin/python`) | The pipeline CLI: audio in → `.json/.srt/.ass/.txt` |
| `caption_service.py` | `claude` `--user` (system `python3`, stdlib only) | HTTP job queue on `127.0.0.1:8001`; wraps bursts in one `gpu-lease` cycle |
| `caption-service.service` | — | systemd `--user` unit (linger is on for `claude`) |
| `../bbb-caption.py` | client (itpietraining) | extract 16 kHz wav → submit → poll → download |

## GPU sharing

The service serializes GPU use through **`gpu-lease`**: it `claim --wait`s the
card (stopping Voxtral once VRAM is idle), drains the whole job queue (with a
`CAPTION_IDLE_LINGER`-second grace so a burst costs one Voxtral bounce), then
`release`s (restarting Voxtral, waiting for `/health=200`). Each job is marked
`done` the instant its artifacts are written — *before* the ~133 s Voxtral cold
restart — so clients download immediately. `gpu-lease`'s safety timer
auto-releases if the worker dies.

Note: since the 2026-07-24 model migration, Voxtral cold-loads from
`/mnt/models` (~160 MB/s), so its restart is ~133 s; `gpu-lease` `HEALTH_TIMEOUT`
was raised to 300 s accordingly.

## HTTP API (behind haproxy `voxtral8443`, path `/caption*`, same Bearer token)

```
POST /caption/submit?<opts>   body = audio bytes (16 kHz mono wav preferred) → {job_id}
GET  /caption/status/<id>     → {status: queued|running|done|error, ...}
GET  /caption/result/<id>/<fmt>   fmt ∈ json|srt|ass|txt
GET  /caption/health          → ok
```

Submit opts (query string): `language`, `min_speakers`, `max_speakers`,
`prompt`, `names`, `model`, `formats`, `width`, `height`, `filename`.

## Deploy on pony

```bash
git -C ~/asr pull
mkdir -p ~/.config/systemd/user
ln -sf ~/asr/caption/caption-service.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now caption-service.service
```

haproxy (`/etc/haproxy/haproxy.cfg`, in `frontend voxtral8443`):

```
    use_backend caption_be if { path_beg /caption }
backend caption_be
    timeout server 5m
    server caption 127.0.0.1:8001
```

## Client usage (itpietraining)

```bash
VOXTRAL_TOKEN=$(cat /etc/bbb-transcribe.token) \
  bbb-caption.py recording.mp4 --min-speakers 2 --max-speakers 2
# → recording.srt, recording.ass, recording.json, recording.txt
```
