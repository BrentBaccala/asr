# freesoft-interpret-web

A browser interpreter that connects clients to the `freesoft-asr`
pipeline (Voxtral realtime ASR + NLLB-200 MT + Kyutai Pocket-TTS / Piper)
over WebRTC. Two modes, push-to-talk with a lock-open option, live
per-client language selection, and a scrolling transcript+translation
pane.

```
browser ──WebRTC(Opus)──▶ aiohttp+aiortc (pony, web-env)
                              ├─ Opus → 16 kHz PCM ─▶ asrpipe.Session
                              │        Voxtral WS (:8000) → NLLB → TTS sidecar
                              ├─ transcript/translation ─▶ data channel "ctl"
                              └─ TTS PCM → Opus track ────▶ browser <audio>
```

## Components

| Path | Role |
|------|------|
| `asrpipe/` (repo root) | Multi-instance pipeline core. `Session` = one independent ASR+MT+TTS stream; `NllbTranslator` + `TtsManager` are process-wide shared singletons. Reuses the exact Voxtral WS, NLLB `translate_multi`, and TTS-sidecar protocols from `freesoft-asr` **without** its globals/TUI. `freesoft-asr` is left untouched. |
| `web/server.py` | aiohttp app: static + `/offer` (WebRTC signaling) + `/langs` + `/ice`. aiortc bridges Opus↔PCM per client; a `Room` wires sessions (solo = self-loop, duo = cross-translate). |
| `web/static/index.html` | Browser UI: getUserMedia (AEC on), PTT button + mic-lock toggle, language dropdowns (live-changeable), transcript pane. |

## Modes

- **solo** (Mode 1) — open `/`. Pick *In* and *Out* languages; speak the
  In language, hear the Out language spoken back, both shown in the pane.
- **duo** (Mode 2) — open `/?room=NAME` in two browsers. Each picks *My
  language*. Each side's speech is translated to the other side's
  language and spoken to it; both panes show both sides.

Language dropdowns are changeable **mid-session** — they send a
`setlang` over the data channel and the server re-targets the session
(`Session.set_langs`) without dropping the Voxtral WS.

## Running (LAN, no TURN — Phase 2/3)

On **pony**, with Voxtral up (`systemctl --user start voxtral`, port 8000):

```bash
cd ~/asr
ASRPIPE_NLLB_DIR=~/asr/models/nllb-600m-ct2 \
  ~/asr/web-env/bin/python web/server.py --host 0.0.0.0 --port 8080
```

The `web-env` venv needs: `aiortc aiohttp av numpy websockets ctranslate2
transformers sentencepiece`. Open `http://pony:8080/`. `/ice` returns an
empty ICE list + policy `all` (host candidates) when no TURN is
configured.

## All-over-443 (Phase 4 — edge mux, not yet stood up)

Set these on the server so `/ice` mints coturn HMAC ephemeral creds and
the browser forces `iceTransportPolicy: relay`:

```
ASRPIPE_TURN_SECRET=<coturn static-auth-secret>
ASRPIPE_TURN_HOST=osito.freesoft.org
ASRPIPE_TURN_PORT=443
```

Then on **edge** (osito.freesoft.org), per the BBB `install_haproxy`
pattern:

- **haproxy** `mode tcp`, `bind *:443 ssl crt /etc/haproxy/certbundle.pem
  alpn h2,http/1.1,stun.turn`; `use_backend turn if { ssl_fc_alpn
  stun.turn }`, else `web`. `backend web` → pony:8080; `backend turn` →
  coturn `127.0.0.1:3478`. Cert reloadcmd rebuilds `certbundle.pem` from
  the acme.sh ECC cert and reloads haproxy.
- **coturn** plaintext `listening-port=3478` (no TLS — haproxy provides
  it), `use-auth-secret` + `static-auth-secret=<same as ASRPIPE_TURN_SECRET>`,
  `realm=osito.freesoft.org`, and lock relay peers to pony:
  `denied-peer-ip=0.0.0.0-255.255.255.255` then
  `allowed-peer-ip=<pony LAN IP>`.

aiortc on pony reaches the same coturn at edge's LAN IP:3478 (plaintext),
so both ends meet at the relay → all media over 443.

## Validated (2026-06-29)

- Two concurrent Voxtral `/v1/realtime` sessions within the 16384 KV
  budget; GPU ~20 GiB used, ~4.5 GiB free; +1 `_24l` TTS model ⇒ ~21.8
  GiB (≈2.7 GiB free).
- `asrpipe.Session` live: Spanish audiobook → Spanish finals → English
  translations → English TTS PCM.
- Mode 1 over aiortc (headless client): transcript events on the data
  channel + inbound TTS audio frames.
- Mode 2 two-client room: A's speech → both panes (English translation) →
  A's TTS routed to **B** (not A).

## Not yet done

- Real-browser Playwright run of the UI (PTT/lock/dropdowns) — the JS is
  straightforward; the server-side bridge is proven via aiortc.
- Phase 4 edge coturn + haproxy 443 mux + remote-over-443 validation.
- Long-call Voxtral session recycle is a plain WS reconnect in
  `_ws_loop`; the VAD-driven recycle from `freesoft-asr` is not ported.
