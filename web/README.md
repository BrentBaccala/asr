# freesoft-interpret-web

A browser interpreter that connects clients to the `freesoft-asr`
pipeline (Voxtral realtime ASR + NLLB-200 MT + Pocket-TTS / Piper /
MeloTTS) over WebRTC. Solo and paired modes, a four-mode microphone
control, live per-client language selection across **13 languages**, and
a scrolling transcript+translation pane. Deployed in production on
**pony** at `https://osito.freesoft.org/` (everything over port 443).

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
| `web/server.py` | aiohttp app: static + `/offer` (WebRTC signaling) + `/langs` + `/ice`. aiortc bridges Opus↔PCM per client; a `Room` wires sessions (solo = self-loop, duo = cross-translate) and broadcasts join/leave notices; `/offer` returns the client's room `label`. |
| `web/static/index.html` | Browser UI: getUserMedia (AEC on), the round talk button, a 4-mode mic selector, a Pair/Unpair toggle, live language dropdowns, and a LOCAL/REMOTE colour-coded transcript pane. |
| `web/deploy/` | Deployment configs: `interpret-web.service` (cosine `--user` unit), `haproxy.cfg`, `turnserver.conf`. |
| `systemd/voxtral.service` | vLLM Voxtral backend unit (see repo-root `systemd/`). |

## Conversation modes

- **solo** — the default. Open `/`. Pick *In* and *Out* languages; speak
  the In language, hear the Out language spoken back, both shown in the
  pane. Two browsers opened independently are just two solo sessions.
- **paired (duo)** — click **Pair** on both clients. They join a shared
  room (`?room=pair`), each keeps its own *My language*, and each side's
  speech is translated to the **other** side's language and spoken to it;
  both panes show both sides. **Unpair** returns to independent solo. The
  transcript labels each utterance **LOCAL** (you, blue) or **REMOTE**
  (peer, orange). Status reads `waiting for peer` until the second client
  joins, then `paired`; it reverts if the peer drops.

Language dropdowns are changeable **mid-session** — they send a
`setlang` over the data channel and the server re-targets the session
(`Session.set_langs`) without dropping the Voxtral WS.

## Microphone modes

A `Microphone mode` dropdown selects how the mic is actuated; the round
button is the talk actuator for the two push modes and glows red whenever
audio is being transmitted:

| Mode | Behaviour | Round button |
|------|-----------|--------------|
| **Hold to talk** (default) | Transmit only while the button is held | hold |
| **Tap to talk / stop** | Tap to start transmitting, tap again to stop | tap toggles |
| **Locked on** | Always transmitting | disabled (glows = live) |
| **Disconnected** | OS mic fully released so other phone apps can use it | disabled |

Transmitting (`live`) is decoupled from holding the OS mic
(`micConnected`). The three non-disconnected modes keep the mic and swap
it on/off the live `RTCRtpSender` via `replaceTrack()` — no
renegotiation. **Background auto-release:** when the page is hidden (you
switch apps or lock the phone) the mic is released; on return the prior
mode is restored (unless you had explicitly chosen Disconnected).

## Running

In production it runs as two **cosine `--user` systemd services** on pony
(`linger` on, so they start at boot):

- `voxtral.service` — vLLM Voxtral-Mini-4B-Realtime on `127.0.0.1:8000`
  (~70–95 s to warm up).
- `interpret-web.service` — this app on `0.0.0.0:8090`
  (`Wants=voxtral`). Reads `web/interpret-web.env` for `ASRPIPE_NLLB_DIR`
  + the `ASRPIPE_TURN_*` vars.

```bash
systemctl --user status voxtral interpret-web
systemctl --user restart interpret-web        # picks up server.py changes
journalctl --user -u interpret-web -f
```

Manual / dev invocation (with Voxtral already up):

```bash
cd ~/asr
ASRPIPE_NLLB_DIR=~/asr/models/nllb-600m-ct2 \
  ~/asr/web-env/bin/python web/server.py --host 0.0.0.0 --port 8090
```

The `web-env` venv needs: `aiortc aiohttp av numpy websockets ctranslate2
transformers sentencepiece`. With no `ASRPIPE_TURN_*` set, `/ice` returns
an empty ICE list + policy `all` (host candidates) for LAN dev.

## All-over-443 (production, on pony)

The public site is served entirely over 443 (with 80 for ACME), both
forwarded by the router to pony. TTS models and Voxtral share pony's GPU;
aiortc and coturn are co-located on pony, so relayed media never leaves
the box. (Historically this ran on edge; the DMZ was migrated edge→pony
on 2026-06-30 and edge powered off, so all of haproxy/coturn/cert/acme.sh
now live on pony with localhost wiring.)

`/ice` mints coturn HMAC ephemeral creds and forces the browser to
`iceTransportPolicy: relay` when these are set (in `interpret-web.env`):

```
ASRPIPE_TURN_SECRET=<coturn static-auth-secret>
ASRPIPE_TURN_HOST=osito.freesoft.org
ASRPIPE_TURN_PORT=443
```

**haproxy** (`/etc/haproxy/haproxy.cfg`, `mode tcp`) terminates TLS on
443 and muxes by ALPN:

```
frontend mux443
    bind *:443 ssl crt /etc/haproxy/certbundle.pem alpn http/1.1,stun.turn
    use_backend web  if { ssl_fc_alpn -m str http/1.1 }   # → 127.0.0.1:8090
    use_backend turn if { ssl_fc_alpn -m str stun.turn }  # → 127.0.0.1:3478
    acl http_first req.payload(0,1),hex -m str 47 50 48 44 4F 43 54
    use_backend web if http_first                         # ALPN-less fallback
    default_backend turn
```

The `http_first` hex is the leading byte of the HTTP verbs
(`G`/`P`/`H`/`D`/`O`/`C`/`T`), so a browser that doesn't set ALPN still
routes to the web backend; anything else defaults to TURN.

**coturn** (`/etc/turnserver.conf`) is plaintext — haproxy provides the
TLS: `listening-ip=127.0.0.1`, `listening-port=3478`, `use-auth-secret`
+ `static-auth-secret=<same as ASRPIPE_TURN_SECRET>`,
`realm=osito.freesoft.org`, `relay-ip=192.168.0.108`, and the relay is
locked to pony (`denied-peer-ip=0.0.0.0-255.255.255.255` then
`allowed-peer-ip=192.168.0.108`).

**Cert**: acme.sh renews the Let's Encrypt cert via HTTP-01 on port 80,
rebuilds `certbundle.pem`, and reloads haproxy.

## Deploying updates

Edit under `~/asr` on samsung, commit, `git push`. On pony, as cosine:
`git -C ~/asr pull`. Static changes (`web/static/index.html`) are served
fresh on the next request — no restart. `web/server.py` changes need
`systemctl --user restart interpret-web`.

## Status / known gaps

- **Live in production** on pony over 443: all 13 languages
  (transcribe → translate → speak), solo + paired modes, the 4 mic modes,
  Pair/Unpair, and LOCAL/REMOTE labels — manually verified on desktop and
  Android.
- No automated Playwright run of the browser UI yet; the server-side
  bridge is proven via aiortc.
- Long-call handling is a plain WS reconnect in `Session._ws_loop`; the
  VAD-driven session recycle from `freesoft-asr` is not ported.
- NLLB can degenerate into n-gram repetition on very long run-on inputs
  (`no_repeat_ngram_size` mitigation identified but not applied).
