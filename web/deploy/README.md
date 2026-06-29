# Phase 4 deployment — WebRTC all over 443 (edge mux + pony service)

Validated 2026-06-29: Mode 1 and Mode 2 both work from a client with all
WebRTC media relayed through coturn-on-443. Reference configs in this dir.

## Topology

```
browser ──TLS:443──▶ edge (osito.freesoft.org, public 100.36.129.3)
                       haproxy (mode tcp, terminates TLS, ALPN demux)
   ├─ alpn http/1.1 ─▶ backend web  → pony 192.168.0.108:8090 (aiohttp app)
   └─ alpn stun.turn ▶ backend turn → coturn 127.0.0.1:3478 (plaintext)
                                        └─UDP relay (relay-ip 192.168.0.119,
                                          ports 49160-49200, peer-locked to
                                          pony 192.168.0.108)─▶ pony aiortc
```

Key design point: the **browser** forces `iceTransportPolicy:relay`; **aiortc
on pony uses its LAN host candidate** (NOT TURN). coturn is on the same /24
as pony, so it relays browser↔pony with **pony as the single
`allowed-peer-ip`**. (Double-relay would conflict with locking the peer to
pony; single-relay is cleaner and is what `allowed-peer-ip=pony` wants.)

## edge (Ubuntu 20.04, passwordless sudo)

- `apt install coturn haproxy`. ufw is inactive (no firewall to open).
- `/etc/turnserver.conf` ← `deploy/turnserver.conf` (set a real
  `static-auth-secret`, stored at `/etc/coturn-shared-secret` mode 600).
  `echo TURNSERVER_ENABLED=1 > /etc/default/coturn`; `systemctl restart coturn`.
- `/etc/haproxy/haproxy.cfg` ← `deploy/haproxy.cfg`.
- Cert bundle: `cat /root/.acme.sh/osito.freesoft.org_ecc/fullchain.cer
  /root/.acme.sh/osito.freesoft.org_ecc/osito.freesoft.org.key >
  /etc/haproxy/certbundle.pem` (mode 600). Reload script
  `/usr/local/bin/haproxy-cert-reload.sh` rebuilds the bundle + reloads
  haproxy; registered as the acme.sh reloadcmd:
  `acme.sh --install-cert -d osito.freesoft.org --ecc --reloadcmd
  /usr/local/bin/haproxy-cert-reload.sh`. acme.sh renewal stays on HTTP-01
  port 80 (haproxy binds 443 only; nothing else uses 80).
- ALPN note: only `http/1.1,stun.turn` are advertised (NOT `h2`) because the
  web backend is aiohttp (HTTP/1.1-only); advertising h2 would let browsers
  negotiate a protocol aiohttp can't speak.

## pony (cosine user)

- App runs as the `cosine --user` unit `interpret-web.service`
  (`deploy/interpret-web.service`), in `/home/cosine/asr/web-env` (aiortc +
  aiohttp + av + numpy + websockets + ctranslate2 + transformers +
  sentencepiece). Port 8090 (8080 is taken by docker-proxy on pony).
- `web/interpret-web.env` (mode 600): `ASRPIPE_NLLB_DIR`,
  `ASRPIPE_TURN_SECRET` (== edge secret), `ASRPIPE_TURN_HOST=osito.freesoft.org`,
  `ASRPIPE_TURN_PORT=443`. The `/ice` endpoint mints coturn HMAC ephemeral
  creds and forces `iceTransportPolicy:relay`.
- `Wants=voxtral.service`; both enabled for the cosine user (linger on).

## Real-mic browser test (human step)

Headless Playwright has no audio device (`getUserMedia` →
"Requested device not found"), so the full mic→transcript loop needs a
human with a real microphone/headset:

1. Open `https://osito.freesoft.org/` (Mode 1) or
   `https://osito.freesoft.org/?room=NAME` in two browsers (Mode 2).
2. Allow microphone when prompted.
3. Mode 1: set In=English, Out=Spanish. Hold **Hold to talk**, speak
   English; the transcript pane shows English + Spanish and Spanish TTS
   plays back. Toggle **Lock mic** for open-mic.
4. Mode 2: each side picks **My language**; speak and the other side hears
   the translation. Change the dropdown mid-call to re-target live.
