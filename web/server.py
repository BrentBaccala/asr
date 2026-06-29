#!/usr/bin/env python3
"""freesoft-interpret-web — browser interpreter bridge.

aiohttp serves the static web app + the WebRTC signaling (HTTP
offer/answer). aiortc terminates one peer connection per browser client,
bridges Opus<->PCM, and drives an asrpipe.Session (Voxtral ASR + NLLB MT
+ Pocket-TTS/Piper). Transcript/translation events ride a WebRTC data
channel; synthesized speech rides an outbound Opus track.

Modes
-----
* **solo** (Mode 1): one client, input_lang -> output_lang. The client
  hears the translation of its own speech.
* **duo** (Mode 2): two clients share a room via URL; each picks its own
  language. Each side's speech is translated to the *other* side's
  language and spoken to it. Both panes show both sides.

Run (on pony, as cosine, with Voxtral up on :8000)::

    ASRPIPE_NLLB_DIR=/home/cosine/asr/models/nllb-600m-ct2 \\
      /home/cosine/asr/web-env/bin/python web/server.py --port 8080

Then open http://<host>:8080/ . For the all-over-443 path see web/README.md.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import av
from aiohttp import web
from aiortc import (RTCPeerConnection, RTCSessionDescription,
                    RTCConfiguration, RTCIceServer, MediaStreamTrack)
from aiortc.mediastreams import MediaStreamError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asrpipe import PipelineConfig, Session, LANG_CHOICES  # noqa: E402

CFG = PipelineConfig()
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RTC_RATE = 48000          # Opus / aiortc native rate
FRAME_SAMPLES = 960       # 20 ms @ 48 kHz


# --------------------------------------------------------------------------
# Outbound audio track: a paced queue of 48 kHz mono PCM -> 20 ms frames.
# --------------------------------------------------------------------------
class PCMTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.buf = bytearray()
        self._ts = 0
        self._start = None
        self._resamplers = {}

    def push_pcm(self, pcm: bytes, sr: int):
        """Append TTS PCM (s16le mono at `sr`), resampled to 48 kHz."""
        if not pcm:
            return
        if sr == RTC_RATE:
            self.buf.extend(pcm)
            return
        rs = self._resamplers.get(sr)
        if rs is None:
            rs = av.AudioResampler(format="s16", layout="mono", rate=RTC_RATE)
            self._resamplers[sr] = rs
        arr = np.frombuffer(pcm, np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        frame.sample_rate = sr
        for out in rs.resample(frame):
            self.buf.extend(out.to_ndarray().astype(np.int16).tobytes())

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        self._ts += FRAME_SAMPLES
        target = self._start + self._ts / RTC_RATE
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        nbytes = FRAME_SAMPLES * 2
        if len(self.buf) >= nbytes:
            chunk = bytes(self.buf[:nbytes])
            del self.buf[:nbytes]
        else:
            chunk = bytes(self.buf) + b"\x00" * (nbytes - len(self.buf))
            self.buf.clear()
        arr = np.frombuffer(chunk, np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        frame.sample_rate = RTC_RATE
        frame.pts = self._ts
        frame.time_base = Fraction(1, RTC_RATE)
        return frame


async def consume_audio(track, session):
    """Decode the browser's inbound track to 16 kHz mono PCM -> session."""
    rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            break
        for out in rs.resample(frame):
            pcm = out.to_ndarray().astype(np.int16).tobytes()
            session.feed_pcm(pcm)


# --------------------------------------------------------------------------
# Rooms — one Member per browser; solo = self-loop, duo = cross-wire.
# --------------------------------------------------------------------------
class Member:
    def __init__(self, room, pc, label):
        self.room = room
        self.pc = pc
        self.label = label
        self.out = PCMTrack()
        self.dc = None
        self.session = None
        self.lang = "eng_Latn"        # duo: this member's language
        self.input_lang = "eng_Latn"  # solo: input language
        self.output_lang = "spa_Latn"  # solo: output language
        self._ev_backlog = []

    def send_event(self, ev_dict):
        msg = json.dumps({"t": "event", **ev_dict})
        if self.dc and self.dc.readyState == "open":
            try:
                self.dc.send(msg)
            except Exception:
                pass
        else:
            self._ev_backlog.append(msg)

    def flush_backlog(self):
        if self.dc and self.dc.readyState == "open":
            for m in self._ev_backlog:
                try:
                    self.dc.send(m)
                except Exception:
                    pass
            self._ev_backlog.clear()


class Room:
    def __init__(self, room_id, mode):
        self.id = room_id
        self.mode = mode          # "solo" | "duo"
        self.members = []

    def other(self, m):
        for x in self.members:
            if x is not m:
                return x
        return None

    def broadcast_event(self, ev_dict):
        for m in self.members:
            m.send_event(ev_dict)

    def _route_audio(self, src_member):
        """Where src_member's synthesized speech should play."""
        if self.mode == "solo":
            return src_member            # hear your own translation
        dest = self.other(src_member)
        return dest or src_member

    def session_langs(self, m):
        """(src, targets, speak) for member m's session."""
        if self.mode == "solo":
            return m.input_lang, [m.output_lang], [m.output_lang]
        dest = self.other(m)
        if dest is None:
            return m.lang, [], []        # nobody to translate for yet
        return m.lang, [dest.lang], [dest.lang]

    async def make_session(self, m, loop):
        src, targets, speak = self.session_langs(m)

        def on_event(ev):
            self.broadcast_event(ev.to_dict())

        def on_audio(code, pcm, sr):
            # Resolve the destination per-utterance: in duo mode the peer
            # may have joined (or changed) after this session was built.
            self._route_audio(m).out.push_pcm(pcm, sr)

        m.session = Session(CFG, src_lang=src, targets=targets,
                            speak_targets=speak, on_event=on_event,
                            on_audio=on_audio, label=m.label, loop=loop)
        await m.session.start()

    async def rewire(self, loop):
        """Recompute every member's session langs (after a join or a
        live language change)."""
        for m in self.members:
            if m.session is None:
                continue
            src, targets, speak = self.session_langs(m)
            await m.session.set_langs(src, targets, speak)


class RoomManager:
    def __init__(self):
        self.rooms = {}

    def get(self, room_id, mode):
        r = self.rooms.get(room_id)
        if r is None:
            r = Room(room_id, mode)
            self.rooms[room_id] = r
        return r


ROOMS = RoomManager()


# --------------------------------------------------------------------------
# HTTP handlers
# --------------------------------------------------------------------------
async def index(request):
    return web.FileResponse(os.path.join(STATIC, "index.html"))


async def langs(request):
    return web.json_response({"choices": LANG_CHOICES})


async def ice(request):
    """ICE config for the browser. With ASRPIPE_TURN_SECRET +
    ASRPIPE_TURN_HOST set (Phase 4), mint a coturn HMAC ephemeral
    credential for the muxed `turns:...:443?transport=tcp` relay and
    force `iceTransportPolicy: relay` (all media over 443). Without
    them (LAN dev), return an empty list + policy 'all'."""
    secret = os.environ.get("ASRPIPE_TURN_SECRET")
    host = os.environ.get("ASRPIPE_TURN_HOST")
    if not (secret and host):
        return web.json_response({"iceServers": [], "iceTransportPolicy": "all"})
    import hashlib
    import hmac
    import base64 as b64
    ttl = int(os.environ.get("ASRPIPE_TURN_TTL", "600"))
    username = f"{int(time.time()) + ttl}:web"
    digest = hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    credential = b64.b64encode(digest).decode()
    port = os.environ.get("ASRPIPE_TURN_PORT", "443")
    return web.json_response({
        "iceServers": [{
            "urls": [f"turns:{host}:{port}?transport=tcp"],
            "username": username, "credential": credential,
        }],
        "iceTransportPolicy": "relay",
    })


async def offer(request):
    params = await request.json()
    room_id = params.get("room") or "solo-" + os.urandom(3).hex()
    mode = params.get("mode") or ("duo" if params.get("room") else "solo")
    room = ROOMS.get(room_id, mode)
    if len(room.members) >= (2 if mode == "duo" else 1):
        return web.json_response({"error": "room full"}, status=409)

    loop = asyncio.get_event_loop()
    pc = RTCPeerConnection()
    label = params.get("label") or chr(ord("A") + len(room.members))
    m = Member(room, pc, label)
    m.input_lang = params.get("input_lang", "eng_Latn")
    m.output_lang = params.get("output_lang", "spa_Latn")
    m.lang = params.get("lang", m.input_lang)
    room.members.append(m)

    pc.addTrack(m.out)

    @pc.on("datachannel")
    def on_datachannel(channel):
        m.dc = channel
        m.flush_backlog()

        @channel.on("message")
        def on_message(msg):
            try:
                data = json.loads(msg)
            except Exception:
                return
            if data.get("t") == "setlang":
                if "input_lang" in data:
                    m.input_lang = data["input_lang"]
                if "output_lang" in data:
                    m.output_lang = data["output_lang"]
                if "lang" in data:
                    m.lang = data["lang"]
                asyncio.ensure_future(room.rewire(loop))

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(consume_audio(track, _ensure_session(m, loop, room)))

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await _cleanup(room, m)

    # Build the pipeline session BEFORE setRemoteDescription: the latter
    # synchronously fires on_track, whose consume_audio() needs the
    # session to already exist.
    await room.make_session(m, loop)
    await room.rewire(loop)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({
        "sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
        "room": room_id, "mode": mode, "label": label,
    })


def _ensure_session(m, loop, room):
    # make_session already created it in offer(); this just returns it.
    return m.session


async def _cleanup(room, m):
    try:
        if m.session:
            await m.session.close()
    except Exception:
        pass
    if m in room.members:
        room.members.remove(m)
    if not room.members:
        ROOMS.rooms.pop(room.id, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/langs", langs)
    app.router.add_get("/ice", ice)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", STATIC)
    print(f"freesoft-interpret-web on http://{args.host}:{args.port}/  "
          f"(voxtral={CFG.voxtral_uri} nllb={CFG.nllb_dir})", flush=True)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
