#!/home/cosine/asr/vllm-env/bin/python
"""
Live streaming Spanish transcription via Voxtral-Mini-4B-Realtime (vLLM).

Reads raw s16le mono 16 kHz PCM from stdin and streams it to the local
vLLM /v1/realtime WebSocket, delta-printing the transcript append-only
(no carriage-return overwrite, so terminal line-wrap is harmless).

Wire it to the samsung->pony RTP call tap exactly like the other
~/asr/stream-*.py scripts:

  ssh cosine@pony
  pw-record --target rtp_call_remote_source --format=s16 --rate=16000 \
            --channels=1 - | ~/asr/stream-voxtral.py

Prerequisite: the vLLM realtime server must be running on pony:
  setsid bash /tmp/serve_voxtral.sh </dev/null >/tmp/vllm.log 2>&1 &
  (listens on 127.0.0.1:8000; ~40 s to become ready)

The audio arrives at real wall-clock rate from the live tap, so no
pacing is done here — chunks are forwarded as they arrive. Stdin EOF
(call/pipe ends) flushes a final commit and exits.
"""
import asyncio
import base64
import json
import sys
import threading

import websockets

HOST, PORT = "127.0.0.1", 8000
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CHUNK = 2048 * 2  # 2048 samples = 128 ms of s16le @ 16 kHz


def stdin_reader(q, loop):
    """Blocking stdin read in a thread; hand bytes to the asyncio loop."""
    while True:
        b = sys.stdin.buffer.read(CHUNK)
        if not b:
            loop.call_soon_threadsafe(q.put_nowait, None)  # EOF sentinel
            return
        loop.call_soon_threadsafe(q.put_nowait, b)


async def main():
    uri = f"ws://{HOST}:{PORT}/v1/realtime"
    try:
        ws = await websockets.connect(uri, max_size=None)
    except Exception as e:
        print(f"[stream-voxtral] cannot reach vLLM at {uri}: {e}",
              file=sys.stderr)
        print("  start it:  setsid bash /tmp/serve_voxtral.sh "
              "</dev/null >/tmp/vllm.log 2>&1 &", file=sys.stderr)
        return

    async with ws:
        r = json.loads(await ws.recv())
        if r.get("type") != "session.created":
            print(f"[stream-voxtral] unexpected: {r}", file=sys.stderr)
            return
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("[stream-voxtral] connected, streaming... (Ctrl-C to stop)",
              file=sys.stderr, flush=True)

        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        threading.Thread(target=stdin_reader, args=(q, loop),
                         daemon=True).start()

        async def receiver():
            while True:
                m = json.loads(await ws.recv())
                t = m.get("type")
                if t == "transcription.delta":
                    sys.stdout.write(m["delta"])
                    sys.stdout.flush()
                elif t == "transcription.done":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                elif t == "error":
                    print(f"\n[stream-voxtral] server error: {m}",
                          file=sys.stderr)
                    return

        rx = asyncio.create_task(receiver())
        while True:
            chunk = await q.get()
            if chunk is None:  # stdin EOF -> call/pipe ended
                await ws.send(json.dumps(
                    {"type": "input_audio_buffer.commit", "final": True}))
                break
            await ws.send(json.dumps(
                {"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(chunk).decode()}))

        try:
            await asyncio.wait_for(rx, timeout=15)
        except asyncio.TimeoutError:
            pass
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("[stream-voxtral] bye.", file=sys.stderr)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
