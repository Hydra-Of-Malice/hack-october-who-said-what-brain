"""Stream an audio file into a live meeting over the same WebSocket the browser uses.

Simulates a microphone at real-time pace, so the meeting room, the speaker lanes and the FalkorDB
write counter update exactly as they do for a live mic. Useful for rehearsals and for testing.

  uv run --project apps/api python scripts/stream_wav.py demo/audio/02-launch-sync.ogg \
      --title "Launch sync (replay)" --seconds 60 [--server http://127.0.0.1:8000] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request

import websockets

from wsw_brain.listen import decode_file
from wsw_brain.listen.audio import float_to_pcm16

FRAME_S = 0.1


def create_meeting(server: str, title: str) -> dict:
    req = urllib.request.Request(
        f"{server}/api/meetings",
        data=json.dumps({"title": title}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


async def stream(server: str, meeting_id: str, samples, speed: float) -> None:
    ws_url = server.replace("http", "ws", 1) + f"/ws/meetings/{meeting_id}/audio"
    frame = int(16_000 * FRAME_S)
    async with websockets.connect(ws_url, max_size=None) as ws:

        async def drain() -> None:
            # The server sends a status frame every ~2 s. Unread frames fill the client's queue, which
            # stops it from reading pongs and ends the connection with a keepalive timeout.
            try:
                async for message in ws:
                    status = json.loads(message) if isinstance(message, str) else {}
                    if status.get("type") == "error":
                        print(f"server: {status.get('detail')}", flush=True)
            except websockets.ConnectionClosed:
                pass

        reader = asyncio.create_task(drain())
        start = time.perf_counter()
        for i in range(0, len(samples), frame):
            await ws.send(float_to_pcm16(samples[i : i + frame]))
            due = start + (i / 16_000) / speed
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
        await ws.send(json.dumps({"type": "stop", "end": True}))
        try:
            await asyncio.wait_for(reader, timeout=60)  # until the server closes the socket
        except asyncio.TimeoutError:
            reader.cancel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--title", default="Live replay")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=0.0, help="stream only the first N seconds (0 = all)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    args = ap.parse_args()

    samples = decode_file(args.file)
    if args.seconds > 0:
        samples = samples[: int(args.seconds * 16_000)]
    meeting = create_meeting(args.server, args.title)
    print(f"meeting {meeting['id']}: streaming {len(samples) / 16_000:.1f} s at {args.speed}x", flush=True)
    asyncio.run(stream(args.server, meeting["id"], samples, args.speed))
    print(f"done: {args.server}/m/{meeting['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
