"""Live channels: Server-Sent Events for every tab (/api/events) and the microphone socket
(/ws/meetings/{id}/audio)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..events import EventBus
from .access import Access
from .capture import Capture, CaptureError
from .services import Services

log = logging.getLogger(__name__)

router = APIRouter()

PING_EVERY_S = 15.0
STATUS_EVERY_S = 2.0
PROCESSING_LAG_S = 1.0


# ---- SSE -----------------------------------------------------------------------------------------------------------
@router.get("/api/events")
async def events(
    request: Request,
    since_id: Annotated[int, Query(ge=0)] = 0,
    meeting_id: Annotated[str | None, Query(max_length=64)] = None,
    last_event_id: Annotated[str | None, Header(max_length=20)] = None,
) -> StreamingResponse:
    services: Services = request.app.state.services
    if since_id == 0 and last_event_id and last_event_id.strip().isdigit():
        since_id = int(last_event_id)  # EventSource's own reconnect sends the last id as a header
    stream = event_stream(
        request, services.bus, since_id=since_id, meeting_id=(meeting_id or "").strip() or None
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def event_stream(
    request: Request,
    bus: EventBus,
    *,
    since_id: int = 0,
    meeting_id: str | None = None,
    ping_every_s: float = PING_EVERY_S,
) -> AsyncIterator[str]:
    """Replays events after `since_id`, then streams new ones; a comment every `ping_every_s` keeps
    proxies from closing the connection. The first bytes go out at once so EventSource opens."""
    queue = bus.subscribe(since_id)
    try:
        yield "retry: 2000\n: connected\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), ping_every_s)
            except TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": ping\n\n"
                continue
            if meeting_id is not None and event.get("meeting_id") != meeting_id:
                continue
            yield format_event(event)
    finally:
        bus.unsubscribe(queue)


def format_event(event: dict[str, Any]) -> str:
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"id: {event['id']}\nevent: {event['type']}\ndata: {data}\n\n"


# ---- live audio --------------------------------------------------------------------------------------------------
@router.websocket("/ws/meetings/{meeting_id}/audio")
async def meeting_audio(websocket: WebSocket, meeting_id: str) -> None:
    services: Services = websocket.app.state.services
    access: Access = websocket.app.state.access
    if not access.origin_allowed(websocket):  # CORS does not cover WebSockets: refuse other sites' pages
        log.warning("audio socket for %s refused: origin %s", meeting_id, websocket.headers.get("origin"))
        await websocket.close(code=1008)  # before accept(): the handshake is answered with HTTP 403
        return
    await websocket.accept()
    if access.required and not access.is_operator(websocket):
        await _fail(websocket, "live capture needs the operator token (WSW_ADMIN_TOKEN)", 4401)
        return
    try:
        capture = await services.captures.attach(meeting_id)
    except CaptureError as exc:
        log.info("audio socket for %s refused: %s", meeting_id, exc)
        await _fail(websocket, str(exc), exc.code)
        return
    except Exception as exc:  # GraphUnavailable and other infrastructure failures
        log.exception("audio socket for %s could not start", meeting_id)
        await _fail(websocket, f"could not start capturing: {exc}", 1011)
        return

    sender = asyncio.create_task(_send_status(websocket, capture))
    end: bool | None = None  # None: the client vanished without a stop frame
    error: CaptureError | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                capture.feed(data)
                continue
            command = _parse_command(message.get("text"))
            if command.get("type") == "stop":
                end = command.get("end", True) is not False
                break
            log.info("audio socket for %s: ignoring unknown text frame %r", meeting_id, command.get("type"))
    except WebSocketDisconnect:
        pass
    except CaptureError as exc:
        error = exc
    except ValueError as exc:  # odd-length PCM frame
        error = CaptureError(f"bad audio frame: {exc}; send 16 kHz mono s16le PCM", 1003)
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await sender
        services.captures.detach(capture, end=end if error is None else False)
    if error is not None:
        await _fail(websocket, str(error), error.code)
    else:
        await _close(websocket, 1000)


async def _send_status(websocket: WebSocket, capture: Capture) -> None:
    """{"type": "status", "state": "listening"|"processing", "lag_s"} every STATUS_EVERY_S."""
    while True:
        lag = capture.lag_s
        state = "processing" if lag > PROCESSING_LAG_S else "listening"
        try:
            await websocket.send_json({"type": "status", "state": state, "lag_s": round(lag, 2)})
        except (WebSocketDisconnect, RuntimeError):  # socket already closing
            return
        await asyncio.sleep(STATUS_EVERY_S)


def _parse_command(text: str | None) -> dict[str, Any]:
    try:
        command = json.loads(text or "")
    except ValueError:
        return {}
    return command if isinstance(command, dict) else {}


async def _fail(websocket: WebSocket, detail: str, code: int) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": detail})
    await _close(websocket, code)


async def _close(websocket: WebSocket, code: int) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(Exception):
            await websocket.close(code)
