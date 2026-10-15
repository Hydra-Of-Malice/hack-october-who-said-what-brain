"""Talking to the brain: typed questions (/api/ask) and spoken ones (/api/ask/voice)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..brain import transcribe_question
from ..contracts import AskResult
from .rest import ServicesDep
from .schemas import AskRequest
from .services import Services

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MAX_QUESTION_BYTES = 8 * 1024 * 1024  # a 30 s clip is ~1 MB as 16 kHz PCM16; allow 48 kHz stereo too
MAX_HINT_NAMES = 24
# The agent's spoken commands. Without them Gemma 4 E4B hears a short "Mark it done." as "Market done"
# (3 of 3 tries on a SAPI clip); with them 3 of 3 were right and longer questions were unchanged.
COMMAND_TERMS = "mark it done, action item, decision, due date, follow-up"


@router.post("/ask")
async def ask(body: AskRequest, services: ServicesDep) -> dict[str, Any]:
    result = await _answer(services, body.session_id, body.question, body.meeting_id)
    return result.to_dict()


@router.post("/ask/voice")
async def ask_voice(
    services: ServicesDep,
    audio: Annotated[UploadFile, File(description="16 kHz mono WAV, at most 30 s")],
    session_id: Annotated[str, Form(min_length=1, max_length=128)],
    meeting_id: Annotated[str | None, Form(max_length=64)] = None,
) -> dict[str, Any]:
    session = " ".join(session_id.split())
    if not session:
        raise HTTPException(status_code=422, detail="session_id must not be blank")
    data = await audio.read(MAX_QUESTION_BYTES + 1)
    if len(data) > MAX_QUESTION_BYTES:
        raise HTTPException(
            status_code=413, detail="question audio is too large; keep questions under 30 seconds"
        )
    hint = await asyncio.to_thread(_question_hint, services)
    try:
        transcript = await transcribe_question(data, hint=hint)
    except ValueError as exc:
        raise _audio_error(str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"could not transcribe the question: {exc}") from exc
    text = transcript.text.strip()
    if not text:
        result = AskResult(
            answer="",
            citations=[],
            actions=[],
            tool_trace=[],
            cypher=[],
            model=services.llm.model,
            provider=services.llm.provider,
            session_id=session,
        )
    else:
        result = await _answer(services, session, text, (meeting_id or "").strip() or None)
    result.transcript = text
    result.transcript_provider = transcript.provider
    result.timings["transcribe"] = round(transcript.latency_ms, 1)
    result.timings["total"] = round(result.timings.get("total", 0.0) + transcript.latency_ms, 1)
    return result.to_dict()


async def _answer(services: Services, session_id: str, question: str, meeting_id: str | None) -> AskResult:
    started_ms = int(time.time() * 1000)
    try:
        result = await services.agent.ask(session_id, question, meeting_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if any(a.get("type") == "refresh" and a.get("what") == "action_items" for a in result.actions):
        await services.announce_completed_items(started_ms)
    return result


def _question_hint(services: Services) -> str:
    """People's names and the command words help Gemma 4 spell what it hears."""
    try:
        names = [p["name"] for p in services.store.list_people() if p.get("name")]
    except Exception as exc:  # noqa: BLE001 - names are optional
        log.debug("no names for the spoken-question hint: %s", exc)
        names = []
    return ", ".join([*names[:MAX_HINT_NAMES], COMMAND_TERMS])


def _audio_error(detail: str) -> HTTPException:
    if "limit is" in detail:
        return HTTPException(status_code=413, detail=detail)
    if "silent" in detail:
        return HTTPException(status_code=400, detail=f"No speech heard: {detail}")
    return HTTPException(status_code=400, detail=detail)
