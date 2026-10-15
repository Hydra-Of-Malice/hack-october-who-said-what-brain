"""REST endpoints (docs/API_CONTRACT.md). Sync handlers run in the threadpool because GraphStore is
blocking; the bus is thread-safe."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from ..brand import brand
from ..listen import get_asr
from .access import TOKEN_COOKIE, TOKEN_COOKIE_MAX_AGE_S, Access
from .schemas import CreateMeeting, NameSpeaker, SetActionStatus, SignIn
from .services import MEETING_ID_RE, Services, new_meeting_id

router = APIRouter(prefix="/api")

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_SUFFIXES = (".wav", ".mp3", ".m4a", ".webm", ".ogg", ".oga", ".opus", ".flac", ".mp4", ".aac")
_CONTENT_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
}
_CHUNK = 1024 * 1024


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_access(request: Request) -> Access:
    return request.app.state.access


AccessDep = Annotated[Access, Depends(get_access)]


def require_operator(request: Request, access: AccessDep) -> None:
    access.check_operator(request)


def count_new_meeting(request: Request, access: AccessDep) -> None:
    access.check_create(request)


OPERATOR = [Depends(require_operator)]  # every change; a no-op unless WSW_ADMIN_TOKEN is set
NEW_MEETING = [*OPERATOR, Depends(count_new_meeting)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} not found")


async def _check_meeting_change(
    request: Request, access: Access, services: Services, meeting_id: str
) -> None:
    access.check_meeting_change(request, await asyncio.to_thread(services.meeting_summary, meeting_id))


# ---- service info --------------------------------------------------------------------------------------------
@router.get("/health")
def health(services: ServicesDep) -> dict[str, Any]:
    falkordb = services.store.ping()
    if falkordb["ok"] and not services.schema_ready:
        services.ensure_graph()
    if services.graph_error:
        falkordb["error"] = services.graph_error
    asr = get_asr()
    return {
        "ok": bool(falkordb["ok"] and services.schema_ready),
        "brand": brand(),
        "falkordb": falkordb,
        "llm": {
            "provider": services.llm.provider,
            "model": services.llm.model,
            "audio": services.audio_llm is not None,
        },
        "question_audio": "gemma4-audio:ollama" if services.audio_llm is not None else "whisper",
        "embedder": {"name": services.embedder.name, "dim": services.embedder.dim},
        "asr": {"name": asr.name, "loaded": asr.loaded},
    }


@router.get("/brand")
def get_brand() -> dict[str, str]:
    return brand()


@router.get("/auth")
def auth_status(request: Request, access: AccessDep) -> dict[str, bool]:
    return {"required": access.required, "operator": access.is_operator(request)}


@router.post("/auth")
def sign_in(body: SignIn, request: Request, response: Response, access: AccessDep) -> dict[str, bool]:
    """Exchange the operator token (WSW_ADMIN_TOKEN) for an HttpOnly cookie, so this browser may make changes."""
    if not access.required:
        return {"required": False, "operator": False}
    if not access.token_matches(body.token):
        raise HTTPException(status_code=401, detail="that is not the operator token")
    response.set_cookie(
        TOKEN_COOKIE,
        body.token.strip(),
        max_age=TOKEN_COOKIE_MAX_AGE_S,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return {"required": True, "operator": True}


@router.get("/stats")
def stats(services: ServicesDep) -> dict[str, int]:
    return services.store.stats()


# ---- meetings ---------------------------------------------------------------------------------------------------
@router.get("/meetings")
def list_meetings(services: ServicesDep) -> list[dict[str, Any]]:
    return services.store.list_meetings()


@router.post("/meetings", dependencies=NEW_MEETING)
def create_meeting(body: CreateMeeting, services: ServicesDep) -> dict[str, Any]:
    return services.new_meeting(body.title, "mic")


@router.post("/meetings/upload", dependencies=NEW_MEETING)
async def upload_meeting(
    services: ServicesDep,
    file: Annotated[UploadFile, File(description="wav, mp3, m4a, webm or ogg")],
    title: Annotated[str, Form(max_length=200)] = "",
    pace: Annotated[Literal["fast", "realtime"], Form()] = "fast",
) -> dict[str, Any]:
    suffix = _upload_suffix(file)
    await asyncio.to_thread(services.require_graph)
    meeting_id = new_meeting_id()
    path = services.uploads_dir / f"{meeting_id}{suffix}"
    try:
        await asyncio.to_thread(_save_upload, file.file, path)
        await asyncio.to_thread(_probe_audio, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    name = " ".join(title.split()) or _title_from_filename(file.filename)
    meeting = await asyncio.to_thread(services.new_meeting, name, "upload", meeting_id=meeting_id)
    services.files.start_upload(meeting_id, path, pace)
    return meeting


@router.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str, services: ServicesDep) -> dict[str, Any]:
    detail = services.store.get_meeting(meeting_id)
    if detail is None:
        raise _not_found(f"meeting {meeting_id}")
    return detail


@router.post("/meetings/{meeting_id}/end", dependencies=OPERATOR)
async def end_meeting(
    meeting_id: str, request: Request, services: ServicesDep, access: AccessDep
) -> dict[str, Any]:
    await _check_meeting_change(request, access, services, meeting_id)
    meeting = await services.end_meeting(meeting_id)
    if meeting is None:
        raise _not_found(f"meeting {meeting_id}")
    return meeting


@router.delete("/meetings/{meeting_id}", dependencies=OPERATOR)
async def delete_meeting(
    meeting_id: str, request: Request, services: ServicesDep, access: AccessDep
) -> dict[str, bool]:
    await _check_meeting_change(request, access, services, meeting_id)
    return {"deleted": await services.delete_meeting(meeting_id)}


@router.get("/meetings/{meeting_id}/audio")
def meeting_audio(meeting_id: str, services: ServicesDep) -> FileResponse:
    path = services.recording_path(meeting_id)
    if not MEETING_ID_RE.match(meeting_id) or not path.is_file():
        raise _not_found(f"audio for meeting {meeting_id}")
    return FileResponse(
        path, media_type="audio/wav", filename=f"{meeting_id}.wav", content_disposition_type="inline"
    )


@router.get("/meetings/{meeting_id}/utterances")
def utterances_after(
    meeting_id: str, services: ServicesDep, after_seq: Annotated[int, Query(ge=-1)] = -1
) -> list[dict[str, Any]]:
    utterances = services.store.utterances_since(meeting_id, after_seq)
    if not utterances and services.meeting_summary(meeting_id) is None:
        raise _not_found(f"meeting {meeting_id}")
    return utterances


@router.patch("/meetings/{meeting_id}/speakers/{track}", dependencies=OPERATOR)
def name_speaker(
    meeting_id: str, track: str, body: NameSpeaker, request: Request, services: ServicesDep, access: AccessDep
) -> dict[str, Any]:
    access.check_meeting_change(request, services.meeting_summary(meeting_id))
    try:
        return services.name_speaker(meeting_id, track, body.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---- memory across meetings ----------------------------------------------------------------------------------
@router.get("/action-items")
def list_action_items(
    services: ServicesDep,
    status: Annotated[str | None, Query(max_length=8)] = None,
    owner: Annotated[str | None, Query(max_length=80)] = None,
    meeting_id: Annotated[str | None, Query(max_length=64)] = None,
) -> list[dict[str, Any]]:
    wanted = (status or "").strip().lower() or None
    if wanted == "all":
        wanted = None
    if wanted not in (None, "open", "done"):
        raise HTTPException(status_code=400, detail="status must be 'open', 'done' or 'all'")
    return services.store.list_action_items(
        wanted, (owner or "").strip() or None, (meeting_id or "").strip() or None
    )


@router.patch("/action-items/{item_id}", dependencies=OPERATOR)
def set_action_item(item_id: str, body: SetActionStatus, services: ServicesDep) -> dict[str, Any]:
    item = services.store.set_action_item_status(item_id, body.status)
    if item is None:
        raise _not_found(f"action item {item_id}")
    services.bus.publish("action.updated", item, item["meeting_id"])
    return item


@router.get("/decisions")
def list_decisions(
    services: ServicesDep, meeting_id: Annotated[str | None, Query(max_length=64)] = None
) -> list[dict[str, Any]]:
    return services.store.list_decisions((meeting_id or "").strip() or None)


@router.get("/people")
def list_people(services: ServicesDep) -> list[dict[str, Any]]:
    return services.store.list_people()


@router.get("/who-knows")
def who_knows(
    services: ServicesDep,
    topic: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
) -> list[dict[str, Any]]:
    return services.store.who_knows(topic, limit)


@router.get("/graph")
def subgraph(
    services: ServicesDep,
    meeting_id: Annotated[str | None, Query(max_length=64)] = None,
    utterance_ids: Annotated[str | None, Query(max_length=8000)] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 250,
) -> dict[str, Any]:
    ids = [i.strip() for i in (utterance_ids or "").split(",") if i.strip()]
    return services.store.subgraph((meeting_id or "").strip() or None, ids or None, limit)


@router.get("/search")
def search(
    services: ServicesDep,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    meeting_id: Annotated[str | None, Query(max_length=64)] = None,
    k: Annotated[int, Query(ge=1, le=50)] = 8,
) -> list[dict[str, Any]]:
    hits, _cypher = services.retriever.retrieve(q, (meeting_id or "").strip() or None, k)
    return hits


@router.get("/sessions/{session_id}")
def session_turns(
    session_id: str, services: ServicesDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[dict[str, Any]]:
    return services.store.get_turns(session_id, limit)


# ---- upload helpers --------------------------------------------------------------------------------------------
def _upload_suffix(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in UPLOAD_SUFFIXES:
        return suffix
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type in _CONTENT_SUFFIX:
        return _CONTENT_SUFFIX[content_type]
    shown = suffix or content_type or "unknown"
    raise HTTPException(
        status_code=400, detail=f"unsupported audio file type ({shown}); upload wav, mp3, m4a, webm or ogg"
    )


def _save_upload(source: BinaryIO, path: Path) -> None:
    """Copy the (already spooled) upload to data/uploads, enforcing the size limit."""
    size = 0
    with path.open("wb") as out:
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"file is larger than {MAX_UPLOAD_BYTES // 2**20} MB"
                )
            out.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")


def _probe_audio(path: Path) -> None:
    """Reject files that hold no decodable audio stream before a meeting is created for them."""
    try:
        import av
    except ImportError:
        import soundfile as sf

        try:
            sf.info(str(path))
        except (sf.LibsndfileError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=f"could not read audio from the file: {exc}") from exc
        return
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                raise HTTPException(status_code=400, detail="the file has no audio stream")
    except av.error.FFmpegError as exc:
        raise HTTPException(status_code=400, detail=f"could not read audio from the file: {exc}") from exc


def _title_from_filename(filename: str | None) -> str:
    stem = Path(filename or "").stem
    words = re.sub(r"[_\-.]+", " ", stem).split()
    return " ".join(words)[:200] or "Uploaded meeting"
