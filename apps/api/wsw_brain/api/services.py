"""Everything one running API process shares: the graph store, models, the ingest pipeline, live captures
and background jobs. Built once in the app lifespan and reached from routes as `request.app.state.services`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from ..brain import Extractor, Retriever, VoiceAgent, get_audio_llm, get_embedder, get_llm
from ..config import Settings, settings
from ..contracts import LLM, Embedder
from ..events import EventBus, bus
from ..graph import GraphError, GraphStore, GraphUnavailable
from ..listen import get_asr, get_speaker_embedder
from ..listen.asr import WhisperASR
from ..listen.speakers import SherpaSpeakerEmbedder
from ..pipeline import IngestPipeline
from .capture import CaptureRegistry
from .jobs import FileIngest

log = logging.getLogger(__name__)

MEETING_ID_RE = re.compile(r"^m_[0-9a-f]{10}$")
SCHEMA_RETRY_S = 5.0
# Gemma 4 pretty-prints the extraction JSON (with a per-utterance "mentions" map): a 14-16 utterance window
# measured 1.4-3.4k characters, past the Extractor's default 700 tokens, and a cut-off object cannot be parsed.
EXTRACT_MAX_TOKENS = 1800
DISCONNECT_GRACE_S = 30.0  # a dropped audio socket ends the meeting unless the client reconnects in time


def new_meeting_id() -> str:
    return "m_" + secrets.token_hex(5)


def now_ms() -> int:
    return int(time.time() * 1000)


class Services:
    def __init__(
        self,
        cfg: Settings,
        store: GraphStore,
        events: EventBus,
        embedder: Embedder,
        llm: LLM,
        audio_llm: LLM | None,
        loop: asyncio.AbstractEventLoop,
        *,
        disconnect_grace_s: float = DISCONNECT_GRACE_S,
    ) -> None:
        self.settings = cfg
        self.store = store
        self.bus = events
        self.embedder = embedder
        self.llm = llm
        self.audio_llm = audio_llm
        self.loop = loop
        self.retriever = Retriever(store, embedder)
        self.agent = VoiceAgent(store, llm, self.retriever)
        self.pipeline = IngestPipeline(
            store, embedder, Extractor(llm, max_tokens=EXTRACT_MAX_TOKENS), events, cfg
        )
        self.captures = CaptureRegistry(self, disconnect_grace_s=disconnect_grace_s)
        self.files = FileIngest(self)
        self.schema_ready = False
        self.graph_error: str | None = None
        self._schema_lock = threading.Lock()
        self._schema_retry_at = 0.0
        self._models_lock = threading.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    # ---- lifecycle --------------------------------------------------------------------------------------
    @classmethod
    async def start(cls, *, disconnect_grace_s: float = DISCONNECT_GRACE_S) -> Services:
        cfg = settings()
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        store = GraphStore(cfg.falkordb_url, cfg.graph)
        t0 = time.perf_counter()
        embedder = await asyncio.to_thread(get_embedder)
        log.info("embedder %s (%d-d) ready in %.1f s", embedder.name, embedder.dim, time.perf_counter() - t0)
        t0 = time.perf_counter()
        llm = await asyncio.to_thread(get_llm)
        audio_llm = await asyncio.to_thread(get_audio_llm)
        log.info(
            "LLM %s %s; spoken questions heard by %s (routing took %.1f s)",
            llm.provider,
            llm.model,
            "Gemma 4 audio" if audio_llm else "Whisper",
            time.perf_counter() - t0,
        )
        services = cls(cfg, store, bus, embedder, llm, audio_llm, loop, disconnect_grace_s=disconnect_grace_s)
        if await asyncio.to_thread(services.ensure_graph):
            log.info("FalkorDB graph '%s' ready at %s", cfg.graph, store.url)
        services.pipeline.start(loop)
        if cfg.seed_demo:
            services.files.start_demo_seed()
        return services

    async def shutdown(self) -> None:
        self.files.cancel_all()
        await asyncio.to_thread(self.files.join, 15.0)
        await self.captures.shutdown()
        await self.pipeline.aclose()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self.store.close()

    # ---- graph availability ----------------------------------------------------------------------------
    def ensure_graph(self) -> bool:
        """Create the indexes if that has not happened yet (FalkorDB may have been down at startup). After
        a failure, callers within SCHEMA_RETRY_S get the cached answer instead of another slow attempt."""
        with self._schema_lock:
            if self.schema_ready:
                return True
            if time.monotonic() < self._schema_retry_at:
                return False
            try:
                self.store.ensure_schema(self.embedder.dim)
            except GraphError as exc:
                self.graph_error = str(exc)
                self._schema_retry_at = time.monotonic() + SCHEMA_RETRY_S
                log.warning("FalkorDB graph not ready: %s", exc)
                return False
            self.schema_ready, self.graph_error = True, None
            return True

    def require_graph(self) -> None:
        if not self.ensure_graph():
            raise GraphUnavailable(self.graph_error or "FalkorDB is not reachable")

    # ---- models --------------------------------------------------------------------------------------------
    def listener_models(self) -> tuple[WhisperASR, SherpaSpeakerEmbedder]:
        """The shared Whisper ASR and speaker embedder, loaded (and warmed up) on first use."""
        with self._models_lock:
            asr = get_asr()
            if not asr.loaded:
                t0 = time.perf_counter()
                asr.load()
                log.info("ASR %s loaded and warmed up in %.1f s", asr.name, time.perf_counter() - t0)
            t0 = time.perf_counter()
            speaker = get_speaker_embedder()
            if (elapsed := time.perf_counter() - t0) > 0.05:
                log.info("speaker embedder %s (%d-d) loaded in %.1f s", speaker.name, speaker.dim, elapsed)
            return asr, speaker

    # ---- meetings ----------------------------------------------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.settings.data_dir / "uploads"

    def recording_path(self, meeting_id: str) -> Path:
        return self.settings.audio_dir / f"{meeting_id}.wav"

    def meeting_summary(self, meeting_id: str) -> dict[str, Any] | None:
        return next((m for m in self.store.list_meetings() if m["id"] == meeting_id), None)

    def new_meeting(
        self,
        title: str,
        source: str,
        *,
        meeting_id: str | None = None,
        started_at_ms: int | None = None,
    ) -> dict[str, Any]:
        self.require_graph()
        mid = meeting_id or new_meeting_id()
        meeting = self.store.create_meeting(mid, title, source, started_at_ms or now_ms())
        self.pipeline.begin_meeting(mid)
        self.bus.publish("meeting.started", meeting, mid)
        return meeting

    def name_speaker(self, meeting_id: str, track: str, name: str) -> dict[str, Any]:
        """Raises LookupError (no such speaker) or ValueError (blank name)."""
        speaker = self.store.name_speaker(meeting_id, track, name)
        self.pipeline.speaker_named(meeting_id, speaker["track"], speaker["name"])
        self.bus.publish(
            "speaker.named",
            {"track": speaker["track"], "name": speaker["name"], "person_id": speaker["person_id"]},
            meeting_id,
        )
        return speaker

    async def end_meeting(
        self,
        meeting_id: str,
        *,
        duration_s: float | None = None,
        wait_final: bool = False,
        ended_at_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Stop any live capture (flushing the listener), then end the meeting through the pipeline."""
        recorded = await self.captures.release(meeting_id)
        known = [d for d in (duration_s, recorded) if d is not None]
        return await self.pipeline.end_meeting(
            meeting_id, max(known) if known else None, wait_final=wait_final, ended_at_ms=ended_at_ms
        )

    async def delete_meeting(self, meeting_id: str) -> bool:
        self.files.cancel(meeting_id)
        await self.captures.release(meeting_id)
        self.pipeline.forget(meeting_id)
        await asyncio.to_thread(self.pipeline.flush)  # a batch being written right now finishes first
        deleted = await asyncio.to_thread(self.store.delete_meeting, meeting_id)
        if MEETING_ID_RE.match(meeting_id):
            for path in [self.recording_path(meeting_id), *self.uploads_dir.glob(f"{meeting_id}.*")]:
                _unlink(path)
        return deleted

    async def announce_completed_items(self, since_ms: int) -> None:
        """The agent's complete_action_item tool writes to the graph directly; tell every tab."""
        items = await asyncio.to_thread(self.store.list_action_items, "done")
        for item in items:
            if (item.get("done_at") or 0) >= since_ms:
                self.bus.publish("action.updated", item, item["meeting_id"])

    # ---- tasks ----------------------------------------------------------------------------------------------
    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run a coroutine in the background (loop thread only); failures are logged, shutdown cancels it."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("background task failed", exc_info=task.exception())

    def call(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """Run a coroutine on the service loop from a worker thread and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # Windows: still open by a reader
        log.warning("could not delete %s: %s", path, exc)
