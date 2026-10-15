"""Live ingest: listener utterances -> micro-batched FalkorDB writes -> bus events, plus live Gemma 4 extraction.

Listener worker threads call the ListenerCallbacks from `callbacks()`. Everything they report goes through
one queue, so a speaker merge is applied only after the utterances that came before it are written. One
writer thread drains the queue: it batches utterances (every `flush_interval_s` or `batch_size` items),
embeds them with EmbeddingGemma, writes them in one UNWIND batch and publishes `utterance.created` and
`graph.write`. Every `settings.extract_every` new utterances of a meeting it schedules an extraction on the
asyncio loop; extractions of one meeting never overlap. Ending a meeting flushes, marks it ended and then
runs the final extraction and the summary.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

from .brain.extract import Extractor
from .brain.summary import summarize_meeting
from .config import Settings
from .contracts import ActionItemDraft, DecisionDraft, Embedder, Insights, ListenerCallbacks, Utterance
from .events import EventBus
from .graph import GraphError, GraphStore, GraphUnavailable
from .graph.text import canonical_track, content_tokens, normalise, parse_track

log = logging.getLogger(__name__)

WINDOW = 14  # utterances an extraction sees (new ones plus earlier context)
MAX_NEW_PER_CALL = 16  # a backlog larger than this is extracted in several calls (bounded model output)
STATS_EVERY_S = 3.0
WRITE_ATTEMPTS = 3
WRITE_RETRY_S = 1.0
_STOP = object()


@dataclasses.dataclass
class _MeetingState:
    names: dict[str, str]  # track -> speaker name
    last_seq: int = -1  # highest seq written to the graph
    last_t_end: float = 0.0
    extracted_seq: int = -1  # highest seq an extraction has covered
    scheduled: bool = False
    extract_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    end_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class IngestPipeline:
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        extractor: Extractor,
        bus: EventBus,
        settings: Settings,
        *,
        flush_interval_s: float = 1.0,
        batch_size: int = 8,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.extractor = extractor
        self.bus = bus
        self.settings = settings
        self.extract_every = max(1, int(settings.extract_every))
        self.flush_interval_s = flush_interval_s
        self.batch_size = max(1, batch_size)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._states: dict[str, _MeetingState] = {}
        self._forgotten: set[str] = set()
        self._futures: set[concurrent.futures.Future[Any]] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_stats = 0.0
        self._thread = threading.Thread(target=self._run, name="ingest-writer", daemon=True)

    # ---- lifecycle --------------------------------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the loop extractions run on and start the writer thread."""
        self._loop = loop
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Write what is queued, stop the writer and cancel scheduled extractions (any thread)."""
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("ingest writer still busy after %.0f s; leaving it to finish", timeout)
        for future in list(self._futures):
            future.cancel()  # thread-safe: the loop cancels the task behind it

    async def aclose(self, timeout: float = 15.0) -> None:
        """stop(), then cancel and await the final-extraction tasks (loop thread)."""
        await asyncio.to_thread(self.stop, timeout)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- listener side (any thread) --------------------------------------------------------------------
    def begin_meeting(self, meeting_id: str) -> None:
        """Register a meeting created just now (no utterances, no names yet)."""
        with self._lock:
            self._forgotten.discard(meeting_id)
            self._states[meeting_id] = _MeetingState(names={})

    def callbacks(
        self,
        meeting_id: str,
        *,
        time_offset_s: float = 0.0,
        on_status: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ListenerCallbacks:
        """Listener callbacks for one stream. `time_offset_s` is added to the stream's utterance times
        (a reconnecting microphone continues the meeting's timeline)."""

        def on_utterance(u: Utterance) -> None:
            shifted = dataclasses.replace(
                u, t_start=round(u.t_start + time_offset_s, 3), t_end=round(u.t_end + time_offset_s, 3)
            )
            self.submit(shifted)

        def status(state: str, detail: dict[str, Any]) -> None:
            self.status(meeting_id, state, detail)
            if on_status is not None:
                on_status(state, detail)

        return ListenerCallbacks(
            on_utterance=on_utterance,
            on_speaker_merged=self.speaker_merged,
            on_status=status,
        )

    def submit(self, utterance: Utterance) -> None:
        self._queue.put(("utterance", utterance))

    def speaker_merged(self, meeting_id: str, src: str, dst: str) -> None:
        self._queue.put(("merge", meeting_id, src, dst))

    def status(self, meeting_id: str, state: str, detail: dict[str, Any]) -> None:
        data = {
            "state": state,
            "lag_s": float(detail.get("lag_s") or 0.0),
            "detail": str(detail.get("detail") or ""),
        }
        self.bus.publish("listener.status", data, meeting_id)

    def speaker_named(self, meeting_id: str, track: str, name: str) -> None:
        """Keep the name cache used for utterance.created events in step with PATCH /speakers."""
        state = self._state(meeting_id)
        if state is not None:
            with self._lock:
                state.names[canonical_track(track)] = name

    def next_seq(self, meeting_id: str) -> int:
        """The seq a new listener stream for this meeting should start at (call after flush())."""
        state = self._state(meeting_id)
        return state.last_seq + 1 if state is not None else 0

    def flush(self, timeout: float | None = 60.0) -> bool:
        """Block until everything queued before this call is written. False on timeout."""
        if not self._thread.is_alive():
            return self._queue.empty()
        done = threading.Event()
        self._queue.put(("barrier", done))
        return done.wait(timeout)

    def forget(self, meeting_id: str) -> None:
        """A deleted meeting: drop its queued utterances and state."""
        with self._lock:
            self._forgotten.add(meeting_id)
            self._states.pop(meeting_id, None)

    # ---- ending --------------------------------------------------------------------------------------------
    async def end_meeting(
        self,
        meeting_id: str,
        duration_s: float | None = None,
        *,
        wait_final: bool = False,
        ended_at_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Flush, mark the meeting ended and publish meeting.ended, then run the final extraction and the
        summary (awaited when `wait_final`, otherwise in the background; a second meeting.ended carries the
        summary). Idempotent: an ended meeting is returned as it is. None if the meeting does not exist."""
        await asyncio.to_thread(self.flush)
        state = await asyncio.to_thread(self._state, meeting_id)
        if state is None:
            return None
        async with state.end_lock:
            meeting = await asyncio.to_thread(self._summary, meeting_id)
            if meeting is None:
                return None
            if meeting["status"] == "ended":
                return meeting
            duration = max(float(duration_s or 0.0), float(meeting["duration_s"] or 0.0), state.last_t_end)
            ended_at = ended_at_ms if ended_at_ms is not None else int(time.time() * 1000)
            ended = await asyncio.to_thread(self.store.end_meeting, meeting_id, ended_at, round(duration, 3))
            if ended is None:
                return None
            self.bus.publish("meeting.ended", ended, meeting_id)
        task = asyncio.ensure_future(self._finalize(meeting_id, state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if not wait_final:
            return ended
        return await task or ended

    async def _finalize(self, meeting_id: str, state: _MeetingState) -> dict[str, Any] | None:
        try:
            async with state.extract_lock:
                await self._extract_pending(meeting_id, state)
            detail = await asyncio.to_thread(self.store.get_meeting, meeting_id)
            if not detail:
                return None
            summary = await summarize_meeting(self.extractor.llm, detail)
            if not summary:
                return detail["meeting"]
            await asyncio.to_thread(self.store.set_meeting_summary, meeting_id, summary)
            meeting = await asyncio.to_thread(self._summary, meeting_id)
            if meeting is not None:
                self.bus.publish("meeting.ended", meeting, meeting_id)
            return meeting
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # background task: report; the meeting is already ended
            log.exception("final extraction/summary for %s failed", meeting_id)
            self.status(meeting_id, "error", {"detail": f"final extraction or summary failed: {exc}"})
            return None

    async def extract(self, meeting_id: str) -> None:
        """Extract whatever has not been extracted yet, now (waits for a running extraction first)."""
        state = await asyncio.to_thread(self._state, meeting_id)
        if state is None:
            return
        async with state.extract_lock:
            await self._extract_pending(meeting_id, state)

    # ---- writer thread -------------------------------------------------------------------------------------
    def _run(self) -> None:
        batch: list[Utterance] = []
        deadline = 0.0
        while True:
            timeout = max(0.0, deadline - time.monotonic()) if batch else 0.5
            try:
                op = self._queue.get(timeout=timeout)
            except queue.Empty:
                batch = self._write_batch(batch)
                continue
            if op is _STOP:
                self._write_batch(batch)
                return
            kind = op[0]
            if kind == "utterance":
                if not batch:
                    deadline = time.monotonic() + self.flush_interval_s
                batch.append(op[1])
                if len(batch) >= self.batch_size:
                    batch = self._write_batch(batch)
                continue
            batch = self._write_batch(batch)
            if kind == "merge":
                self._guard("merging speakers", self._merge, *op[1:])
            elif kind == "barrier":
                op[1].set()

    def _write_batch(self, batch: list[Utterance]) -> list[Utterance]:
        """Write and publish a batch; returns the new (empty) batch."""
        groups: dict[str, list[Utterance]] = {}
        for u in batch:
            if u.meeting_id not in self._forgotten:
                groups.setdefault(u.meeting_id, []).append(u)
        for meeting_id, utterances in groups.items():
            self._guard("writing utterances", self._write_meeting, meeting_id, utterances)
        return []

    def _write_meeting(self, meeting_id: str, utterances: list[Utterance]) -> None:
        t0 = time.perf_counter()
        vectors = self._embed([u.text for u in utterances])
        embed_ms = (time.perf_counter() - t0) * 1000
        result = self._write_with_retries(utterances, vectors)
        if result is None:
            return
        state = self._state(meeting_id)
        if state is None:  # deleted while we were writing
            return
        with self._lock:
            for u in utterances:
                state.last_seq = max(state.last_seq, u.seq)
                state.last_t_end = max(state.last_t_end, float(u.t_end))
            names = dict(state.names)
        for u in sorted(utterances, key=lambda x: x.seq):
            self.bus.publish("utterance.created", _utterance_out(u, names), meeting_id)
        data: dict[str, Any] = {
            "utterances": int(result["written"]),
            "ms": float(result["ms"]),
            "embed_ms": round(embed_ms, 1),
        }
        now = time.monotonic()
        if now - self._last_stats >= STATS_EVERY_S:
            self._last_stats = now
            try:
                data["totals"] = self.store.stats()
            except GraphError as exc:
                log.warning("could not read graph totals: %s", exc)
        self.bus.publish("graph.write", data, meeting_id)
        self._maybe_schedule(meeting_id)

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            vectors = self.embedder.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001 - a missing vector must not lose the utterance
            log.warning("embedding %d utterances failed; writing them without vectors: %s", len(texts), exc)
            return None
        return vectors if len(vectors) == len(texts) else None

    def _write_with_retries(
        self, utterances: list[Utterance], vectors: list[list[float]] | None
    ) -> dict[str, Any] | None:
        meeting_id = utterances[0].meeting_id
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                return self.store.write_utterances(utterances, vectors)
            except ValueError as exc:  # embedding shape the index rejects: keep the text
                if vectors is None:
                    raise
                log.warning("writing utterances with vectors failed (%s); retrying without", exc)
                vectors = None
            except GraphUnavailable as exc:
                if attempt == WRITE_ATTEMPTS:
                    detail = f"could not write {len(utterances)} utterances to FalkorDB: {exc}"
                    log.error("%s", detail)
                    self.status(meeting_id, "error", {"detail": detail})
                    return None
                time.sleep(WRITE_RETRY_S * attempt)
        return None

    def _merge(self, meeting_id: str, src: str, dst: str) -> None:
        if meeting_id in self._forgotten:
            return
        moved = self.store.merge_speakers(meeting_id, src, dst)
        state = self._state(meeting_id)
        if state is not None:
            with self._lock:
                src_t, dst_t = canonical_track(src), canonical_track(dst)
                name = state.names.pop(src_t, None)
                if name and dst_t not in state.names:
                    state.names[dst_t] = name
        log.info("meeting %s: speaker %s merged into %s (%d utterances moved)", meeting_id, src, dst, moved)
        self.bus.publish("speaker.merged", {"from": src, "to": dst}, meeting_id)

    def _guard(self, what: str, fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # the writer thread must survive any one failure
            log.exception("ingest: %s failed", what)
            meeting_id = args[0] if args and isinstance(args[0], str) else None
            if meeting_id:
                self.status(meeting_id, "error", {"detail": f"{what} failed: {type(exc).__name__}: {exc}"})

    # ---- extraction ----------------------------------------------------------------------------------------
    def _maybe_schedule(self, meeting_id: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            state = self._states.get(meeting_id)
            if state is None or state.scheduled or state.last_seq - state.extracted_seq < self.extract_every:
                return
            state.scheduled = True
        self._submit_coroutine(self._scheduled_extraction(meeting_id, state), loop)

    def _submit_coroutine(self, coro: Coroutine[Any, Any, Any], loop: asyncio.AbstractEventLoop) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)

    async def _scheduled_extraction(self, meeting_id: str, state: _MeetingState) -> None:
        try:
            async with state.extract_lock:
                with self._lock:
                    state.scheduled = False
                await self._extract_pending(meeting_id, state)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("live extraction for %s failed", meeting_id)
            return
        self._maybe_schedule(meeting_id)  # more arrived while we were extracting

    async def _extract_pending(self, meeting_id: str, state: _MeetingState) -> None:
        """Extract every utterance after state.extracted_seq, in windows of WINDOW (new + context)."""
        detail = await asyncio.to_thread(self.store.get_meeting, meeting_id)
        if not detail:
            return
        utterances: list[dict[str, Any]] = detail["utterances"]
        pending = [u for u in utterances if int(u["seq"]) > state.extracted_seq]
        if not pending:
            return
        decisions = await asyncio.to_thread(self.store.list_decisions)
        existing = list(reversed(decisions))  # oldest first: the extractor keeps the most recent ones
        held = _MeetingRecords.from_detail(detail)
        position = {u["id"]: i for i, u in enumerate(utterances)}
        for start in range(0, len(pending), MAX_NEW_PER_CALL):
            chunk = pending[start : start + MAX_NEW_PER_CALL]
            first = position[chunk[0]["id"]]
            context = utterances[max(0, first - max(0, WINDOW - len(chunk))) : first]
            insights, meta = await self.extractor.extract_with_meta(
                meeting_id, [*context, *chunk], detail["speakers"], existing
            )
            kept, dropped = held.admit(insights, {u["id"] for u in chunk})
            created = await asyncio.to_thread(self.store.write_insights, meeting_id, kept)
            self._publish_insights(meeting_id, created, meta)
            existing.extend(created.get("decisions", []))
            held.add(created)
            state.extracted_seq = max(state.extracted_seq, int(chunk[-1]["seq"]))
            log.info(
                "meeting %s: extracted seq<=%d via %s in %.0f ms: %d decisions, %d action items, %d topics "
                "(%d repeats dropped)",
                meeting_id,
                state.extracted_seq,
                meta.get("provider"),
                float(meta.get("latency_ms") or 0.0),
                len(created.get("decisions", [])),
                len(created.get("action_items", [])),
                len(created.get("topics", [])),
                dropped,
            )

    def _publish_insights(
        self, meeting_id: str, created: dict[str, list[dict[str, Any]]], meta: dict[str, Any]
    ) -> None:
        by = {k: meta.get(k) for k in ("provider", "model", "latency_ms", "fallback")}
        for kind, key in (("decision", "decisions"), ("action_item", "action_items"), ("topic", "topics")):
            for item in created.get(key, []):
                self.bus.publish("insight.created", {"kind": kind, "item": item, "by": by}, meeting_id)

    # ---- state ------------------------------------------------------------------------------------------------
    def _state(self, meeting_id: str) -> _MeetingState | None:
        """State for a meeting; loaded from the graph the first time this process sees it. None if the
        meeting was deleted (or does not exist)."""
        with self._lock:
            if meeting_id in self._forgotten:
                return None
            state = self._states.get(meeting_id)
            if state is not None:
                return state
            detail = self.store.get_meeting(meeting_id)
            if detail is None:
                return None
            utterances = detail["utterances"]
            last_seq = max((int(u["seq"]) for u in utterances), default=-1)
            state = _MeetingState(
                names={s["track"]: s["name"] for s in detail["speakers"] if s.get("name")},
                last_seq=last_seq,
                last_t_end=max((float(u["t_end"]) for u in utterances), default=0.0),
                extracted_seq=last_seq,  # earlier utterances were extracted by the process that wrote them
            )
            self._states[meeting_id] = state
            return state

    def _summary(self, meeting_id: str) -> dict[str, Any] | None:
        return next((m for m in self.store.list_meetings() if m["id"] == meeting_id), None)


@dataclasses.dataclass
class _MeetingRecords:
    """What a meeting already holds, so overlapping windows do not add it again in new words.

    Windows overlap (context lines tell the model which question an answer belongs to), so the model sees
    extracted speech again and restates it ("The annual discount is 15%" after "The annual discount will
    be 15%, reviewed after 90 days"). It also sometimes misses an item in the window where it is new and
    finds it one window later, citing context lines, so context-only drafts cannot simply be dropped.
    A decision or action item is a repeat of a held one (action items: with a compatible owner) when the
    shorter text's content words are contained in the other at least RESTATE, or at least LOOSE_RESTATE
    when it shares an evidence utterance or cites only context. Speaker names and labels are not content.
    Topics, entities and mentions are idempotent graph merges and pass unchanged."""

    decisions: list[dict[str, Any]]  # {"text", "evidence", ...}
    items: list[dict[str, Any]]  # {"text", "evidence", "owner", "owner_track", ...}
    speaker_words: frozenset[str]

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> _MeetingRecords:
        words = {"speaker"}
        for speaker in detail["speakers"]:
            for value in (speaker.get("name"), speaker.get("label"), speaker.get("track")):
                words.update(normalise(value or "").split())
        return cls(
            decisions=[d for d in detail["decisions"] if d.get("status") == "active"],
            items=list(detail["action_items"]),
            speaker_words=frozenset(words),
        )

    def admit(self, insights: Insights, new_ids: set[str]) -> tuple[Insights, int]:
        decisions: list[DecisionDraft] = []
        for draft in insights.decisions:
            held = [*self.decisions, *({"text": d.text, "evidence": d.evidence} for d in decisions)]
            if not self._repeats(draft.text, draft.evidence, new_ids, held):
                decisions.append(draft)
        items: list[ActionItemDraft] = []
        for draft in insights.action_items:
            accepted = (
                {"text": i.text, "evidence": i.evidence, "owner": i.owner, "owner_track": None} for i in items
            )
            held = [item for item in [*self.items, *accepted] if _same_owner(draft.owner, item)]
            if not self._repeats(draft.text, draft.evidence, new_ids, held, owner=draft.owner):
                items.append(draft)
        dropped = len(insights.decisions) - len(decisions) + len(insights.action_items) - len(items)
        return dataclasses.replace(insights, decisions=decisions, action_items=items), dropped

    def add(self, created: dict[str, list[dict[str, Any]]]) -> None:
        self.decisions.extend(created.get("decisions", []))
        self.items.extend(created.get("action_items", []))

    def _repeats(
        self,
        text: str,
        evidence: list[str],
        new_ids: set[str],
        held: list[dict[str, Any]],
        owner: str | None = None,
    ) -> bool:
        words = self._words(text, owner)
        fresh = not evidence or any(e in new_ids for e in evidence)
        for record in held:
            overlap = _containment(words, self._words(record["text"], record.get("owner")))
            shared = bool(set(evidence) & set(record.get("evidence") or []))
            if overlap >= RESTATE or (overlap >= LOOSE_RESTATE and (shared or not fresh)):
                return True
        return False

    def _words(self, text: str, owner: str | None = None) -> set[str]:
        """Content words without speaker names/labels or the owner's name ("Priya has the page by Monday"
        and "Speaker 1 will draft the page by Monday" are about the same page)."""
        return content_tokens(text or "") - self.speaker_words - set(normalise(owner or "").split())


RESTATE = 0.8
LOOSE_RESTATE = 0.5


def _containment(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _same_owner(owner: str | None, item: dict[str, Any]) -> bool:
    """Could a draft's owner (track or name) be the held item's owner (name or label, and/or track)? Only
    two tracks, or two real names, tell people apart; a name against an unnamed track ("Speaker 1") cannot."""
    if not owner or not owner.strip():
        return True
    track = parse_track(owner)
    held_owner = " ".join(str(item.get("owner") or "").lower().split())
    held_track = item.get("owner_track") or parse_track(held_owner)
    if track is not None and held_track is not None:
        return track == held_track
    if track is None and held_owner and not _LABEL.fullmatch(held_owner) and not parse_track(held_owner):
        name = " ".join(owner.lower().split())
        return name == held_owner or name.split()[0] == held_owner.split()[0]
    return True


_LABEL = re.compile(r"speaker \d+")


def _utterance_out(u: Utterance, names: dict[str, str]) -> dict[str, Any]:
    track = canonical_track(u.speaker)
    return {
        "id": u.id,
        "meeting_id": u.meeting_id,
        "seq": u.seq,
        "speaker": track,
        "speaker_name": names.get(track),
        "t_start": float(u.t_start),
        "t_end": float(u.t_end),
        "text": u.text,
        "confidence": float(u.confidence),
    }
