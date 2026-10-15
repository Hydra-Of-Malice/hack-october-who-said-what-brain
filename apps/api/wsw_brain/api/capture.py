"""Live microphone capture: one LiveSession per connected audio socket, one recording per meeting.

The recording outlives a socket. A client that stops with {"end": false} (or drops) and reconnects keeps
appending to the same WAV, and the gap is filled with silence so the recording, the utterance times and
the UI's wall-clock playhead all share the meeting's timeline (time 0 = meeting.started_at). The speaker
tracks outlive a socket too (Services.speaker_state_path), so a voice keeps its track (S1, S2, ...) across
a pause, a reconnect or a server restart.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from ..contracts import SAMPLE_RATE
from ..listen import ListenerOverloaded, LiveSession, WavAppender, pcm16_to_float

if TYPE_CHECKING:
    from .services import Services

log = logging.getLogger(__name__)

MAX_GAP_FILL_S = 600.0  # a longer pause is not padded with silence (the timeline then skips the gap)
CLOSE_TIMEOUT_S = 120.0


class CaptureError(Exception):
    """A socket cannot capture audio for this meeting. `code` is the WebSocket close code."""

    def __init__(self, detail: str, code: int = 1011) -> None:
        super().__init__(detail)
        self.code = code


class CaptureClosed(CaptureError):
    """The capture was released (the meeting ended or was deleted) while audio was still arriving."""

    def __init__(self, meeting_id: str) -> None:
        super().__init__(f"meeting {meeting_id} has ended; no more audio is accepted", 1000)


class Capture:
    def __init__(self, meeting_id: str, recorder: WavAppender) -> None:
        self.meeting_id = meeting_id
        self.recorder = recorder
        self.session: LiveSession | None = None
        self.attached = False
        self.released = False
        self.end_timer: asyncio.TimerHandle | None = None
        self.attaching: asyncio.Future[None] | None = None  # done once the latest attach() has finished
        self._close_task: asyncio.Future[None] | None = None

    @property
    def lag_s(self) -> float:
        return self.session.lag_s if self.session is not None else 0.0

    def feed(self, data: bytes) -> None:
        """One binary frame of 16 kHz mono s16le PCM. Raises ValueError for an odd byte count."""
        session = self.session
        if self.released or session is None or session.closed:
            raise CaptureClosed(self.meeting_id)
        samples = pcm16_to_float(data)
        if samples.size == 0:
            return
        try:
            session.feed(samples)
        except ListenerOverloaded as exc:  # the listener is too far behind real time to keep up
            raise CaptureError(f"{exc}; the listener is overloaded, reconnect in a moment", 1013) from exc
        self.recorder.append(samples)

    def start_session(self, session: LiveSession) -> None:
        self.session = session
        self._close_task = None

    async def close_session(self) -> None:
        """Flush and stop the current listener session (idempotent; concurrent callers share the work)."""
        session = self.session
        if session is None:
            return
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(asyncio.to_thread(_close_listener, session))
        await asyncio.shield(self._close_task)

    async def wait_session_stopped(self) -> None:
        """Wait until the current session's worker has stopped. close() gives up after CLOSE_TIMEOUT_S but the
        worker keeps emitting utterances; a new session must not start before that (it would reuse their
        seqs, and miss the speaker tracks the old one saves when it finishes)."""
        session = self.session
        if session is not None and session.running:
            log.info("meeting %s: waiting for the previous listener to finish its backlog", self.meeting_id)
            await asyncio.to_thread(session.close, None)

    def cancel_end_timer(self) -> None:
        if self.end_timer is not None:
            self.end_timer.cancel()
            self.end_timer = None


class CaptureRegistry:
    def __init__(self, services: Services, *, disconnect_grace_s: float) -> None:
        self.services = services
        self.disconnect_grace_s = disconnect_grace_s
        self._captures: dict[str, Capture] = {}
        self._releasing: dict[str, asyncio.Future[float]] = {}  # meeting id -> the release in progress
        self._ended: set[str] = set()  # released for good (ended or deleted): no socket may attach again
        self._lock = asyncio.Lock()

    def is_capturing(self, meeting_id: str) -> bool:
        capture = self._captures.get(meeting_id)
        return capture is not None and capture.attached

    async def attach(self, meeting_id: str) -> Capture:
        """Start a listener session for a socket. Raises CaptureError when the meeting cannot take audio."""
        services = self.services
        connected_at = time.time()
        await asyncio.to_thread(services.require_graph)
        meeting = await asyncio.to_thread(services.meeting_summary, meeting_id)
        if meeting is None:
            raise CaptureError(f"meeting {meeting_id} not found", 4404)
        if meeting["status"] != "live":
            raise CaptureError(f"meeting {meeting_id} has ended; start a new meeting to record again", 4409)
        if meeting["source"] != "mic":
            raise CaptureError(
                f"meeting {meeting_id} is a {meeting['source']} meeting and takes no live audio", 4409
            )
        async with self._lock:
            if meeting_id in self._ended:  # ended or deleted since the status was read
                raise CaptureError(
                    f"meeting {meeting_id} has ended; start a new meeting to record again", 4409
                )
            capture = self._captures.get(meeting_id)
            if capture is not None and capture.attached:
                raise CaptureError("this meeting is already being captured from another tab or device", 4409)
            if capture is None:
                recorder = await asyncio.to_thread(_open_recorder, services.recording_path(meeting_id))
                capture = Capture(meeting_id, recorder)
                self._captures[meeting_id] = capture
            capture.attached = True
            capture.cancel_end_timer()
            attaching = capture.attaching = asyncio.get_running_loop().create_future()
        try:
            await capture.close_session()  # the previous socket's session may still be flushing
            await capture.wait_session_stopped()
            if capture.released:
                raise CaptureClosed(meeting_id)
            elapsed = connected_at - meeting["started_at"] / 1000.0
            await asyncio.to_thread(_fill_gap, capture.recorder, elapsed)
            offset = capture.recorder.duration_s
            asr, speaker_embedder = await asyncio.to_thread(services.listener_models)
            await asyncio.to_thread(services.pipeline.flush)
            start_seq = await asyncio.to_thread(services.pipeline.next_seq, meeting_id)
            cfg = services.settings
            session = await asyncio.to_thread(
                functools.partial(
                    LiveSession,
                    meeting_id,
                    asr,
                    speaker_embedder,
                    services.pipeline.callbacks(meeting_id, time_offset_s=offset),
                    threshold=cfg.speaker_threshold,
                    max_utterance_s=cfg.max_utterance_s,
                    start_seq=start_seq,
                    # the speaker tracks continue from the earlier sessions (and a server restart)
                    speaker_state_path=services.speaker_state_path(meeting_id),
                )
            )
            if capture.released:  # ended or deleted while the listener was starting: nobody else closes it
                await asyncio.to_thread(_close_listener, session)
                raise CaptureClosed(meeting_id)
            capture.start_session(session)
            await asyncio.to_thread(services.store.set_meeting_has_audio, meeting_id, True)
        except CaptureError:
            capture.attached = False
            raise
        except Exception as exc:
            capture.attached = False
            raise CaptureError(f"could not start the listener: {type(exc).__name__}: {exc}") from exc
        finally:
            attaching.set_result(None)
        log.info("meeting %s: capture started at %.1f s, seq %d", meeting_id, offset, start_seq)
        return capture

    def detach(self, capture: Capture, *, end: bool | None) -> None:
        """The socket is gone. end=True ends the meeting, False keeps it live, None (no stop frame: the
        client dropped) ends it unless a socket reattaches within the grace period. Loop thread only."""
        capture.attached = False
        if capture.released:
            return
        services = self.services
        services.spawn(capture.close_session())
        if end:
            services.spawn(services.end_meeting(capture.meeting_id))
        elif end is None:
            loop = asyncio.get_running_loop()
            capture.cancel_end_timer()
            capture.end_timer = loop.call_later(self.disconnect_grace_s, self._end_if_abandoned, capture)

    def _end_if_abandoned(self, capture: Capture) -> None:
        capture.end_timer = None
        if not capture.attached and not capture.released:
            log.info("meeting %s: audio socket did not come back; ending the meeting", capture.meeting_id)
            self.services.spawn(self.services.end_meeting(capture.meeting_id))

    async def release(self, meeting_id: str) -> float | None:
        """Stop capturing for good (meeting ended or deleted): flush the session and close the recording.
        Concurrent callers (an end and a delete, two tabs, the disconnect timer) all wait for the same flush.
        Returns the recording's duration in seconds, or None if nothing was captured in this process."""
        async with self._lock:
            self._ended.add(meeting_id)
            capture = self._captures.pop(meeting_id, None)
            if capture is not None:
                capture.released = True
                capture.cancel_end_timer()
                self._releasing[meeting_id] = asyncio.ensure_future(self._finish_release(capture))
            releasing = self._releasing.get(meeting_id)
        if releasing is None:
            return None
        return await asyncio.shield(releasing)

    async def _finish_release(self, capture: Capture) -> float:
        try:
            if capture.attaching is not None:
                await capture.attaching  # a socket attaching right now sees `released` and stops its listener
            await capture.close_session()
            await asyncio.to_thread(capture.recorder.close)
            return capture.recorder.duration_s
        finally:
            self._releasing.pop(capture.meeting_id, None)

    async def shutdown(self) -> None:
        """Server stopping: flush every session and close recordings; meetings stay live."""
        for meeting_id in [*self._captures, *self._releasing]:
            try:
                await self.release(meeting_id)
            except Exception:
                log.exception("could not close the capture of meeting %s", meeting_id)


def _open_recorder(path: Path) -> WavAppender:
    """A recorder for the meeting's WAV; audio recorded before a server restart is kept."""
    previous: np.ndarray | None = None
    if path.is_file() and path.stat().st_size > 44:
        try:
            previous, _ = sf.read(str(path), dtype="float32")
        except (sf.LibsndfileError, RuntimeError) as exc:
            log.warning("could not read the earlier recording %s (%s); starting a new one", path, exc)
    recorder = WavAppender(path)
    if previous is not None and previous.size:
        recorder.append(previous.reshape(-1))
    return recorder


def _fill_gap(recorder: WavAppender, elapsed_s: float) -> None:
    gap = min(elapsed_s - recorder.duration_s, MAX_GAP_FILL_S)
    if gap > 0.05:
        recorder.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))


def _close_listener(session: LiveSession) -> None:
    try:
        session.close(timeout=CLOSE_TIMEOUT_S)
    except TimeoutError as exc:
        log.warning("%s; it keeps flushing in the background", exc)
