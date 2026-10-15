"""Background ingestion of audio files through the real listener: uploads and the demo seed."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import soundfile as sf

from ..contracts import Utterance
from ..listen import process_file

if TYPE_CHECKING:
    from .services import Services

log = logging.getLogger(__name__)

Pace = Literal["fast", "realtime"]
END_TIMEOUT_S = 600.0
_DEMO_KEY = "__demo_seed__"


class FileIngest:
    def __init__(self, services: Services) -> None:
        self.services = services
        self._jobs: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._deleted: set[str] = set()
        self._lock = threading.Lock()

    # ---- job control --------------------------------------------------------------------------------------
    def start_upload(self, meeting_id: str, path: Path, pace: Pace) -> None:
        self._start(meeting_id, f"upload-{meeting_id}", self._run_upload, meeting_id, path, pace)

    def start_demo_seed(self) -> None:
        from .demo import seed_demo  # demo assets are optional; keep the import local

        self._start(_DEMO_KEY, "demo-seed", seed_demo, self.services)

    def _start(self, key: str, name: str, target: Any, *args: Any) -> None:
        cancel = threading.Event()
        thread = threading.Thread(
            target=self._guarded, args=(key, target, *args, cancel), name=name, daemon=True
        )
        with self._lock:
            self._jobs[key] = (thread, cancel)
        thread.start()

    def _guarded(self, key: str, target: Any, *args: Any) -> None:
        try:
            target(*args)
        except Exception:
            log.exception("background job %s failed", key)
        finally:
            with self._lock:
                self._jobs.pop(key, None)

    def cancel(self, meeting_id: str) -> None:
        """The meeting is being deleted: stop feeding its file and remove what the job leaves behind."""
        with self._lock:
            job = self._jobs.get(meeting_id)
            if job is not None:
                self._deleted.add(meeting_id)
        if job is not None:
            job[1].set()

    def cancel_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for _, cancel in jobs:
            cancel.set()

    def join(self, timeout: float) -> None:
        with self._lock:
            threads = [thread for thread, _ in self._jobs.values()]
        for thread in threads:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("background job %s still running at shutdown", thread.name)

    def busy(self, meeting_id: str) -> bool:
        with self._lock:
            return meeting_id in self._jobs

    # ---- work (worker threads) --------------------------------------------------------------------------
    def run_file(
        self,
        meeting_id: str,
        path: Path,
        *,
        pace: Pace,
        cancel: threading.Event,
        on_utterance: Callable[[Utterance], None] | None = None,
    ) -> tuple[list[Utterance], float | None]:
        """Stream a file through a LiveSession into the pipeline, recording it as the meeting's WAV.
        `on_utterance` also sees each utterance (listener thread), after the pipeline has queued it.
        Returns (utterances with speaker merges applied, recorded duration in seconds)."""
        services = self.services
        cfg = services.settings
        asr, speaker_embedder = services.listener_models()
        recording = services.recording_path(meeting_id)
        callbacks = services.pipeline.callbacks(meeting_id)
        if on_utterance is not None:
            callbacks = dataclasses.replace(
                callbacks, on_utterance=_both(callbacks.on_utterance, on_utterance)
            )
        services.store.set_meeting_has_audio(meeting_id, True)
        try:
            utterances = process_file(
                path,
                meeting_id,
                asr,
                speaker_embedder,
                callbacks,
                pace=pace,
                threshold=cfg.speaker_threshold,
                max_utterance_s=cfg.max_utterance_s,
                recording_path=recording,
                cancel=cancel,
            )
        except Exception:
            if not recording.is_file():
                services.store.set_meeting_has_audio(meeting_id, False)
            raise
        return utterances, _wav_duration(recording)

    def end_meeting(
        self,
        meeting_id: str,
        duration_s: float | None,
        *,
        wait_final: bool = False,
        ended_at_ms: int | None = None,
    ) -> None:
        """End a meeting from a worker thread (waits for the pipeline)."""
        services = self.services
        ending = services.end_meeting(
            meeting_id, duration_s=duration_s, wait_final=wait_final, ended_at_ms=ended_at_ms
        )
        try:
            services.call(ending, END_TIMEOUT_S)
        except (concurrent.futures.CancelledError, TimeoutError) as exc:
            log.warning("ending meeting %s did not finish: %r", meeting_id, exc)

    def _run_upload(self, meeting_id: str, path: Path, pace: Pace, cancel: threading.Event) -> None:
        duration: float | None = None
        try:
            utterances, duration = self.run_file(meeting_id, path, pace=pace, cancel=cancel)
            log.info("upload %s: %d utterances from %s", meeting_id, len(utterances), path.name)
        except Exception as exc:
            log.exception("upload %s could not be processed", meeting_id)
            self.services.pipeline.status(
                meeting_id, "error", {"detail": f"could not process {path.name}: {type(exc).__name__}: {exc}"}
            )
        if cancel.is_set():  # deleted, or the server is stopping
            if meeting_id in self._deleted:  # the listener held the files open while the meeting was deleted
                for leftover in (self.services.recording_path(meeting_id), path):
                    _unlink(leftover)
            return
        self.end_meeting(meeting_id, duration)


def _both(
    first: Callable[[Utterance], None], second: Callable[[Utterance], None]
) -> Callable[[Utterance], None]:
    def call(u: Utterance) -> None:
        first(u)
        second(u)

    return call


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not delete %s: %s", path, exc)


def _wav_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
    except (sf.LibsndfileError, RuntimeError, FileNotFoundError):
        return None
    return info.frames / info.samplerate if info.samplerate else None
