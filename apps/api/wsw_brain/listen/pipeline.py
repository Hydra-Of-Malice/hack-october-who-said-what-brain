"""Run an audio file through a LiveSession, as if it were streamed (uploads and the demo seed)."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np

from ..contracts import ASR, SAMPLE_RATE, ListenerCallbacks, SpeakerEmbedder, Utterance
from .audio import iter_decode
from .session import LiveSession

Pace = Literal["fast", "realtime"]
CHUNK_S = 0.1
MAX_FILE_S = 2 * 3600.0  # longer files are refused: decoding, ASR and the recording all grow with it
FEED_AHEAD_S = 30.0  # a file is fed at most this far ahead of the listener (bounded memory, prompt cancel)
MAX_CONCURRENT_FILES = 2  # files processed at once; more wait for a free slot
POLL_S = 0.2
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_FILES)


def process_file(
    path: str | Path,
    meeting_id: str,
    asr: ASR,
    embedder: SpeakerEmbedder,
    callbacks: ListenerCallbacks,
    *,
    pace: Pace = "fast",
    threshold: float,
    max_utterance_s: float,
    recording_path: str | Path | None = None,
    start_seq: int = 0,
    cancel: threading.Event | None = None,
    close_timeout: float | None = None,
    max_duration_s: float | None = MAX_FILE_S,
) -> list[Utterance]:
    """Decode `path` block by block, stream it through a LiveSession in 100 ms chunks and return every
    utterance.

    pace="realtime" sleeps so the audio arrives at wall-clock speed (a live demo from a file);
    "fast" feeds as quickly as the session takes it, never more than FEED_AHEAD_S ahead of it.
    Callbacks fire as utterances are produced. The returned list reflects speaker merges that happened
    later in the stream. At most MAX_CONCURRENT_FILES files run at once; a call waits for a free slot.
    Setting `cancel` (while waiting, feeding or finishing) abandons the file: audio not yet transcribed
    is dropped and what was produced so far is returned. Raises AudioTooLong (a ValueError) for audio
    longer than `max_duration_s`, before anything is recorded when the file declares its duration.
    """
    if pace not in ("fast", "realtime"):
        raise ValueError(f"pace must be 'fast' or 'realtime', got {pace!r}")

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    while not _slots.acquire(timeout=POLL_S):
        if cancelled():
            return []
    try:
        if cancelled():
            return []
        # a missing, undecodable or too long file fails here, before the session records anything
        blocks = iter_decode(path, max_duration_s=max_duration_s)
        first = next(blocks, None)
        collected: list[Utterance] = []

        def on_utterance(u: Utterance) -> None:
            collected.append(u)
            callbacks.on_utterance(u)

        def on_speaker_merged(meeting: str, src: str, dst: str) -> None:
            for u in collected:
                if u.speaker == src:
                    u.speaker = dst
            callbacks.on_speaker_merged(meeting, src, dst)

        session = LiveSession(
            meeting_id,
            asr,
            embedder,
            ListenerCallbacks(
                on_utterance=on_utterance, on_speaker_merged=on_speaker_merged, on_status=callbacks.on_status
            ),
            threshold=threshold,
            max_utterance_s=max_utterance_s,
            start_seq=start_seq,
            recording_path=recording_path,
        )
        fed = 0
        started = time.monotonic()
        try:
            for chunk in _chunks(first, blocks, int(CHUNK_S * SAMPLE_RATE)):
                while not cancelled() and not session.wait_queued(FEED_AHEAD_S, POLL_S):
                    pass
                if cancelled():
                    break
                session.feed(chunk)
                fed += chunk.size
                if pace == "realtime":
                    ahead = fed / SAMPLE_RATE - (time.monotonic() - started)
                    if ahead > 0:
                        time.sleep(ahead)
        finally:
            blocks.close()
            timeout = close_timeout if close_timeout is not None else 120.0 + 5.0 * fed / SAMPLE_RATE
            session.close(timeout=timeout, cancel=cancel)
        return collected
    finally:
        _slots.release()


def _chunks(first: np.ndarray | None, rest: Iterator[np.ndarray], size: int) -> Iterator[np.ndarray]:
    for block in itertools.chain([] if first is None else [first], rest):
        for offset in range(0, block.size, size):
            yield block[offset : offset + size]
