"""Run an audio file through a LiveSession, as if it were streamed (uploads and the demo seed)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Literal

from ..contracts import ASR, SAMPLE_RATE, ListenerCallbacks, SpeakerEmbedder, Utterance
from .audio import decode_file
from .session import LiveSession

Pace = Literal["fast", "realtime"]
CHUNK_S = 0.1


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
) -> list[Utterance]:
    """Decode `path`, stream it through a LiveSession in 100 ms chunks and return every utterance.

    pace="realtime" sleeps so the audio arrives at wall-clock speed (a live demo from a file);
    "fast" feeds as quickly as the session accepts it. Callbacks fire as utterances are produced.
    The returned list reflects speaker merges that happened later in the stream. Setting `cancel`
    stops feeding; what was fed so far is still processed and returned.
    """
    if pace not in ("fast", "realtime"):
        raise ValueError(f"pace must be 'fast' or 'realtime', got {pace!r}")
    samples = decode_file(path)
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
    chunk = int(CHUNK_S * SAMPLE_RATE)
    started = time.monotonic()
    try:
        for offset in range(0, samples.size, chunk):
            if cancel is not None and cancel.is_set():
                break
            session.feed(samples[offset : offset + chunk])
            if pace == "realtime":
                ahead = (offset + chunk) / SAMPLE_RATE - (time.monotonic() - started)
                if ahead > 0:
                    time.sleep(ahead)
    finally:
        duration_s = samples.size / SAMPLE_RATE
        session.close(timeout=close_timeout if close_timeout is not None else 120.0 + 5.0 * duration_s)
    return collected
