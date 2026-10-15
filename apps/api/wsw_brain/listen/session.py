"""LiveSession: a live audio stream -> speaker-attributed, timestamped utterances ("who said what").

feed() only enqueues. One worker thread per session takes whatever audio is queued and:

1. runs Silero VAD over it (finished speech segments come out);
2. splits a segment where the voice changes inside it (back-to-back speakers, no pause for the VAD);
3. embeds each part and assigns a speaker track with the online clusterer. A short part that matches
   nobody well is held until the next attributed part arrives, then joins whichever neighbour (the
   speaker before or after it) it sounds closer to; it is released after `hold_s` of stream time;
4. transcribes and emits. Consecutive parts of one speaker that are already waiting together (the
   worker fell behind, or a file is fed faster than real time) share one ASR call: Whisper's cost is
   per call (a fixed 30 s encoder window), not per second of speech.

All callbacks run on the worker thread. Errors are reported through on_status("error", ...) and the
worker keeps going.
"""

from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from ..contracts import ASR, SAMPLE_RATE, ListenerCallbacks, SpeakerEmbedder, Utterance
from ._sherpa import import_sherpa
from .audio import WavAppender, as_mono_float32, pcm16_to_float
from .factory import vad_model_path as default_vad_model_path
from .speakers import OnlineClusterer, speaker_change_points

log = logging.getLogger(__name__)

VAD_WINDOW = 512  # samples per Silero window at 16 kHz
STATUS_EVERY_S = 2.0
JOIN_SILENCE_S = 0.3  # silence placed between parts that share one ASR call
_CLOSE = object()

# Phrases Whisper is known to produce on breath, noise or near-silence. They are only dropped when the
# segment is quiet or the ASR is unsure, so a clearly spoken "Thank you." survives.
_HALLUCINATIONS = frozenset(
    {
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thank you for watching",
        "you",
        "bye",
        "please subscribe",
        "the end",
        "subtitles by the amara org community",
        "thanks",
    }
)
_NON_WORD = re.compile(r"[^\w\s]+")
_SPACES = re.compile(r"\s+")


@dataclass
class _Part:
    """A speech segment (or a piece of one) on its way to becoming an Utterance."""

    start: int  # sample index from stream start
    samples: np.ndarray
    rms: float
    embedding: np.ndarray | None = None
    speaker: str | None = None  # None while held for its neighbours
    confidence: float = 0.0
    prev: str | None = None  # speaker before this part (for held parts)

    @property
    def end(self) -> int:
        return self.start + self.samples.size

    @property
    def duration(self) -> float:
        return self.samples.size / SAMPLE_RATE


class LiveSession:
    def __init__(
        self,
        meeting_id: str,
        asr: ASR,
        embedder: SpeakerEmbedder,
        callbacks: ListenerCallbacks,
        *,
        threshold: float,
        max_utterance_s: float,
        start_seq: int = 0,
        recording_path: str | Path | None = None,
        vad_model_path: str | Path | None = None,
        max_speakers: int = 8,
        min_silence_s: float = 0.45,
        min_speech_s: float = 0.25,
        min_embed_s: float = 0.3,
        short_s: float = 0.8,
        hold_s: float = 2.0,
        split_turns: bool = True,
        join_gap_s: float = 1.5,
        max_batch_s: float = 15.0,
        min_rms: float = 0.003,
    ) -> None:
        if max_utterance_s <= min_speech_s:
            raise ValueError(f"max_utterance_s must exceed {min_speech_s} s, got {max_utterance_s}")
        self.meeting_id = meeting_id
        self.asr = asr
        self.embedder = embedder
        self.callbacks = callbacks
        self.clusterer = OnlineClusterer(threshold, max_speakers=max_speakers, short_s=short_s)
        self.min_embed_s = min_embed_s
        self.short_s = short_s
        self.split_turns = split_turns
        self.change_threshold = max(0.1, threshold - 0.1)  # stricter than clustering: only clear changes
        self.min_rms = min_rms
        self._hold = int(hold_s * SAMPLE_RATE)
        self._join_gap = int(join_gap_s * SAMPLE_RATE)
        self._max_utterance = int(max_utterance_s * SAMPLE_RATE)
        self._max_batch = int(max_batch_s * SAMPLE_RATE)

        self._vad = _build_vad(
            Path(vad_model_path) if vad_model_path else default_vad_model_path(),
            min_silence_s=min_silence_s,
            min_speech_s=min_speech_s,
            max_speech_s=max_utterance_s,
        )
        self._recording = WavAppender(recording_path) if recording_path else None

        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._idle = threading.Event()  # set when the worker has handled everything queued so far
        self._idle.set()
        self._closed = False
        self._seq = start_seq
        self._pending = np.zeros(0, dtype=np.float32)  # tail shorter than one VAD window
        self._backlog: list[_Part] = []  # attributed or held parts not yet transcribed, in time order
        self._received = 0  # samples accepted by feed()
        self._vad_fed = 0  # samples given to the VAD
        self._done = 0  # samples fully handled: before the VAD tail and before any part still in the backlog
        self._last_status = 0.0
        self._last_error: tuple[str, float] = ("", 0.0)
        self._timing = {"vad_s": 0.0, "embed_s": 0.0, "asr_s": 0.0}
        self._counts = {
            "utterances": 0,
            "segments": 0,
            "splits": 0,
            "held": 0,
            "merges": 0,
            "skipped_quiet": 0,
            "skipped_empty": 0,
            "skipped_unattributed": 0,
            "errors": 0,
        }

        self._thread = threading.Thread(target=self._run, name=f"listen-{meeting_id}", daemon=True)
        self._thread.start()

    # ---- producer side (any thread) --------------------------------------------------
    def feed(self, samples: np.ndarray) -> None:
        """Queue 16 kHz mono float32 samples; returns immediately. Any length is fine (large arrays
        are queued as 1 s pieces so the worker keeps streaming results)."""
        chunk = np.nan_to_num(
            np.array(as_mono_float32(samples), dtype=np.float32, copy=True), nan=0.0, posinf=1.0, neginf=-1.0
        )
        if chunk.size == 0:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError(f"listener for meeting {self.meeting_id} is closed")
            self._received += chunk.size
            self._idle.clear()
            for i in range(0, chunk.size, SAMPLE_RATE):
                self._queue.put(chunk[i : i + SAMPLE_RATE])

    def feed_pcm16(self, data: bytes) -> None:
        self.feed(pcm16_to_float(data))

    def close(self, timeout: float | None = 60.0) -> None:
        """Flush the VAD, process what is left, stop the worker and finalise the recording.

        Raises TimeoutError if the worker is still busy after `timeout` seconds (it keeps going and
        finishes in the background)."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_CLOSE)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"listener for meeting {self.meeting_id} still processing after {timeout} s "
                f"(backlog {self.lag_s:.1f} s of audio)"
            )

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the worker has handled all audio fed so far (held parts may still be pending).
        Returns False on timeout."""
        return self._idle.wait(timeout)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- introspection --------------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def next_seq(self) -> int:
        return self._seq

    @property
    def lag_s(self) -> float:
        """Seconds of received audio not yet turned into utterances (or discarded)."""
        return max(0, self._received - self._done) / SAMPLE_RATE

    @property
    def recording_path(self) -> Path | None:
        return self._recording.path if self._recording else None

    def speaker_centroids(self) -> dict[str, np.ndarray]:
        return self.clusterer.centroids()

    def metrics(self) -> dict[str, Any]:
        done_s = self._done / SAMPLE_RATE
        compute_s = sum(self._timing.values())
        return {
            "audio_s": round(self._received / SAMPLE_RATE, 3),
            "processed_s": round(done_s, 3),
            "lag_s": round(self.lag_s, 3),
            "speakers": self.clusterer.tracks,
            **self._counts,
            **{k: round(v, 3) for k, v in self._timing.items()},
            "rtf": round(compute_s / done_s, 4) if done_s else None,
        }

    # ---- worker loop ------------------------------------------------------------------
    def _run(self) -> None:
        self._status("listening", force=True)
        closing = False
        while not closing:
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._status("listening")
                continue
            chunks, closing = self._take_batch(first)
            if chunks:
                self._guarded("processing audio", self._process, chunks)
            if closing:
                self._guarded("flushing the stream", self._finish)
            else:
                self._status("processing" if self.lag_s > 1.0 else "listening")
            with self._lock:
                if self._queue.empty():
                    self._idle.set()
        self._status("idle", force=True)

    def _take_batch(self, first: Any) -> tuple[list[np.ndarray], bool]:
        chunks: list[np.ndarray] = []
        total, item = 0, first
        while item is not _CLOSE:
            chunks.append(item)
            total += item.size
            if total >= self._max_batch:
                return chunks, False
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return chunks, False
        return chunks, True

    def _process(self, chunks: list[np.ndarray]) -> None:
        for chunk in chunks:
            if self._recording is not None:
                self._guarded("recording audio", self._recording.append, chunk)
            self._feed_vad(chunk)
        self._take_segments()
        self._release_held(final=False)
        self._emit(final=False)

    def _finish(self) -> None:
        try:
            if self._pending.size:
                padded = np.zeros(VAD_WINDOW, dtype=np.float32)
                padded[: self._pending.size] = self._pending
                self._vad_fed += self._pending.size
                self._pending = np.zeros(0, dtype=np.float32)
                self._vad.accept_waveform(padded)
            self._vad.flush()
            self._take_segments()
            self._release_held(final=True)
            self._emit(final=True)
        finally:
            self._backlog.clear()
            self._done = self._received
            if self._recording is not None:
                self._recording.close()

    def _guarded(self, what: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Run fn(*args); on failure report it and return None (the worker keeps going)."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - the worker must survive anything a part or callback throws
            self._report_error(f"{what}: {type(exc).__name__}: {exc}", exc)
            return None

    # ---- VAD ------------------------------------------------------------------------------
    def _feed_vad(self, chunk: np.ndarray) -> None:
        samples = np.concatenate((self._pending, chunk)) if self._pending.size else chunk
        whole = (samples.size // VAD_WINDOW) * VAD_WINDOW
        self._pending = samples[whole:].copy()
        t0 = time.perf_counter()
        for i in range(0, whole, VAD_WINDOW):
            self._vad.accept_waveform(samples[i : i + VAD_WINDOW])
        self._timing["vad_s"] += time.perf_counter() - t0
        self._vad_fed += whole
        self._update_done()

    def _take_segments(self) -> None:
        while not self._vad.empty():
            segment = self._vad.front
            start, samples = int(segment.start), np.asarray(segment.samples, dtype=np.float32)
            self._vad.pop()
            self._guarded("attributing a speech segment", self._admit, start, samples)
        self._update_done()

    # ---- speaker attribution ------------------------------------------------------------
    def _admit(self, start: int, samples: np.ndarray) -> None:
        self._counts["segments"] += 1
        if _rms(samples) < self.min_rms:
            self._counts["skipped_quiet"] += 1
            return
        for offset, piece in self._split(samples):
            part = _Part(start=start + offset, samples=piece, rms=_rms(piece))
            self._backlog.append(part)
            self._attribute(part)

    def _split(self, samples: np.ndarray) -> list[tuple[int, np.ndarray]]:
        if not self.split_turns:
            return [(0, samples)]
        t0 = time.perf_counter()
        try:
            cuts = speaker_change_points(samples, self.embedder, self.change_threshold)
        finally:
            self._timing["embed_s"] += time.perf_counter() - t0
        self._counts["splits"] += len(cuts)
        bounds = [0, *cuts, samples.size]
        return [(a, samples[a:b]) for a, b in itertools.pairwise(bounds) if b > a]

    def _attribute(self, part: _Part) -> None:
        part.prev = self.clusterer.previous
        part.embedding = self._embed(part.samples) if part.duration >= self.min_embed_s else None
        if part.embedding is None:
            self._counts["held"] += 1
            return
        if part.duration < self.short_s:
            best, sim = self.clusterer.best_match(part.embedding)
            if best is not None and sim < self.clusterer.threshold:
                self._counts["held"] += 1  # too short to trust on its own: let the neighbours decide
                return
        track, confidence = self.clusterer.assign(part.embedding, part.duration)
        self._apply_merges()
        part.speaker, part.confidence = self.clusterer.canonical(track), confidence
        self._resolve_held(next_track=part.speaker)

    def _embed(self, samples: np.ndarray) -> np.ndarray | None:
        t0 = time.perf_counter()
        try:
            return self.embedder.embed(samples)
        except ValueError as exc:  # too short / degenerate audio for the speaker model
            log.debug("no speaker embedding for %.2f s of audio: %s", samples.size / SAMPLE_RATE, exc)
            return None
        finally:
            self._timing["embed_s"] += time.perf_counter() - t0

    def _resolve_held(self, next_track: str | None) -> None:
        for part in self._backlog:
            if part.speaker is None:
                self._decide(part, [c for c in (part.prev, next_track) if c])

    def _release_held(self, *, final: bool) -> None:
        """Held parts fall back to the speaker before them (or their own best match) when nobody
        spoke within hold_s after them, or at the end of the stream. While the VAD is inside speech
        the next segment is still being collected and will resolve them, so they keep waiting."""
        if not final and self._vad.is_speech_detected():
            return
        for part in list(self._backlog):
            if part.speaker is not None or not (final or part.end + self._hold <= self._vad_fed):
                continue
            candidates = [part.prev] if part.prev else []
            if part.embedding is not None:
                best, _ = self.clusterer.best_match(part.embedding)
                candidates += [best] if best else []
            if candidates:
                self._decide(part, candidates)
            elif part.embedding is not None:  # nobody has spoken yet: this is the first speaker
                track, confidence = self.clusterer.assign(part.embedding, part.duration)
                part.speaker, part.confidence = self.clusterer.canonical(track), confidence
            else:
                self._counts["skipped_unattributed"] += 1
                self._backlog.remove(part)

    def _decide(self, part: _Part, candidates: list[str]) -> None:
        part.speaker, part.confidence = self.clusterer.choose(part.embedding, candidates)

    def _apply_merges(self) -> None:
        for src, dst in self.clusterer.merge_close():
            self._counts["merges"] += 1
            log.info("meeting %s: speaker %s merged into %s", self.meeting_id, src, dst)
            self._guarded(
                "reporting a speaker merge", self.callbacks.on_speaker_merged, self.meeting_id, src, dst
            )

    # ---- transcription and output ---------------------------------------------------------
    def _emit(self, *, final: bool) -> None:
        backlog = self._backlog
        i = 0
        while i < len(backlog) and backlog[i].speaker is not None:
            j = i + 1
            while j < len(backlog) and self._joinable(backlog[i], backlog[j - 1], backlog[j]):
                j += 1
            may_grow = backlog[j - 1].end - backlog[i].start < self._max_utterance
            if j == len(backlog) and not final and may_grow and not self._queue.empty():
                break  # more audio is already queued: this speaker's turn may continue
            utterance = self._guarded("transcribing", self._transcribe, backlog[i:j])
            if utterance is not None:
                self._guarded("delivering an utterance", self.callbacks.on_utterance, utterance)
            i = j
            self._done = min(self._vad_fed, backlog[i].start) if i < len(backlog) else self._vad_fed
            self._status("processing" if self.lag_s > 1.0 else "listening")
        del backlog[:i]
        self._update_done()

    def _joinable(self, first: _Part, last: _Part, nxt: _Part) -> bool:
        return (
            nxt.speaker is not None
            and self.clusterer.canonical(nxt.speaker) == self.clusterer.canonical(first.speaker or "")
            and nxt.start - last.end <= self._join_gap
            and nxt.end - first.start <= self._max_utterance
        )

    def _transcribe(self, group: list[_Part]) -> Utterance | None:
        pieces: list[np.ndarray] = []
        for k, part in enumerate(group):
            if k:
                gap = min(max(0, part.start - group[k - 1].end), int(JOIN_SILENCE_S * SAMPLE_RATE))
                pieces.append(np.zeros(gap, dtype=np.float32))
            pieces.append(part.samples)
        audio = np.concatenate(pieces)
        t0 = time.perf_counter()
        text, asr_confidence = self.asr.transcribe(audio)
        self._timing["asr_s"] += time.perf_counter() - t0
        text = _SPACES.sub(" ", text).strip()
        loudest = max(p.rms for p in group)
        if _is_garbage(text, rms=loudest, asr_confidence=asr_confidence, quiet_rms=4 * self.min_rms):
            self._counts["skipped_empty"] += 1
            return None
        weight = sum(p.duration for p in group)
        confidence = sum(p.confidence * p.duration for p in group) / weight if weight else 0.0
        t_start = group[0].start / SAMPLE_RATE
        utterance = Utterance(
            meeting_id=self.meeting_id,
            seq=self._seq,
            speaker=self.clusterer.canonical(group[0].speaker or ""),
            t_start=round(t_start, 3),
            t_end=round(group[-1].end / SAMPLE_RATE, 3),
            text=text,
            confidence=round(float(confidence), 3),
        )
        self._seq += 1
        self._counts["utterances"] += 1
        return utterance

    # ---- bookkeeping ----------------------------------------------------------------------
    def _update_done(self) -> None:
        self._done = min(self._vad_fed, self._backlog[0].start) if self._backlog else self._vad_fed

    def _status(self, state: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status < STATUS_EVERY_S:
            return
        self._last_status = now
        try:
            self.callbacks.on_status(state, {"lag_s": round(self.lag_s, 2)})
        except Exception:
            log.exception("listener on_status callback failed")

    def _report_error(self, detail: str, exc: BaseException) -> None:
        self._counts["errors"] += 1
        log.error("listener %s: %s", self.meeting_id, detail, exc_info=exc)
        now = time.monotonic()
        last_detail, last_at = self._last_error
        if detail == last_detail and now - last_at < STATUS_EVERY_S:
            return  # the same failure on every chunk: report it at most every 2 s
        self._last_error = (detail, now)
        try:
            self.callbacks.on_status("error", {"lag_s": round(self.lag_s, 2), "detail": detail})
        except Exception:
            log.exception("listener on_status callback failed while reporting an error")


def _build_vad(model: Path, *, min_silence_s: float, min_speech_s: float, max_speech_s: float) -> Any:
    if not model.is_file():
        raise FileNotFoundError(f"Silero VAD model not found: {model} (run scripts/fetch_models.py)")
    sherpa_onnx = import_sherpa()
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = min_silence_s
    config.silero_vad.min_speech_duration = min_speech_s
    config.silero_vad.max_speech_duration = max_speech_s
    config.silero_vad.window_size = VAD_WINDOW
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    if not config.validate():
        raise ValueError(f"invalid Silero VAD configuration for {model}")
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max(60.0, 4 * max_speech_s))


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def _is_garbage(text: str, *, rms: float, asr_confidence: float, quiet_rms: float) -> bool:
    words = _SPACES.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()
    if not words:
        return True  # empty, or punctuation / symbols only
    return words in _HALLUCINATIONS and (rms < quiet_rms or asr_confidence < 0.45)
