"""Audio helpers: decoding, PCM16 <-> float32 conversion and the growing meeting recording."""

from __future__ import annotations

import threading
import wave
from collections.abc import Generator
from pathlib import Path
from typing import Any, Self

import numpy as np
import soundfile as sf

from ..contracts import SAMPLE_RATE

_PCM16_SCALE = 32768.0


class AudioTooLong(ValueError):
    """The audio is longer than the caller's `max_duration_s`."""


def decode_file(path: str | Path, *, max_duration_s: float | None = None) -> np.ndarray:
    """Decode any container FFmpeg understands (wav/mp3/m4a/webm/ogg/...) to 16 kHz mono float32.

    Holds the whole decoded file in memory (about 230 MB per hour of audio); `iter_decode` streams it.
    Raises AudioTooLong past `max_duration_s`.
    """
    blocks = list(iter_decode(path, max_duration_s=max_duration_s))
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)


def iter_decode(
    path: str | Path, *, max_duration_s: float | None = None, block_s: float = 10.0
) -> Generator[np.ndarray, None, None]:
    """Decode like `decode_file`, yielding 16 kHz mono float32 blocks of `block_s` seconds (the last one
    shorter), so a long file is never held in memory at once.

    Raises FileNotFoundError, ValueError when the file cannot be decoded, and AudioTooLong when the
    audio runs past `max_duration_s`: checked against the container's declared duration before any audio
    is decoded, and again while decoding, because that header can be missing or wrong.

    Uses PyAV (bundled with faster-whisper) directly: faster_whisper.decode_audio 1.2.1 passes
    `metadata_errors=` to av.open, which PyAV >= 15 rejects. Falls back to soundfile + soxr
    (wav/flac/ogg only) if PyAV is unavailable.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    try:
        import av  # noqa: F401
    except ImportError:
        pieces = _decode_soundfile(p, max_duration_s)
    else:
        pieces = _decode_pyav(p, max_duration_s)
    limit = None if max_duration_s is None else int(max_duration_s * SAMPLE_RATE)
    block = max(1, int(block_s * SAMPLE_RATE))
    buffered: list[np.ndarray] = []
    size = total = 0
    try:
        while True:
            try:
                piece = next(pieces, None)
            except AudioTooLong:
                raise
            except Exception as exc:  # PyAV and libsndfile raise their own error types
                raise ValueError(f"could not decode audio file {p.name}: {exc}") from exc
            if piece is None:
                break
            piece = as_mono_float32(piece)
            total += piece.size
            if limit is not None and total > limit:
                raise AudioTooLong(f"{p.name} is longer than the {_hours(max_duration_s)} limit")
            buffered.append(piece)
            size += piece.size
            while size >= block:
                joined = np.concatenate(buffered) if len(buffered) > 1 else buffered[0]
                yield joined[:block]
                rest = joined[block:]
                buffered, size = ([rest] if rest.size else []), rest.size
        if size:
            yield np.concatenate(buffered) if len(buffered) > 1 else buffered[0]
    finally:
        pieces.close()


def _decode_pyav(path: Path, max_duration_s: float | None) -> Generator[np.ndarray, None, None]:
    import av

    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio stream in the file")
        stream = container.streams.audio[0]
        _check_declared(path, _pyav_duration_s(av, container, stream), max_duration_s)
        # Keep the source channel layout and average channels ourselves: FFmpeg's default stereo->mono
        # matrix adds both channels at -3 dB, which makes correlated audio 1.4x louder (and can clip).
        resampler = av.AudioResampler(format="fltp", rate=SAMPLE_RATE)
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue  # one corrupt packet (common at the end of browser recordings): keep the rest
            for frame in frames:
                for out in resampler.resample(frame):
                    yield _mono(out.to_ndarray())
        for out in resampler.resample(None):
            yield _mono(out.to_ndarray())


def _pyav_duration_s(av: Any, container: Any, stream: Any) -> float | None:
    """The duration the container declares (None when it does not say, e.g. browser webm)."""
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / av.time_base
    return None


def _check_declared(path: Path, duration_s: float | None, max_duration_s: float | None) -> None:
    if max_duration_s is not None and duration_s is not None and duration_s > max_duration_s:
        raise AudioTooLong(f"{path.name} is {_hours(duration_s)} long; the limit is {_hours(max_duration_s)}")


def _hours(seconds: float | None) -> str:
    s = int(seconds or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _mono(planar: np.ndarray) -> np.ndarray:
    """(channels, samples) planar frame -> mono by channel average."""
    return planar.mean(axis=0) if planar.ndim == 2 and planar.shape[0] > 1 else planar.reshape(-1)


def _decode_soundfile(path: Path, max_duration_s: float | None) -> Generator[np.ndarray, None, None]:
    import soxr

    info = sf.info(str(path))
    _check_declared(path, info.duration, max_duration_s)
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = soxr.resample(mono, rate, SAMPLE_RATE)
    yield mono.astype(np.float32, copy=False)


def as_mono_float32(samples: np.ndarray) -> np.ndarray:
    """Validate and normalise an incoming sample array to a contiguous 1-D float32 array."""
    arr = np.asarray(samples)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"expected mono samples (1-D), got shape {arr.shape}")
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / _PCM16_SCALE
    return np.ascontiguousarray(arr, dtype=np.float32)


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM bytes -> float32 in [-1, 1)."""
    if len(data) % 2:
        raise ValueError(f"PCM16 payload has an odd number of bytes ({len(data)})")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / _PCM16_SCALE


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> signed 16-bit little-endian PCM bytes (clipped)."""
    arr = as_mono_float32(samples)
    return np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), as_mono_float32(samples), sample_rate, subtype="PCM_16", format="WAV")
    return p


class WavAppender:
    """Appends float32 chunks to a 16-bit mono WAV that grows while a meeting runs.

    The RIFF header is patched after every append (the stdlib `wave` writer does this whenever the
    declared length differs from what was written), so a reader opening the file mid-meeting sees a
    valid WAV of the audio so far. `close()` finalises it. Calls are serialised by a lock, so a
    reader thread may call `duration_s` while the listener thread appends.
    """

    def __init__(self, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._frames = 0
        self._file = open(self.path, "wb")  # noqa: SIM115 (kept open for appends; closed in close())
        self._writer: wave.Wave_write | None = wave.open(self._file, "wb")  # noqa: SIM115 (same)
        self._writer.setnchannels(1)
        self._writer.setsampwidth(2)
        self._writer.setframerate(sample_rate)
        self._writer.writeframes(b"")  # write the header now so the file is valid from the start
        self._file.flush()

    @property
    def closed(self) -> bool:
        return self._writer is None

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def duration_s(self) -> float:
        return self._frames / self.sample_rate

    def append(self, samples: np.ndarray) -> None:
        data = float_to_pcm16(samples)
        with self._lock:
            if self._writer is None:
                raise RuntimeError(f"recording already closed: {self.path}")
            self._writer.writeframes(data)
            self._file.flush()
            self._frames += len(data) // 2

    def close(self) -> Path:
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.close()  # patches the final header; leaves our file object open
                finally:
                    self._writer = None
                    self._file.close()
        return self.path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
