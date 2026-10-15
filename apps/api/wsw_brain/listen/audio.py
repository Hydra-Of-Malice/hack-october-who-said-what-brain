"""Audio helpers: decoding, PCM16 <-> float32 conversion and the growing meeting recording."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Self

import numpy as np
import soundfile as sf

from ..contracts import SAMPLE_RATE

_PCM16_SCALE = 32768.0


def decode_file(path: str | Path) -> np.ndarray:
    """Decode any container FFmpeg understands (wav/mp3/m4a/webm/ogg/...) to 16 kHz mono float32.

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
        decoder = _decode_soundfile
    else:
        decoder = _decode_pyav
    try:
        samples = decoder(p)
    except Exception as exc:  # PyAV and libsndfile raise their own error types
        raise ValueError(f"could not decode audio file {p.name}: {exc}") from exc
    return as_mono_float32(samples)


def _decode_pyav(path: Path) -> np.ndarray:
    import av

    pieces: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio stream in the file")
        stream = container.streams.audio[0]
        # Keep the source channel layout and average channels ourselves: FFmpeg's default stereo->mono
        # matrix adds both channels at -3 dB, which makes correlated audio 1.4x louder (and can clip).
        resampler = av.AudioResampler(format="fltp", rate=SAMPLE_RATE)
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue  # one corrupt packet (common at the end of browser recordings): keep the rest
            for frame in frames:
                pieces.extend(_mono(out.to_ndarray()) for out in resampler.resample(frame))
        pieces.extend(_mono(out.to_ndarray()) for out in resampler.resample(None))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _mono(planar: np.ndarray) -> np.ndarray:
    """(channels, samples) planar frame -> mono by channel average."""
    return planar.mean(axis=0) if planar.ndim == 2 and planar.shape[0] > 1 else planar.reshape(-1)


def _decode_soundfile(path: Path) -> np.ndarray:
    import soxr

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = soxr.resample(mono, rate, SAMPLE_RATE)
    return mono.astype(np.float32, copy=False)


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
