"""Spoken questions: Gemma 4 E4B hears the clip natively (Ollama); the listener's Whisper ASR is the
fallback."""

from __future__ import annotations

import asyncio
import io
import logging
import time

import numpy as np

from ..contracts import SAMPLE_RATE, Transcript
from .llm import get_audio_llm

log = logging.getLogger(__name__)

MAX_QUESTION_S = 30.0  # Gemma 4 audio limit per clip
MIN_QUESTION_S = 0.3
AUDIO_TIMEOUT_S = 60.0
_SILENCE_PEAK = 1e-3


def prepare_question_audio(wav_bytes: bytes) -> tuple[np.ndarray, bytes]:
    """Decode and validate a question clip.
    Returns (16 kHz mono float32 samples, 16 kHz mono PCM16 WAV bytes).
    Raises ValueError for empty, undecodable, silent, too short or too long audio."""
    import soundfile as sf

    if not wav_bytes:
        raise ValueError("question audio is empty")
    try:
        data, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise ValueError(f"could not decode question audio (send a 16 kHz mono WAV): {exc}") from exc
    if data.size == 0 or rate <= 0:
        raise ValueError("question audio has no samples")
    duration = data.shape[0] / rate
    if duration > MAX_QUESTION_S + 0.05:
        raise ValueError(f"question audio is {duration:.1f} s; the limit is {MAX_QUESTION_S:.0f} s")
    if duration < MIN_QUESTION_S:
        raise ValueError(f"question audio is {duration:.2f} s; speak for at least {MIN_QUESTION_S} s")
    samples = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != SAMPLE_RATE:
        import soxr

        samples = soxr.resample(samples, rate, SAMPLE_RATE)
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    if float(np.max(np.abs(samples))) < _SILENCE_PEAK:
        raise ValueError("question audio is silent")
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return samples, buffer.getvalue()


async def transcribe_question(wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
    """Gemma 4 native audio first (provider "gemma4-audio:ollama"), then Whisper (provider "whisper")."""
    samples, wav16 = prepare_question_audio(wav_bytes)
    problems: list[str] = []
    audio_llm = get_audio_llm()
    if audio_llm is not None:
        try:
            transcript = await asyncio.wait_for(audio_llm.transcribe(wav16, hint=hint), AUDIO_TIMEOUT_S)
            if transcript.text.strip():
                return transcript
            problems.append("Gemma 4 audio returned an empty transcript")
        except TimeoutError:
            problems.append(f"Gemma 4 audio timed out after {AUDIO_TIMEOUT_S:.0f} s")
        except Exception as exc:  # noqa: BLE001 - any audio failure falls through to Whisper
            problems.append(f"Gemma 4 audio failed: {exc}")
        log.warning("question transcription falling back to Whisper: %s", problems[-1])
    else:
        problems.append("no audio-capable Gemma 4 runtime")
    return await _whisper(samples, problems)


async def _whisper(samples: np.ndarray, problems: list[str]) -> Transcript:
    try:
        from ..listen import get_asr  # the listener package owns the Whisper model
    except ImportError as exc:
        raise RuntimeError(
            f"cannot transcribe the question: {'; '.join(problems)}; "
            f"and the listener ASR is unavailable ({exc})"
        ) from exc
    t0 = time.perf_counter()
    asr = get_asr()
    text, _confidence = await asyncio.to_thread(asr.transcribe, samples)
    return Transcript(
        text=" ".join(str(text).split()),
        model=getattr(asr, "name", "whisper"),
        provider="whisper",
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
