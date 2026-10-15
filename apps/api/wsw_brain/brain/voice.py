"""Spoken questions: Gemma 4 E4B hears the clip natively (Ollama); the listener's Whisper ASR is the
fallback."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time

import numpy as np

from ..contracts import SAMPLE_RATE, Transcript
from .llm import get_audio_llm

log = logging.getLogger(__name__)

MAX_QUESTION_S = 30.0  # Gemma 4 audio limit per clip
MIN_QUESTION_S = 0.3
MAX_CHANNELS = 2
MAX_SAMPLE_RATE = 48_000
MIN_SPEECH_S = 0.2  # voiced audio the VAD must find before a clip counts as a question
AUDIO_TIMEOUT_S = 60.0
_SILENCE_PEAK = 1e-3
_NO_SPEECH = "question audio is silent or only noise; no speech detected"
_WORD = re.compile(r"[a-z0-9]+")


def prepare_question_audio(wav_bytes: bytes) -> tuple[np.ndarray, bytes]:
    """Decode and validate a question clip.
    Returns (16 kHz mono float32 samples, 16 kHz mono PCM16 WAV bytes).
    Raises ValueError for empty, undecodable, silent, speechless, too short or too long audio.
    Blocking (decode, resample, VAD): call it off the event loop."""
    import soundfile as sf

    if not wav_bytes:
        raise ValueError("question audio is empty")
    try:
        # Check the header before decoding: a few hundred KB of compressed silence (FLAC) can claim
        # hours of audio and decode to gigabytes of float32.
        info = sf.info(io.BytesIO(wav_bytes))
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise ValueError(f"could not decode question audio (send a 16 kHz mono WAV): {exc}") from exc
    rate, channels = int(info.samplerate), int(info.channels)
    if info.frames <= 0 or rate <= 0 or channels <= 0:
        raise ValueError("question audio has no samples")
    if channels > MAX_CHANNELS or rate > MAX_SAMPLE_RATE:
        raise ValueError(
            f"question audio has {channels} channel(s) at {rate} Hz; send mono or stereo at most "
            f"{MAX_SAMPLE_RATE // 1000} kHz (16 kHz mono WAV is best)"
        )
    limit_frames = int((MAX_QUESTION_S + 0.05) * rate)
    if info.frames > limit_frames:
        raise ValueError(f"question audio is {info.frames / rate:.1f} s; the limit is {MAX_QUESTION_S:.0f} s")
    try:
        # Bounded read: a header that understates the length cannot make the decode unbounded either.
        data, rate = sf.read(io.BytesIO(wav_bytes), frames=limit_frames + 1, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise ValueError(f"could not decode question audio (send a 16 kHz mono WAV): {exc}") from exc
    if data.size == 0 or rate <= 0:
        raise ValueError("question audio has no samples")
    duration = data.shape[0] / rate
    if data.shape[0] > limit_frames:
        raise ValueError(
            f"question audio is over {MAX_QUESTION_S:.0f} s; the limit is {MAX_QUESTION_S:.0f} s"
        )
    if duration < MIN_QUESTION_S:
        raise ValueError(f"question audio is {duration:.2f} s; speak for at least {MIN_QUESTION_S} s")
    samples = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != SAMPLE_RATE:
        import soxr

        samples = soxr.resample(samples, rate, SAMPLE_RATE)
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    if float(np.max(np.abs(samples))) < _SILENCE_PEAK:
        raise ValueError("question audio is silent")
    # Room tone, hum or a key click passes the peak check; given the hint words, Gemma 4 then "hears" a
    # command ("Mark it done.") in it. Only clips with speech in them go on to be transcribed.
    if speech_seconds(samples) < MIN_SPEECH_S:
        raise ValueError(_NO_SPEECH)
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return samples, buffer.getvalue()


def speech_seconds(samples: np.ndarray) -> float:
    """Seconds of speech in a 16 kHz mono clip: the listener's Silero VAD, or an energy-envelope check
    when that model is unavailable."""
    try:
        return _silero_speech_seconds(samples)
    except Exception as exc:  # noqa: BLE001 - missing model / sherpa-onnx: use the envelope check
        log.debug("Silero VAD unavailable for the question clip (%s); using the envelope check", exc)
        return envelope_speech_seconds(samples)


def _silero_speech_seconds(samples: np.ndarray) -> float:
    from ..listen.factory import vad_model_path
    from ..listen.session import VAD_WINDOW, _build_vad

    vad = _build_vad(
        vad_model_path(), min_silence_s=0.3, min_speech_s=0.15, max_speech_s=MAX_QUESTION_S + 1.0
    )
    whole = (len(samples) // VAD_WINDOW) * VAD_WINDOW
    for i in range(0, whole, VAD_WINDOW):
        vad.accept_waveform(samples[i : i + VAD_WINDOW])
    vad.flush()
    voiced = 0
    while not vad.empty():
        voiced += len(vad.front.samples)
        vad.pop()
    return voiced / SAMPLE_RATE


def envelope_speech_seconds(samples: np.ndarray, frame: int = 320, rise_db: float = 12.0) -> float:
    """Fallback detector: speech rises and falls by syllable, so it has 20 ms frames well above the clip's
    own floor; steady noise and hum stay within a few dB of it. Returns the seconds of such frames."""
    frames = len(samples) // frame
    if frames < 5:
        return 0.0
    blocks = np.asarray(samples[: frames * frame], dtype=np.float64).reshape(frames, frame)
    level = 10 * np.log10(np.mean(np.square(blocks), axis=1) + 1e-12)
    floor = float(np.percentile(level, 10))
    loud = (level > floor + rise_db) & (level > -60.0)
    return float(np.count_nonzero(loud)) * frame / SAMPLE_RATE


def echoes_hint(text: str, hint: str | None) -> bool:
    """True when a transcript of two or more words uses only words from the hint. Given noise, Gemma 4
    reads the hint back ("Mark it done action item decision due date follow-up"), so such a transcript
    needs a second opinion before it can act."""
    if not hint:
        return False
    words = _WORD.findall(text.lower())
    return len(words) >= 2 and set(words) <= set(_WORD.findall(hint.lower()))


async def transcribe_question(wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
    """Gemma 4 native audio first (provider "gemma4-audio:ollama"), then Whisper (provider "whisper")."""
    samples, wav16 = await asyncio.to_thread(prepare_question_audio, wav_bytes)
    problems: list[str] = []
    audio_llm = await asyncio.to_thread(get_audio_llm)
    if audio_llm is not None:
        try:
            transcript = await asyncio.wait_for(audio_llm.transcribe(wav16, hint=hint), AUDIO_TIMEOUT_S)
        except TimeoutError:
            problems.append(f"Gemma 4 audio timed out after {AUDIO_TIMEOUT_S:.0f} s")
        except Exception as exc:  # noqa: BLE001 - any audio failure falls through to Whisper
            problems.append(f"Gemma 4 audio failed: {exc}")
        else:
            if transcript.text.strip():
                if echoes_hint(transcript.text, hint):
                    await _confirm_with_whisper(transcript.text, samples)
                return transcript
            problems.append("Gemma 4 audio returned an empty transcript")
        log.warning("question transcription falling back to Whisper: %s", problems[-1])
    else:
        problems.append("no audio-capable Gemma 4 runtime")
    return await _whisper(samples, problems)


async def _confirm_with_whisper(text: str, samples: np.ndarray) -> None:
    """Raise the no-speech ValueError when Whisper, which gets no hint, hears none of the transcript's
    words. Keeps the transcript if Whisper is unavailable."""
    try:
        heard = await _whisper(samples, [])
    except Exception as exc:  # noqa: BLE001 - no second opinion available; keep Gemma's transcript
        log.warning("could not double-check the hint-like transcript %r with Whisper: %s", text, exc)
        return
    if not set(_WORD.findall(heard.text.lower())) & set(_WORD.findall(text.lower())):
        log.warning("dropping transcript %r: Whisper heard %r in the same clip", text, heard.text)
        raise ValueError(_NO_SPEECH)


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
