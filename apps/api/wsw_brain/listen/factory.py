"""Process-wide listener models built from settings (loaded once, shared by every session)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from ..config import settings
from .asr import WhisperASR
from .speakers import SherpaSpeakerEmbedder

_lock = threading.Lock()
_asr: WhisperASR | None = None
_embedder: SherpaSpeakerEmbedder | None = None


def get_asr() -> WhisperASR:
    """The shared faster-whisper ASR. The model itself loads lazily (call .load() to warm it up)."""
    global _asr
    with _lock:
        if _asr is None:
            s = settings()
            # 8 threads measured ~10% faster than CTranslate2's default on the i9-13900HX; more did not help.
            threads = min(8, os.cpu_count() or 4)
            _asr = WhisperASR(
                s.asr_model, s.asr_device, s.asr_compute_type, s.asr_language, cpu_threads=threads
            )
        return _asr


def speaker_model_path() -> Path:
    s = settings()
    return s.models_dir / s.speaker_model  # an absolute WSW_SPEAKER_MODEL wins over models_dir


def vad_model_path() -> Path:
    return settings().models_dir / "silero_vad.onnx"


def get_speaker_embedder() -> SherpaSpeakerEmbedder:
    """The shared speaker embedder (raises FileNotFoundError if the model file is missing)."""
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = SherpaSpeakerEmbedder(speaker_model_path())
        return _embedder


def reset_listener_models() -> None:
    """Drop the cached models (tests, or after changing settings)."""
    global _asr, _embedder
    with _lock:
        _asr = None
        _embedder = None
