"""faster-whisper speech recognition for VAD segments (implements contracts.ASR)."""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import SAMPLE_RATE
from .audio import as_mono_float32

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class WhisperASR:
    """Lazy-loaded faster-whisper model shared by every live session.

    CTranslate2 models are not safe for concurrent `transcribe` calls on one instance, so calls are
    serialised by a lock. Segments come from the VAD already, so Whisper's own VAD and timestamps are
    off, and each segment is decoded independently (no conditioning on previous text, which is what
    makes Whisper repeat itself on long streams).
    """

    def __init__(
        self,
        model_size_or_path: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
        *,
        cpu_threads: int = 0,
        temperatures: tuple[float, ...] = (0.0, 0.2, 0.4),
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.cpu_threads = cpu_threads
        self.temperatures = temperatures
        size = Path(model_size_or_path).name if _looks_like_path(model_size_or_path) else model_size_or_path
        self.name = f"faster-whisper-{size}"
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model and run one tiny decode so the first real utterance is not slowed by
        CTranslate2's first-call setup (otherwise the first `transcribe` call loads it)."""
        with self._lock:
            model = self._ensure_model()
            segments, _ = model.transcribe(
                np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                language=self.language or "en",
                beam_size=1,
                without_timestamps=True,
            )
            for _ in segments:
                pass

    def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        kwargs = {"device": self.device, "compute_type": self.compute_type, "cpu_threads": self.cpu_threads}
        try:
            self._model = WhisperModel(self.model_size_or_path, local_files_only=True, **kwargs)
        except Exception as cached_exc:  # noqa: BLE001 - any cache miss/corruption: try a download
            log.info(
                "faster-whisper %s not in the local cache (%s); downloading",
                self.model_size_or_path,
                cached_exc,
            )
            try:
                self._model = WhisperModel(self.model_size_or_path, local_files_only=False, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"could not load faster-whisper model {self.model_size_or_path!r} on {self.device}/"
                    f"{self.compute_type}: {exc} (cache lookup failed first with: {cached_exc})"
                ) from exc
        return self._model

    def transcribe(self, samples: np.ndarray) -> tuple[str, float]:
        audio = as_mono_float32(samples)
        if audio.size == 0:
            return "", 0.0
        with self._lock:
            model = self._ensure_model()
            segments, _info = model.transcribe(
                audio,
                language=self.language,
                beam_size=1,
                temperature=list(self.temperatures),
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            parts = [(seg.text.strip(), seg.avg_logprob) for seg in segments]  # the generator decodes lazily
        texts = [text for text, _ in parts if text]
        if not texts:
            return "", 0.0
        confidence = float(np.mean([math.exp(lp) for text, lp in parts if text]))
        return " ".join(texts), min(1.0, max(0.0, confidence))


def _looks_like_path(value: str) -> bool:
    return any(sep in value for sep in ("/", "\\")) or Path(value).exists()
