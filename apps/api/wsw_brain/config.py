"""Runtime settings, all overridable with environment variables (or a .env file at the repo root)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .brand import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    # Storage
    falkordb_url: str = Field("falkor://127.0.0.1:6379", alias="WSW_FALKORDB_URL")
    graph: str = Field("wsw_demo", alias="WSW_GRAPH")
    data_dir: Path = Field(REPO_ROOT / "data", alias="WSW_DATA_DIR")
    models_dir: Path = Field(REPO_ROOT / "models", alias="WSW_MODELS_DIR")

    # Gemma 4 / LLM routing. "auto": gemini if a key is set, else ollama if reachable, else fake.
    llm_provider: Literal["auto", "ollama", "gemini", "openai", "fake"] = Field("auto", alias="WSW_LLM_PROVIDER")
    ollama_url: str = Field("http://127.0.0.1:11434", alias="WSW_OLLAMA_URL")
    ollama_model: str = Field("gemma4:e4b", alias="WSW_OLLAMA_MODEL")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemma-4-31b-it", alias="WSW_GEMINI_MODEL")
    openai_base_url: str | None = Field(None, alias="WSW_OPENAI_BASE_URL")
    openai_api_key: str | None = Field(None, alias="WSW_OPENAI_API_KEY")
    openai_model: str = Field("gemma-4-31B-it", alias="WSW_OPENAI_MODEL")
    # Which runtime hears spoken questions natively. "ollama" = Gemma 4 E4B audio input; "whisper" = server ASR.
    question_audio: Literal["auto", "ollama", "whisper"] = Field("auto", alias="WSW_QUESTION_AUDIO")

    # Embeddings: EmbeddingGemma-300m ONNX, or a deterministic hashing fallback.
    embedder: Literal["gemma", "hash"] = Field("gemma", alias="WSW_EMBEDDER")
    embed_repo: str = Field("onnx-community/embeddinggemma-300m-ONNX", alias="WSW_EMBED_REPO")
    embed_dim: int = Field(768, alias="WSW_EMBED_DIM")

    # Listener
    asr_model: str = Field("small", alias="WSW_ASR_MODEL")
    asr_device: str = Field("cpu", alias="WSW_ASR_DEVICE")
    asr_compute_type: str = Field("int8", alias="WSW_ASR_COMPUTE")
    asr_language: str | None = Field("en", alias="WSW_ASR_LANGUAGE")
    speaker_model: str = Field("speaker_embedding_en.onnx", alias="WSW_SPEAKER_MODEL")
    speaker_threshold: float = Field(0.55, alias="WSW_SPEAKER_THRESHOLD")
    max_utterance_s: float = Field(8.0, alias="WSW_MAX_UTTERANCE_S")

    # Brain
    extract_every: int = Field(6, alias="WSW_EXTRACT_EVERY")
    seed_demo: bool = Field(True, alias="WSW_SEED_DEMO")
    cors_origins: str = Field("*", alias="WSW_CORS_ORIGINS")

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


@lru_cache(maxsize=1)
def settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.audio_dir.mkdir(parents=True, exist_ok=True)
    return s
