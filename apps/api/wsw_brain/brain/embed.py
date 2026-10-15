"""Text embeddings: EmbeddingGemma-300m (ONNX, CPU) with a deterministic hashing fallback."""

from __future__ import annotations

import hashlib
import itertools
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import settings
from ..contracts import Embedder

log = logging.getLogger(__name__)

QUERY_PROMPT = "task: search result | query: {text}"
DOCUMENT_PROMPT = "title: none | text: {text}"
_SNAPSHOT_PATTERNS = ["onnx/model_quantized.onnx*", "tokenizer*", "*.json"]
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _snapshot_dir(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    def has_model(path: Path) -> bool:
        return (path / "onnx" / "model_quantized.onnx").is_file() and (path / "tokenizer.json").is_file()

    try:
        local = Path(snapshot_download(repo, allow_patterns=_SNAPSHOT_PATTERNS, local_files_only=True))
        if has_model(local):
            return local
        log.info("EmbeddingGemma cache for %s is incomplete; downloading", repo)
    except Exception as exc:  # noqa: BLE001 - LocalEntryNotFoundError and friends: not cached
        log.info("EmbeddingGemma %s not in the HF cache (%s); downloading", repo, exc)
    path = Path(snapshot_download(repo, allow_patterns=_SNAPSHOT_PATTERNS))
    if not has_model(path):
        raise FileNotFoundError(
            f"{repo} snapshot at {path} lacks onnx/model_quantized.onnx or tokenizer.json"
        )
    return path


class GemmaEmbedder:
    """EmbeddingGemma-300m. Queries and documents get the model's asymmetric task prompts."""

    name = "embeddinggemma-300m"

    def __init__(
        self,
        repo: str | None = None,
        dim: int | None = None,
        *,
        threads: int = 4,
        max_tokens: int = 256,
        batch_size: int = 16,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cfg = settings()
        folder = _snapshot_dir(repo or cfg.embed_repo)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(folder / "onnx" / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        missing = {"input_ids", "attention_mask"} - self._inputs
        if missing:
            raise RuntimeError(
                f"EmbeddingGemma ONNX lacks inputs {sorted(missing)}; has {sorted(self._inputs)}"
            )
        outputs = {o.name: o for o in self._session.get_outputs()}
        if "sentence_embedding" in outputs:
            self._output, self._pooled = "sentence_embedding", True
        elif "last_hidden_state" in outputs:
            self._output, self._pooled = "last_hidden_state", False
        else:
            raise RuntimeError(f"EmbeddingGemma ONNX has no usable output; has {sorted(outputs)}")
        native = outputs[self._output].shape[-1]
        native_dim = native if isinstance(native, int) else 768
        self.dim = min(dim or cfg.embed_dim, native_dim)

        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        pad_id = self._tokenizer.token_to_id("<pad>")
        self._tokenizer.enable_truncation(max_length=max_tokens)
        self._tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="<pad>")
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PROMPT.format(text=t) for t in texts]).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PROMPT.format(text=text)])[0].tolist()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks = [self._run(texts[i : i + self._batch_size]) for i in range(0, len(texts), self._batch_size)]
        vectors = np.concatenate(chunks, axis=0)[:, : self.dim]
        return _l2_normalise(vectors).astype(np.float32)

    def _run(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        (out,) = self._session.run([self._output], feeds)
        if self._pooled:
            return np.asarray(out, dtype=np.float32)
        weights = mask[..., None].astype(np.float32)  # mean-pool the token states
        return (out * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)


class HashEmbedder:
    """Deterministic hashed bag of words + bigrams (signed feature hashing), L2-normalised."""

    def __init__(self, dim: int = 768) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        features = [(w, 1.0) for w in words] + [(f"{a} {b}", 0.5) for a, b in itertools.pairwise(words)]
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vec[value % self.dim] += weight if (value >> 63) & 1 else -weight
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0  # empty text: a fixed unit vector keeps cosine indexes well-defined
            return vec.tolist()
        return (vec / norm).tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    cfg = settings()
    if cfg.embedder == "gemma":
        try:
            embedder = GemmaEmbedder()
            log.info("embedder: %s (%d-d)", embedder.name, embedder.dim)
            return embedder
        except Exception as exc:  # noqa: BLE001 - degrade to hashing, but log why
            log.warning("EmbeddingGemma unavailable (%s: %s); using HashEmbedder", type(exc).__name__, exc)
    else:
        log.info("WSW_EMBEDDER=%s; using HashEmbedder", cfg.embedder)
    return HashEmbedder(cfg.embed_dim)
