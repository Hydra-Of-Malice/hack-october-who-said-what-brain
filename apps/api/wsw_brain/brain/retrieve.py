"""Hybrid retrieval over the graph: full-text + EmbeddingGemma KNN + speaker and recency Cypher templates,
fused with reciprocal-rank fusion."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from ..contracts import Embedder

log = logging.getLogger(__name__)

RRF_K = 60
_RECENT = re.compile(
    r"\b(just now|right now|latest|most recent(?:ly)?|recently|last (?:few|couple of) minutes|last minute|"
    r"so far|a moment ago|minutes? ago|just said|just mentioned|earlier today)\b",
    re.IGNORECASE,
)


def rrf_fuse(ranked_lists: list[list[dict[str, Any]]], k: int, rrf_k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion of Hit lists. Each fused hit keeps the dict and `via` of the list where it
    ranked best; `score` is the fused score divided by the top fused score (so the best hit is 1.0)."""
    fused: dict[str, dict[str, Any]] = {}
    for list_index, hits in enumerate(ranked_lists):
        seen_here: set[str] = set()
        rank = 0
        for hit in hits:
            uid = hit.get("utterance_id") or hit.get("id")
            if not uid or uid in seen_here:
                continue
            seen_here.add(uid)
            rank += 1
            entry = fused.get(uid)
            if entry is None:
                entry = fused[uid] = {
                    "hit": hit,
                    "score": 0.0,
                    "best": (rank, list_index),
                    "first": len(fused),
                }
            entry["score"] += 1.0 / (rrf_k + rank)
            if (rank, list_index) < entry["best"]:
                entry["best"], entry["hit"] = (rank, list_index), hit
    if not fused:
        return []
    ordered = sorted(fused.values(), key=lambda e: (-e["score"], e["best"], e["first"]))[:k]
    top = ordered[0]["score"]
    results = []
    for entry in ordered:
        hit = dict(entry["hit"])
        hit.setdefault("utterance_id", hit.get("id"))
        hit["score"] = round(entry["score"] / top, 4)
        results.append(hit)
    return results


def refers_to_recent(question: str) -> bool:
    return bool(_RECENT.search(question))


class Retriever:
    def __init__(self, store: Any, embedder: Embedder, *, names_ttl_s: float = 30.0) -> None:
        self.store = store
        self.embedder = embedder
        self.names_ttl_s = names_ttl_s
        self._names: dict[str | None, tuple[float, list[str]]] = {}

    def retrieve(
        self, question: str, meeting_id: str | None = None, k: int = 8
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """(Hit dicts, Cypher statements that produced them)."""
        question = question.strip()
        if not question:
            return [], []
        cypher: list[str] = []
        pool = max(k * 2, 10)
        lists = [
            self._call(cypher, self.store.search_fulltext, question, k=pool, meeting_id=meeting_id),
            self._call(
                cypher,
                self.store.search_vector,
                self.embedder.embed_query(question),
                k=pool,
                meeting_id=meeting_id,
            ),
        ]
        for name in self.mentioned_speakers(question, meeting_id):
            lists.append(
                self._call(cypher, self.store.utterances_by_speaker, name, meeting_id=meeting_id, limit=pool)
            )
        if refers_to_recent(question):
            recent = self._call(cypher, self.store.recent_utterances, meeting_id=meeting_id, limit=k)
            lists.append(list(reversed(recent)) if _oldest_first(recent) else recent)
        return rrf_fuse(lists, k), cypher

    def mentioned_speakers(self, question: str, meeting_id: str | None = None) -> list[str]:
        """Known person/speaker names (or labels like "Speaker 2") that the question mentions."""
        lowered = question.lower()
        found: list[str] = []
        for name in self._known_names(meeting_id):
            candidates = [name]
            first = name.split()[0]
            if " " in name and len(first) >= 3 and not first.lower().startswith("speaker"):
                candidates.append(first)
            mentioned = any(re.search(rf"\b{re.escape(c.lower())}\b", lowered) for c in candidates)
            if mentioned and name.lower() not in {f.lower() for f in found}:
                found.append(name)
        return found

    def _known_names(self, meeting_id: str | None) -> list[str]:
        cached = self._names.get(meeting_id)
        now = time.monotonic()
        if cached and now - cached[0] < self.names_ttl_s:
            return cached[1]
        names: list[str] = []
        try:
            names += [p["name"] for p in self.store.list_people() if p.get("name")]
            if meeting_id:
                detail = self.store.get_meeting(meeting_id) or {}
                for s in detail.get("speakers", []):
                    names += [v for v in (s.get("name"), s.get("label")) if v]
        except Exception as exc:  # noqa: BLE001 - names are an optional boost
            log.warning("could not load speaker names for retrieval: %s", exc)
        unique = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
        self._names[meeting_id] = (now, unique)
        return unique

    def _call(
        self, cypher: list[str], fn: Callable[..., list[dict[str, Any]]], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        try:
            hits = fn(*args, **kwargs) or []
        except Exception as exc:  # noqa: BLE001 - one source must not sink the rest
            log.warning("retrieval source %s failed: %s", getattr(fn, "__name__", fn), exc)
            hits = []
        cypher.extend(getattr(self.store, "last_cypher", None) or [])
        return hits


def _oldest_first(hits: list[dict[str, Any]]) -> bool:
    """recent_utterances may come oldest- or newest-first; rank the newest first for fusion."""
    if len(hits) < 2:
        return False
    first, last = hits[0], hits[-1]
    key = "t_start" if first.get("meeting_id") == last.get("meeting_id") else None
    return key is not None and float(first.get(key) or 0) < float(last.get(key) or 0)
