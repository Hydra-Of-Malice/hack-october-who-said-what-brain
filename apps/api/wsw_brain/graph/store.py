"""GraphStore: the only module that talks to FalkorDB.

Rules (see docs/TECH_NOTES.md §2):
- Writes are batched with UNWIND $rows (one query per batch, never one per utterance).
- Put a WITH barrier between consecutive MATCH clauses; after UNWIND use inline property maps
  (MATCH (n:Label {id: r.id})) rather than WHERE n.id = r.id. This keeps queries correct on the
  C engine (4.x) and the Rust engine (6.0.x, issues #3343/#2972).
- Vector `score` from db.idx.vector.queryNodes is a DISTANCE (lower = closer). Convert to a
  similarity (1 - distance) before returning Hit dicts, whose `score` is "higher = better".
- Every read method that feeds an answer also records the Cypher it ran in `self.last_cypher`
  (list[str], reset at the start of each public read call) so the UI can show "Why this answer".

Return shapes are documented in wsw_brain/contracts.py.
"""

from __future__ import annotations

from typing import Any

from ..contracts import ActionItemDraft, DecisionDraft, Insights, Utterance


class GraphStore:
    def __init__(self, url: str, graph: str) -> None:
        """url like falkor://host:port (or redis://, rediss://). graph = graph (key) name."""
        raise NotImplementedError

    # ---- lifecycle --------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """{"ok": bool, "version": str|None, "graph": str} — never raises."""
        raise NotImplementedError

    def ensure_schema(self, embed_dim: int) -> None:
        """Idempotently create range, full-text and vector indexes (see docs/STRATEGY.md §7)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Delete the whole graph (tests/demo reseed only)."""
        raise NotImplementedError

    # ---- meetings -----------------------------------------------------------
    def create_meeting(self, meeting_id: str, title: str, source: str, started_at_ms: int) -> dict[str, Any]: ...
    def end_meeting(self, meeting_id: str, ended_at_ms: int, duration_s: float) -> dict[str, Any] | None: ...
    def set_meeting_summary(self, meeting_id: str, summary: str) -> None: ...
    def set_meeting_has_audio(self, meeting_id: str, has_audio: bool) -> None: ...
    def list_meetings(self) -> list[dict[str, Any]]:
        """MeetingSummary dicts, newest first."""
        ...
    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        """{"meeting": MeetingSummary, "speakers": [Speaker], "utterances": [UtteranceOut] (by seq),
        "decisions": [Decision], "action_items": [ActionItem], "topics": [Topic]}"""
        ...
    def delete_meeting(self, meeting_id: str) -> bool: ...

    # ---- live writes ------------------------------------------------------------
    def write_utterances(self, utterances: list[Utterance], embeddings: list[list[float]] | None) -> dict[str, Any]:
        """One UNWIND batch: MERGE Utterance + Speaker (+HAS_SPEAKER, SPOKEN_BY, HAS_UTTERANCE) and link
        NEXT from the previous seq. Returns {"written": n, "ms": float}."""
        ...
    def merge_speakers(self, meeting_id: str, src_track: str, dst_track: str) -> int:
        """Re-point src's utterances to dst and delete src. Returns utterances moved."""
        ...
    def name_speaker(self, meeting_id: str, track: str, name: str) -> dict[str, Any]:
        """Set Speaker.name and MERGE (s)-[:IS]->(p:Person {id: slug(name)}). Returns the Speaker dict
        with person_id. Also updates ActionItem.owner where owner_track == track."""
        ...
    def write_insights(self, meeting_id: str, insights: Insights) -> dict[str, list[dict[str, Any]]]:
        """Write decisions, action items, topics, entities, MENTIONS edges. Dedupe by normalised text per
        meeting. A DecisionDraft with supersedes_text marks the best-matching earlier active decision (any
        meeting) as superseded and links (new)-[:SUPERSEDES]->(old). Returns the NEW items only:
        {"decisions": [Decision], "action_items": [ActionItem], "topics": [Topic]}."""
        ...

    # ---- reads used by the UI ---------------------------------------------------
    def utterances_since(self, meeting_id: str, after_seq: int) -> list[dict[str, Any]]: ...
    def list_action_items(self, status: str | None = None, owner: str | None = None,
                          meeting_id: str | None = None) -> list[dict[str, Any]]: ...
    def set_action_item_status(self, item_id: str, status: str) -> dict[str, Any] | None: ...
    def list_decisions(self, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Decisions with supersession links, newest first."""
        ...
    def list_people(self) -> list[dict[str, Any]]:
        """[{id, name, meetings: int, utterances: int, talk_time_s, open_items: int}]"""
        ...
    def who_knows(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        """People (or unnamed speakers) ranked by how much they said about `topic` (full-text match).
        [{name, person_id|None, meeting_id, hits: int, sample: UtteranceOut}]"""
        ...
    def subgraph(self, meeting_id: str | None = None, utterance_ids: list[str] | None = None,
                 limit: int = 250) -> dict[str, Any]:
        """Subgraph dict for the graph inspector. With utterance_ids: those utterances plus their speaker,
        meeting, decisions/action items they evidence and topics they mention."""
        ...
    def stats(self) -> dict[str, int]:
        """Counts per label and total relationships: {"Meeting": n, ..., "relationships": n}."""
        ...

    # ---- retrieval (used by the agent) -----------------------------------------
    def search_fulltext(self, query: str, k: int = 8, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Hit dicts (via="fulltext"). Sanitise the query for RediSearch (strip punctuation, drop
        stopwords, OR the terms, prefix-match long terms)."""
        ...
    def search_vector(self, vector: list[float], k: int = 8, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Hit dicts (via="vector"), score = 1 - distance."""
        ...
    def get_utterances(self, ids: list[str]) -> list[dict[str, Any]]:
        """Hit-shaped dicts for these ids (score=1.0, via="graph"), in the given order."""
        ...
    def context_window(self, utterance_id: str, before: int = 2, after: int = 2) -> list[dict[str, Any]]:
        """Neighbouring utterances via NEXT, as UtteranceOut dicts in seq order."""
        ...
    def recent_utterances(self, meeting_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Latest utterances (of a meeting, or of the latest meeting), Hit-shaped, via="recent"."""
        ...
    def utterances_by_speaker(self, name_or_track: str, meeting_id: str | None = None,
                              limit: int = 20) -> list[dict[str, Any]]:
        """Hit-shaped utterances whose speaker name/track/person matches (case-insensitive)."""
        ...

    # ---- conversation memory ----------------------------------------------------
    def save_turn(self, session_id: str, role: str, text: str, at_ms: int,
                  citations: list[str] | None = None) -> None: ...
    def get_turns(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        """[{role, text, at, citations}] oldest first."""
        ...

    last_cypher: list[str]


__all__ = ["GraphStore", "ActionItemDraft", "DecisionDraft"]
