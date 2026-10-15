"""GraphStore: the only module that talks to FalkorDB.

Rules (see docs/TECH_NOTES.md §2):
- Writes are batched with UNWIND $rows (one query per batch, never one per utterance).
- Put a WITH barrier between consecutive MATCH clauses; after UNWIND use inline property maps
  (MATCH (n:Label {id: r.id})) rather than WHERE n.id = r.id. This keeps queries correct on the
  C engine (4.x) and the Rust engine (6.0.x, issues #3343/#2972).
- Vector `score` from db.idx.vector.queryNodes is a DISTANCE (lower = closer). Convert to a
  similarity (1 - distance) before returning Hit dicts, whose `score` is "higher = better".
- Every read method that feeds an answer also records the Cypher it ran in `self.last_cypher`
  (list[str], replaced by each such call) so the UI can show "Why this answer".

Return shapes are documented in wsw_brain/contracts.py. All Cypher lives in graph/cypher.py.

Thread safety: the redis connection pool is shared; each thread gets its own falkordb Graph handle (the
client caches schema per handle). Multi-step writes (insights, speaker merge/naming, deletes) hold an
in-process lock so their read-then-write sequences do not interleave.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import re
import socket
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Concatenate, NamedTuple

import redis
from falkordb import FalkorDB
from falkordb.graph import Graph
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

from ..contracts import ActionItemDraft, DecisionDraft, Insights, Utterance
from . import cypher as cq
from .text import (
    canonical_track,
    color_idx,
    content_tokens,
    fulltext_query,
    jaccard,
    normalise,
    parse_track,
    person_id,
    query_terms,
    short_hash,
    slug,
    speaker_id,
    speaker_label,
    truncate,
)

log = logging.getLogger(__name__)

SOURCES = ("mic", "upload", "demo")
ITEM_STATUSES = ("open", "done")
ENTITY_KINDS = ("product", "customer", "vendor", "org", "other")
SUPERSEDE_MIN_JACCARD = 0.3
FULLTEXT_SCAN_LIMIT = 2000  # utterances examined by the CONTAINS fallback
WHO_KNOWS_POOL = 300  # full-text hits grouped by who_knows
_SLUG_KINDS = ("Topic", "Entity", "Person")  # ids are bare slugs; prefixed in subgraph output to stay unique
_KIND_ORDER = {k: i for i, k in enumerate(
    ("Meeting", "Speaker", "Person", "Decision", "ActionItem", "Topic", "Entity", "Utterance"))}

class GraphError(RuntimeError):
    """FalkorDB could not run a statement."""


class GraphUnavailable(GraphError):
    """FalkorDB is unreachable (connection refused, reset or timed out)."""


class GraphQueryError(GraphError):
    """FalkorDB rejected a statement (syntax, missing index, constraint...)."""


class _SpeakerRef(NamedTuple):
    id: str
    track: str
    name: str | None
    label: str


def _traced[**P, R](method: Callable[Concatenate[GraphStore, P], R]) -> Callable[Concatenate[GraphStore, P], R]:
    """Public read: collect every statement it runs, then publish them as `last_cypher`. Nested traced
    calls add to the outer call's trace."""

    @functools.wraps(method)
    def wrapper(self: GraphStore, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if getattr(self._local, "trace", None) is not None:
            return method(self, *args, **kwargs)
        trace: list[str] = []
        self._local.trace = trace
        try:
            return method(self, *args, **kwargs)
        finally:
            self._local.trace = None
            self.last_cypher = trace

    return wrapper


class GraphStore:
    last_cypher: list[str]

    def __init__(self, url: str, graph: str) -> None:
        """url like falkor://host:port (or redis://, rediss://). graph = graph (key) name."""
        if not url or not isinstance(url, str):
            raise ValueError("FalkorDB url must be a non-empty string such as falkor://localhost:6379")
        if not graph or not isinstance(graph, str):
            raise ValueError("graph name must be a non-empty string")
        self.url = url
        self.graph_name = graph
        self._db: FalkorDB | None = None  # created on first use: the client constructor talks to the server
        self._db_lock = threading.Lock()
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._embed_dim: int | None = None
        self._schema_pending = False  # set by reset(): indexes are recreated before the next write
        self.last_cypher = []

    # ---- plumbing -----------------------------------------------------------------------------------
    def _client(self) -> FalkorDB:
        if self._db is None:
            with self._db_lock:
                if self._db is None:
                    try:
                        self._db = FalkorDB.from_url(
                            _prefer_ipv4_loopback(self.url),
                            socket_connect_timeout=3,
                            socket_timeout=60,
                            health_check_interval=30,
                            retry=Retry(ExponentialBackoff(cap=0.5, base=0.05), 2),
                        )
                    except redis.exceptions.RedisError as exc:
                        raise GraphUnavailable(f"FalkorDB at {_redact(self.url)} is unreachable: {exc}") from exc
        return self._db

    def _graph(self) -> Graph:
        graph = getattr(self._local, "graph", None)
        if graph is None:
            graph = self._client().select_graph(self.graph_name)
            self._local.graph = graph
        return graph

    def _query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        text = cypher.strip()
        trace = getattr(self._local, "trace", None)
        if trace is not None:
            trace.append(text)
        result = self._execute(text, params)
        names = [column[1] for column in result.header]
        return [dict(zip(names, row)) for row in result.result_set]

    def _execute(self, text: str, params: dict[str, Any] | None) -> Any:
        for attempt in (1, 2):
            try:
                return self._graph().query(text, params)
            except redis.exceptions.ResponseError as exc:
                raise GraphQueryError(f"FalkorDB rejected the query ({exc}): {_first_line(text)}") from exc
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, AttributeError) as exc:
                # AttributeError: redis-py's parser on a socket that was closed under it. Statements are
                # idempotent (MERGE / SET / id-keyed), so one retry on a fresh connection is safe.
                if attempt == 2:
                    raise GraphUnavailable(
                        f"FalkorDB at {_redact(self.url)} is unreachable or dropped the connection: {exc}"
                    ) from exc
                log.warning("FalkorDB connection problem (%s); retrying once", exc)
                self._client().connection.connection_pool.disconnect(inuse_connections=False)
        raise AssertionError("unreachable")

    def close(self) -> None:
        """Release pooled connections (the store stays usable; connections reopen on demand)."""
        if self._db is not None:
            self._db.connection.connection_pool.disconnect()

    # ---- lifecycle --------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """{"ok": bool, "version": str|None, "graph": str} — never raises."""
        try:
            self._client().connection.ping()
        except Exception as exc:  # noqa: BLE001 - contract: report, never raise
            log.debug("FalkorDB ping failed: %s", exc)
            return {"ok": False, "version": None, "graph": self.graph_name}
        version: str | None = None
        try:
            version = _module_version(self._client().connection.execute_command("MODULE", "LIST"))
        except Exception as exc:  # noqa: BLE001 - MODULE LIST may be blocked by ACLs on hosted FalkorDB
            log.debug("could not read the FalkorDB module version: %s", exc)
        return {"ok": True, "version": version, "graph": self.graph_name}

    def ensure_schema(self, embed_dim: int) -> None:
        """Idempotently create range, full-text and vector indexes (see docs/STRATEGY.md §7)."""
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive int, got {embed_dim!r}")
        for statement in cq.schema_statements():
            self._create_index(statement)
        existing = self._vector_dimension()
        if existing is None:
            self._create_index(cq.vector_index(embed_dim))
        elif existing != embed_dim:
            raise GraphError(
                f"graph '{self.graph_name}' already has a {existing}-d vector index on Utterance.embedding but "
                f"the embedder produces {embed_dim}-d vectors; use another WSW_GRAPH or reset the graph"
            )
        self._embed_dim = embed_dim
        self._schema_pending = False

    def _create_index(self, statement: str) -> None:
        try:
            self._query(statement)
        except GraphQueryError as exc:
            if "already indexed" not in str(exc).lower():
                raise

    def _vector_dimension(self) -> int | None:
        for row in self._query(cq.LIST_INDEXES):
            if row["label"] != "Utterance":
                continue
            types = dict(row.get("types") or {})
            if "VECTOR" not in (types.get("embedding") or []):
                continue
            options = dict((row.get("options") or {}).get("embedding") or {})
            dim = options.get("dimension")
            return int(dim) if dim is not None else None
        return None

    def reset(self) -> None:
        """Delete the whole graph (tests/demo reseed only). Nothing is left behind; if ensure_schema ran
        before, the indexes are recreated automatically by the next create_meeting/write_utterances."""
        with self._write_lock:
            try:
                self._client().connection.execute_command("GRAPH.DELETE", self.graph_name)
            except redis.exceptions.ResponseError as exc:
                if "empty key" not in str(exc).lower():
                    raise GraphQueryError(f"could not delete graph '{self.graph_name}': {exc}") from exc
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
                raise GraphUnavailable(f"FalkorDB at {_redact(self.url)} is unreachable: {exc}") from exc
            self._schema_pending = self._embed_dim is not None

    def _restore_schema(self) -> None:
        if self._schema_pending and self._embed_dim is not None:
            self.ensure_schema(self._embed_dim)

    # ---- meetings -----------------------------------------------------------
    def create_meeting(self, meeting_id: str, title: str, source: str, started_at_ms: int) -> dict[str, Any]:
        if source not in SOURCES:
            raise ValueError(f"meeting source must be one of {SOURCES}, got {source!r}")
        _require_id(meeting_id, "meeting_id")
        self._restore_schema()
        params = {"meeting_id": meeting_id, "title": (title or "").strip() or "Untitled meeting",
                  "source": source, "started_at": int(started_at_ms)}
        self._query(cq.CREATE_MEETING, params)
        meeting = self._meeting(meeting_id)
        if meeting is None:
            raise GraphError(f"meeting {meeting_id} was not readable right after it was created")
        return meeting

    def end_meeting(self, meeting_id: str, ended_at_ms: int, duration_s: float) -> dict[str, Any] | None:
        params = {"meeting_id": meeting_id, "ended_at": int(ended_at_ms), "duration_s": float(duration_s)}
        if not self._query(cq.END_MEETING, params):
            return None
        return self._meeting(meeting_id)

    def set_meeting_summary(self, meeting_id: str, summary: str) -> None:
        self._query(cq.SET_MEETING_SUMMARY, {"meeting_id": meeting_id, "summary": summary})

    def set_meeting_has_audio(self, meeting_id: str, has_audio: bool) -> None:
        self._query(cq.SET_MEETING_HAS_AUDIO, {"meeting_id": meeting_id, "has_audio": bool(has_audio)})

    @_traced
    def list_meetings(self) -> list[dict[str, Any]]:
        """MeetingSummary dicts, newest first."""
        return [_meeting_dict(row) for row in self._query(cq.meeting_summaries(cq.ALL_MEETINGS))]

    @_traced
    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        """{"meeting": MeetingSummary, "speakers": [Speaker], "utterances": [UtteranceOut] (by seq),
        "decisions": [Decision], "action_items": [ActionItem], "topics": [Topic]}"""
        meeting = self._meeting(meeting_id)
        if meeting is None:
            return None
        return {
            "meeting": meeting,
            "speakers": self._speakers(cq.MEETING_SPEAKERS, {"meeting_id": meeting_id}),
            "utterances": self._utterances_after(meeting_id, -1),
            "decisions": self._decisions(cq.DECISIONS_BY_MEETING, {"meeting_id": meeting_id}),
            "action_items": self._action_items(cq.action_items_match(False, True), {"meeting_id": meeting_id}),
            "topics": self._topics(cq.MEETING_TOPICS, {"meeting_id": meeting_id}),
        }

    def delete_meeting(self, meeting_id: str) -> bool:
        params = {"meeting_id": meeting_id}
        with self._write_lock:
            for label in ("Utterance", "Speaker", "Decision", "ActionItem"):
                self._query(cq.delete_by_meeting(label), params)
            deleted = self._query(cq.DELETE_MEETING, params)
            self._query(cq.REACTIVATE_UNSUPERSEDED)
            for label in ("Topic", "Entity", "Person"):
                self._query(cq.delete_orphans(label))
        return bool(deleted and deleted[0]["deleted"])

    def _meeting(self, meeting_id: str) -> dict[str, Any] | None:
        rows = self._query(cq.meeting_summaries(cq.ONE_MEETING), {"meeting_id": meeting_id})
        return _meeting_dict(rows[0]) if rows else None

    # ---- live writes ------------------------------------------------------------
    def write_utterances(self, utterances: list[Utterance], embeddings: list[list[float]] | None) -> dict[str, Any]:
        """One UNWIND batch: MERGE Utterance + Speaker (+HAS_SPEAKER, SPOKEN_BY, HAS_UTTERANCE) and link
        NEXT from the previous seq. Returns {"written": n, "ms": float}."""
        if not utterances:
            return {"written": 0, "ms": 0.0}
        if embeddings is not None and len(embeddings) != len(utterances):
            raise ValueError(f"got {len(embeddings)} embeddings for {len(utterances)} utterances")
        self._restore_schema()
        started = time.perf_counter()
        rows = [
            self._utterance_row(u, embeddings[i] if embeddings is not None else None)
            for i, u in enumerate(utterances)
        ]
        self._query(cq.write_utterances(embeddings is not None), {"rows": rows, "now": _now_ms()})
        self._query(cq.LINK_NEXT, {"links": _next_links(utterances)})
        return {"written": len(rows), "ms": round((time.perf_counter() - started) * 1000.0, 2)}

    def _utterance_row(self, u: Utterance, embedding: Sequence[float] | None) -> dict[str, Any]:
        _require_id(u.meeting_id, "Utterance.meeting_id")
        if isinstance(u.seq, bool) or not isinstance(u.seq, int) or u.seq < 0:
            raise ValueError(f"Utterance.seq must be a non-negative int, got {u.seq!r}")
        track = canonical_track(u.speaker)
        return {
            "id": u.id,
            "meeting_id": u.meeting_id,
            "seq": u.seq,
            "speaker": track,
            "speaker_id": speaker_id(u.meeting_id, track),
            "label": speaker_label(track),
            "color_idx": color_idx(track),
            "t_start": float(u.t_start),
            "t_end": float(u.t_end),
            "text": u.text or "",
            "confidence": float(u.confidence),
            "embedding": self._embedding(u.id, embedding),
        }

    def _embedding(self, utterance_id: str, values: Sequence[float] | None) -> list[float] | None:
        if values is None:
            return None
        vector = [float(v) for v in values]
        if not vector:
            return None
        if self._embed_dim is not None and len(vector) != self._embed_dim:
            raise ValueError(
                f"embedding for {utterance_id} has {len(vector)} dims; the vector index expects {self._embed_dim}"
            )
        if not all(math.isfinite(v) for v in vector):
            log.warning("embedding for %s has non-finite values; stored without a vector", utterance_id)
            return None
        return vector

    def merge_speakers(self, meeting_id: str, src_track: str, dst_track: str) -> int:
        """Re-point src's utterances to dst and delete src. Returns utterances moved."""
        src, dst = canonical_track(src_track), canonical_track(dst_track)
        if src == dst:
            return 0
        src_id, dst_id = speaker_id(meeting_id, src), speaker_id(meeting_id, dst)
        with self._write_lock:
            refs = self._speaker_refs(meeting_id)
            if src not in refs:
                return 0
            if dst not in refs:
                self._query(cq.ENSURE_SPEAKER, {"meeting_id": meeting_id, "speaker_id": dst_id, "track": dst,
                                                "label": speaker_label(dst), "color_idx": color_idx(dst)})
            params = {"meeting_id": meeting_id, "src_id": src_id, "dst_id": dst_id,
                      "src_track": src, "dst_track": dst}
            moved = self._query(cq.MOVE_UTTERANCES, params)
            self._query(cq.MOVE_ASSIGNMENTS, params)
            self._query(cq.MOVE_PROPOSALS, params)
            src_name = refs[src].name
            dst_name = refs[dst].name if dst in refs else None
            if src_name and not dst_name:  # dst inherits the name (and Person) the user gave src
                self._apply_name(meeting_id, dst, src_name)
            elif dst_name:  # src's items the transcript gave to another name go back to that name
                self._sync_owners(meeting_id, dst, dst_name)
            self._query(cq.DELETE_SPEAKER, {"speaker_id": src_id})
            self._query(cq.delete_orphans("Person"))
        return int(moved[0]["moved"]) if moved else 0

    def name_speaker(self, meeting_id: str, track: str, name: str) -> dict[str, Any]:
        """Set Speaker.name and MERGE (s)-[:IS]->(p:Person {id: slug(name)}). Returns the Speaker dict
        with person_id. Also updates ActionItem.owner where owner_track == track.
        Raises LookupError when the meeting has no such speaker."""
        clean = " ".join((name or "").split())
        if not clean:
            raise ValueError("speaker name must not be empty")
        track = canonical_track(track)
        with self._write_lock:
            if not self._apply_name(meeting_id, track, clean):
                raise LookupError(f"meeting {meeting_id} has no speaker {track}")
            self._query(cq.delete_orphans("Person"))  # a rename can leave the previous Person unreferenced
            speakers = self._speakers(cq.ONE_SPEAKER, {"speaker_id": speaker_id(meeting_id, track)})
        if not speakers:
            raise LookupError(f"meeting {meeting_id} has no speaker {track}")
        return speakers[0]

    def _apply_name(self, meeting_id: str, track: str, name: str) -> bool:
        sid = speaker_id(meeting_id, track)
        named = self._query(cq.NAME_SPEAKER, {"speaker_id": sid, "name": name, "person_id": person_id(name)})
        if not named:
            return False
        self._sync_owners(meeting_id, track, name)
        return True

    def _sync_owners(self, meeting_id: str, track: str, name: str) -> None:
        """Items the transcript gave to a different name leave the track (a corrected speaker name hands
        them back); items owned by this name (but no track) now belong to the speaker; then every item on
        the track shows the speaker's name."""
        params = {"meeting_id": meeting_id, "track": track, "speaker_id": speaker_id(meeting_id, track),
                  "name": name, "lowered_name": name.lower()}
        self._query(cq.RELEASE_NAMED_OWNERS, params)
        self._query(cq.ASSIGN_NAMED_OWNERS, params)
        self._query(cq.RENAME_TRACK_OWNERS, params)

    def write_insights(self, meeting_id: str, insights: Insights) -> dict[str, list[dict[str, Any]]]:
        """Write decisions, action items, topics, entities, MENTIONS edges. Dedupe by normalised text per
        meeting. A DecisionDraft with supersedes_text marks the best-matching earlier active decision (any
        meeting) as superseded and links (new)-[:SUPERSEDES]->(old). Returns the NEW items only:
        {"decisions": [Decision], "action_items": [ActionItem], "topics": [Topic]}.
        Raises LookupError when the meeting does not exist."""
        created: dict[str, list[dict[str, Any]]] = {"decisions": [], "action_items": [], "topics": []}
        if insights.is_empty() and not insights.mentions:
            return created
        with self._write_lock:
            if not self._query(cq.existing_ids("Meeting"), {"ids": [meeting_id]}):
                raise LookupError(f"meeting {meeting_id} not found")
            speakers = self._speaker_refs(meeting_id)
            evidence_ids = [e for d in [*insights.decisions, *insights.action_items] for e in d.evidence or []]
            evidence = self._evidence_times(evidence_ids)
            now = _now_ms()
            new_decisions = self._write_decisions(meeting_id, insights.decisions, speakers, evidence, now)
            new_items = self._write_action_items(meeting_id, insights.action_items, speakers, evidence, now)
            new_topics = self._write_topics(meeting_id, insights.topics)
            self._write_entities(insights.entities)
            self._write_mentions(insights.mentions)
            if new_decisions:
                created["decisions"] = _in_order(
                    self._decisions(cq.DECISIONS_BY_IDS, {"ids": new_decisions}), new_decisions)
            if new_items:
                created["action_items"] = _in_order(
                    self._action_items(cq.ACTION_ITEMS_BY_IDS, {"ids": new_items}), new_items)
            if new_topics:
                created["topics"] = _in_order(
                    self._topics(cq.MEETING_TOPICS_BY_IDS, {"meeting_id": meeting_id, "ids": new_topics}),
                    new_topics)
        return created

    def _speaker_refs(self, meeting_id: str) -> dict[str, _SpeakerRef]:
        rows = self._query(cq.speakers(cq.MEETING_SPEAKERS), {"meeting_id": meeting_id})
        return {
            row["track"]: _SpeakerRef(row["id"], row["track"], row["name"], row["label"] or speaker_label(row["track"]))
            for row in rows
            if row["track"]
        }

    def _evidence_times(self, ids: Iterable[str]) -> dict[str, float]:
        unique = [i for i in dict.fromkeys(ids) if isinstance(i, str) and i]
        if not unique:
            return {}
        rows = self._query(cq.EVIDENCE_TIMES, {"ids": unique})
        return {row["id"]: float(row["t_start"] or 0.0) for row in rows}

    def _write_decisions(self, meeting_id: str, drafts: list[DecisionDraft], speakers: dict[str, _SpeakerRef],
                         evidence: dict[str, float], now: int) -> list[str]:
        rows: list[dict[str, Any]] = []
        evidence_pairs: list[dict[str, str]] = []
        proposer_pairs: list[dict[str, str]] = []
        supersedes: dict[str, str] = {}
        last_t: float | None = None
        for draft in drafts:
            text = " ".join((draft.text or "").split())
            key = normalise(text)
            if not key:
                continue
            did = f"{meeting_id}:d-{short_hash(key)}"
            if any(r["id"] == did for r in rows):
                continue
            cited = [e for e in dict.fromkeys(draft.evidence or []) if e in evidence]
            if cited:
                made_at = min(evidence[e] for e in cited)
            else:
                if last_t is None:
                    last_t = self._meeting_last_t(meeting_id)
                made_at = last_t
            proposer = _resolve_speaker(draft.proposed_by, speakers)
            rows.append({"id": did, "meeting_id": meeting_id, "text": text, "made_at": made_at, "created_at": now,
                         "proposed_by_name": None if proposer else _fallback_name(draft.proposed_by)})
            evidence_pairs += [{"src": did, "dst": e} for e in cited]
            if proposer:
                proposer_pairs.append({"src": did, "dst": proposer.id})
            if draft.supersedes_text and draft.supersedes_text.strip():
                supersedes[did] = draft.supersedes_text
        if not rows:
            return []
        existing = {r["id"] for r in self._query(cq.existing_ids("Decision"), {"ids": [r["id"] for r in rows]})}
        new_ids = [r["id"] for r in rows if r["id"] not in existing]
        self._query(cq.UPSERT_DECISIONS, {"rows": rows})
        if evidence_pairs:
            self._query(cq.link("Decision", "EVIDENCED_BY", "Utterance"), {"pairs": evidence_pairs})
        new_proposers = [p for p in proposer_pairs if p["src"] in new_ids]
        if new_proposers:
            self._query(cq.link("Decision", "PROPOSED_BY", "Speaker"), {"pairs": new_proposers})
        self._supersede([(did, supersedes[did]) for did in new_ids if did in supersedes], new_ids)
        return new_ids

    def _supersede(self, requests: list[tuple[str, str]], batch_ids: list[str]) -> None:
        """For each (new decision, supersedes_text) pick the active decision with the highest token Jaccard
        (>= SUPERSEDE_MIN_JACCARD; newest wins ties). Decisions later in this batch are not candidates."""
        if not requests:
            return
        candidates = {row["id"]: content_tokens(row["text"] or "") for row in self._query(cq.ACTIVE_DECISIONS)}
        pairs: list[dict[str, str]] = []
        for new_id, text in requests:
            wanted = content_tokens(text)
            excluded = set(batch_ids[batch_ids.index(new_id):])
            best_id, best = None, 0.0
            for cid, tokens in candidates.items():
                if cid in excluded:
                    continue
                score = jaccard(wanted, tokens)
                if score > best:
                    best_id, best = cid, score
            if best_id is not None and best >= SUPERSEDE_MIN_JACCARD:
                pairs.append({"src": new_id, "dst": best_id})
                candidates.pop(best_id)
            else:
                log.info("no active decision matches supersedes_text %r (best Jaccard %.2f)", text, best)
        if pairs:
            self._query(cq.SUPERSEDE, {"pairs": pairs})

    def _write_action_items(self, meeting_id: str, drafts: list[ActionItemDraft],
                            speakers: dict[str, _SpeakerRef], evidence: dict[str, float], now: int) -> list[str]:
        named_owners = [d.owner for d in drafts if d.owner and not _resolve_speaker(d.owner, speakers)
                        and not parse_track(d.owner)]
        persons = self._persons([person_id(" ".join(o.split())) for o in named_owners])
        rows: list[dict[str, Any]] = []
        evidence_pairs: list[dict[str, str]] = []
        assignee_pairs: list[dict[str, str]] = []
        for draft in drafts:
            text = " ".join((draft.text or "").split())
            key = normalise(text)
            if not key:
                continue
            aid = f"{meeting_id}:a-{short_hash(key)}"
            if any(r["id"] == aid for r in rows):
                continue
            owner_name, owner_track, owner_sid, owner_said = _resolve_owner(draft.owner, speakers, persons)
            due = " ".join((draft.due or "").split()) or None
            rows.append({"id": aid, "meeting_id": meeting_id, "text": text, "created_at": now,
                         "owner_name": owner_name, "owner_track": owner_track, "owner_said": owner_said,
                         "due": due})
            evidence_pairs += [{"src": aid, "dst": e} for e in dict.fromkeys(draft.evidence or []) if e in evidence]
            if owner_sid:
                assignee_pairs.append({"src": aid, "dst": owner_sid})
        if not rows:
            return []
        existing = {r["id"] for r in self._query(cq.existing_ids("ActionItem"), {"ids": [r["id"] for r in rows]})}
        new_ids = [r["id"] for r in rows if r["id"] not in existing]
        self._query(cq.UPSERT_ACTION_ITEMS, {"rows": rows})
        if evidence_pairs:
            self._query(cq.link("ActionItem", "EVIDENCED_BY", "Utterance"), {"pairs": evidence_pairs})
        new_assignees = [p for p in assignee_pairs if p["src"] in new_ids]
        if new_assignees:
            self._query(cq.link("ActionItem", "ASSIGNED_TO", "Speaker"), {"pairs": new_assignees})
        return new_ids

    def _persons(self, ids: list[str]) -> dict[str, str]:
        unique = [i for i in dict.fromkeys(ids) if i]
        if not unique:
            return {}
        return {row["id"]: row["name"] for row in self._query(cq.PERSONS_BY_IDS, {"ids": unique})}

    def _meeting_last_t(self, meeting_id: str) -> float:
        rows = self._query(cq.MEETING_LAST_T, {"meeting_id": meeting_id})
        return float(rows[0]["last_t"] or 0.0) if rows else 0.0

    def _write_topics(self, meeting_id: str, names: list[str]) -> list[str]:
        rows: dict[str, dict[str, str]] = {}
        for raw in names or []:
            name = " ".join(str(raw or "").split())
            tid = slug(name)
            if tid and tid not in rows:
                rows[tid] = {"id": tid, "name": name}
        if not rows:
            return []
        before = {r["id"] for r in self._query(cq.MEETING_TOPIC_IDS, {"meeting_id": meeting_id})}
        self._query(cq.UPSERT_TOPICS, {"meeting_id": meeting_id, "rows": list(rows.values())})
        return [tid for tid in rows if tid not in before]

    def _write_entities(self, entities: list[dict[str, str]]) -> None:
        rows: dict[str, dict[str, str]] = {}
        for entity in entities or []:
            if not isinstance(entity, dict):
                continue
            name = " ".join(str(entity.get("name") or "").split())
            eid = slug(name)
            if not eid or eid in rows:
                continue
            kind = str(entity.get("kind") or "other").strip().lower()
            rows[eid] = {"id": eid, "name": name, "kind": kind if kind in ENTITY_KINDS else "other"}
        if rows:
            self._query(cq.UPSERT_ENTITIES, {"rows": list(rows.values())})

    def _write_mentions(self, mentions: dict[str, list[str]]) -> None:
        """MENTIONS edges to Topic and/or Entity nodes whose slug matches; unknown names are skipped."""
        pairs: dict[tuple[str, str], dict[str, str]] = {}
        for utterance_id, names in (mentions or {}).items():
            for name in names or []:
                target = slug(str(name or ""))
                if utterance_id and target:
                    pairs[(utterance_id, target)] = {"src": utterance_id, "dst": target}
        if not pairs:
            return
        params = {"pairs": list(pairs.values())}
        self._query(cq.link("Utterance", "MENTIONS", "Topic"), params)
        self._query(cq.link("Utterance", "MENTIONS", "Entity"), params)

    # ---- reads used by the UI ---------------------------------------------------
    def utterances_since(self, meeting_id: str, after_seq: int) -> list[dict[str, Any]]:
        return self._utterances_after(meeting_id, int(after_seq))

    def _utterances_after(self, meeting_id: str, after_seq: int) -> list[dict[str, Any]]:
        rows = self._query(cq.UTTERANCES_AFTER, {"meeting_id": meeting_id, "after_seq": after_seq})
        return _unique_utterances(rows)

    @_traced
    def list_action_items(self, status: str | None = None, owner: str | None = None,
                          meeting_id: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in ITEM_STATUSES:
            raise ValueError(f"action item status must be one of {ITEM_STATUSES}, got {status!r}")
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if meeting_id is not None:
            params["meeting_id"] = meeting_id
        who = " ".join((owner or "").split()).lower()
        if who:
            who = (parse_track(who) or who).lower()
            params.update(owner=who, owner_prefix=who + " ")
        match = cq.action_items_match(status is not None, meeting_id is not None)
        return self._action_items(match, params, by_owner=bool(who))

    @_traced
    def set_action_item_status(self, item_id: str, status: str) -> dict[str, Any] | None:
        if status not in ITEM_STATUSES:
            raise ValueError(f"action item status must be one of {ITEM_STATUSES}, got {status!r}")
        if not self._query(cq.SET_ACTION_STATUS, {"item_id": item_id, "status": status, "now": _now_ms()}):
            return None
        items = self._action_items(cq.ACTION_ITEMS_BY_IDS, {"ids": [item_id]})
        return items[0] if items else None

    @_traced
    def list_decisions(self, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Decisions with supersession links, newest first."""
        if meeting_id is None:
            return self._decisions(cq.ALL_DECISIONS, {})
        return self._decisions(cq.DECISIONS_BY_MEETING, {"meeting_id": meeting_id})

    @_traced
    def list_people(self) -> list[dict[str, Any]]:
        """[{id, name, meetings: int, utterances: int, talk_time_s, open_items: int}]"""
        rows = self._query(cq.PEOPLE)
        by_name: dict[str, set[str]] = {}
        for row in self._query(cq.OPEN_ITEMS_BY_NAME):
            by_name.setdefault(person_id(row["owner_name"]), set()).add(row["id"])
        people = [
            {
                "id": row["id"],
                "name": row["name"] or row["id"],
                "meetings": int(row["meetings"] or 0),
                "utterances": int(row["utterances"] or 0),
                "talk_time_s": round(float(row["talk_time_s"] or 0.0), 2),
                "open_items": len(set(row["assigned_open"] or []) | by_name.get(row["id"], set())),
            }
            for row in rows
        ]
        people.sort(key=lambda p: (-p["utterances"], p["name"].lower()))
        return people

    @_traced
    def who_knows(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        """People (or unnamed speakers) ranked by how much they said about `topic` (full-text match).
        [{name, person_id|None, meeting_id, hits: int, sample: UtteranceOut}]"""
        rows = self._search_rows(topic, cq.fulltext_speakers, cq.scan_speakers, WHO_KNOWS_POOL, None)
        rows.sort(key=lambda r: -float(r["score"] or 0.0))
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row["person_id"] or row["speaker_id"] or f"{row['meeting_id']}:{row['speaker']}"
            group = groups.get(key)
            if group is None:
                name = (row["person_name"] or row["speaker_name"] or row["speaker_label"]
                        or speaker_label(row["speaker"] or "?"))
                groups[key] = {"name": name, "person_id": row["person_id"], "meeting_id": row["meeting_id"],
                               "hits": 1, "sample": _utterance_dict(row), "_score": float(row["score"] or 0.0)}
            else:
                group["hits"] += 1
                group["_score"] += float(row["score"] or 0.0)
        ranked = sorted(groups.values(), key=lambda g: (-g["hits"], -g["_score"]))[: max(1, int(limit))]
        for group in ranked:
            del group["_score"]
        return ranked

    @_traced
    def subgraph(self, meeting_id: str | None = None, utterance_ids: list[str] | None = None,
                 limit: int = 250) -> dict[str, Any]:
        """Subgraph dict for the graph inspector. With utterance_ids: those utterances plus their speaker,
        meeting, decisions/action items they evidence and topics they mention. With meeting_id: the
        meeting, its speakers/people, decisions, action items, topics, entities and utterances (by seq)
        up to `limit` nodes. Neither: a cross-meeting overview without utterances.
        Topic/Entity/Person node ids are prefixed ("topic:pricing") so slugs cannot collide; `ref` keeps
        the raw id."""
        limit = _clamp(limit, 1, 2000)
        nodes: dict[int, dict[str, Any]] = {}
        ids = [i for i in dict.fromkeys(utterance_ids or []) if isinstance(i, str) and i]
        if ids:
            _collect(nodes, self._query(cq.SUBGRAPH_UTTERANCES, {"ids": ids}), limit)
        elif meeting_id:
            _collect(nodes, self._query(cq.SUBGRAPH_MEETING_CONTEXT, {"meeting_id": meeting_id}), limit)
            room = limit - len(nodes)
            if nodes and room > 0:
                rows = self._query(cq.SUBGRAPH_MEETING_UTTERANCES, {"meeting_id": meeting_id, "limit": room})
                _collect(nodes, rows, limit)
        else:
            _collect(nodes, self._query(cq.SUBGRAPH_OVERVIEW, {"limit": limit}), limit)
        if not nodes:
            return {"nodes": [], "links": []}
        keys = {nid: node["id"] for nid, node in nodes.items()}
        links: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in self._query(cq.SUBGRAPH_EDGES, {"nids": list(nodes)}):
            source, target = keys.get(row["source"]), keys.get(row["target"])
            if source and target:
                links[(source, target, row["type"])] = {"source": source, "target": target, "type": row["type"]}
        return {"nodes": list(nodes.values()), "links": list(links.values())}

    def stats(self) -> dict[str, int]:
        """Counts per label and total relationships: {"Meeting": n, ..., "relationships": n}."""
        counts = dict.fromkeys([*cq.LABELS, "relationships"], 0)  # empty labels may yield no row at all
        for row in self._query(cq.stats()):
            counts[row["label"]] = int(row["n"] or 0)
        return counts

    # ---- retrieval (used by the agent) -----------------------------------------
    @_traced
    def search_fulltext(self, query: str, k: int = 8, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Hit dicts (via="fulltext"). Sanitise the query for RediSearch (strip punctuation, drop
        stopwords, OR the terms, prefix-match long terms). Empty queries give []; a query RediSearch
        rejects falls back to a bounded CONTAINS scan instead of raising."""
        k = _clamp(k, 1, 500)
        rows = self._search_rows(query, cq.fulltext_hits, cq.scan_hits, k, meeting_id)
        hits = _unique_hits(rows, "fulltext")
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]

    def _search_rows(self, query: str, fulltext: Callable[[bool], str], scan: Callable[[bool], str], k: int,
                     meeting_id: str | None) -> list[dict[str, Any]]:
        terms = query_terms(query)
        if not terms:
            return []
        by_meeting = meeting_id is not None
        scoped = {"meeting_id": meeting_id} if by_meeting else {}
        try:
            return self._query(fulltext(by_meeting), {"query": fulltext_query(terms), "k": k, **scoped})
        except GraphQueryError as exc:
            log.warning("full-text search for %r failed (%s); falling back to a substring scan", query, exc)
        params = {"terms": terms, "k": k, "scan_limit": FULLTEXT_SCAN_LIMIT, **scoped}
        return self._query(scan(by_meeting), params)

    @_traced
    def search_vector(self, vector: list[float], k: int = 8, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Hit dicts (via="vector"), score = 1 - distance (cosine: 1 = same direction)."""
        values = [float(v) for v in (vector if vector is not None else [])]
        if not values or not all(math.isfinite(v) for v in values):
            return []
        if self._embed_dim is not None and len(values) != self._embed_dim:
            raise ValueError(f"query vector has {len(values)} dims; the vector index expects {self._embed_dim}")
        k = _clamp(k, 1, 500)
        by_meeting = meeting_id is not None
        # KNN runs before the meeting filter, so over-fetch when filtering.
        params: dict[str, Any] = {"vector": values, "k": k, "knn": min(max(k * 10, 100), 1000) if by_meeting else k}
        if by_meeting:
            params["meeting_id"] = meeting_id
        try:
            rows = self._query(cq.vector_hits(by_meeting), params)
        except GraphQueryError as exc:
            if "undefined attribute" in str(exc).lower():  # no utterance has an embedding yet
                return []
            raise
        hits = _unique_hits(rows, "vector")
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]

    @_traced
    def get_utterances(self, ids: list[str]) -> list[dict[str, Any]]:
        """Hit-shaped dicts for these ids (score=1.0, via="graph"), in the given order."""
        wanted = [i for i in dict.fromkeys(ids or []) if isinstance(i, str) and i]
        if not wanted:
            return []
        hits = {h["utterance_id"]: h for h in _unique_hits(self._query(cq.HITS_BY_IDS, {"ids": wanted}), "graph")}
        return [hits[i] for i in wanted if i in hits]

    @_traced
    def context_window(self, utterance_id: str, before: int = 2, after: int = 2) -> list[dict[str, Any]]:
        """Neighbouring utterances via NEXT, as UtteranceOut dicts in seq order."""
        statement = cq.context_window(_clamp(before, 0, 50), _clamp(after, 0, 50))
        return _unique_utterances(self._query(statement, {"utterance_id": utterance_id}))

    @_traced
    def recent_utterances(self, meeting_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Latest utterances (of a meeting, or of the latest meeting), Hit-shaped, via="recent",
        oldest first."""
        if meeting_id is None:
            latest = self._query(cq.LATEST_MEETING)
            if not latest:
                return []
            meeting_id = latest[0]["id"]
        rows = self._query(cq.RECENT_HITS, {"meeting_id": meeting_id, "limit": _clamp(limit, 1, 500)})
        rows.sort(key=lambda r: int(r["_seq"] or 0))
        return _unique_hits(rows, "recent")

    @_traced
    def utterances_by_speaker(self, name_or_track: str, meeting_id: str | None = None,
                              limit: int = 20) -> list[dict[str, Any]]:
        """Hit-shaped utterances whose speaker name/track/person matches (case-insensitive). A first name
        matches a full name ("priya" -> "Priya Sharma"). Newest meeting first, then by seq."""
        who = " ".join((name_or_track or "").split()).lower()
        if not who:
            return []
        who = (parse_track(who) or who).lower()
        by_meeting = meeting_id is not None
        params: dict[str, Any] = {"who": who, "who_slug": slug(who), "who_prefix": who + " ",
                                  "limit": _clamp(limit, 1, 500)}
        if by_meeting:
            params["meeting_id"] = meeting_id
        rows = self._query(cq.speaker_hits(by_meeting), params)
        rows.sort(key=lambda r: (-int(r["_started"] or 0), int(r["_seq"] or 0)))
        return _unique_hits(rows, "graph")

    # ---- conversation memory ----------------------------------------------------
    def save_turn(self, session_id: str, role: str, text: str, at_ms: int,
                  citations: list[str] | None = None) -> None:
        _require_id(session_id, "session_id")
        if not role or not isinstance(role, str):
            raise ValueError("turn role must be a non-empty string such as 'user' or 'assistant'")
        params = {"session_id": session_id, "turn_id": uuid.uuid4().hex, "role": role, "text": text or "",
                  "at": int(at_ms), "citations": [str(c) for c in citations or []]}
        self._query(cq.SAVE_TURN, params)

    def get_turns(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        """[{role, text, at, citations}] oldest first."""
        rows = self._query(cq.GET_TURNS, {"session_id": session_id, "limit": _clamp(limit, 1, 500)})
        return [
            {"role": row["role"], "text": row["text"] or "", "at": int(row["at"] or 0),
             "citations": list(row["citations"] or [])}
            for row in reversed(rows)
        ]

    # ---- shared row readers -----------------------------------------------------
    def _speakers(self, match: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self._query(cq.speakers(match), params)
        return sorted((_speaker_dict(row) for row in rows), key=_track_order)

    def _decisions(self, match: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [_decision_dict(row) for row in self._query(cq.decisions(match), params)]

    def _action_items(self, match: str, params: dict[str, Any], by_owner: bool = False) -> list[dict[str, Any]]:
        return [_action_item_dict(row) for row in self._query(cq.action_items(match, by_owner), params)]

    def _topics(self, match: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self._query(cq.topics(match), params)
        return [{"id": row["id"], "name": row["name"] or row["id"], "mentions": int(row["mentions"] or 0)}
                for row in rows]


# ---- row -> dict ------------------------------------------------------------------------------------


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _meeting_dict(row: dict[str, Any]) -> dict[str, Any]:
    status = row["status"] or "live"
    stored = _float(row["duration_s"])
    last_t = _float(row["last_t"])
    duration = stored if status == "ended" and stored > 0 else max(stored, last_t)
    return {
        "id": row["id"],
        "title": row["title"] or "Untitled meeting",
        "status": status,
        "source": row["source"] or "mic",
        "started_at": int(row["started_at"] or 0),
        "ended_at": int(row["ended_at"]) if row["ended_at"] is not None else None,
        "duration_s": round(duration, 2),
        "utterance_count": int(row["utterance_count"] or 0),
        "speaker_count": int(row["speaker_count"] or 0),
        "has_audio": bool(row["has_audio"]),
        "summary": row["summary"],
    }


def _speaker_dict(row: dict[str, Any]) -> dict[str, Any]:
    track = row["track"] or ""
    return {
        "track": track,
        "label": row["label"] or speaker_label(track),
        "name": row["name"],
        "person_id": row["person_id"],
        "color_idx": int(row["color_idx"]) if row["color_idx"] is not None else color_idx(track or "?"),
        "utterance_count": int(row["utterance_count"] or 0),
        "talk_time_s": round(_float(row["talk_time_s"]), 2),
    }


def _track_order(speaker: dict[str, Any]) -> tuple[int, str]:
    parsed = parse_track(speaker["track"])
    return (int(parsed[1:]) if parsed else 10**6, speaker["track"])


def _utterance_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "meeting_id": row["meeting_id"],
        "seq": int(row["seq"] or 0),
        "speaker": row["speaker"],
        "speaker_name": row["speaker_name"],
        "t_start": _float(row["t_start"]),
        "t_end": _float(row["t_end"]),
        "text": row["text"] or "",
        "confidence": _float(row["confidence"], 1.0),
    }


def _unique_utterances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["id"], _utterance_dict(row))
    return sorted(unique.values(), key=lambda u: u["seq"])


def _hit_dict(row: dict[str, Any], via: str) -> dict[str, Any]:
    return {
        "utterance_id": row["utterance_id"],
        "meeting_id": row["meeting_id"],
        "meeting_title": row["meeting_title"] or "Untitled meeting",
        "speaker": row["speaker"],
        "speaker_name": row["speaker_name"],
        "t_start": _float(row["t_start"]),
        "t_end": _float(row["t_end"]),
        "text": row["text"] or "",
        "score": round(_float(row["score"]), 6),
        "via": via,
    }


def _unique_hits(rows: list[dict[str, Any]], via: str) -> list[dict[str, Any]]:
    """Hit dicts in row order, one per utterance (stale duplicate edges must not duplicate hits)."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["utterance_id"], _hit_dict(row, via))
    return list(unique.values())


def _decision_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "meeting_id": row["meeting_id"],
        "text": row["text"] or "",
        "status": row["status"] or "active",
        "made_at": _float(row["made_at"]),
        "proposed_by": row["proposed_by"],
        "evidence": sorted(set(row["evidence"] or [])),
        "superseded_by": row["superseded_by"],
        "supersedes": row["supersedes"],
    }


def _action_item_dict(row: dict[str, Any]) -> dict[str, Any]:
    owner_track = row["owner_track"]
    return {
        "id": row["id"],
        "meeting_id": row["meeting_id"],
        "meeting_title": row["meeting_title"] or "Untitled meeting",
        "text": row["text"] or "",
        "owner": row["owner"] or (speaker_label(owner_track) if owner_track else None),
        "owner_track": owner_track,
        "due": row["due"],
        "status": row["status"] or "open",
        "created_at": int(row["created_at"] or 0),
        "done_at": int(row["done_at"]) if row["done_at"] is not None else None,
        "evidence": sorted(set(row["evidence"] or [])),
    }


def _graph_node(row: dict[str, Any]) -> dict[str, Any]:
    kind = row["kind"] or "Node"
    raw_id = str(row["id"])
    node: dict[str, Any] = {
        "id": f"{kind.lower()}:{raw_id}" if kind in _SLUG_KINDS else raw_id,
        "label": truncate(str(row["label"] if row["label"] is not None else raw_id)),
        "kind": kind,
    }
    for key in ("status", "meeting_id", "track", "color_idx", "t_start"):
        if row.get(key) is not None:
            node[key] = row[key]
    if kind in _SLUG_KINDS:
        node["ref"] = raw_id
    return node


def _collect(nodes: dict[int, dict[str, Any]], rows: list[dict[str, Any]], limit: int) -> None:
    """Add subgraph node rows (keyed by internal node id) in kind priority order until `limit`."""
    for row in sorted(rows, key=lambda r: _KIND_ORDER.get(r["kind"], len(_KIND_ORDER))):
        if len(nodes) >= limit:
            return
        if row["id"] is not None and row["nid"] not in nodes:
            nodes[row["nid"]] = _graph_node(row)


# ---- small helpers -----------------------------------------------------------------------------------


def _resolve_speaker(value: str | None, speakers: dict[str, _SpeakerRef]) -> _SpeakerRef | None:
    """A track ("S2"), a speaker label ("Speaker 2") or a speaker's name in this meeting."""
    if not value or not value.strip():
        return None
    track = parse_track(value)
    if track:
        return speakers.get(track)
    lowered = " ".join(value.split()).lower()
    for ref in speakers.values():
        if (ref.name and ref.name.lower() == lowered) or ref.label.lower() == lowered:
            return ref
    return None


def _resolve_owner(value: str | None, speakers: dict[str, _SpeakerRef],
                   persons: dict[str, str]) -> tuple[str | None, str | None, str | None, str | None]:
    """(owner_name, owner_track, speaker node id, owner_said) for an ActionItemDraft.owner. owner_said is
    the name the transcript gave (None for a track or speaker label), kept so a corrected speaker name can
    hand the item back to that name."""
    if not value or not value.strip():
        return None, None, None, None
    clean = " ".join(value.split())
    ref = _resolve_speaker(value, speakers)
    if ref:
        by_name = ref.name is not None and not parse_track(value) and ref.name.lower() == clean.lower()
        return ref.name, ref.track, ref.id, ref.name if by_name else None
    track = parse_track(value)
    if track:  # a track this meeting has not produced yet
        return None, track, None, None
    name = persons.get(person_id(clean)) or clean
    return name, None, None, name


def _fallback_name(value: str | None) -> str | None:
    """Display name for a proposer that is not a known speaker of the meeting."""
    if not value or not value.strip():
        return None
    track = parse_track(value)
    return speaker_label(track) if track else " ".join(value.split())


def _next_links(utterances: list[Utterance]) -> list[dict[str, str]]:
    """seq-1 -> seq and seq -> seq+1 candidates; the query links only pairs whose ends both exist, so an
    utterance that arrives late still gets linked from both sides."""
    links: dict[tuple[str, str], dict[str, str]] = {}
    for u in utterances:
        if u.seq > 0:
            prev = dataclasses.replace(u, seq=u.seq - 1).id
            links[(prev, u.id)] = {"prev": prev, "next": u.id}
        nxt = dataclasses.replace(u, seq=u.seq + 1).id
        links[(u.id, nxt)] = {"prev": u.id, "next": nxt}
    return list(links.values())


def _in_order(items: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in items}
    return [by_id[i] for i in ids if i in by_id]


def _module_version(modules: Any) -> str | None:
    """'graph' module ver 41803 -> '4.18.3'."""
    for module in modules or []:
        info = module if isinstance(module, dict) else dict(zip(module[::2], module[1::2]))
        name = info.get("name") or info.get(b"name")
        if name in ("graph", b"graph"):
            ver = int(info.get("ver") or info.get(b"ver") or 0)
            return f"{ver // 10000}.{(ver // 100) % 100}.{ver % 100}"
    return None


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer, got {value!r}") from exc
    return max(low, min(high, number))


def _require_id(value: Any, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:120] if text.strip() else ""


def _redact(url: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", url)


def _prefer_ipv4_loopback(url: str) -> str:
    """Windows resolves "localhost" to ::1 first and spends ~2 s on every new connection before falling
    back to IPv4 when FalkorDB (WSL/Docker) listens on IPv4 only. Use 127.0.0.1 when it answers."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != "localhost":
        return url
    try:
        with socket.create_connection(("127.0.0.1", parts.port or 6379), timeout=0.5):
            pass
    except OSError:
        return url
    userinfo, _, hostport = parts.netloc.rpartition("@")
    hostport = re.sub(r"(?i)^localhost", "127.0.0.1", hostport)
    return urllib.parse.urlunsplit(parts._replace(netloc=f"{userinfo}@{hostport}" if userinfo else hostport))


__all__ = ["ActionItemDraft", "DecisionDraft", "GraphError", "GraphQueryError", "GraphStore", "GraphUnavailable"]
