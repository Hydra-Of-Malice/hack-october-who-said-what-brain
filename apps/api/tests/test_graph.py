"""GraphStore against the live FalkorDB (WSW_FALKORDB_URL, default falkor://localhost:6379).

One throwaway graph per test session ("wsw_test_<hex>"), reset before every test and deleted at the end.
The whole module is skipped when FalkorDB is not reachable.
"""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from typing import Any

import pytest
import redis
from falkordb import FalkorDB

from wsw_brain.contracts import ActionItemDraft, DecisionDraft, Insights, Utterance
from wsw_brain.graph import GraphError, GraphStore
from wsw_brain.graph import store as store_module

URL = os.environ.get("WSW_FALKORDB_URL", "falkor://localhost:6379")
RAW_URL = URL.replace("//localhost:", "//127.0.0.1:")  # skip the ~2 s IPv6-first localhost lookup on Windows
DIM = 8

MEETING_KEYS = {"id", "title", "status", "source", "started_at", "ended_at", "duration_s", "utterance_count",
                "speaker_count", "has_audio", "summary"}
SPEAKER_KEYS = {"track", "label", "name", "person_id", "color_idx", "utterance_count", "talk_time_s"}
UTTERANCE_KEYS = {"id", "meeting_id", "seq", "speaker", "speaker_name", "t_start", "t_end", "text", "confidence"}
DECISION_KEYS = {"id", "meeting_id", "text", "status", "made_at", "proposed_by", "evidence", "superseded_by",
                 "supersedes"}
ACTION_KEYS = {"id", "meeting_id", "meeting_title", "text", "owner", "owner_track", "due", "status", "created_at",
               "done_at", "evidence"}
TOPIC_KEYS = {"id", "name", "mentions"}
HIT_KEYS = {"utterance_id", "meeting_id", "meeting_title", "speaker", "speaker_name", "t_start", "t_end", "text",
            "score", "via"}


# ---- fixtures ------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def graph_name() -> str:
    return "wsw_test_" + uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def raw(graph_name: str):
    """A plain falkordb Graph handle on the test graph, for assertions that bypass GraphStore."""
    db = FalkorDB.from_url(RAW_URL, socket_connect_timeout=3)
    try:
        db.connection.ping()
    except redis.exceptions.RedisError as exc:  # no database, nothing to test against
        pytest.skip(f"FalkorDB not reachable at {URL}: {exc}")
    yield db.select_graph(graph_name)
    with contextlib.suppress(redis.exceptions.ResponseError):  # already gone
        db.connection.execute_command("GRAPH.DELETE", graph_name)
    db.connection.close()


@pytest.fixture(scope="session")
def session_store(graph_name: str, raw) -> GraphStore:
    store = GraphStore(URL, graph_name)
    yield store
    store.close()


@pytest.fixture
def store(session_store: GraphStore) -> GraphStore:
    session_store.reset()
    session_store.ensure_schema(DIM)
    return session_store


def rows(raw, query: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
    return raw.query(query, params).result_set


def one_hot(i: int) -> list[float]:
    vec = [0.0] * DIM
    vec[i % DIM] = 1.0
    return vec


def utt(meeting_id: str, seq: int, speaker: str, text: str) -> Utterance:
    return Utterance(meeting_id=meeting_id, seq=seq, speaker=speaker, t_start=seq * 5.0, t_end=seq * 5.0 + 4.0,
                     text=text, confidence=0.9)


PRICING = [
    ("S1", "Let's start with the pricing review for the enterprise tier."),
    ("S2", "I think the enterprise discount should be fifteen percent."),
    ("S1", "We decided to launch the beta on March 3."),
    ("S2", "Priya will send the pricing deck to the board by Friday."),
    ("S3", "Can we also talk about the onboarding emails?"),
]


def seed(store: GraphStore, meeting_id: str = "m1", title: str = "Pricing sync", started_at: int = 1_000,
         lines: list[tuple[str, str]] = PRICING) -> list[Utterance]:
    store.create_meeting(meeting_id, title, "mic", started_at)
    utterances = [utt(meeting_id, i, spk, text) for i, (spk, text) in enumerate(lines)]
    store.write_utterances(utterances, [one_hot(i) for i in range(len(utterances))])
    return utterances


# ---- schema & lifecycle --------------------------------------------------------------------------------


def test_ping_reports_engine_version(store: GraphStore, graph_name: str) -> None:
    status = store.ping()
    assert status["ok"] is True
    assert status["graph"] == graph_name
    assert status["version"] and status["version"][0].isdigit()


def test_ping_never_raises_when_unreachable() -> None:
    dead = GraphStore("falkor://127.0.0.1:1", "wsw_test_dead")
    assert dead.ping() == {"ok": False, "version": None, "graph": "wsw_test_dead"}


def test_ensure_schema_is_idempotent_and_complete(store: GraphStore, raw) -> None:
    store.ensure_schema(DIM)
    store.ensure_schema(DIM)
    indexes = {r[0]: (r[2], r[3]) for r in rows(raw, "CALL db.indexes() YIELD label, properties, types, options "
                                                    "RETURN label, properties, types, options")}
    types, options = indexes["Utterance"]
    assert types["id"] == ["RANGE"] and types["t_start"] == ["RANGE"] and types["meeting_id"] == ["RANGE"]
    assert types["text"] == ["FULLTEXT"] and types["embedding"] == ["VECTOR"]
    assert options["embedding"]["dimension"] == DIM
    assert options["embedding"]["similarityFunction"] == "cosine"
    assert indexes["ActionItem"][0]["status"] == ["RANGE"]
    for label in ("Meeting", "Speaker", "Person", "Topic", "Decision", "ActionItem", "Entity", "Turn"):
        assert indexes[label][0]["id"] == ["RANGE"], label


def test_reset_deletes_graph_and_next_write_restores_schema(store: GraphStore, raw, graph_name: str) -> None:
    seed(store)
    store.reset()
    assert raw.client.connection.exists(graph_name) == 0  # nothing left behind
    seed(store, "m2", "Reseeded", 2_000)
    vector_dims = [r[1]["embedding"]["dimension"] for r in rows(
        raw, "CALL db.indexes() YIELD label, options RETURN label, options") if r[0] == "Utterance"]
    assert vector_dims == [DIM]
    assert [h["utterance_id"] for h in store.search_vector(one_hot(1), k=1)] == ["m2:u00001"]


def test_ensure_schema_rejects_bad_or_conflicting_dimensions(store: GraphStore) -> None:
    with pytest.raises(ValueError):
        store.ensure_schema(0)
    with pytest.raises(GraphError, match="8-d vector index"):
        store.ensure_schema(16)


# ---- live writes ---------------------------------------------------------------------------------------


def test_write_batches_out_of_order_builds_next_chain(store: GraphStore, raw) -> None:
    store.create_meeting("m1", "Standup", "mic", 1_000)
    batch = [utt("m1", s, "S1" if s % 2 == 0 else "S2", f"line {s}") for s in range(5)]
    first = store.write_utterances([batch[0], batch[1], batch[3]], [one_hot(0), one_hot(1), one_hot(3)])
    assert first["written"] == 3 and isinstance(first["ms"], float)
    store.write_utterances([batch[4], batch[2]], None)  # seq 2 arrives after 3 and 4

    chain = rows(raw, "MATCH (a:Utterance)-[:NEXT]->(b:Utterance) RETURN a.seq, b.seq ORDER BY a.seq")
    assert chain == [[0, 1], [1, 2], [2, 3], [3, 4]]

    store.write_utterances(batch, None)  # replaying a batch must not duplicate anything
    assert rows(raw, "MATCH (u:Utterance) RETURN count(u)") == [[5]]
    assert rows(raw, "MATCH ()-[r:NEXT]->() RETURN count(r)") == [[4]]
    assert rows(raw, "MATCH (:Meeting {id: 'm1'})-[r:HAS_SPEAKER]->(:Speaker) RETURN count(r)") == [[2]]
    assert rows(raw, "MATCH (u:Utterance {id: 'm1:u00000'}) RETURN u.meeting_id, u.speaker") == [["m1", "S1"]]
    assert rows(raw, "MATCH (s:Speaker {id: 'm1:S2'}) RETURN s.meeting_id, s.label, s.color_idx") == \
        [["m1", "Speaker 2", 1]]
    # the embedding written in the first batch survives the embedding-less replay
    assert rows(raw, "MATCH (u:Utterance {id: 'm1:u00001'}) RETURN u.embedding")[0][0] == one_hot(1)


def test_write_utterances_validates_input(store: GraphStore) -> None:
    store.create_meeting("m1", "Standup", "mic", 1_000)
    assert store.write_utterances([], None) == {"written": 0, "ms": 0.0}
    with pytest.raises(ValueError, match="embeddings"):
        store.write_utterances([utt("m1", 0, "S1", "hi")], [])
    with pytest.raises(ValueError, match="dims"):
        store.write_utterances([utt("m1", 0, "S1", "hi")], [[1.0, 0.0]])


def test_rewritten_utterance_moves_to_new_speaker(store: GraphStore, raw) -> None:
    seed(store)
    store.write_utterances([utt("m1", 4, "S1", PRICING[4][1])], None)
    assert rows(raw, "MATCH (:Utterance {id: 'm1:u00004'})-[:SPOKEN_BY]->(s) RETURN s.id") == [["m1:S1"]]


# ---- meetings --------------------------------------------------------------------------------------------


def test_get_meeting_shape(store: GraphStore) -> None:
    seed(store)
    store.set_meeting_summary("m1", "Pricing and beta launch.")
    store.set_meeting_has_audio("m1", True)
    detail = store.get_meeting("m1")
    assert detail is not None
    assert set(detail) == {"meeting", "speakers", "utterances", "decisions", "action_items", "topics"}

    meeting = detail["meeting"]
    assert set(meeting) == MEETING_KEYS
    assert meeting["status"] == "live" and meeting["source"] == "mic" and meeting["started_at"] == 1_000
    assert meeting["utterance_count"] == 5 and meeting["speaker_count"] == 3
    assert meeting["has_audio"] is True and meeting["summary"] == "Pricing and beta launch."
    assert meeting["duration_s"] == 24.0  # live: last t_end

    speakers = detail["speakers"]
    assert [s["track"] for s in speakers] == ["S1", "S2", "S3"]
    assert all(set(s) == SPEAKER_KEYS for s in speakers)
    assert speakers[0] == {"track": "S1", "label": "Speaker 1", "name": None, "person_id": None, "color_idx": 0,
                           "utterance_count": 2, "talk_time_s": 8.0}

    utterances = detail["utterances"]
    assert [u["seq"] for u in utterances] == [0, 1, 2, 3, 4]
    assert all(set(u) == UTTERANCE_KEYS for u in utterances)
    assert utterances[1]["id"] == "m1:u00001" and utterances[1]["speaker"] == "S2"
    assert utterances[1]["speaker_name"] is None and utterances[1]["confidence"] == pytest.approx(0.9)
    assert detail["decisions"] == [] and detail["action_items"] == [] and detail["topics"] == []
    assert store.get_meeting("nope") is None


def test_meeting_lifecycle_and_listing(store: GraphStore) -> None:
    seed(store, "m1", "Older", 1_000)
    seed(store, "m2", "Newer", 2_000)
    assert [m["id"] for m in store.list_meetings()] == ["m2", "m1"]
    ended = store.end_meeting("m1", 9_000, 61.5)
    assert ended is not None
    assert ended["status"] == "ended" and ended["ended_at"] == 9_000 and ended["duration_s"] == 61.5
    assert store.end_meeting("nope", 1, 1.0) is None
    with pytest.raises(ValueError):
        store.create_meeting("m3", "x", "fax", 1)


def test_utterances_since(store: GraphStore) -> None:
    seed(store)
    assert [u["seq"] for u in store.utterances_since("m1", 2)] == [3, 4]
    assert store.utterances_since("m1", 4) == []


# ---- speakers ------------------------------------------------------------------------------------------


def test_merge_speakers_repoints_utterances_and_items(store: GraphStore, raw) -> None:
    seed(store)
    store.name_speaker("m1", "S3", "Rahul")
    store.write_insights("m1", Insights(action_items=[ActionItemDraft("Fix onboarding emails", ["m1:u00004"], "S3")]))

    assert store.merge_speakers("m1", "S3", "S1") == 1
    assert rows(raw, "MATCH (s:Speaker {id: 'm1:S3'}) RETURN count(s)") == [[0]]
    detail = store.get_meeting("m1")
    assert [s["track"] for s in detail["speakers"]] == ["S1", "S2"]
    moved = detail["utterances"][4]
    assert moved["speaker"] == "S1" and moved["speaker_name"] == "Rahul"  # S1 inherited the name
    item = detail["action_items"][0]
    assert item["owner_track"] == "S1" and item["owner"] == "Rahul"
    assert rows(raw, "MATCH (:ActionItem)-[:ASSIGNED_TO]->(s:Speaker) RETURN s.id") == [["m1:S1"]]

    assert store.merge_speakers("m1", "S9", "S1") == 0
    assert store.merge_speakers("m1", "S1", "S1") == 0
    assert store.merge_speakers("m1", "S2", "S7") == 2  # absent destination is created
    assert [s["track"] for s in store.get_meeting("m1")["speakers"]] == ["S1", "S7"]


def test_name_speaker_links_person_and_updates_owners(store: GraphStore, raw) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Board prep", 2_000, [("S1", "The pricing deck goes to the board."),
                                             ("S2", "I'll review it tonight.")])
    store.write_insights("m1", Insights(action_items=[ActionItemDraft("Send the pricing deck", ["m1:u00003"], "S2")]))
    store.write_insights("m2", Insights(action_items=[ActionItemDraft("Review the deck", ["m2:u00001"], "priya")]))
    before = store.list_action_items(meeting_id="m1")[0]
    assert before["owner"] == "Speaker 2" and before["owner_track"] == "S2"

    speaker = store.name_speaker("m1", "S2", "  Priya ")
    assert set(speaker) == SPEAKER_KEYS
    assert speaker["name"] == "Priya" and speaker["person_id"] == "priya" and speaker["track"] == "S2"
    assert store.list_action_items(meeting_id="m1")[0]["owner"] == "Priya"
    assert store.get_meeting("m1")["utterances"][1]["speaker_name"] == "Priya"

    store.name_speaker("m2", "s2", "Priya")  # same person in another meeting; the name-owned item gets linked
    item = store.list_action_items(meeting_id="m2")[0]
    assert item["owner_track"] == "S2" and item["owner"] == "Priya"
    assert rows(raw, "MATCH (:ActionItem {meeting_id: 'm2'})-[:ASSIGNED_TO]->(s) RETURN s.id") == [["m2:S2"]]
    assert rows(raw, "MATCH (s:Speaker)-[:IS]->(p:Person {id: 'priya'}) RETURN count(s)") == [[2]]
    people = store.list_people()
    assert people == [{"id": "priya", "name": "Priya", "meetings": 2, "utterances": 3, "talk_time_s": 12.0,
                       "open_items": 2}]

    renamed = store.name_speaker("m1", "S2", "Priya Sharma")
    assert renamed["person_id"] == "priya-sharma"
    assert rows(raw, "MATCH (:Speaker {id: 'm1:S2'})-[r:IS]->() RETURN count(r)") == [[1]]
    assert store.list_action_items(meeting_id="m1")[0]["owner"] == "Priya Sharma"
    store.name_speaker("m1", "S1", "Temp Name")
    store.name_speaker("m1", "S1", "Final Name")  # the unreferenced "temp-name" Person is dropped
    assert sorted(r[0] for r in rows(raw, "MATCH (p:Person) RETURN p.id")) == ["final-name", "priya",
                                                                               "priya-sharma"]
    with pytest.raises(LookupError):
        store.name_speaker("m1", "S9", "Nobody")
    with pytest.raises(ValueError):
        store.name_speaker("m1", "S1", "   ")


# ---- insights ------------------------------------------------------------------------------------------


def test_write_insights_dedupes_and_returns_new_items_only(store: GraphStore, raw) -> None:
    seed(store)
    first = store.write_insights("m1", Insights(
        decisions=[DecisionDraft("We will launch the beta on March 3.", ["m1:u00002", "m1:u99999"], "S1")],
        action_items=[ActionItemDraft("Send the pricing deck to the board", ["m1:u00003"], "S2", "Friday")],
        topics=["Pricing", "pricing ", "Beta launch"],
        entities=[{"name": "Acme Corp", "kind": "customer"}, {"name": "Stripe", "kind": "weird"}],
        mentions={"m1:u00000": ["Pricing"], "m1:u00003": ["pricing", "Acme Corp"]},
    ))
    assert set(first) == {"decisions", "action_items", "topics"}
    decision = first["decisions"][0]
    assert set(decision) == DECISION_KEYS
    assert decision["status"] == "active" and decision["evidence"] == ["m1:u00002"]  # unknown id dropped
    assert decision["made_at"] == 10.0 and decision["proposed_by"] == "Speaker 1"
    assert decision["supersedes"] is None and decision["superseded_by"] is None
    item = first["action_items"][0]
    assert set(item) == ACTION_KEYS
    assert item["owner"] == "Speaker 2" and item["owner_track"] == "S2" and item["due"] == "Friday"
    assert item["status"] == "open" and item["done_at"] is None and item["meeting_title"] == "Pricing sync"
    assert [t["id"] for t in first["topics"]] == ["pricing", "beta-launch"]
    assert all(set(t) == TOPIC_KEYS for t in first["topics"])
    assert {t["id"]: t["mentions"] for t in first["topics"]} == {"pricing": 2, "beta-launch": 0}
    assert rows(raw, "MATCH (e:Entity) RETURN e.id, e.kind ORDER BY e.id") == [["acme-corp", "customer"],
                                                                              ["stripe", "other"]]
    assert rows(raw, "MATCH (:Utterance {id: 'm1:u00003'})-[:MENTIONS]->(n) RETURN count(n)") == [[2]]

    again = store.write_insights("m1", Insights(
        decisions=[DecisionDraft("we will launch the BETA on march 3", ["m1:u00002"], "S1")],
        action_items=[ActionItemDraft("Send the pricing deck to the board!", ["m1:u00003"], "S2")],
        topics=["PRICING", "Onboarding"],
    ))
    assert again["decisions"] == [] and again["action_items"] == []
    assert [t["id"] for t in again["topics"]] == ["onboarding"]
    assert rows(raw, "MATCH (d:Decision) RETURN count(d)") == [[1]]
    assert rows(raw, "MATCH (a:ActionItem) RETURN count(a)") == [[1]]
    assert store.write_insights("m1", Insights()) == {"decisions": [], "action_items": [], "topics": []}
    with pytest.raises(LookupError):
        store.write_insights("missing", Insights(topics=["x"]))


def test_supersession_across_meetings(store: GraphStore) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Launch review", 2_000, [("S1", "Let's move the beta launch to March 10 instead."),
                                                ("S2", "Agreed, March 10.")])
    old = store.write_insights("m1", Insights(decisions=[
        DecisionDraft("Launch the beta on March 3", ["m1:u00002"]),
        DecisionDraft("Enterprise discount is fifteen percent", ["m1:u00001"]),
    ]))["decisions"]
    old_launch = next(d for d in old if "beta" in d["text"])

    new = store.write_insights("m2", Insights(decisions=[
        DecisionDraft("Launch the beta on March 10", ["m2:u00000"], "S1",
                      supersedes_text="launch the beta on march 3"),
        DecisionDraft("Hire a designer", ["m2:u00001"], supersedes_text="something nobody decided"),
    ]))["decisions"]
    launch = new[0]
    assert launch["supersedes"] == old_launch["id"] and launch["status"] == "active"
    assert new[1]["supersedes"] is None

    by_id = {d["id"]: d for d in store.list_decisions()}
    assert by_id[old_launch["id"]]["status"] == "superseded"
    assert by_id[old_launch["id"]]["superseded_by"] == launch["id"]
    assert sum(d["status"] == "active" for d in by_id.values()) == 3
    # same batch, so newest first falls back to the later meeting time (made_at 5.0 before 0.0)
    assert [d["id"] for d in store.list_decisions("m2")] == [new[1]["id"], launch["id"]]
    assert set(store.list_decisions()[0]) == DECISION_KEYS


def test_action_item_status_toggle_and_filters(store: GraphStore) -> None:
    seed(store)
    store.name_speaker("m1", "S2", "Priya Sharma")
    store.write_insights("m1", Insights(action_items=[
        ActionItemDraft("Send the pricing deck", ["m1:u00003"], "S2", "Friday"),
        ActionItemDraft("Draft onboarding emails", ["m1:u00004"], "S3"),
        ActionItemDraft("Call Acme legal", [], "Dana"),
    ]))
    items = {i["text"]: i for i in store.list_action_items()}
    assert items["Call Acme legal"]["owner"] == "Dana" and items["Call Acme legal"]["owner_track"] is None
    deck = items["Send the pricing deck"]

    done = store.set_action_item_status(deck["id"], "done")
    assert done is not None and done["status"] == "done" and isinstance(done["done_at"], int)
    assert any("$item_id" in c for c in store.last_cypher)
    assert [i["id"] for i in store.list_action_items(status="done")] == [deck["id"]]
    assert len(store.list_action_items(status="open")) == 2
    reopened = store.set_action_item_status(deck["id"], "open")
    assert reopened["status"] == "open" and reopened["done_at"] is None
    assert store.set_action_item_status("nope", "done") is None
    with pytest.raises(ValueError):
        store.set_action_item_status(deck["id"], "finished")
    with pytest.raises(ValueError):
        store.list_action_items(status="finished")

    assert [i["id"] for i in store.list_action_items(owner="priya sharma")] == [deck["id"]]
    assert [i["id"] for i in store.list_action_items(owner="Priya")] == [deck["id"]]  # first name
    assert [i["text"] for i in store.list_action_items(owner="s3")] == ["Draft onboarding emails"]
    assert [i["text"] for i in store.list_action_items(owner="speaker 3")] == ["Draft onboarding emails"]
    assert len(store.list_action_items(meeting_id="m1")) == 3
    assert store.list_action_items(meeting_id="other") == []


# ---- retrieval ---------------------------------------------------------------------------------------------


def test_fulltext_search_shapes_and_sanitising(store: GraphStore) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Pricing follow-up", 2_000, [("S1", "Pricing for startups stays flat.")])
    hits = store.search_fulltext("What did they say about PRICING?!")
    assert hits and all(set(h) == HIT_KEYS for h in hits)
    assert all(h["via"] == "fulltext" for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert {h["utterance_id"] for h in hits} >= {"m1:u00000", "m1:u00003", "m2:u00000"}
    first = next(h for h in hits if h["utterance_id"] == "m1:u00003")
    assert first["meeting_title"] == "Pricing sync" and first["speaker"] == "S2" and first["t_start"] == 15.0

    assert {h["meeting_id"] for h in store.search_fulltext("pricing", meeting_id="m2")} == {"m2"}
    assert len(store.search_fulltext("pricing", k=1)) == 1
    assert {h["utterance_id"] for h in store.search_fulltext("discounts")} == {"m1:u00001"}  # prefix match
    assert store.search_fulltext("") == []
    assert store.search_fulltext("what did they say?") == []  # stopwords only
    assert store.search_fulltext('pricing" OR ((* -_ |') != []  # punctuation never reaches RediSearch

    assert store.last_cypher and "$query" in store.last_cypher[0]
    assert "pricing" not in store.last_cypher[0].lower()


def test_fulltext_falls_back_to_contains_scan(store: GraphStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed(store)
    monkeypatch.setattr(store_module, "fulltext_query", lambda terms: "((")  # RediSearch syntax error
    hits = store.search_fulltext("onboarding emails")
    assert [h["utterance_id"] for h in hits] == ["m1:u00004"]
    assert hits[0]["via"] == "fulltext"
    assert any("CONTAINS" in c for c in store.last_cypher)


def test_vector_search_scores_are_similarities(store: GraphStore) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    store.create_meeting("m2", "Other", "upload", 2_000)
    store.write_utterances([utt("m2", 0, "S1", "unrelated")], [one_hot(2)])

    hits = store.search_vector(one_hot(2), k=3)
    assert hits and all(set(h) == HIT_KEYS for h in hits) and all(h["via"] == "vector" for h in hits)
    assert {h["utterance_id"] for h in hits[:2]} == {"m1:u00002", "m2:u00000"}
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)  # identical direction: 1 - distance 0
    assert hits[-1]["score"] == pytest.approx(0.0, abs=1e-5)  # orthogonal: 1 - distance 1
    assert [h["utterance_id"] for h in store.search_vector(one_hot(2), k=3, meeting_id="m2")] == ["m2:u00000"]
    assert "$vector" in store.last_cypher[0]
    assert store.search_vector([], k=3) == []
    with pytest.raises(ValueError, match="dims"):
        store.search_vector([1.0, 0.0], k=3)


def test_get_utterances_context_recent_and_by_speaker(store: GraphStore) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Latest", 2_000, [("S1", "hello"), ("S2", "world"), ("S1", "again")])
    store.name_speaker("m1", "S2", "Priya Sharma")

    hits = store.get_utterances(["m1:u00003", "missing", "m1:u00001", "m1:u00003"])
    assert [h["utterance_id"] for h in hits] == ["m1:u00003", "m1:u00001"]
    assert all(h["via"] == "graph" and h["score"] == 1.0 for h in hits)
    assert hits[0]["speaker_name"] == "Priya Sharma"
    assert len(store.last_cypher) == 1

    window = store.context_window("m1:u00002", before=1, after=1)
    assert [u["seq"] for u in window] == [1, 2, 3] and all(set(u) == UTTERANCE_KEYS for u in window)
    assert [u["seq"] for u in store.context_window("m1:u00000", before=2, after=0)] == [0]
    assert store.context_window("missing") == []

    recent = store.recent_utterances(limit=2)
    assert [h["utterance_id"] for h in recent] == ["m2:u00001", "m2:u00002"]
    assert all(h["via"] == "recent" for h in recent)
    assert [h["utterance_id"] for h in store.recent_utterances("m1", limit=1)] == ["m1:u00004"]

    by_track = store.utterances_by_speaker("s1", meeting_id="m1")
    assert [h["utterance_id"] for h in by_track] == ["m1:u00000", "m1:u00002"]
    assert [h["utterance_id"] for h in store.utterances_by_speaker("priya")] == ["m1:u00001", "m1:u00003"]
    assert [h["utterance_id"] for h in store.utterances_by_speaker("Speaker 3")] == ["m1:u00004"]
    across = store.utterances_by_speaker("S1")
    assert [h["meeting_id"] for h in across] == ["m2", "m2", "m1", "m1"]  # newest meeting first
    assert store.utterances_by_speaker("  ") == []


def test_who_knows_groups_by_person(store: GraphStore) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Pricing again", 2_000, [("S1", "Pricing pages need a refresh."),
                                                ("S2", "Pricing for students too.")])
    store.name_speaker("m1", "S2", "Priya")
    store.name_speaker("m2", "S2", "Priya")
    ranked = store.who_knows("pricing")
    assert ranked[0]["name"] == "Priya" and ranked[0]["person_id"] == "priya" and ranked[0]["hits"] == 2
    assert set(ranked[0]) == {"name", "person_id", "meeting_id", "hits", "sample"}
    assert set(ranked[0]["sample"]) == UTTERANCE_KEYS
    assert {r["name"] for r in ranked} == {"Priya", "Speaker 1"}
    assert sum(r["hits"] for r in ranked) == 4
    assert all(r["person_id"] is None for r in ranked if r["name"] == "Speaker 1")
    assert store.who_knows("the") == []
    assert len(store.who_knows("pricing", limit=1)) == 1


def test_subgraph_views(store: GraphStore) -> None:
    seed(store)
    store.name_speaker("m1", "S2", "Priya")
    store.write_insights("m1", Insights(
        decisions=[DecisionDraft("Launch the beta on March 3", ["m1:u00002"], "S1")],
        action_items=[ActionItemDraft("Send the pricing deck", ["m1:u00003"], "S2")],
        topics=["Pricing"], entities=[{"name": "Pricing", "kind": "product"}],
        mentions={"m1:u00003": ["Pricing"]},
    ))

    full = store.subgraph(meeting_id="m1")
    kinds = {n["kind"] for n in full["nodes"]}
    assert kinds == {"Meeting", "Speaker", "Person", "Utterance", "Decision", "ActionItem", "Topic", "Entity"}
    ids = [n["id"] for n in full["nodes"]]
    assert len(ids) == len(set(ids))
    assert {"topic:pricing", "entity:pricing", "person:priya", "m1", "m1:S2"} <= set(ids)
    assert all(set(n) >= {"id", "label", "kind"} and len(n["label"]) <= 60 for n in full["nodes"])
    keys = [(link["source"], link["target"], link["type"]) for link in full["links"]]
    assert len(keys) == len(set(keys))
    assert all(link["source"] in ids and link["target"] in ids for link in full["links"])
    types = {link["type"] for link in full["links"]}
    assert {"HAS_UTTERANCE", "HAS_SPEAKER", "SPOKEN_BY", "NEXT", "IS", "DECIDED_IN", "EVIDENCED_BY",
            "PROPOSED_BY", "RAISED_IN", "ASSIGNED_TO", "ABOUT", "MENTIONS"} <= types

    capped = store.subgraph(meeting_id="m1", limit=6)
    assert len(capped["nodes"]) == 6 and all(n["kind"] != "Utterance" for n in capped["nodes"])

    focus = store.subgraph(utterance_ids=["m1:u00003"])
    focus_ids = {n["id"] for n in focus["nodes"]}
    assert {"m1:u00003", "m1:S2", "m1", "person:priya", "topic:pricing", "entity:pricing"} <= focus_ids
    assert any(n["kind"] == "ActionItem" for n in focus["nodes"])
    assert ("m1:S2", "person:priya", "IS") in {(lk["source"], lk["target"], lk["type"]) for lk in focus["links"]}

    overview = store.subgraph()
    assert overview["nodes"] and all(n["kind"] != "Utterance" for n in overview["nodes"])
    assert store.subgraph(meeting_id="missing") == {"nodes": [], "links": []}


# ---- memory, stats, deletion ----------------------------------------------------------------------------


def test_turns_round_trip(store: GraphStore) -> None:
    store.save_turn("sess", "user", "What did Priya commit to?", 1_000)
    store.save_turn("sess", "assistant", "She will send the deck.", 1_000, ["m1:u00003"])
    store.save_turn("sess", "user", "By when?", 2_000)
    store.save_turn("other", "user", "Unrelated", 500)
    turns = store.get_turns("sess")
    assert [t["text"] for t in turns] == ["What did Priya commit to?", "She will send the deck.", "By when?"]
    assert turns[1] == {"role": "assistant", "text": "She will send the deck.", "at": 1_000,
                        "citations": ["m1:u00003"]}
    assert turns[0]["citations"] == []
    assert [t["text"] for t in store.get_turns("sess", limit=2)] == ["She will send the deck.", "By when?"]
    assert store.get_turns("nobody") == []


def test_stats_counts_labels_and_relationships(store: GraphStore) -> None:
    empty = store.stats()
    assert empty == {label: 0 for label in (*store_module.cq.LABELS, "relationships")}
    seed(store)
    counts = store.stats()
    assert counts["Meeting"] == 1 and counts["Utterance"] == 5 and counts["Speaker"] == 3
    # 5 HAS_UTTERANCE + 5 SPOKEN_BY + 3 HAS_SPEAKER + 4 NEXT
    assert counts["relationships"] == 17


def test_delete_meeting_cleans_up_and_reactivates(store: GraphStore, raw) -> None:
    seed(store, "m1", "Pricing sync", 1_000)
    seed(store, "m2", "Launch review", 2_000, [("S1", "Move the beta launch to March 10.")])
    old = store.write_insights("m1", Insights(decisions=[DecisionDraft("Launch the beta on March 3", [])],
                                              topics=["Pricing"]))["decisions"][0]
    store.write_insights("m2", Insights(
        decisions=[DecisionDraft("Launch the beta on March 10", ["m2:u00000"], supersedes_text="beta March 3")],
        action_items=[ActionItemDraft("Update the launch plan", ["m2:u00000"], "S1")],
        topics=["Launch"]))
    store.name_speaker("m2", "S1", "Solo Person")
    assert store.list_decisions("m1")[0]["status"] == "superseded"

    assert store.delete_meeting("m2") is True
    assert store.get_meeting("m2") is None
    assert rows(raw, "MATCH (n) WHERE n.meeting_id = 'm2' RETURN count(n)") == [[0]]
    assert store.list_decisions("m1")[0] == {**old, "status": "active", "superseded_by": None}
    assert rows(raw, "MATCH (t:Topic) RETURN t.id") == [["pricing"]]
    assert rows(raw, "MATCH (p:Person) RETURN count(p)") == [[0]]
    assert store.delete_meeting("m2") is False
    assert store.get_meeting("m1")["meeting"]["utterance_count"] == 5


def test_concurrent_writers_from_threads(store: GraphStore) -> None:
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            mid = f"t{n}"
            store.create_meeting(mid, f"Thread {n}", "upload", 1_000 + n)
            for seq in range(10):
                store.write_utterances([utt(mid, seq, f"S{seq % 3 + 1}", f"thread {n} line {seq}")], [one_hot(seq)])
                store.search_fulltext("thread line")
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(m["utterance_count"] == 10 for m in store.list_meetings())
