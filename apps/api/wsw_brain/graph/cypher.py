"""Every Cypher statement GraphStore runs. Values always travel as $params; only labels, relationship types
and small validated integers (index dimension, path bounds) are formatted into the text.

Query rules (docs/TECH_NOTES.md §2.1): a WITH barrier between consecutive MATCH clauses, and inline property
maps after UNWIND, so the statements stay correct on the C engine (4.x) and the Rust engine (6.0.x).
"""

from __future__ import annotations

LABELS = ("Meeting", "Speaker", "Person", "Utterance", "Topic", "Decision", "ActionItem", "Entity", "Turn")

# ---------------------------------------------------------------------------------------------- schema

_RANGE_INDEXES = [(label, "id") for label in LABELS] + [
    ("Utterance", "t_start"),
    ("Utterance", "meeting_id"),
    ("Speaker", "meeting_id"),
    ("Decision", "meeting_id"),
    ("Decision", "status"),
    ("ActionItem", "meeting_id"),
    ("ActionItem", "status"),
    ("Turn", "session_id"),
]


def schema_statements() -> list[str]:
    """Range and full-text index statements (the vector index needs the embedding dimension)."""
    ranges = [f"CREATE INDEX FOR (n:{label}) ON (n.{prop})" for label, prop in _RANGE_INDEXES]
    return ranges + ["CREATE FULLTEXT INDEX FOR (u:Utterance) ON (u.text)"]


def vector_index(dim: int) -> str:
    return (
        "CREATE VECTOR INDEX FOR (u:Utterance) ON (u.embedding) "
        f"OPTIONS {{dimension: {int(dim)}, similarityFunction: 'cosine'}}"
    )


LIST_INDEXES = "CALL db.indexes() YIELD label, properties, types, options RETURN label, properties, types, options"


def stats() -> str:
    per_label = [f"MATCH (n:{label}) RETURN '{label}' AS label, count(n) AS n" for label in LABELS]
    edges = "MATCH ()-[r]->() RETURN 'relationships' AS label, count(r) AS n"
    return "\nUNION ALL\n".join(per_label + [edges])


# -------------------------------------------------------------------------------------------- meetings

CREATE_MEETING = """
MERGE (m:Meeting {id: $meeting_id})
ON CREATE SET m.status = 'live', m.has_audio = false, m.duration_s = 0.0
SET m.title = $title, m.source = $source, m.started_at = $started_at
"""

END_MEETING = """
MATCH (m:Meeting {id: $meeting_id})
SET m.status = 'ended', m.ended_at = $ended_at, m.duration_s = $duration_s
RETURN m.id AS id
"""

SET_MEETING_SUMMARY = "MATCH (m:Meeting {id: $meeting_id}) SET m.summary = $summary"

SET_MEETING_HAS_AUDIO = "MATCH (m:Meeting {id: $meeting_id}) SET m.has_audio = $has_audio"

LATEST_MEETING = "MATCH (m:Meeting) RETURN m.id AS id ORDER BY m.started_at DESC LIMIT 1"

ALL_MEETINGS = "MATCH (m:Meeting)"
ONE_MEETING = "MATCH (m:Meeting {id: $meeting_id})"


def meeting_summaries(match: str) -> str:
    """MeetingSummary rows; `match` binds m."""
    return f"""
{match}
WITH m
OPTIONAL MATCH (m)-[:HAS_UTTERANCE]->(u:Utterance)
WITH m, count(u) AS utterance_count, max(u.t_end) AS last_t
OPTIONAL MATCH (m)-[:HAS_SPEAKER]->(s:Speaker)
WITH m, utterance_count, last_t, count(s) AS speaker_count
RETURN m.id AS id, m.title AS title, m.status AS status, m.source AS source, m.started_at AS started_at,
       m.ended_at AS ended_at, m.duration_s AS duration_s, m.has_audio AS has_audio, m.summary AS summary,
       utterance_count, speaker_count, last_t
ORDER BY started_at DESC
"""


def delete_by_meeting(label: str) -> str:
    return f"MATCH (n:{label} {{meeting_id: $meeting_id}}) DETACH DELETE n"


DELETE_MEETING = """
MATCH (m:Meeting {id: $meeting_id})
WITH m, m.id AS id
DETACH DELETE m
RETURN count(id) AS deleted
"""

REACTIVATE_UNSUPERSEDED = """
MATCH (o:Decision {status: 'superseded'})
WITH o
OPTIONAL MATCH (n:Decision)-[:SUPERSEDES]->(o)
WITH o, count(n) AS superseders
WHERE superseders = 0
SET o.status = 'active'
"""


def delete_orphans(label: str) -> str:
    """Remove global nodes (Topic, Entity, Person) nothing points at any more."""
    return f"""
MATCH (n:{label})
WITH n
OPTIONAL MATCH (n)<-[r]-()
WITH n, count(r) AS refs
WHERE refs = 0
DETACH DELETE n
"""


# -------------------------------------------------------------------------------------------- speakers

MEETING_SPEAKERS = "MATCH (s:Speaker {meeting_id: $meeting_id})"
ONE_SPEAKER = "MATCH (s:Speaker {id: $speaker_id})"


def speakers(match: str) -> str:
    """Speaker rows; `match` binds s."""
    return f"""
{match}
WITH s
OPTIONAL MATCH (s)-[:IS]->(p:Person)
WITH s, head(collect(p.id)) AS person_id
OPTIONAL MATCH (u:Utterance)-[:SPOKEN_BY]->(s)
RETURN s.id AS id, s.track AS track, s.label AS label, s.name AS name, person_id, s.color_idx AS color_idx,
       count(u) AS utterance_count, sum(u.t_end - u.t_start) AS talk_time_s
"""


ENSURE_SPEAKER = """
MERGE (m:Meeting {id: $meeting_id})
MERGE (s:Speaker {id: $speaker_id})
ON CREATE SET s.meeting_id = $meeting_id, s.track = $track, s.label = $label, s.color_idx = $color_idx
MERGE (m)-[:HAS_SPEAKER]->(s)
"""

MOVE_UTTERANCES = """
MATCH (src:Speaker {id: $src_id})
WITH src
MATCH (dst:Speaker {id: $dst_id})
WITH src, dst
MATCH (u:Utterance)-[r:SPOKEN_BY]->(src)
DELETE r
MERGE (u)-[:SPOKEN_BY]->(dst)
SET u.speaker = $dst_track
RETURN count(u) AS moved
"""

MOVE_ASSIGNMENTS = """
MATCH (a:ActionItem {meeting_id: $meeting_id, owner_track: $src_track})
SET a.owner_track = $dst_track
WITH a
OPTIONAL MATCH (a)-[r:ASSIGNED_TO]->(:Speaker {id: $src_id})
DELETE r
WITH DISTINCT a
MATCH (dst:Speaker {id: $dst_id})
MERGE (a)-[:ASSIGNED_TO]->(dst)
SET a.owner_name = coalesce(dst.name, a.owner_name)
"""

MOVE_PROPOSALS = """
MATCH (d:Decision)-[r:PROPOSED_BY]->(:Speaker {id: $src_id})
DELETE r
WITH DISTINCT d
MATCH (dst:Speaker {id: $dst_id})
MERGE (d)-[:PROPOSED_BY]->(dst)
"""

DELETE_SPEAKER = "MATCH (s:Speaker {id: $speaker_id}) DETACH DELETE s"

NAME_SPEAKER = """
MATCH (s:Speaker {id: $speaker_id})
SET s.name = $name
WITH s
OPTIONAL MATCH (s)-[r:IS]->(:Person)
DELETE r
WITH DISTINCT s
MERGE (p:Person {id: $person_id})
ON CREATE SET p.name = $name
MERGE (s)-[:IS]->(p)
RETURN s.id AS id
"""

RENAME_TRACK_OWNERS = """
MATCH (a:ActionItem {meeting_id: $meeting_id, owner_track: $track})
SET a.owner_name = $name
"""

ASSIGN_NAMED_OWNERS = """
MATCH (a:ActionItem {meeting_id: $meeting_id})
WHERE a.owner_track IS NULL AND toLower(a.owner_name) = $lowered_name
SET a.owner_track = $track
WITH a
MATCH (s:Speaker {id: $speaker_id})
MERGE (a)-[:ASSIGNED_TO]->(s)
"""

# ---------------------------------------------------------------------------------------- utterances


def write_utterances(with_embeddings: bool) -> str:
    """One UNWIND batch: Meeting/Speaker/Utterance + HAS_SPEAKER, HAS_UTTERANCE, SPOKEN_BY. A rewritten
    utterance that changed speaker loses its stale SPOKEN_BY edge."""
    embedding = (
        "FOREACH (e IN CASE WHEN r.embedding IS NULL THEN [] ELSE [r.embedding] END |"
        " SET u.embedding = vecf32(e))\n"
        if with_embeddings
        else ""
    )
    return f"""
UNWIND $rows AS r
MERGE (m:Meeting {{id: r.meeting_id}})
ON CREATE SET m.title = 'Untitled meeting', m.status = 'live', m.source = 'mic', m.started_at = $now,
              m.has_audio = false, m.duration_s = 0.0
MERGE (s:Speaker {{id: r.speaker_id}})
ON CREATE SET s.meeting_id = r.meeting_id, s.track = r.speaker, s.label = r.label, s.color_idx = r.color_idx
MERGE (u:Utterance {{id: r.id}})
SET u.meeting_id = r.meeting_id, u.seq = r.seq, u.speaker = r.speaker, u.t_start = r.t_start,
    u.t_end = r.t_end, u.text = r.text, u.confidence = r.confidence
MERGE (m)-[:HAS_SPEAKER]->(s)
MERGE (m)-[:HAS_UTTERANCE]->(u)
MERGE (u)-[:SPOKEN_BY]->(s)
{embedding}WITH u, s
OPTIONAL MATCH (u)-[old:SPOKEN_BY]->(other:Speaker)
WHERE other.id <> s.id
DELETE old
"""


LINK_NEXT = """
UNWIND $links AS l
MATCH (a:Utterance {id: l.prev})
WITH a, l
MATCH (b:Utterance {id: l.next})
MERGE (a)-[:NEXT]->(b)
"""

_UTTERANCE_TAIL = """
WITH u
OPTIONAL MATCH (u)-[:SPOKEN_BY]->(s:Speaker)
RETURN u.id AS id, u.meeting_id AS meeting_id, u.seq AS seq, u.speaker AS speaker, s.name AS speaker_name,
       u.t_start AS t_start, u.t_end AS t_end, u.text AS text, u.confidence AS confidence
ORDER BY seq
"""

UTTERANCES_AFTER = """
MATCH (u:Utterance {meeting_id: $meeting_id})
WHERE u.seq > $after_seq
""" + _UTTERANCE_TAIL


def context_window(before: int, after: int) -> str:
    """Neighbours of $utterance_id along NEXT (bounds are validated ints; paths cannot take $params)."""
    return f"""
MATCH (c:Utterance {{id: $utterance_id}})
WITH c
OPTIONAL MATCH (b:Utterance)-[:NEXT*0..{int(before)}]->(c)
WITH c, collect(b) AS earlier
OPTIONAL MATCH (c)-[:NEXT*0..{int(after)}]->(a:Utterance)
WITH earlier, collect(a) AS later
UNWIND earlier + later AS u
WITH DISTINCT u
""" + _UTTERANCE_TAIL


# ------------------------------------------------------------------------------------------ retrieval

_HIT_TAIL = """
WITH u, score
OPTIONAL MATCH (u)-[:SPOKEN_BY]->(s:Speaker)
WITH u, score, s
OPTIONAL MATCH (m:Meeting)-[:HAS_UTTERANCE]->(u)
RETURN u.id AS utterance_id, u.meeting_id AS meeting_id, m.title AS meeting_title, u.speaker AS speaker,
       s.name AS speaker_name, u.t_start AS t_start, u.t_end AS t_end, u.text AS text, score,
       u.seq AS _seq, m.started_at AS _started
"""

_WHO_TAIL = """
WITH u, score
OPTIONAL MATCH (u)-[:SPOKEN_BY]->(s:Speaker)
WITH u, score, s
OPTIONAL MATCH (s)-[:IS]->(p:Person)
RETURN u.id AS id, u.meeting_id AS meeting_id, u.seq AS seq, u.speaker AS speaker, s.name AS speaker_name,
       u.t_start AS t_start, u.t_end AS t_end, u.text AS text, u.confidence AS confidence, score,
       s.id AS speaker_id, s.label AS speaker_label, p.id AS person_id, p.name AS person_name
"""


def _fulltext_source(by_meeting: bool) -> str:
    where = "WHERE u.meeting_id = $meeting_id\n" if by_meeting else ""
    return (
        "CALL db.idx.fulltext.queryNodes('Utterance', $query) YIELD node, score\n"
        "WITH node AS u, score\n"
        f"{where}"
        "WITH u, score ORDER BY score DESC LIMIT $k"
    )


def _scan_source(by_meeting: bool) -> str:
    match = "MATCH (u:Utterance {meeting_id: $meeting_id})" if by_meeting else "MATCH (u:Utterance)"
    return (
        f"{match}\n"
        "WITH u LIMIT $scan_limit\n"
        "WITH u, size([t IN $terms WHERE toLower(u.text) CONTAINS t]) AS matched\n"
        "WHERE matched > 0\n"
        "WITH u, toFloat(matched) AS score ORDER BY score DESC LIMIT $k"
    )


def fulltext_hits(by_meeting: bool) -> str:
    return _fulltext_source(by_meeting) + _HIT_TAIL


def scan_hits(by_meeting: bool) -> str:
    """Fallback when RediSearch rejects the query: substring scan over a bounded number of utterances."""
    return _scan_source(by_meeting) + _HIT_TAIL


def fulltext_speakers(by_meeting: bool) -> str:
    return _fulltext_source(by_meeting) + _WHO_TAIL


def scan_speakers(by_meeting: bool) -> str:
    return _scan_source(by_meeting) + _WHO_TAIL


def vector_hits(by_meeting: bool) -> str:
    where = "WHERE u.meeting_id = $meeting_id\n" if by_meeting else ""
    return (
        "CALL db.idx.vector.queryNodes('Utterance', 'embedding', $knn, vecf32($vector)) YIELD node, score\n"
        "WITH node AS u, 1.0 - score AS score\n"
        f"{where}"
        "WITH u, score ORDER BY score DESC LIMIT $k" + _HIT_TAIL
    )


HITS_BY_IDS = """
UNWIND $ids AS uid
MATCH (u:Utterance {id: uid})
WITH u, 1.0 AS score
""" + _HIT_TAIL

RECENT_HITS = """
MATCH (u:Utterance {meeting_id: $meeting_id})
WITH u ORDER BY u.seq DESC LIMIT $limit
WITH u, 1.0 AS score
""" + _HIT_TAIL


def speaker_hits(by_meeting: bool) -> str:
    match = "MATCH (s:Speaker {meeting_id: $meeting_id})" if by_meeting else "MATCH (s:Speaker)"
    return f"""
{match}
WITH s
OPTIONAL MATCH (s)-[:IS]->(p:Person)
WITH s, p
WHERE toLower(s.track) = $who OR toLower(s.label) = $who
   OR toLower(coalesce(s.name, '')) = $who OR toLower(coalesce(p.name, '')) = $who OR p.id = $who_slug
   OR toLower(coalesce(s.name, '')) STARTS WITH $who_prefix
   OR toLower(coalesce(p.name, '')) STARTS WITH $who_prefix
WITH DISTINCT s
MATCH (u:Utterance)-[:SPOKEN_BY]->(s)
WITH u
OPTIONAL MATCH (m:Meeting)-[:HAS_UTTERANCE]->(u)
WITH u, m ORDER BY m.started_at DESC, u.seq ASC LIMIT $limit
WITH u, 1.0 AS score
""" + _HIT_TAIL


# ------------------------------------------------------------------------------------------- insights


def existing_ids(label: str) -> str:
    return f"UNWIND $ids AS i MATCH (n:{label} {{id: i}}) RETURN n.id AS id"


EVIDENCE_TIMES = "UNWIND $ids AS i MATCH (u:Utterance {id: i}) RETURN u.id AS id, u.t_start AS t_start"

PERSONS_BY_IDS = "UNWIND $ids AS i MATCH (p:Person {id: i}) RETURN p.id AS id, p.name AS name"

MEETING_LAST_T = "MATCH (u:Utterance {meeting_id: $meeting_id}) RETURN max(u.t_end) AS last_t"

ACTIVE_DECISIONS = """
MATCH (d:Decision {status: 'active'})
RETURN d.id AS id, d.text AS text, d.created_at AS created_at
ORDER BY created_at DESC
"""

MEETING_TOPIC_IDS = """
MATCH (m:Meeting {id: $meeting_id})
WITH m
MATCH (m)-[:ABOUT]->(t:Topic)
RETURN t.id AS id
"""

UPSERT_DECISIONS = """
UNWIND $rows AS r
MERGE (d:Decision {id: r.id})
ON CREATE SET d.meeting_id = r.meeting_id, d.text = r.text, d.status = 'active', d.made_at = r.made_at,
              d.created_at = r.created_at, d.proposed_by_name = r.proposed_by_name
WITH d, r
MATCH (m:Meeting {id: r.meeting_id})
MERGE (d)-[:DECIDED_IN]->(m)
"""

UPSERT_ACTION_ITEMS = """
UNWIND $rows AS r
MERGE (a:ActionItem {id: r.id})
ON CREATE SET a.meeting_id = r.meeting_id, a.text = r.text, a.status = 'open', a.created_at = r.created_at,
              a.owner_name = r.owner_name, a.owner_track = r.owner_track, a.due = r.due
ON MATCH SET a.due = coalesce(a.due, r.due)
WITH a, r
MATCH (m:Meeting {id: r.meeting_id})
MERGE (a)-[:RAISED_IN]->(m)
"""

UPSERT_TOPICS = """
MATCH (m:Meeting {id: $meeting_id})
WITH m
UNWIND $rows AS r
MERGE (t:Topic {id: r.id})
ON CREATE SET t.name = r.name
MERGE (m)-[:ABOUT]->(t)
"""

UPSERT_ENTITIES = """
UNWIND $rows AS r
MERGE (e:Entity {id: r.id})
ON CREATE SET e.name = r.name, e.kind = r.kind
"""

SUPERSEDE = """
UNWIND $pairs AS p
MATCH (old:Decision {id: p.dst})
SET old.status = 'superseded'
WITH old, p
MATCH (new:Decision {id: p.src})
MERGE (new)-[:SUPERSEDES]->(old)
"""


def link(src_label: str, rel: str, dst_label: str) -> str:
    """MERGE (src)-[:rel]->(dst) for each {src, dst} id pair in $pairs; missing endpoints are skipped."""
    return f"""
UNWIND $pairs AS p
MATCH (a:{src_label} {{id: p.src}})
WITH a, p
MATCH (b:{dst_label} {{id: p.dst}})
MERGE (a)-[:{rel}]->(b)
"""


DECISIONS_BY_MEETING = "MATCH (d:Decision {meeting_id: $meeting_id})"
ALL_DECISIONS = "MATCH (d:Decision)"
DECISIONS_BY_IDS = "MATCH (d:Decision) WHERE d.id IN $ids"


def decisions(match: str) -> str:
    """Decision rows, newest first; `match` binds d."""
    return f"""
{match}
WITH d
OPTIONAL MATCH (d)-[:EVIDENCED_BY]->(u:Utterance)
WITH d, collect(u.id) AS evidence
OPTIONAL MATCH (d)-[:PROPOSED_BY]->(s:Speaker)
WITH d, evidence, head(collect(coalesce(s.name, s.label))) AS proposer
OPTIONAL MATCH (n:Decision)-[:SUPERSEDES]->(d)
WITH d, evidence, proposer, head(collect(n.id)) AS superseded_by
OPTIONAL MATCH (d)-[:SUPERSEDES]->(o:Decision)
WITH d, evidence, proposer, superseded_by, head(collect(o.id)) AS supersedes
RETURN d.id AS id, d.meeting_id AS meeting_id, d.text AS text, d.status AS status, d.made_at AS made_at,
       coalesce(proposer, d.proposed_by_name) AS proposed_by, evidence, superseded_by, supersedes,
       d.created_at AS created_at
ORDER BY created_at DESC, made_at DESC
"""


ACTION_ITEMS_BY_IDS = "MATCH (a:ActionItem) WHERE a.id IN $ids"


def action_items_match(status: bool, meeting: bool) -> str:
    props = [p for p, on in (("status: $status", status), ("meeting_id: $meeting_id", meeting)) if on]
    return "MATCH (a:ActionItem {" + ", ".join(props) + "})" if props else "MATCH (a:ActionItem)"


def action_items(match: str, by_owner: bool = False) -> str:
    """ActionItem rows, newest first; `match` binds a. Owner = stored name, else the assignee's name/label."""
    owner_filter = (
        "WHERE toLower(coalesce(owner, '')) = $owner OR toLower(coalesce(a.owner_track, '')) = $owner\n"
        "   OR toLower(coalesce(owner, '')) STARTS WITH $owner_prefix\n"
        if by_owner
        else ""
    )
    return f"""
{match}
WITH a
OPTIONAL MATCH (a)-[:RAISED_IN]->(m:Meeting)
WITH a, head(collect(m.title)) AS meeting_title
OPTIONAL MATCH (a)-[:ASSIGNED_TO]->(s:Speaker)
WITH a, meeting_title, head(collect(coalesce(s.name, s.label))) AS assignee
WITH a, meeting_title, coalesce(a.owner_name, assignee) AS owner
{owner_filter}OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(u:Utterance)
WITH a, meeting_title, owner, collect(u.id) AS evidence
RETURN a.id AS id, a.meeting_id AS meeting_id, meeting_title, a.text AS text, owner,
       a.owner_track AS owner_track, a.due AS due, a.status AS status, a.created_at AS created_at,
       a.done_at AS done_at, evidence
ORDER BY created_at DESC
"""


SET_ACTION_STATUS = """
MATCH (a:ActionItem {id: $item_id})
SET a.done_at = CASE WHEN $status = 'done' THEN coalesce(a.done_at, $now) ELSE null END, a.status = $status
RETURN a.id AS id
"""

MEETING_TOPICS = """
MATCH (m:Meeting {id: $meeting_id})
WITH m
MATCH (m)-[:ABOUT]->(t:Topic)
"""

MEETING_TOPICS_BY_IDS = MEETING_TOPICS.rstrip() + "\nWHERE t.id IN $ids\n"


def topics(match: str) -> str:
    """Topic rows with mention counts inside $meeting_id; `match` binds t."""
    return f"""
{match}
WITH t
OPTIONAL MATCH (u:Utterance {{meeting_id: $meeting_id}})-[:MENTIONS]->(t)
RETURN t.id AS id, t.name AS name, count(u) AS mentions
ORDER BY mentions DESC, name
"""


# --------------------------------------------------------------------------------------------- people

PEOPLE = """
MATCH (p:Person)
WITH p
MATCH (s:Speaker)-[:IS]->(p)
WITH p, s
OPTIONAL MATCH (u:Utterance)-[:SPOKEN_BY]->(s)
WITH p, collect(DISTINCT s.meeting_id) AS meetings, count(u) AS utterances,
     sum(u.t_end - u.t_start) AS talk_time_s
OPTIONAL MATCH (a:ActionItem {status: 'open'})-[:ASSIGNED_TO]->(:Speaker)-[:IS]->(p)
WITH p, meetings, utterances, talk_time_s, collect(DISTINCT a.id) AS assigned_open
RETURN p.id AS id, p.name AS name, size(meetings) AS meetings, utterances, talk_time_s, assigned_open
"""

OPEN_ITEMS_BY_NAME = """
MATCH (a:ActionItem {status: 'open'})
WHERE a.owner_track IS NULL AND a.owner_name IS NOT NULL
RETURN a.id AS id, a.owner_name AS owner_name
"""

# --------------------------------------------------------------------------------------------- memory

SAVE_TURN = """
OPTIONAL MATCH (x:Turn {session_id: $session_id})
WITH count(x) AS n
MERGE (t:Turn {id: $turn_id})
ON CREATE SET t.session_id = $session_id, t.role = $role, t.text = $text, t.at = $at,
              t.citations = $citations, t.n = n
"""

GET_TURNS = """
MATCH (t:Turn {session_id: $session_id})
RETURN t.role AS role, t.text AS text, t.at AS at, t.citations AS citations, t.n AS n
ORDER BY n DESC LIMIT $limit
"""

# ------------------------------------------------------------------------------------------- subgraph

_NODE_COLUMNS = (
    "id(n) AS nid, n.id AS id, labels(n)[0] AS kind, "
    "coalesce(n.title, n.name, n.text, n.label, n.id) AS label, n.status AS status, "
    "n.meeting_id AS meeting_id, coalesce(n.track, n.speaker) AS track, n.color_idx AS color_idx, "
    "n.t_start AS t_start"
)


def _nodes(*matches: str) -> str:
    return "\nUNION\n".join(f"{m}\nRETURN {_NODE_COLUMNS}" for m in matches)


SUBGRAPH_MEETING_CONTEXT = _nodes(
    "MATCH (n:Meeting {id: $meeting_id})",
    "MATCH (m:Meeting {id: $meeting_id})\nWITH m\nMATCH (m)-[:HAS_SPEAKER|ABOUT|DECIDED_IN|RAISED_IN]-(n)",
    "MATCH (s:Speaker {meeting_id: $meeting_id})\nWITH s\nMATCH (s)-[:IS]->(n:Person)",
    "MATCH (d:Decision {meeting_id: $meeting_id})\nWITH d\nMATCH (d)-[:SUPERSEDES]-(n:Decision)",
    "MATCH (u:Utterance {meeting_id: $meeting_id})\nWITH u\nMATCH (u)-[:MENTIONS]->(n)",
)

SUBGRAPH_MEETING_UTTERANCES = _nodes("MATCH (n:Utterance {meeting_id: $meeting_id})\nWITH n ORDER BY n.seq LIMIT $limit")

SUBGRAPH_UTTERANCES = _nodes(
    "UNWIND $ids AS uid\nMATCH (n:Utterance {id: uid})",
    "UNWIND $ids AS uid\nMATCH (u:Utterance {id: uid})\nWITH u\n"
    "MATCH (u)-[:SPOKEN_BY|MENTIONS|EVIDENCED_BY|HAS_UTTERANCE]-(n)",
    "UNWIND $ids AS uid\nMATCH (u:Utterance {id: uid})\nWITH u\nMATCH (u)-[:SPOKEN_BY]->(:Speaker)-[:IS]->(n:Person)",
)

OVERVIEW_LABELS = ("Meeting", "Person", "Speaker", "Topic", "Decision", "ActionItem", "Entity")

SUBGRAPH_OVERVIEW = "\nUNION ALL\n".join(
    f"MATCH (n:{label})\nWITH n "
    + ("ORDER BY n.started_at DESC " if label == "Meeting" else "")
    + f"LIMIT $limit\nRETURN {_NODE_COLUMNS}"
    for label in OVERVIEW_LABELS
)

SUBGRAPH_EDGES = """
UNWIND $nids AS nid
MATCH (a) WHERE id(a) = nid
WITH a
MATCH (a)-[r]->(b)
WHERE id(b) IN $nids
RETURN id(a) AS source, type(r) AS type, id(b) AS target
"""
