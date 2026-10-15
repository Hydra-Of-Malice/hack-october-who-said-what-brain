"""Live extraction: Gemma 4 reads a window of speaker-attributed utterances and returns decisions,
action items, topics and entities, each tied to evidence utterance ids that exist in the window."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..contracts import LLM, ActionItemDraft, DecisionDraft, Insights
from .llm import LLMError, get_extraction_llm, llm_priority

log = logging.getLogger(__name__)

MAX_DECISIONS = 5
MAX_ACTION_ITEMS = 8
MAX_TOPICS = 8
MAX_ENTITIES = 10
ENTITY_KINDS = {"product", "customer", "vendor", "org", "person", "other"}

SYSTEM_PROMPT = """You extract structured meeting notes from a transcript window. Reply with STRICT JSON only:
{"decisions": [{"text": str, "evidence": [utterance ids], "proposed_by": speaker track or null,
                "supersedes": "exact text of an earlier decision this replaces" or null}],
 "action_items": [{"text": str, "owner": speaker track or person name or null, "due": str or null,
                   "evidence": [utterance ids]}],
 "topics": [short noun phrases],
 "entities": [{"name": str, "kind": "product"|"customer"|"vendor"|"org"|"other"}],
 "mentions": {"<utterance id>": [topic or entity names]}}
Rules:
- A decision is something the group settled on: stated as final ("we'll go with", "agreed", "locked",
  "final") or accepted without objection. A proposal, preference or opinion ("I want", "I'd suggest",
  "what if") is NOT a decision until someone agrees to it.
- An action item is a concrete task someone committed to ("I'll…", "I will…") or accepted when asked
  ("can you…?" "yes"). Wishes and positions ("Meera wants…", "Rahul thinks…") are NOT action items.
- Only set "supersedes" when this window explicitly changes an earlier decision to something different. A
  decision that "still holds", "stays" or "remains" the same is a restatement: leave "supersedes" null.
- Refer to people by name in the text; never write speaker tracks like "S2" inside "text".
- Write each item as one short sentence in plain words. Owners and proposers use the speaker track (S1, S2...)
  when the person is in the speaker list, otherwise the name as spoken.
- "due" is the deadline as spoken ("Friday", "next week"), or null.
- Evidence ids must be copied exactly from the [brackets] at the start of transcript lines.
- Only include what the transcript actually says. Empty lists are fine.
- At most 5 decisions and 8 action items."""


# A decision that keeps something as it was is not a change ("The launch date still holds").
_REAFFIRM = re.compile(r"\b(still|stays|stay|remains?|unchanged|holds|reaffirm\w*|as before|as agreed)\b", re.IGNORECASE)
# Positions and wishes are not tasks ("Meera wants the starter tier at 3,999").
_WISH = re.compile(r"\b(wants?|would like|prefers?|thinks?|believes?|feels?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _speaker_table(speakers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["track"]: s for s in speakers if s.get("track")}


def speaker_display(track: str, table: dict[str, dict[str, Any]], name: str | None = None) -> str:
    speaker = table.get(track) or {}
    if name or speaker.get("name"):
        return name or speaker["name"]
    if speaker.get("label"):
        return speaker["label"]
    match = re.fullmatch(r"S(\d+)", track or "")
    return f"Speaker {match.group(1)}" if match else (track or "Unknown")


_TRACK_TOKEN = re.compile(r"\bS(\d+)\b")
_WORDS_AFTER_PERSON = frozenset({
    "will", "would", "should", "can", "could", "must", "said", "says", "wants", "agreed", "agrees",
    "proposed", "proposes", "suggested", "suggests", "owns", "promised", "offered", "volunteered", "decided",
})  # fmt: skip
_WORDS_BEFORE_PERSON = frozenset({"ask", "asks", "asked", "tell", "tells", "told", "remind", "ping", "cc"})


def speaker_tracks_to_names(
    text: str,
    table: dict[str, dict[str, Any]],
    person: str | None = None,
    spoken: frozenset[str] = frozenset(),
) -> str:
    """Replace speaker tracks ("S2") with names where the text uses them for a person: a track of this
    meeting that nobody said aloud, at the start, before a verb ("S2 will", "S2'll"), after "ask"/"tell",
    or the record's own owner/proposer (`person`) before "'s". Anything else, such as "AWS S3" or
    "Galaxy S23", stays as written."""

    def replace(match: re.Match[str]) -> str:
        track = match.group(0)
        if track not in table or track in spoken:
            return track
        before = re.findall(r"[A-Za-z]+", text[: match.start()])
        after = re.match(r"(?:['’](\w+)|\s+([A-Za-z]+))?", text[match.end() :])
        suffix, next_word = (after.group(1), after.group(2)) if after else (None, None)
        suffix = (suffix or "").lower()
        as_person = (
            not before
            or before[-1].lower() in _WORDS_BEFORE_PERSON
            or suffix in ("ll", "d")
            or (suffix == "s" and track == person)
            or (next_word or "").lower() in _WORDS_AFTER_PERSON
        )
        return speaker_display(track, table) if as_person else track

    return _TRACK_TOKEN.sub(replace, text)


def format_window(utterances: list[dict[str, Any]], speakers: list[dict[str, Any]]) -> str:
    table = _speaker_table(speakers)
    legend = ", ".join(f"{t} = {speaker_display(t, table)}" for t in sorted(table)) or "unknown"
    lines = [
        f"[{u['id']}] {speaker_display(u.get('speaker', ''), table, u.get('speaker_name'))} "
        f"({mmss(float(u.get('t_start') or 0))}): {u.get('text', '').strip()}"
        for u in utterances
    ]
    return f"Speakers: {legend}\nTranscript:\n" + "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def content_words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if len(w) > 2 and w not in _STOPWORDS}


def overlap_score(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the outermost JSON object in a model reply (tolerates code fences, prose, trailing commas)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    for candidate in (cleaned, _outermost_object(cleaned)):
        if not candidate:
            continue
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _outermost_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Validation of model output against the window
# ---------------------------------------------------------------------------


class _Window:
    def __init__(self, utterances: list[dict[str, Any]], speakers: list[dict[str, Any]]) -> None:
        self.ids = [u["id"] for u in utterances]
        self.id_set = set(self.ids)
        self.by_suffix: dict[str, list[str]] = {}
        for uid in self.ids:
            self.by_suffix.setdefault(uid.rsplit(":", 1)[-1], []).append(uid)
        self.speakers = speakers
        self.tracks = {s["track"] for s in speakers if s.get("track")}
        self.track_of_utterance = {u["id"]: u.get("speaker") for u in utterances}

    def evidence(self, raw: Any) -> list[str]:
        values = raw if isinstance(raw, list) else [raw] if raw else []
        out: list[str] = []
        for value in values:
            uid = self.resolve_id(str(value))
            if uid and uid not in out:
                out.append(uid)
        return out

    def resolve_id(self, value: str) -> str | None:
        value = value.strip().strip("[]").strip()
        if value.lower().startswith("u:") and value[2:] in self.id_set:
            value = value[2:]
        if value in self.id_set:
            return value
        if ":" in value:  # a full id from another meeting (or invented) is never remapped
            return None
        matches = self.by_suffix.get(value, [])  # bare "u00003": accept when unambiguous
        if not matches and value.isdigit():  # Gemma 4 E4B often abbreviates "mx:u00003" to 3
            matches = self.by_suffix.get(f"u{int(value):05d}", [])
        return matches[0] if len(matches) == 1 else None

    def track(self, raw: Any) -> str | None:
        """A speaker track for a track/label/name; None if it names nobody in the speaker list."""
        if not isinstance(raw, str) or not raw.strip() or raw.strip().lower() in ("null", "none", "unknown"):
            return None
        value = raw.strip()
        if value.upper() in self.tracks:
            return value.upper()
        lowered = value.lower()
        for s in self.speakers:
            if lowered in {str(s.get("label") or "").lower(), str(s.get("name") or "").lower()}:
                return s["track"]
        return None

    def owner(self, raw: Any) -> str | None:
        if not isinstance(raw, str) or raw.strip().lower() in ("", "null", "none", "unknown"):
            return None
        return self.track(raw) or raw.strip()[:60]


def _text(raw: Any, limit: int = 200) -> str:
    return " ".join(str(raw).split())[:limit] if isinstance(raw, str) else ""


def _optional_text(raw: Any, limit: int = 60) -> str | None:
    value = _text(raw, limit)
    return value if value and value.lower() not in ("null", "none", "n/a", "unknown") else None


def insights_from_json(
    data: dict[str, Any],
    utterances: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    existing_decisions: list[dict[str, Any]] | None = None,
) -> Insights:
    window = _Window(utterances, speakers)
    table = _speaker_table(speakers)
    earlier = [d.get("text", "") for d in existing_decisions or [] if d.get("text")]
    earlier_norm = {_norm(t) for t in earlier}

    # Tokens like "S3" that someone actually said ("move the backups to S3") are words, not speakers.
    spoken = frozenset(
        t.upper()
        for u in utterances
        for t in re.findall(r"\bS\d+\b", str(u.get("text") or ""), re.IGNORECASE)
    )

    def readable(raw: Any, person: str | None = None) -> str:
        # The prompt's legend maps tracks to names; models still write "S2 will…" sometimes.
        return speaker_tracks_to_names(_text(raw), table, person, spoken)

    decisions: list[DecisionDraft] = []
    for item in _dicts(data.get("decisions")):
        evidence = window.evidence(item.get("evidence"))
        proposer = window.track(item.get("proposed_by")) or (
            window.track_of_utterance.get(evidence[0]) if evidence else None
        )
        text = readable(item.get("text"), proposer)
        if not text or not evidence or _norm(text) in earlier_norm:
            continue
        decisions.append(
            DecisionDraft(
                text=text,
                evidence=evidence,
                proposed_by=proposer,
                supersedes_text=None if _REAFFIRM.search(text) else _match_earlier(item.get("supersedes"), earlier, text),
            )
        )
        if len(decisions) == MAX_DECISIONS:
            break

    actions: list[ActionItemDraft] = []
    for item in _dicts(data.get("action_items")):
        owner = window.owner(item.get("owner"))
        text, evidence = readable(item.get("text"), owner), window.evidence(item.get("evidence"))
        if not text or not evidence or _WISH.search(text):
            continue
        actions.append(
            ActionItemDraft(
                text=text,
                evidence=evidence,
                owner=owner,
                due=_optional_text(item.get("due")),
            )
        )
        if len(actions) == MAX_ACTION_ITEMS:
            break

    topics = _unique(
        [t for t in (_text(x, 40) for x in _list(data.get("topics"))) if 0 < len(t.split()) <= 5]
    )
    topics = topics[:MAX_TOPICS]
    entities: list[dict[str, str]] = []
    for item in _dicts(data.get("entities")):
        name = _text(item.get("name"), 60)
        if name and name.lower() not in {e["name"].lower() for e in entities}:
            kind = str(item.get("kind") or "other").lower()
            entities.append({"name": name, "kind": kind if kind in ENTITY_KINDS else "other"})
        if len(entities) == MAX_ENTITIES:
            break

    known = {n.lower(): n for n in [*topics, *(e["name"] for e in entities)]}
    mentions: dict[str, list[str]] = {}
    raw_mentions = data.get("mentions") if isinstance(data.get("mentions"), dict) else {}
    for raw_id, names in raw_mentions.items():
        uid = window.resolve_id(str(raw_id))
        kept = _unique([known[n.lower()] for n in _list(names) if isinstance(n, str) and n.lower() in known])
        if uid and kept:
            mentions[uid] = _unique([*mentions.get(uid, []), *kept])
    return Insights(
        decisions=decisions, action_items=actions, topics=topics, entities=entities, mentions=mentions
    )


def _match_earlier(raw: Any, earlier: list[str], new_text: str) -> str | None:
    """Map the model's `supersedes` to the closest real earlier decision text (or drop it)."""
    claimed = _optional_text(raw, 300)
    if not claimed or not earlier:
        return None
    best = max(earlier, key=lambda t: overlap_score(claimed, t))
    if overlap_score(claimed, best) < 0.35 or _norm(best) == _norm(new_text):
        return None
    # The new decision must be about the same thing: "Self-serve will launch on Monday" does not replace
    # "The self-serve pricing is locked" (0.33), while "Starter moves from 4999 to 3999" replaces "The
    # starter tier will be 4,999 with 1500 shipments" (0.5).
    if overlap_score(new_text, best) < SUPERSEDE_MIN_OVERLAP:
        return None
    return best


SUPERSEDE_MIN_OVERLAP = 0.4


def _dicts(raw: Any) -> list[dict[str, Any]]:
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Heuristic fallback (fake LLM, invalid JSON, model errors)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "about",
        "into",
        "over",
        "after",
        "before",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
        "okay",
        "ok",
        "yeah",
        "yes",
        "right",
        "well",
        "like",
        "really",
        "think",
        "know",
        "going",
        "get",
        "got",
        "let",
        "lets",
        "let's",
        "thing",
        "things",
        "one",
        "two",
        "sure",
        "good",
        "great",
        "need",
        "want",
        "make",
        "sounds",
        "sound",
        "alright",
        "actually",
        "maybe",
        "something",
        "there",
        "here",
        "now",
    }
)
_DECISION_CUE = re.compile(
    r"\b(we(?:'ve| have)? decided|decided to|let'?s go with|let us go with|we'?ll go with|we will go with|"
    r"we'?re going with|we are going with|(?<!not )agreed|final answer is|the decision is)\b",
    re.IGNORECASE,
)
_SELF_CUE = re.compile(r"\b(i'?ll|i will|i'?m going to|i am going to|i can take|i'?ll take)\b", re.IGNORECASE)
_REQUEST_CUE = re.compile(r"\b(can you|could you|would you|please)\b", re.IGNORECASE)
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DUE_BODY = (
    rf"(?:end of (?:the )?(?:day|week|month|sprint|quarter)|eod|eow|tomorrow|tonight|today|"
    rf"next (?:week|month|sprint|quarter|{_WEEKDAY})|this (?:week|month|{_WEEKDAY})|{_WEEKDAY}|"
    rf"{_MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)? (?:of )?{_MONTH})"
)
_DUE = re.compile(
    rf"\b(?:by|before|until|on)\s+(?:the\s+)?({_DUE_BODY})\b|\b(tomorrow|tonight|eod|end of day|next week)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"^(?:(?:so|okay|ok|alright|all right|well|then|and|yeah|right|um|uh)[,\s]+)+", re.IGNORECASE
)
_SELF_LEAD = re.compile(r"^(?:i'?ll|i will|i'?m going to|i am going to|i can)\s+", re.IGNORECASE)
_REQUEST_LEAD = re.compile(
    r"^(?:(?P<name>[A-Z][a-z]+)(?P<comma>,)?\s+)?(?i:can you|could you|would you|please)\s+"
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _clean_sentence(sentence: str) -> str:
    sentence = _FILLER.sub("", sentence.strip()).rstrip(" .!?,;")
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def _format_due(raw: str) -> str:
    lowered = raw.lower()
    if lowered in ("eod", "end of day", "end of the day"):
        return "EOD"
    if lowered == "eow":
        return "end of week"
    return " ".join(
        w.capitalize() if re.fullmatch(_WEEKDAY + "|" + _MONTH, w) else w for w in lowered.split()
    )


def parse_due(text: str) -> str | None:
    match = _DUE.search(text)
    if not match:
        return None
    return _format_due(match.group(1) or match.group(2))


def _action_text(
    sentence: str, known_names: set[str] | frozenset[str] = frozenset()
) -> tuple[str, str | None]:
    """Turn a commitment sentence into an imperative item; returns (text, addressed name or None).
    "Rahul, can you ..." addresses Rahul; without the comma only a known speaker name counts."""
    cleaned = _clean_sentence(sentence).rstrip("?")
    addressed = None
    request = _REQUEST_LEAD.match(cleaned)
    if request:
        name = request.group("name")
        if name and (request.group("comma") or name.lower() in known_names):
            addressed = name
        elif name:
            request = None
    if request:
        cleaned = cleaned[request.end() :]
    else:
        cleaned = _SELF_LEAD.sub("", cleaned)
    cleaned = cleaned.strip()
    return (cleaned[:1].upper() + cleaned[1:])[:200], addressed


def heuristic_extract(
    meeting_id: str,
    utterances: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    existing_decisions: list[dict[str, Any]] | None = None,
) -> Insights:
    """Cue-phrase extraction used when no model is available or the model's JSON is unusable."""
    known_names = {
        str(s.get("name")).lower(): s["track"] for s in speakers if s.get("name") and s.get("track")
    }
    earlier = [d.get("text", "") for d in existing_decisions or [] if d.get("status", "active") == "active"]
    decisions: list[DecisionDraft] = []
    actions: list[ActionItemDraft] = []
    for u in utterances:
        uid, track = u["id"], u.get("speaker")
        for sentence in _sentences(u.get("text", "")):
            if _DECISION_CUE.search(sentence) and len(decisions) < MAX_DECISIONS:
                text = _clean_sentence(sentence)
                supersedes = None
                if earlier and re.search(
                    r"\b(instead|actually|change|changing|no longer|moving|move|push)\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    best = max(earlier, key=lambda t: overlap_score(t, sentence))
                    supersedes = best if overlap_score(best, sentence) >= 0.3 else None
                decisions.append(
                    DecisionDraft(text=text, evidence=[uid], proposed_by=track, supersedes_text=supersedes)
                )
                continue
            due = parse_due(sentence)
            is_self = bool(_SELF_CUE.search(sentence))
            is_request = bool(_REQUEST_CUE.search(sentence))
            if not (is_self or is_request or due) or len(actions) >= MAX_ACTION_ITEMS:
                continue
            text, addressed = _action_text(sentence, set(known_names))
            if len(text.split()) < 2:
                continue
            owner = (
                track if is_self else (known_names.get(addressed.lower(), addressed) if addressed else None)
            )
            actions.append(ActionItemDraft(text=text, evidence=[uid], owner=owner, due=due))
    topics, mentions = _heuristic_topics(utterances)
    return Insights(decisions=decisions, action_items=actions, topics=topics, mentions=mentions)


def _heuristic_topics(utterances: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    counts: dict[str, int] = {}
    for u in utterances:
        for word in set(re.findall(r"[a-z][a-z-]{3,}", u.get("text", "").lower())):
            if word not in _STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
    topics = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= 2][:5]
    mentions: dict[str, list[str]] = {}
    for u in utterances:
        words = set(re.findall(r"[a-z][a-z-]{3,}", u.get("text", "").lower()))
        hit = [t for t in topics if t in words]
        if hit:
            mentions[u["id"]] = hit
    return topics, mentions


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class Extractor:
    def __init__(self, llm: LLM | None = None, *, max_tokens: int = 1800) -> None:
        self.llm = llm or get_extraction_llm()
        self.max_tokens = max_tokens

    async def extract(
        self,
        meeting_id: str,
        utterances: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        existing_decisions: list[dict[str, Any]] | None = None,
    ) -> Insights:
        insights, _ = await self.extract_with_meta(meeting_id, utterances, speakers, existing_decisions)
        return insights

    async def extract_with_meta(
        self,
        meeting_id: str,
        utterances: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        existing_decisions: list[dict[str, Any]] | None = None,
    ) -> tuple[Insights, dict[str, Any]]:
        """Insights plus {"provider", "model", "latency_ms", "fallback": reason or None}."""
        if not utterances:
            return Insights(), {
                "provider": "none",
                "model": None,
                "latency_ms": 0.0,
                "fallback": "empty window",
            }
        if self.llm.provider == "fake":
            return self._heuristic(meeting_id, utterances, speakers, existing_decisions, "fake llm")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._prompt(utterances, speakers, existing_decisions)},
        ]
        try:
            with llm_priority():  # meeting notes go ahead of queued questions on a shared model
                result = await self.llm.chat(
                    messages, json_mode=True, temperature=0.1, max_tokens=self.max_tokens
                )
        except LLMError as exc:
            log.warning("extraction model call failed for %s: %s", meeting_id, exc)
            return self._heuristic(meeting_id, utterances, speakers, existing_decisions, f"llm error: {exc}")
        data = parse_json_object(result.text)
        if data is None:
            log.warning("extraction for %s returned invalid JSON: %.200s", meeting_id, result.text)
            return self._heuristic(meeting_id, utterances, speakers, existing_decisions, "invalid json")
        insights = insights_from_json(data, utterances, speakers, existing_decisions)
        meta = {
            "provider": result.provider,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "fallback": None,
        }
        return insights, meta

    @staticmethod
    def _prompt(
        utterances: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        existing_decisions: list[dict[str, Any]] | None,
    ) -> str:
        parts = [format_window(utterances, speakers)]
        earlier = [
            d["text"] for d in existing_decisions or [] if d.get("text") and d.get("status") != "superseded"
        ]
        if earlier:
            listed = "\n".join(f"- {t}" for t in earlier[-10:])
            parts.append(
                f'Earlier decisions (set "supersedes" to one of these exact texts if a new decision '
                f"replaces it):\n{listed}"
            )
        parts.append("Return the JSON now.")
        return "\n\n".join(parts)

    @staticmethod
    def _heuristic(
        meeting_id: str,
        utterances: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        existing_decisions: list[dict[str, Any]] | None,
        reason: str,
    ) -> tuple[Insights, dict[str, Any]]:
        insights = heuristic_extract(meeting_id, utterances, speakers, existing_decisions)
        return insights, {
            "provider": "heuristic",
            "model": "cue-phrases",
            "latency_ms": 0.0,
            "fallback": reason,
        }
