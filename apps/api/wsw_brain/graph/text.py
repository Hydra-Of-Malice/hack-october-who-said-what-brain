"""Pure text helpers for the graph layer: ids, slugs, dedupe keys and full-text query sanitising."""

from __future__ import annotations

import hashlib
import re
import zlib

SPEAKER_COLORS = 8

# Words that carry no retrieval signal in a spoken question. Kept small on purpose: anything domain-like
# ("meeting", "price", "deadline") stays searchable.
STOPWORDS = frozenset((
    "a", "about", "all", "also", "an", "and", "any", "anything", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "let", "lets", "me", "my", "of",
    "on", "or", "our", "ours", "she", "so", "some", "something", "tell", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "told", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "will", "with", "would", "you", "your", "yours",
    "said", "say", "says",
))

_TRACK_RE = re.compile(r"\s*[Ss](\d{1,4})\s*")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_ASCII_TERM_RE = re.compile(r"[A-Za-z0-9]+")
_MAX_QUERY_TERMS = 12
_PREFIX_MIN_LEN = 4


def normalise(text: str) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed. The dedupe key for decisions/action items."""
    return " ".join(_NON_WORD_RE.sub(" ", text.lower()).split())


def slug(text: str, max_len: int = 80) -> str:
    """URL-safe id from a display name ("Q3 Pricing!" -> "q3-pricing"). Empty when nothing survives."""
    return "-".join(normalise(text).split())[:max_len].strip("-")


def person_id(name: str) -> str:
    """Person ids are slug(name); names with no slug-able characters get a stable hash id."""
    return slug(name) or "p-" + short_hash(name.strip().lower())


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def content_tokens(text: str) -> set[str]:
    """Normalised tokens without stopwords (falls back to all tokens if only stopwords remain)."""
    tokens = normalise(text).split()
    content = {t for t in tokens if t not in STOPWORDS}
    return content or set(tokens)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---- speakers -------------------------------------------------------------------


def parse_track(value: str) -> str | None:
    """'S2', 's2', ' S02 ' -> 'S2'. None when the value is not a speaker track."""
    match = _TRACK_RE.fullmatch(value or "")
    return f"S{int(match.group(1))}" if match else None


def canonical_track(track: str) -> str:
    """Normalise a track for storage; non-standard tracks are kept as given (stripped)."""
    parsed = parse_track(track)
    if parsed:
        return parsed
    cleaned = (track or "").strip()
    if not cleaned:
        raise ValueError("speaker track must be a non-empty string such as 'S1'")
    return cleaned


def speaker_id(meeting_id: str, track: str) -> str:
    return f"{meeting_id}:{track}"


def speaker_label(track: str) -> str:
    parsed = parse_track(track)
    return f"Speaker {parsed[1:]}" if parsed else track


def color_idx(track: str) -> int:
    parsed = parse_track(track)
    if parsed:
        return (int(parsed[1:]) - 1) % SPEAKER_COLORS
    return zlib.crc32(track.encode("utf-8")) % SPEAKER_COLORS


# ---- full-text --------------------------------------------------------------------


def query_terms(query: str) -> list[str]:
    """Lowercased ASCII alphanumeric terms of a question, stopwords and 1-char tokens dropped, deduped."""
    terms: list[str] = []
    for raw in _ASCII_TERM_RE.findall(query or ""):
        term = raw.lower()
        if len(term) < 2 or term in STOPWORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= _MAX_QUERY_TERMS:
            break
    return terms


def fulltext_query(terms: list[str]) -> str:
    """RediSearch OR query. Long terms also match as prefixes ("price" -> "price|price*")."""
    parts: list[str] = []
    for term in terms:
        parts.append(term)
        if len(term) >= _PREFIX_MIN_LEN:
            parts.append(f"{term}*")
    return "|".join(parts)


def truncate(text: str, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
