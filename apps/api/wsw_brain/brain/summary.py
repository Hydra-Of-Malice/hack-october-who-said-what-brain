"""Short meeting summaries (<= 80 words): who took part, what was decided, who owns what."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..contracts import LLM
from .extract import mmss, speaker_display
from .llm import LLMError, strip_thoughts

log = logging.getLogger(__name__)

MAX_WORDS = 80
MAX_TRANSCRIPT_CHARS = 6000


def _participants(detail: dict[str, Any]) -> list[str]:
    table = {s["track"]: s for s in detail.get("speakers", []) if s.get("track")}
    return [speaker_display(t, table) for t in table]


def _owner(item: dict[str, Any]) -> str:
    return str(item.get("owner") or "someone")


def clip_words(text: str, limit: int = MAX_WORDS) -> str:
    """At most `limit` words, cut at a sentence end when one is available."""
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    clipped = " ".join(words[:limit])
    end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if end >= len(clipped) // 2:
        return clipped[: end + 1]
    return clipped.rstrip(",;:") + "..."


def heuristic_summary(detail: dict[str, Any]) -> str:
    meeting = detail.get("meeting") or {}
    people = _participants(detail)
    decisions = [d for d in detail.get("decisions", []) if d.get("status") != "superseded"]
    items = detail.get("action_items", [])
    minutes = round(float(meeting.get("duration_s") or 0) / 60)
    who = (
        ", ".join(people[:-1]) + f" and {people[-1]}"
        if len(people) > 1
        else (people[0] if people else "The team")
    )
    parts = [f"{who} met" + (f" for about {minutes} minutes." if minutes else ".")]
    if decisions:
        parts.append("Decided: " + "; ".join(d["text"].rstrip(".") for d in decisions[:3]) + ".")
    if items:
        owned = [
            f"{_owner(i)}: {i['text'].rstrip('.')}" + (f" (due {i['due']})" if i.get("due") else "")
            for i in items[:4]
        ]
        parts.append("Actions: " + "; ".join(owned) + ".")
    if not decisions and not items:
        utterances = detail.get("utterances", [])
        if utterances:
            parts.append(f"No decisions or action items were recorded across {len(utterances)} utterances.")
    return clip_words(" ".join(parts))


def _transcript(detail: dict[str, Any]) -> str:
    table = {s["track"]: s for s in detail.get("speakers", []) if s.get("track")}
    lines = [
        f"{speaker_display(u.get('speaker', ''), table, u.get('speaker_name'))} "
        f"({mmss(float(u.get('t_start') or 0))}): {u.get('text', '').strip()}"
        for u in detail.get("utterances", [])
    ]
    text = "\n".join(lines)
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    head, tail = lines[:40], lines[-40:]
    return "\n".join([*head, "[...]", *tail])[: MAX_TRANSCRIPT_CHARS * 2]


async def summarize_meeting(llm: LLM | None, meeting_detail: dict[str, Any]) -> str:
    """<= 80-word summary naming the participants, decisions and owners. Empty string for an empty meeting."""
    if not meeting_detail.get("utterances"):
        return ""
    if llm is None or llm.provider == "fake":
        return heuristic_summary(meeting_detail)
    decisions = [d for d in meeting_detail.get("decisions", []) if d.get("status") != "superseded"]
    items = meeting_detail.get("action_items", [])
    facts = [f"Participants: {', '.join(_participants(meeting_detail)) or 'unknown'}"]
    if decisions:
        facts.append("Decisions:\n" + "\n".join(f"- {d['text']}" for d in decisions[:8]))
    if items:
        facts.append(
            "Action items:\n"
            + "\n".join(
                f"- {i['text']} (owner: {_owner(i)}{', due ' + i['due'] if i.get('due') else ''})"
                for i in items[:10]
            )
        )
    title = (meeting_detail.get("meeting") or {}).get("title") or "the meeting"
    messages = [
        {
            "role": "system",
            "content": "You summarise meetings in one plain paragraph of at most 80 words. Name the "
            "people involved, the decisions, and each action item with its owner. No "
            "markdown, no bullet points, no preamble.",
        },
        {
            "role": "user",
            "content": f"Meeting: {title}\n"
            + "\n\n".join(facts)
            + f"\n\nTranscript:\n{_transcript(meeting_detail)}\n\nWrite the summary.",
        },
    ]
    try:
        result = await llm.chat(messages, temperature=0.2, max_tokens=200)
    except LLMError as exc:
        log.warning("summary model call failed; using the heuristic summary: %s", exc)
        return heuristic_summary(meeting_detail)
    text = re.sub(r"^(?:summary\s*:\s*)", "", strip_thoughts(result.text), flags=re.IGNORECASE)
    text = re.sub(r"[*#`]+", "", text)
    return clip_words(text) if text.strip() else heuristic_summary(meeting_detail)
