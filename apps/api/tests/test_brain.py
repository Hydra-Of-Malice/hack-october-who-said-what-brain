"""Tests for the Gemma 4 intelligence layer (wsw_brain.brain).

Unit tests use FakeLLM / a scripted LLM, the HashEmbedder and an in-memory fake GraphStore.
`slow` loads EmbeddingGemma from the HF cache. `live` talks to the local Ollama gemma4:e4b.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from wsw_brain.brain import agent as agent_mod
from wsw_brain.brain import voice as voice_mod
from wsw_brain.brain.agent import VoiceAgent, resolve_citations
from wsw_brain.brain.embed import GemmaEmbedder, HashEmbedder
from wsw_brain.brain.extract import Extractor, heuristic_extract, parse_due, parse_json_object
from wsw_brain.brain.llm import (
    FakeLLM,
    GeminiLLM,
    OllamaLLM,
    OpenAICompatLLM,
    assistant_tool_message,
    ollama_model_hears_audio,
    probe_ollama_model,
    select_audio_llm,
    select_llm,
    strip_thoughts,
    to_gemini_contents,
    tool_result_message,
)
from wsw_brain.brain.retrieve import Retriever, rrf_fuse
from wsw_brain.brain.summary import heuristic_summary, summarize_meeting
from wsw_brain.config import settings
from wsw_brain.contracts import LLMResult, ToolCall

OLLAMA_URL = settings().ollama_url
OLLAMA_MODEL = settings().ollama_model

# ---------------------------------------------------------------------------
# In-memory fake GraphStore (only the methods the brain calls)
# ---------------------------------------------------------------------------

MEETINGS = {
    "m1": {
        "id": "m1",
        "title": "Pricing review",
        "status": "ended",
        "source": "demo",
        "started_at": 2_000,
        "ended_at": 3_000,
        "duration_s": 600.0,
        "utterance_count": 6,
        "speaker_count": 3,
        "has_audio": True,
        "summary": None,
    },
    "m2": {
        "id": "m2",
        "title": "Hiring sync",
        "status": "ended",
        "source": "demo",
        "started_at": 5_000,
        "ended_at": 6_000,
        "duration_s": 300.0,
        "utterance_count": 2,
        "speaker_count": 2,
        "has_audio": False,
        "summary": "Asha and Priya planned backend hiring.",
    },
}
SPEAKERS = {
    "m1": [
        {"track": "S1", "label": "Speaker 1", "name": "Priya", "person_id": "priya", "color_idx": 0},
        {"track": "S2", "label": "Speaker 2", "name": "Rahul", "person_id": "rahul", "color_idx": 1},
        {"track": "S3", "label": "Speaker 3", "name": None, "person_id": None, "color_idx": 2},
    ],
    "m2": [
        {"track": "S1", "label": "Speaker 1", "name": "Asha", "person_id": "asha", "color_idx": 0},
        {"track": "S2", "label": "Speaker 2", "name": "Priya", "person_id": "priya", "color_idx": 1},
    ],
}
LINES = {
    "m1": [
        ("S1", 0, 5, "Let's start with the pricing review for the Pro plan."),
        ("S2", 5, 11, "I think twenty dollars a month is too low for the Pro plan."),
        ("S1", 11, 18, "Agreed, we decided to price the Pro plan at twenty five dollars a month."),
        ("S1", 18, 25, "I'll send the updated pricing deck to the sales team by Friday."),
        ("S2", 25, 30, "Great, and I will update the billing page next week."),
        ("S3", 30, 36, "We also need to hire two backend engineers next quarter."),
    ],
    "m2": [
        ("S1", 0, 6, "We should open the backend engineer role this week."),
        ("S2", 6, 12, "I can review the candidates on Monday."),
    ],
}


def _utterances(mid: str) -> list[dict[str, Any]]:
    names = {s["track"]: s["name"] for s in SPEAKERS[mid]}
    return [
        {
            "id": f"{mid}:u{seq:05d}",
            "meeting_id": mid,
            "seq": seq,
            "speaker": track,
            "speaker_name": names[track],
            "t_start": float(t0),
            "t_end": float(t1),
            "text": text,
            "confidence": 0.9,
        }
        for seq, (track, t0, t1, text) in enumerate(LINES[mid])
    ]


class FakeStore:
    def __init__(self) -> None:
        self.embedder = HashEmbedder(768)
        self.utts = {mid: _utterances(mid) for mid in LINES}
        self.all = [u for mid in ("m1", "m2") for u in self.utts[mid]]
        self.vectors = {
            u["id"]: np.array(v)
            for u, v in zip(
                self.all, self.embedder.embed_documents([u["text"] for u in self.all]), strict=True
            )
        }
        self.items = [
            {
                "id": "ai1",
                "meeting_id": "m1",
                "meeting_title": "Pricing review",
                "text": "Send the updated pricing deck to the sales team",
                "owner": "Priya",
                "owner_track": "S1",
                "due": "Friday",
                "status": "open",
                "created_at": 1,
                "done_at": None,
                "evidence": ["m1:u00003"],
            },
            {
                "id": "ai2",
                "meeting_id": "m1",
                "meeting_title": "Pricing review",
                "text": "Update the billing page",
                "owner": "Rahul",
                "owner_track": "S2",
                "due": "next week",
                "status": "open",
                "created_at": 2,
                "done_at": None,
                "evidence": ["m1:u00004"],
            },
        ]
        self.decisions = [
            {
                "id": "d1",
                "meeting_id": "m1",
                "text": "Price the Pro plan at $25 a month",
                "status": "active",
                "made_at": 11.0,
                "proposed_by": "S1",
                "evidence": ["m1:u00002"],
                "superseded_by": None,
                "supersedes": None,
            },
        ]
        self.turns: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.last_cypher: list[str] = []

    # helpers
    def _hit(self, u: dict[str, Any], score: float, via: str) -> dict[str, Any]:
        return {
            "utterance_id": u["id"],
            "meeting_id": u["meeting_id"],
            "meeting_title": MEETINGS[u["meeting_id"]]["title"],
            "speaker": u["speaker"],
            "speaker_name": u["speaker_name"],
            "t_start": u["t_start"],
            "t_end": u["t_end"],
            "text": u["text"],
            "score": score,
            "via": via,
        }

    def _scope(self, meeting_id: str | None) -> list[dict[str, Any]]:
        return self.utts.get(meeting_id, []) if meeting_id else self.all

    def _log(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        self.last_cypher = [f"/* {name} */ MATCH (u:Utterance) RETURN u"]

    # retrieval
    def search_fulltext(self, query: str, k: int = 8, meeting_id: str | None = None) -> list[dict[str, Any]]:
        self._log("search_fulltext", query, meeting_id)
        terms = {w.strip("?.,!").lower() for w in query.split() if len(w) > 3}
        scored = [
            (len(terms & {w.strip("?.,!").lower() for w in u["text"].split()}), u)
            for u in self._scope(meeting_id)
        ]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:k]
        return [self._hit(u, float(s), "fulltext") for s, u in ranked]

    def search_vector(
        self, vector: list[float], k: int = 8, meeting_id: str | None = None
    ) -> list[dict[str, Any]]:
        self._log("search_vector", meeting_id)
        q = np.array(vector)
        scored = sorted(
            ((float(self.vectors[u["id"]] @ q), u) for u in self._scope(meeting_id)), key=lambda s: -s[0]
        )[:k]
        return [self._hit(u, s, "vector") for s, u in scored]

    def utterances_by_speaker(
        self, name_or_track: str, meeting_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        self._log("utterances_by_speaker", name_or_track, meeting_id)
        key = name_or_track.lower()
        return [
            self._hit(u, 1.0, "graph")
            for u in self._scope(meeting_id)
            if key in ((u["speaker_name"] or "").lower(), u["speaker"].lower())
        ][:limit]

    def recent_utterances(self, meeting_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self._log("recent_utterances", meeting_id)
        rows = self.utts[meeting_id or "m2"][-limit:]
        return [self._hit(u, 1.0, "recent") for u in reversed(rows)]

    def get_utterances(self, ids: list[str]) -> list[dict[str, Any]]:
        self._log("get_utterances", tuple(ids))
        by_id = {u["id"]: u for u in self.all}
        return [self._hit(by_id[i], 1.0, "graph") for i in ids if i in by_id]

    def context_window(self, utterance_id: str, before: int = 2, after: int = 2) -> list[dict[str, Any]]:
        self._log("context_window", utterance_id)
        mid = utterance_id.split(":")[0]
        rows = self.utts.get(mid, [])
        idx = next((i for i, u in enumerate(rows) if u["id"] == utterance_id), None)
        return [] if idx is None else [dict(u) for u in rows[max(0, idx - before) : idx + after + 1]]

    # reads
    def list_people(self) -> list[dict[str, Any]]:
        return [
            {"id": n.lower(), "name": n, "meetings": 1, "utterances": 2, "talk_time_s": 10.0, "open_items": 0}
            for n in ("Priya", "Rahul", "Asha")
        ]

    def list_meetings(self) -> list[dict[str, Any]]:
        return [dict(MEETINGS["m2"]), dict(MEETINGS["m1"])]

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        if meeting_id not in MEETINGS:
            return None
        return {
            "meeting": dict(MEETINGS[meeting_id]),
            "speakers": SPEAKERS[meeting_id],
            "utterances": self.utts[meeting_id],
            "decisions": [d for d in self.decisions if d["meeting_id"] == meeting_id],
            "action_items": [i for i in self.items if i["meeting_id"] == meeting_id],
            "topics": [],
        }

    def list_action_items(
        self, status: str | None = None, owner: str | None = None, meeting_id: str | None = None
    ) -> list[dict[str, Any]]:
        self._log("list_action_items", status, owner, meeting_id)
        return [
            dict(i)
            for i in self.items
            if (status is None or i["status"] == status)
            and (owner is None or (i["owner"] or "").lower() == owner.lower())
            and (meeting_id is None or i["meeting_id"] == meeting_id)
        ]

    def set_action_item_status(self, item_id: str, status: str) -> dict[str, Any] | None:
        self._log("set_action_item_status", item_id, status)
        for item in self.items:
            if item["id"] == item_id:
                item["status"], item["done_at"] = status, 99 if status == "done" else None
                return dict(item)
        return None

    def list_decisions(self, meeting_id: str | None = None) -> list[dict[str, Any]]:
        self._log("list_decisions", meeting_id)
        return [dict(d) for d in self.decisions if meeting_id is None or d["meeting_id"] == meeting_id]

    def who_knows(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        self._log("who_knows", topic)
        hits = [u for u in self.all if topic.lower() in u["text"].lower()]
        return [
            {
                "name": u["speaker_name"],
                "person_id": None,
                "meeting_id": u["meeting_id"],
                "hits": 1,
                "sample": u,
            }
            for u in hits[:limit]
        ]

    # memory
    def save_turn(
        self, session_id: str, role: str, text: str, at_ms: int, citations: list[str] | None = None
    ) -> None:
        self.turns.setdefault(session_id, []).append(
            {"role": role, "text": text, "at": at_ms, "citations": list(citations or [])}
        )

    def get_turns(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        return [dict(t) for t in self.turns.get(session_id, [])[-limit:]]


class ScriptedLLM:
    """Plays back scripted replies (a string, or a list of ToolCalls) and records every request."""

    provider = "scripted"
    model = "scripted-gemma"
    supports_audio = False

    def __init__(self, replies: list[str | list[ToolCall]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        self.requests.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "json_mode": json_mode}
        )
        reply = self.replies.pop(0) if self.replies else "Done."
        if isinstance(reply, list):
            return LLMResult(
                text="", tool_calls=reply, model=self.model, provider=self.provider, latency_ms=1.0
            )
        return LLMResult(text=reply, tool_calls=[], model=self.model, provider=self.provider, latency_ms=1.0)

    async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Any:
        raise NotImplementedError


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def make_agent(store: FakeStore, llm: Any) -> VoiceAgent:
    return VoiceAgent(store, llm, Retriever(store, store.embedder))


def all_ids(store: FakeStore) -> set[str]:
    return {u["id"] for u in store.all}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


async def test_agent_answer_cites_only_valid_ids(store: FakeStore) -> None:
    result = await make_agent(store, FakeLLM()).ask("s1", "What did Priya commit to on the pricing deck?")
    assert result.citations, "the fake LLM quotes an evidence line, so there must be a citation"
    assert {c["utterance_id"] for c in result.citations} <= all_ids(store)
    assert "[u:" not in result.answer
    assert result.provider == "fake" and result.session_id == "s1"
    assert result.cypher and set(result.timings) >= {"retrieve", "llm", "total"}
    roles = [t["role"] for t in store.turns["s1"]]
    assert roles == ["user", "assistant"]
    assert store.turns["s1"][1]["citations"] == [c["utterance_id"] for c in result.citations]


async def test_agent_strips_invalid_citations(store: FakeStore) -> None:
    llm = ScriptedLLM(
        ["Priya will send the deck [u:m1:u00003] and the budget doubled [u:m9:u00001] (u:bogus)."]
    )
    result = await make_agent(store, llm).ask("s1", "What did Priya commit to on pricing?")
    assert [c["utterance_id"] for c in result.citations] == ["m1:u00003"]
    assert "u:" not in result.answer and "[" not in result.answer
    assert result.answer.startswith("Priya will send the deck and the budget doubled")


async def test_agent_attaches_top_hit_when_nothing_cited(store: FakeStore) -> None:
    result = await make_agent(store, ScriptedLLM(["Priya will send the pricing deck by Friday."])).ask(
        "s1", "What did Priya commit to on the pricing deck?"
    )
    assert len(result.citations) == 1
    assert result.citations[0]["utterance_id"] in all_ids(store)


async def test_agent_dont_know_has_no_citations(store: FakeStore) -> None:
    result = await make_agent(store, ScriptedLLM(["The meetings don't cover the office move."])).ask(
        "s1", "When is the office move happening?"
    )
    assert result.citations == []
    assert result.answer == "The meetings don't cover the office move."


async def test_follow_up_uses_previous_citations(store: FakeStore) -> None:
    llm = ScriptedLLM(
        ["Priya will send the updated pricing deck [u:m1:u00003].", "It is due Friday [u:m1:u00003]."]
    )
    agent = make_agent(store, llm)
    await agent.ask("s2", "What did Priya commit to in the pricing review?")
    second = await agent.ask("s2", "And when is that due?")

    messages = llm.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "What did Priya commit to in the pricing review?"
    final_prompt = messages[-1]["content"]
    assert final_prompt.startswith("Sources of your previous answer:\n[u:m1:u00003]")
    queries = [args[0] for name, args in store.calls if name == "search_fulltext"]
    assert "Priya" in queries[-1] and "when is that due" in queries[-1]
    assert [c["utterance_id"] for c in second.citations] == ["m1:u00003"]


async def test_complete_action_item_tool(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            [ToolCall("complete_action_item", {"description": "pricing deck"})],
            "Done, Priya's pricing deck item is marked done [u:m1:u00003].",
        ]
    )
    result = await make_agent(store, llm).ask("s3", "Mark Priya's pricing deck item as done.")
    assert store.items[0]["status"] == "done" and store.items[1]["status"] == "open"
    assert {"type": "refresh", "what": "action_items"} in result.actions
    assert (
        result.tool_trace[0]["tool"] == "complete_action_item" and result.tool_trace[0]["result_count"] == 1
    )
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_messages and json.loads(tool_messages[0]["content"])["item"]["id"] == "ai1"
    assistant = [m for m in llm.requests[1]["messages"] if m.get("tool_calls")]
    assert assistant[0]["tool_calls"][0]["id"] == tool_messages[0]["tool_call_id"]
    assert [c["utterance_id"] for c in result.citations] == ["m1:u00003"]
    assert any("set_action_item_status" in c for c in result.cypher)


async def test_mark_it_done_uses_conversation_evidence(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            "Priya will send the updated pricing deck by Friday [u:m1:u00003].",
            [ToolCall("complete_action_item", {})],
            "Marked it done.",
        ]
    )
    agent = make_agent(store, llm)
    await agent.ask("s4", "What did Priya commit to?")
    result = await agent.ask("s4", "Mark it done.")
    assert store.items[0]["status"] == "done"
    assert result.citations[0]["utterance_id"] == "m1:u00003"


async def test_mark_it_done_guard_when_model_asks_back(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            "Rahul will update the billing page next week [u:m1:u00004].",
            "Which action item do you mean?",
        ]
    )
    agent = make_agent(store, llm)
    await agent.ask("s8", "When is Rahul's billing page due?")
    result = await agent.ask("s8", "Mark it done.")
    assert [i["status"] for i in store.items] == ["open", "done"]
    assert result.answer == 'Done. I marked "Update the billing page" as done.'
    assert [c["utterance_id"] for c in result.citations] == ["m1:u00004"]
    assert result.tool_trace[0]["fallback"] == "intent"
    assert {"type": "refresh", "what": "action_items"} in result.actions
    assert agent_mod.completion_request("Please mark the pricing deck item as done") == "pricing deck"
    assert agent_mod.completion_request("What is done so far?") is None
    assert agent_mod.completion_request("Check if the pricing deck is done") is None
    assert agent_mod.completion_request("Okay, can you mark that as complete?") == ""


async def test_linked_records_reach_the_prompt(store: FakeStore) -> None:
    llm = ScriptedLLM(["Priya owns it [item ai1] [u:m1:u00003]."])
    result = await make_agent(store, llm).ask("s9", "What did Priya commit to on the pricing deck?")
    prompt = llm.requests[0]["messages"][-1]["content"]
    assert (
        "- item ai1: Send the updated pricing deck to the sales team (owner Priya, due Friday, open) "
        "[u:m1:u00003]"
    ) in prompt
    assert "- Price the Pro plan at $25 a month (active) [u:m1:u00002]" in prompt
    assert result.answer == "Priya owns it."


async def test_seek_and_draft_tools(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            [
                ToolCall("seek_to", {"utterance_id": "u:m1:u00003"}),
                ToolCall("draft_follow_up", {"meeting_id": "pricing review", "recipient": "Sales"}),
            ],
            "Hi Sales, Priya will send the updated pricing deck by Friday.",
            "Playing it now and I drafted a follow-up [u:m1:u00003].",
        ]
    )
    result = await make_agent(store, llm).ask("s5", "Play Priya's commitment and draft a follow-up.")
    seek = next(a for a in result.actions if a["type"] == "seek")
    assert seek == {"type": "seek", "meeting_id": "m1", "t": 18.0}
    draft = next(a for a in result.actions if a["type"] == "draft")
    assert draft == {
        "type": "draft",
        "title": "Follow-up: Pricing review",
        "body": "Hi Sales, Priya will send the updated pricing deck by Friday.",
    }
    draft_prompt = llm.requests[1]["messages"][1]["content"]
    assert (
        "Recipient: Sales" in draft_prompt
        and "pricing deck" in draft_prompt
        and "(Priya, due Friday)" in draft_prompt
    )
    assert [t["tool"] for t in result.tool_trace] == ["seek_to", "draft_follow_up"]
    assert result.answer == "Playing it now and I drafted a follow-up."
    detail = store.get_meeting("m1")
    assert detail is not None
    template = agent_mod.draft_template("Pricing review", detail["action_items"], detail["decisions"], None)
    assert template.startswith("Hi all,") and "- Update the billing page (Rahul, due next week)" in template


async def test_json_text_tool_call_fallback(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            '{"tool": "list_action_items", "args": {"owner": "priya"}}',
            "Priya owes the pricing deck by Friday [u:m1:u00003].",
        ]
    )
    result = await make_agent(store, llm).ask("s6", "What does Priya owe?")
    assert result.tool_trace[0]["tool"] == "list_action_items"
    assert result.tool_trace[0]["result_count"] == 1
    assert [c["utterance_id"] for c in result.citations] == ["m1:u00003"]


async def test_tool_errors_are_reported_not_raised(store: FakeStore) -> None:
    llm = ScriptedLLM(
        [
            [
                ToolCall("complete_action_item", {"description": "quantum roadmap"}),
                ToolCall("nonexistent_tool", {}),
            ],
            "I couldn't find an open item about a quantum roadmap.",
        ]
    )
    result = await make_agent(store, llm).ask("s7", "Mark the quantum roadmap item done.")
    assert all(i["status"] == "open" for i in store.items)
    assert result.tool_trace[0]["result_count"] == 0
    assert "error" in result.tool_trace[1]
    assert result.citations == []


def test_resolve_citations_variants() -> None:
    evidence = {"m1:u00001": {}, "m1:u00002": {}}
    text, cited = resolve_citations(
        "A [u:m1:u00002, u:m1:u00001] b [m1:u00001] c [laughs] d [u:nope] [drafted:true] [ai1].", evidence
    )
    assert cited == ["m1:u00002", "m1:u00001"]
    assert text == "A b c [laughs] d."
    assert resolve_citations("Bare [u:u00001].", evidence) == ("Bare.", ["m1:u00001"])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_heuristic_extract_finds_decision_and_owned_action(store: FakeStore) -> None:
    utterances = [
        *store.utts["m1"],
        {
            "id": "m1:u00006",
            "meeting_id": "m1",
            "seq": 6,
            "speaker": "S1",
            "speaker_name": "Priya",
            "t_start": 36.0,
            "t_end": 40.0,
            "text": "Rahul, can you check the churn numbers before Monday?",
        },
    ]
    insights = heuristic_extract("m1", utterances, SPEAKERS["m1"])
    decision = insights.decisions[0]
    assert decision.evidence == ["m1:u00002"] and decision.proposed_by == "S1"
    assert "twenty five dollars" in decision.text
    deck = next(a for a in insights.action_items if a.evidence == ["m1:u00003"])
    assert deck.owner == "S1" and deck.due == "Friday"
    assert deck.text == "Send the updated pricing deck to the sales team by Friday"
    billing = next(a for a in insights.action_items if a.evidence == ["m1:u00004"])
    assert billing.owner == "S2" and billing.due == "next week"
    churn = next(a for a in insights.action_items if a.evidence == ["m1:u00006"])
    assert churn.owner == "S2" and churn.due == "Monday" and churn.text.startswith("Check the churn numbers")
    assert "pricing" in insights.topics


def test_parse_due_and_json_helpers() -> None:
    assert parse_due("I'll do it by end of day") == "EOD"
    assert parse_due("ship before next Friday please") == "next Friday"
    assert parse_due("no deadline here") is None
    assert parse_json_object('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert parse_json_object('Sure! Here it is: {"x": {"y": "}"}} trailing') == {"x": {"y": "}"}}
    assert parse_json_object("not json") is None


async def test_extractor_validates_model_json(store: FakeStore) -> None:
    reply = """```json
{"decisions": [
   {"text": "Price the Pro plan at $25/month", "evidence": ["u:m1:u00002"], "proposed_by": "Priya",
    "supersedes": "price pro plan at 20 dollars"},
   {"text": "Invented decision", "evidence": ["m7:u00001"]}],
 "action_items": [
   {"text": "Send the pricing deck", "owner": "Priya", "due": "Friday", "evidence": ["m1:u00003", "bogus"]},
   {"text": "Update billing page", "owner": "Speaker 2", "due": null, "evidence": ["[m1:u00004]"]},
   {"text": "No evidence item", "owner": "S1", "evidence": []}],
 "topics": ["Pro plan pricing", "pro plan pricing", "billing"],
 "entities": [{"name": "Pro plan", "kind": "product"}, {"name": "Acme", "kind": "weird"}],
 "mentions": {"m1:u00002": ["Pro plan pricing", "unknown thing"], "zzz": ["billing"]}}
```"""
    existing = [{"id": "d0", "text": "Price the Pro plan at 20 dollars", "status": "active"}]
    llm = ScriptedLLM([reply])
    insights, meta = await Extractor(llm).extract_with_meta("m1", store.utts["m1"], SPEAKERS["m1"], existing)
    assert llm.requests[0]["json_mode"] is True
    prompt = llm.requests[0]["messages"][1]["content"]
    assert "[m1:u00003] Priya (00:18): I'll send the updated pricing deck" in prompt
    assert "Price the Pro plan at 20 dollars" in prompt
    assert meta["provider"] == "scripted" and meta["fallback"] is None
    assert len(insights.decisions) == 1
    d = insights.decisions[0]
    assert d.proposed_by == "S1" and d.evidence == ["m1:u00002"]
    assert d.supersedes_text == "Price the Pro plan at 20 dollars"
    assert [(a.owner, a.due, a.evidence) for a in insights.action_items] == [
        ("S1", "Friday", ["m1:u00003"]),
        ("S2", None, ["m1:u00004"]),
    ]
    assert insights.topics == ["Pro plan pricing", "billing"]
    assert insights.entities[1] == {"name": "Acme", "kind": "other"}
    assert insights.mentions == {"m1:u00002": ["Pro plan pricing"]}


async def test_extractor_falls_back_on_invalid_json_and_fake(store: FakeStore) -> None:
    insights, meta = await Extractor(ScriptedLLM(["I cannot do that."])).extract_with_meta(
        "m1", store.utts["m1"], SPEAKERS["m1"], []
    )
    assert meta["fallback"] == "invalid json" and insights.decisions
    insights2 = await Extractor(FakeLLM()).extract("m1", store.utts["m1"], SPEAKERS["m1"], [])
    assert insights2.action_items and insights2.decisions


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _h(uid: str, via: str) -> dict[str, Any]:
    return {"utterance_id": uid, "via": via, "text": uid, "score": 0.5}


def test_rrf_fusion_order_and_scores() -> None:
    fused = rrf_fuse(
        [
            [_h("x", "fulltext"), _h("y", "fulltext"), _h("z", "fulltext")],
            [_h("y", "vector"), _h("w", "vector")],
            [_h("y", "graph")],
        ],
        k=10,
    )
    assert [h["utterance_id"] for h in fused] == ["y", "x", "w", "z"]
    assert fused[0]["score"] == 1.0 and fused[0]["via"] == "vector"
    expected_x = (1 / 61) / (1 / 62 + 2 / 61)
    assert math.isclose(fused[1]["score"], round(expected_x, 4))
    assert all(0 < h["score"] <= 1 for h in fused)
    assert rrf_fuse([], k=5) == []


def test_retriever_uses_speaker_and_recent_sources(store: FakeStore) -> None:
    retriever = Retriever(store, store.embedder)
    hits, cypher = retriever.retrieve("What did Rahul say about the billing page?", k=5)
    names = [name for name, _ in store.calls]
    assert "utterances_by_speaker" in names and "recent_utterances" not in names
    assert hits[0]["utterance_id"] == "m1:u00004"
    assert len({h["utterance_id"] for h in hits}) == len(hits) <= 5
    assert any("utterances_by_speaker" in c for c in cypher)

    store.calls.clear()
    hits, _ = retriever.retrieve("What was said just now?", meeting_id="m1", k=4)
    assert ("recent_utterances", ("m1",)) in store.calls
    assert hits[0]["utterance_id"] == "m1:u00005"  # newest utterance ranks first for "just now"


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    e = HashEmbedder(64)
    a, b = e.embed_query("pricing deck"), HashEmbedder(64).embed_query("pricing deck")
    assert a == b and math.isclose(float(np.linalg.norm(a)), 1.0, rel_tol=1e-5)
    assert len(e.embed_documents(["", "x"])[0]) == 64


@pytest.mark.slow
def test_gemma_embedder_ranks_pricing_above_hiring() -> None:
    embedder = GemmaEmbedder()
    assert embedder.dim == 768
    q = np.array(embedder.embed_query("What did we decide about pricing?"))
    docs = np.array(
        embedder.embed_documents(
            [
                "We agreed the Pro plan will cost twenty five dollars a month.",
                "We need to hire two backend engineers next quarter.",
            ]
        )
    )
    assert math.isclose(float(np.linalg.norm(q)), 1.0, rel_tol=1e-4)
    pricing, hiring = docs @ q
    assert pricing > hiring


# ---------------------------------------------------------------------------
# Adapters (no network)
# ---------------------------------------------------------------------------


def test_strip_thoughts_and_audio_names() -> None:
    assert strip_thoughts("<|channel>thought\nhmm<channel|>Answer") == "Answer"
    assert strip_thoughts("<think>x</think> <think>y</think>Hi") == "Hi"
    assert all(
        ollama_model_hears_audio(m) for m in ("gemma4", "gemma4:latest", "gemma4:e4b", "gemma4:12b-it-qat")
    )
    assert not any(ollama_model_hears_audio(m) for m in ("gemma4:31b", "gemma4:26b", "qwen2.5vl:3b"))


async def test_ollama_payload_and_tool_call_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<|channel>thought ok<channel|>",
                    "tool_calls": [{"function": {"name": "seek_to", "arguments": '{"t": 12}'}}],
                }
            },
        )

    llm = OllamaLLM("http://ollama.test/", "gemma4:e4b", transport=httpx.MockTransport(handler))
    call = assistant_tool_message("", [ToolCall("list_action_items", {"owner": "Priya"})])
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        call,
        tool_result_message(call["tool_calls"][0], '{"items":[]}'),
    ]
    tools = [{"name": "seek_to", "description": "seek", "parameters": {"type": "object", "properties": {}}}]
    result = await llm.chat(messages, tools=tools, json_mode=True, temperature=0.1, max_tokens=50)

    body = seen["body"]
    assert seen["url"] == "http://ollama.test/api/chat"
    assert body["stream"] is False and body["think"] is False and body["format"] == "json"
    assert body["options"] == {"temperature": 0.1, "num_predict": 50}
    assert body["tools"][0] == {"type": "function", "function": tools[0]}
    assert body["messages"][2]["tool_calls"] == [
        {"function": {"name": "list_action_items", "arguments": {"owner": "Priya"}}}
    ]
    assert body["messages"][3] == {
        "role": "tool",
        "content": '{"items":[]}',
        "tool_name": "list_action_items",
    }
    assert result.text == "" and result.tool_calls == [ToolCall("seek_to", {"t": 12})]
    assert result.provider == "ollama"


async def test_openai_compat_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "who_knows", "arguments": '{"topic":"pricing"}'},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    llm = OpenAICompatLLM(
        "https://inference.example/v1", "key", "gemma-4-31B-it", transport=httpx.MockTransport(handler)
    )
    call = assistant_tool_message("", [ToolCall("who_knows", {"topic": "x"})])
    result = await llm.chat(
        [{"role": "user", "content": "q"}, call, tool_result_message(call["tool_calls"][0], "{}")],
        tools=[{"name": "who_knows", "description": "", "parameters": {}}],
        json_mode=True,
    )
    assert seen["url"] == "https://inference.example/v1/chat/completions" and seen["auth"] == "Bearer key"
    msgs = seen["body"]["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "who_knows", "arguments": '{"topic": "x"}'}
    assert msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert result.tool_calls == [ToolCall("who_knows", {"topic": "pricing"})]


def test_gemini_contents_mapping() -> None:
    call = assistant_tool_message("", [ToolCall("a", {"x": 1}), ToolCall("b", {})])
    system, contents = to_gemini_contents(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q"},
            call,
            tool_result_message(call["tool_calls"][0], '{"ok": true}'),
            tool_result_message(call["tool_calls"][1], "[1]"),
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert system == "be brief"
    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert [p.function_call.name for p in contents[1].parts] == ["a", "b"]
    responses = [p.function_response for p in contents[2].parts]
    assert [(r.name, r.response) for r in responses] == [("a", {"ok": True}), ("b", {"result": [1]})]


async def test_gemini_chat_config_parsing_and_compat_retry() -> None:
    from google.genai import errors
    from google.genai import types as gtypes

    calls: list[dict[str, Any]] = []

    async def generate_content(*, model: str, contents: list[Any], config: Any) -> Any:
        calls.append({"model": model, "contents": contents, "config": config})
        if len(calls) == 1:
            raise errors.ClientError(
                400,
                {"error": {"code": 400, "message": "Developer instruction is not enabled for models/gemma"}},
            )
        parts = [
            gtypes.Part(text="private reasoning", thought=True),
            gtypes.Part.from_text(text="Seeking now."),
            gtypes.Part.from_function_call(name="seek_to", args={"t": 3}),
        ]
        return gtypes.GenerateContentResponse(
            candidates=[gtypes.Candidate(content=gtypes.Content(role="model", parts=parts))]
        )

    llm = GeminiLLM("test-key", "gemma-4-31b-it")
    llm._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    tools = [{"name": "seek_to", "description": "seek", "parameters": {"type": "object", "properties": {}}}]
    result = await llm.chat(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "play it"}],
        tools=tools,
        json_mode=True,
        temperature=0.1,
        max_tokens=40,
    )
    first, second = calls[0]["config"], calls[1]["config"]
    assert first.system_instruction == "be brief" and first.response_mime_type == "application/json"
    assert first.max_output_tokens == 40 and first.automatic_function_calling.disable is True
    assert first.tools[0].function_declarations[0].name == "seek_to"
    assert second.system_instruction is None and second.response_mime_type is None
    assert calls[1]["contents"][0].parts[0].text == "be brief"
    assert result.text == "Seeking now." and result.tool_calls == [ToolCall("seek_to", {"t": 3})]
    assert result.provider == "gemini" and not llm.supports_audio


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "llm_provider": "auto",
        "gemini_api_key": None,
        "gemini_model": "gemma-4-31b-it",
        "openai_base_url": None,
        "openai_api_key": None,
        "openai_model": "gemma-4-31B-it",
        "ollama_url": "http://ollama.test",
        "ollama_model": "gemma4:e4b",
        "question_audio": "auto",
    }
    return SimpleNamespace(**{**base, **overrides})


def test_llm_routing() -> None:
    present = lambda url, model: {"name": model, "capabilities": ["completion", "audio"]}
    absent = lambda url, model: None
    assert isinstance(select_llm(_settings(), probe=present), OllamaLLM)
    assert isinstance(select_llm(_settings(), probe=absent), FakeLLM)
    assert isinstance(select_llm(_settings(gemini_api_key="k"), probe=absent), GeminiLLM)
    assert isinstance(select_llm(_settings(llm_provider="gemini"), probe=present), FakeLLM)
    assert isinstance(
        select_llm(_settings(llm_provider="openai", openai_base_url="http://x"), probe=absent),
        OpenAICompatLLM,
    )
    text_llm = select_llm(_settings(), probe=present)
    assert select_audio_llm(_settings(), text_llm, probe=present) is text_llm
    assert select_audio_llm(_settings(question_audio="whisper"), text_llm, probe=present) is None
    assert select_audio_llm(_settings(), FakeLLM(), probe=absent) is None
    no_audio = lambda url, model: {"name": model, "capabilities": ["completion"]}
    assert select_audio_llm(_settings(), FakeLLM(), probe=no_audio) is None
    assert select_audio_llm(_settings(ollama_model="gemma4:31b"), FakeLLM(), probe=present) is None


# ---------------------------------------------------------------------------
# Voice + summary
# ---------------------------------------------------------------------------


def _wav(seconds: float, rate: int = 44_100, channels: int = 2, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_prepare_question_audio_validates_and_resamples() -> None:
    samples, wav16 = voice_mod.prepare_question_audio(_wav(1.0))
    assert abs(len(samples) - 16_000) <= 2 and samples.dtype == np.float32
    info = sf.info(io.BytesIO(wav16))
    assert (info.samplerate, info.channels) == (16_000, 1)
    for bad, message in [
        (b"", "empty"),
        (b"RIFFjunk", "decode"),
        (_wav(0.1), "at least"),
        (_wav(31.0, rate=8_000, channels=1), "limit"),
        (_wav(1.0, amplitude=0.0), "silent"),
    ]:
        with pytest.raises(ValueError, match=message):
            voice_mod.prepare_question_audio(bad)


async def test_transcribe_question_falls_back_to_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenAudio:
        async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Any:
            raise RuntimeError("audio projector crashed")

    class FakeASR:
        name = "faster-whisper-small"

        def transcribe(self, samples: np.ndarray) -> tuple[str, float]:
            assert samples.dtype == np.float32 and len(samples) > 10_000
            return "  what did priya commit to  ", 0.9

    fake_listen = types.ModuleType("wsw_brain.listen")
    fake_listen.get_asr = lambda: FakeASR()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wsw_brain.listen", fake_listen)
    monkeypatch.setattr(voice_mod, "get_audio_llm", lambda: BrokenAudio())
    transcript = await voice_mod.transcribe_question(_wav(1.0))
    assert transcript.provider == "whisper" and transcript.text == "what did priya commit to"
    assert transcript.model == "faster-whisper-small"


async def test_summaries(store: FakeStore) -> None:
    detail = store.get_meeting("m1")
    assert detail is not None
    text = heuristic_summary(detail)
    assert len(text.split()) <= 80
    assert "Priya" in text and "Rahul" in text and "Price the Pro plan" in text
    assert await summarize_meeting(FakeLLM(), detail) == text
    long_reply = "Summary: " + " ".join(["Priya and Rahul agreed on pricing."] * 30)
    clipped = await summarize_meeting(ScriptedLLM([long_reply]), detail)
    assert len(clipped.split()) <= 80 and clipped.endswith(".") and not clipped.startswith("Summary")
    assert await summarize_meeting(FakeLLM(), {**detail, "utterances": []}) == ""


# ---------------------------------------------------------------------------
# Live: Ollama gemma4:e4b (sequential, small max_tokens)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama() -> OllamaLLM:
    if probe_ollama_model(OLLAMA_URL, OLLAMA_MODEL) is None:
        pytest.skip(f"Ollama model {OLLAMA_MODEL} not available at {OLLAMA_URL}")
    return OllamaLLM(OLLAMA_URL, OLLAMA_MODEL)


@pytest.mark.live
async def test_live_ollama_chat(ollama: OllamaLLM) -> None:
    result = await ollama.chat(
        [{"role": "user", "content": "Reply with exactly one word: ready"}], max_tokens=10, temperature=0.0
    )
    assert result.text.strip() and "ready" in result.text.lower()
    assert result.provider == "ollama" and result.latency_ms > 0


@pytest.mark.live
async def test_live_ollama_tool_call(ollama: OllamaLLM) -> None:
    tools = [
        {
            "name": "complete_action_item",
            "description": "Mark an action item as done.",
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    ]
    result = await ollama.chat(
        [{"role": "user", "content": "Mark the action item about the pricing deck as done."}],
        tools=tools,
        temperature=0.0,
        max_tokens=60,
    )
    assert result.tool_calls, f"no native tool call; text was {result.text!r}"
    assert result.tool_calls[0].name == "complete_action_item"
    assert "pricing" in json.dumps(result.tool_calls[0].arguments).lower()


def _sapi_wav(path: Path, text: str) -> None:
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if sys.platform != "win32" or not powershell:
        pytest.skip("Windows SAPI is needed to synthesise the question audio")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {}; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        timeout=60,
        capture_output=True,
    )


@pytest.mark.live
async def test_live_ollama_transcribes_spoken_question(ollama: OllamaLLM, tmp_path: Path) -> None:
    wav = tmp_path / "question.wav"
    _sapi_wav(wav, "What did Priya commit to in the pricing review?")
    transcript = await voice_mod.transcribe_question(wav.read_bytes(), hint="Priya, Rahul")
    text = transcript.text.lower()
    assert transcript.provider == "gemma4-audio:ollama", transcript
    assert "priya" in text or "pricing" in text, transcript.text


@pytest.mark.live
async def test_live_extractor_with_gemma(ollama: OllamaLLM) -> None:
    window = [
        ("S1", 0, 4, "Okay, the launch date. Marketing wants the twelfth of November."),
        ("S2", 4, 9, "Engineering can make the twelfth if we cut the export feature."),
        ("S1", 9, 13, "Fine, we decided to launch on November 12th without export."),
        ("S2", 13, 18, "I'll write the release notes by Friday."),
        ("S1", 18, 22, "Rahul, can you brief the support team next week?"),
        ("S2", 22, 25, "Yes, I will do the support briefing."),
    ]
    utterances = [
        {
            "id": f"mx:u{i:05d}",
            "meeting_id": "mx",
            "seq": i,
            "speaker": s,
            "speaker_name": None,
            "t_start": float(a),
            "t_end": float(b),
            "text": t,
        }
        for i, (s, a, b, t) in enumerate(window)
    ]
    speakers = [
        {"track": "S1", "label": "Speaker 1", "name": "Priya"},
        {"track": "S2", "label": "Speaker 2", "name": "Rahul"},
    ]
    insights, meta = await Extractor(ollama).extract_with_meta("mx", utterances, speakers, [])
    assert meta["provider"] == "ollama" and meta["fallback"] is None, meta
    items = [*insights.decisions, *insights.action_items]
    assert items, "Gemma 4 found no decision or action item"
    valid = {u["id"] for u in utterances}
    assert all(item.evidence and set(item.evidence) <= valid for item in items)


@pytest.mark.live
async def test_live_agent_answers_and_acts(ollama: OllamaLLM, store: FakeStore) -> None:
    agent = make_agent(store, ollama)
    answer = await agent.ask("live", "What did Priya commit to in the pricing review?")
    assert answer.answer and answer.citations, answer
    assert {c["utterance_id"] for c in answer.citations} <= all_ids(store)
    assert "[u:" not in answer.answer

    action = await agent.ask("live", "Mark it done.")
    # Reported so the run log shows whether Gemma called the tool natively or the intent guard did.
    print(f"\nmark-it-done path: {action.tool_trace} -> {action.answer!r}")
    assert store.items[0]["status"] == "done", action.tool_trace
    assert store.items[1]["status"] == "open"
    assert {"type": "refresh", "what": "action_items"} in action.actions
    assert action.citations and action.citations[0]["utterance_id"] == "m1:u00003"

    seek = await agent.ask("live", "Play the moment where Priya agreed on the price.")
    print(f"seek path: {seek.tool_trace} -> {seek.actions}")
    assert {"type": "seek", "meeting_id": "m1", "t": 11.0} in seek.actions


@pytest.mark.live
async def test_live_brain_on_real_graph_store(ollama: OllamaLLM) -> None:
    """Extraction -> FalkorDB -> retrieval -> Gemma 4 agent action, on a throwaway graph that is deleted."""
    import time
    import uuid

    from wsw_brain.contracts import Utterance
    from wsw_brain.graph.store import GraphStore

    store = GraphStore(settings().falkordb_url, f"wsw_test_brain_{uuid.uuid4().hex[:8]}")
    if not store.ping().get("ok"):
        pytest.skip(f"FalkorDB not reachable at {settings().falkordb_url}")
    try:
        embedder = HashEmbedder(768)
        store.ensure_schema(embedder.dim)
        store.create_meeting("mlive", "Pricing review", "demo", int(time.time() * 1000))
        utterances = [
            Utterance("mlive", i, s, float(a), float(b), t) for i, (s, a, b, t) in enumerate(LINES["m1"])
        ]
        store.write_utterances(utterances, embedder.embed_documents([u.text for u in utterances]))
        store.name_speaker("mlive", "S1", "Priya")
        store.name_speaker("mlive", "S2", "Rahul")
        detail = store.get_meeting("mlive")
        assert detail is not None
        insights = await Extractor(FakeLLM()).extract("mlive", detail["utterances"], detail["speakers"], [])
        store.write_insights("mlive", insights)
        deck = next(i for i in store.list_action_items() if "pricing deck" in i["text"])

        agent = VoiceAgent(store, ollama, Retriever(store, embedder))
        answer = await agent.ask("live-graph", "What did Priya commit to in the pricing review?")
        assert answer.citations and all(c["utterance_id"].startswith("mlive:") for c in answer.citations)
        assert answer.cypher, "the Why panel needs the Cypher that produced the evidence"
        done = await agent.ask("live-graph", "Mark it done.")
        print(f"\nreal-graph mark-it-done path: {done.tool_trace}")
        status = {i["id"]: i["status"] for i in store.list_action_items()}
        assert status[deck["id"]] == "done", done.tool_trace
        assert len(store.get_turns("live-graph")) == 4
    finally:
        store.reset()


@pytest.mark.live
async def test_live_gemini_when_key_present() -> None:
    key = settings().gemini_api_key
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    result = await GeminiLLM(key, settings().gemini_model).chat(
        [{"role": "user", "content": "Reply with exactly one word: ready"}], max_tokens=10
    )
    assert "ready" in result.text.lower()


def test_agent_module_exports_tool_specs() -> None:
    names = [t["name"] for t in agent_mod.tool_specs()]
    assert names == [
        "search_meetings",
        "list_action_items",
        "complete_action_item",
        "list_decisions",
        "who_knows",
        "summarize_meeting",
        "seek_to",
        "draft_follow_up",
    ]
