"""Voice agent: Gemma 4 answers spoken questions over the meeting graph with cited evidence, session
memory and tool calls (search, action items, decisions, who-knows, summaries, seek, follow-up drafts).

Answers are spoken, so `AskResult.answer` is clean text without citation markers; the cited utterances
are in `AskResult.citations` (Hit dicts, in the order the answer cited them)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..brand import product_name
from ..contracts import LLM, AskResult, LLMResult, ToolCall
from .embed import get_embedder
from .extract import content_words, mmss, parse_json_object, speaker_display
from .llm import LLMError, assistant_tool_message, get_llm, tool_result_message
from .retrieve import Retriever

log = logging.getLogger(__name__)

HISTORY_TURNS = 6
EVIDENCE_K = 8
MAX_TOOL_ROUNDS = 3
TOOL_RESULT_CHARS = 2400

_BRACKET = re.compile(r"\s*(?:\[([^\[\]]{1,300})\]|\((u:[^()]{1,300})\))")
_PRONOUN = re.compile(
    r"\b(she|he|they|them|her|him|his|hers|their|that|it|this|those|these|then|there|same)\b", re.IGNORECASE
)
_DONT_KNOW = re.compile(
    r"\b(?:(?:do not|don't|couldn't|could not|can't|cannot|didn't|did not) "
    r"(?:find|see|know|have|contain|cover|mention|say)|no (?:record|mention|information|evidence)|"
    r"not (?:mentioned|discussed|covered|recorded)|(?:isn't|is not|wasn't|was not) "
    r"(?:in|covered|mentioned|discussed)|doesn't (?:say|mention|contain|cover)|nothing (?:in|about))\b",
    re.IGNORECASE,
)
# Non-citation tokens models echo from prompts and tool results: [drafted:true], [item ai1], [ai1], [d3].
_RECORD_TOKEN = re.compile(
    r"[\w.-]+:\S*|(?:action\s+)?(?:item|decision|id)\s*[:#]?\s*[\w.:-]+|[A-Za-z_-]*\d[\w.-]*", re.IGNORECASE
)
_COMPLETE_LEAD = r"^\s*(?:(?:ok(?:ay)?|so|then|and|now|please|yes|yeah|can you|could you|would you)[,\s]+)*"
# Only a whole imperative sentence: "mark it done", "please set the billing page as complete", "close it".
_COMPLETE_INTENT = re.compile(
    _COMPLETE_LEAD + r"(?:mark|set|tick)\b.{0,80}?\b(?:done|complete|completed|finished|closed)"
    r"(?:\s+(?:now|please|then|for me))?[\s.!?]*$"
    r"|" + _COMPLETE_LEAD + r"(?:complete|close)\s+(?:it|that|this)(?:\s+(?:one|item|task))?"
    r"(?:\s+(?:now|please))?[\s.!?]*$",
    re.IGNORECASE,
)
# Sentences that mention completion without asking for it: negations, reopening, reminders, conditions.
_NOT_COMPLETION = re.compile(
    r"\b(?:not|never|no longer|undo|undone|unmark|uncheck|untick|reopen|re-open|(?:back|as|to) open|"
    r"remind|reminder|reminders|when|whenever|once|if|unless|whether|loop|yet)\b|n['’]t\b",
    re.IGNORECASE,
)
# Words in a completion request that say nothing about which item is meant.
_ITEM_FILLER = frozenset({
    "action", "item", "items", "task", "tasks", "todo", "mark", "done", "complete", "completed",
    "finished", "closed",
})  # fmt: skip
_COMPLETE_FILLER = frozenset({
    "ok", "okay", "so", "then", "and", "now", "please", "can", "could", "would", "you", "yes", "yeah",
    "mark", "set", "tick", "check", "off", "as", "to", "for", "me", "done", "complete", "completed",
    "finished", "closed", "close", "finish", "it", "that", "this", "the", "action", "item", "items", "task",
    "one",
})  # fmt: skip
_QUESTION_WORDS = frozenset({
    "what", "when", "who", "whom", "where", "why", "how", "which", "is", "are", "was", "were", "did", "does",
    "do", "her", "his", "their", "our", "my", "your", "its", "a", "an", "of", "for", "to", "by", "due",
    "about", "on", "in", "at", "be", "will", "would", "should",
})  # fmt: skip


def tool_specs() -> list[dict[str, Any]]:
    def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        return {"type": "object", "properties": props, "required": required or []}

    s = {"type": "string"}
    return [
        {
            "name": "search_meetings",
            "description": "Search what was said in meetings. Returns utterances to cite.",
            "parameters": obj(
                {
                    "query": {**s, "description": "what to look for"},
                    "meeting_id": {**s, "description": "limit to one meeting (optional)"},
                },
                ["query"],
            ),
        },
        {
            "name": "list_action_items",
            "description": "List action items (commitments), optionally by owner/status.",
            "parameters": obj(
                {
                    "owner": {**s, "description": "person name or speaker label"},
                    "status": {"type": "string", "enum": ["open", "done"]},
                }
            ),
        },
        {
            "name": "complete_action_item",
            "description": (
                "Mark an action item as done. No id is needed: describe the item in a few words taken from "
                "the conversation (for example 'pricing deck') and the matching open item is updated."
            ),
            "parameters": obj(
                {
                    "description": {**s, "description": "a few words naming the item, e.g. 'pricing deck'"},
                    "item_id": {**s, "description": "the item id, only if a tool result showed it"},
                }
            ),
        },
        {
            "name": "list_decisions",
            "description": "List decisions, including superseded ones.",
            "parameters": obj({"meeting_id": s}),
        },
        {
            "name": "who_knows",
            "description": "Find who talked most about a topic.",
            "parameters": obj({"topic": s}, ["topic"]),
        },
        {
            "name": "summarize_meeting",
            "description": "Get a short summary of a meeting.",
            "parameters": obj({"meeting_id": s}),
        },
        {
            "name": "seek_to",
            "description": "Play a meeting recording from a moment (a cited utterance or a time).",
            "parameters": obj(
                {
                    "meeting_id": s,
                    "t": {"type": "number", "description": "seconds from start"},
                    "utterance_id": {**s, "description": "an id from the evidence, without 'u:'"},
                }
            ),
        },
        {
            "name": "draft_follow_up",
            "description": "Draft (never send) a follow-up message about a meeting.",
            "parameters": obj({"meeting_id": s, "recipient": s}),
        },
    ]


@dataclass
class _Turn:
    """Per-question state shared by the tool handlers."""

    session_id: str
    meeting_id: str | None
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_evidence: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    cypher: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    llm_ms: float = 0.0
    last: LLMResult | None = None

    def add_evidence(self, hits: list[dict[str, Any]], *, from_tool: bool = False) -> None:
        for hit in hits:
            uid = hit.get("utterance_id")
            if not uid:
                continue
            if uid not in self.evidence:
                self.evidence[uid] = hit
            if from_tool and uid not in self.tool_evidence:
                self.tool_evidence.append(uid)

    def add_action(self, action: dict[str, Any]) -> None:
        if action not in self.actions:
            self.actions.append(action)


def _as_hit(u: dict[str, Any], meeting_title: str | None = None, via: str = "graph") -> dict[str, Any]:
    """UtteranceOut or Hit dict -> Hit dict."""
    return {
        "utterance_id": u.get("utterance_id") or u.get("id"),
        "meeting_id": u.get("meeting_id"),
        "meeting_title": u.get("meeting_title") or meeting_title,
        "speaker": u.get("speaker"),
        "speaker_name": u.get("speaker_name"),
        "t_start": u.get("t_start"),
        "t_end": u.get("t_end"),
        "text": u.get("text", ""),
        "score": float(u.get("score") or 0.0),
        "via": u.get("via") or via,
    }


def _who(hit: dict[str, Any]) -> str:
    return speaker_display(hit.get("speaker") or "", {}, hit.get("speaker_name"))


def evidence_line(hit: dict[str, Any]) -> str:
    title = hit.get("meeting_title") or hit.get("meeting_id") or "a meeting"
    return (
        f'[u:{hit["utterance_id"]}] {_who(hit)} in "{title}" at {mmss(float(hit.get("t_start") or 0))}: '
        f"{' '.join(str(hit.get('text', '')).split())}"
    )


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"u:{hit['utterance_id']}",
        "who": _who(hit),
        "meeting": hit.get("meeting_title"),
        "at": mmss(float(hit.get("t_start") or 0)),
        "text": str(hit.get("text", ""))[:220],
    }


def _dump(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= TOOL_RESULT_CHARS else text[: TOOL_RESULT_CHARS - 3] + "..."


def _arg(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("null", "none", "undefined") else None


def parse_time(value: Any) -> float | None:
    """Seconds from a number, "75", "01:15" or "1:01:15"."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    try:
        if ":" in text:
            total = 0.0
            for part in text.split(":"):
                total = total * 60 + float(part)
            return max(0.0, total)
        return max(0.0, float(text))
    except ValueError:
        return None


def is_dont_know(text: str) -> bool:
    return bool(_DONT_KNOW.search(text))


def resolve_citations(text: str, evidence: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Strip every citation marker from `text`; return (clean text, valid cited ids in first-cited order).
    A bracket is removed if it uses the u: prefix, names an evidence id, or is a record token such as
    [drafted:true] or [ai1]; other brackets ("[laughs]") stay. A bare "u00012" resolves when exactly one
    evidence id ends with it."""
    cited: list[str] = []
    by_suffix: dict[str, list[str]] = {}
    for uid in evidence:
        by_suffix.setdefault(uid.rsplit(":", 1)[-1], []).append(uid)

    def resolve(raw: str) -> str:
        uid = raw[2:].strip() if raw.lower().startswith("u:") else raw
        if uid not in evidence and ":" not in uid and len(by_suffix.get(uid, [])) == 1:
            return by_suffix[uid][0]
        return uid

    def replace(match: re.Match[str]) -> str:
        inner = (match.group(1) or match.group(2)).strip()
        ids = [resolve(p.strip()) for p in re.split(r"[,;]|\s+and\s+", inner) if p.strip()]
        is_marker = (
            inner.lower().startswith("u:")
            or any(i in evidence for i in ids)
            or _RECORD_TOKEN.fullmatch(inner) is not None
        )
        if not is_marker:
            return match.group(0)
        for uid in ids:
            if uid in evidence and uid not in cited:
                cited.append(uid)
        return ""

    clean = _BRACKET.sub(replace, text)
    return tidy_spoken(clean), cited


def tidy_spoken(text: str) -> str:
    text = re.sub(r"[*`]+|^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)
    return " ".join(text.split()).strip()


def is_follow_up(question: str) -> bool:
    return len(question.split()) <= 6 or bool(_PRONOUN.search(question))


def completion_request(question: str) -> str | None:
    """For "mark it done" / "mark the billing page as complete": the words naming the item ("" for "it").
    None when the question is not a request to complete an action item."""
    # Test each sentence: "When is her pricing page due? Mark it done." asks for completion in its second
    # sentence, and names the item ("pricing page") in the first.
    sentences = re.split(r"(?<=[.?!;])\s+", question.strip())
    for i, sentence in enumerate(sentences):
        if not _COMPLETE_INTENT.search(sentence) or _NOT_COMPLETION.search(sentence):
            continue
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        description = " ".join(w for w in words if w.lower() not in _COMPLETE_FILLER)
        if description:
            return description
        others = " ".join(sentences[:i] + sentences[i + 1 :])
        hint = [
            w for w in re.findall(r"[A-Za-z0-9']+", others)
            if w.lower() not in _COMPLETE_FILLER and w.lower() not in _QUESTION_WORDS
        ]
        return " ".join(hint)
    return None


def strip_tool_text(text: str, names: set[str]) -> str:
    """Remove tool calls a model printed into its answer ("[complete_action_item(item_id='x')]")."""
    for name in names:
        text = re.sub(rf"\[?\s*{name}\s*\([^()]*\)\s*\]?", "", text)
        text = re.sub(rf"call:{name}\s*\{{[^{{}}]*\}}", "", text)
    return text


_TOOL_TEXT = re.compile(r"call:(\w+)\s*(\{.*\})", re.DOTALL)
# Python-style pseudo calls a small model may print instead of calling: [complete_action_item(item_id='x')]
_TOOL_PYCALL = re.compile(r"\b(\w+)\(([^()]*)\)")
_TOOL_KWARG = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([\w.:-]+))")


def tool_call_from_text(text: str, names: set[str]) -> ToolCall | None:
    """JSON fallback: some runtimes print the call as text ({"tool": name, "args": {...}} or Gemma's
    `call:name{...}`) instead of returning a structured tool call."""
    if not text or not any(n in text for n in names):
        return None
    match = _TOOL_TEXT.search(text)
    if match and match.group(1) in names:
        args = parse_json_object(match.group(2))
        return ToolCall(name=match.group(1), arguments=args or {})
    for py in _TOOL_PYCALL.finditer(text):
        if py.group(1) in names:
            kwargs = {k: next((v for v in vals if v), "") for k, *vals in _TOOL_KWARG.findall(py.group(2))}
            return ToolCall(name=py.group(1), arguments=kwargs)
    data = parse_json_object(text) if text.lstrip().startswith(("{", "`")) else None
    if not data:
        return None
    name = data.get("tool") or data.get("name") or data.get("function")
    args = data.get("args") or data.get("arguments") or data.get("parameters") or {}
    if isinstance(name, str) and name in names and isinstance(args, dict):
        return ToolCall(name=name, arguments=args)
    return None


class VoiceAgent:
    def __init__(
        self,
        store: Any,
        llm: LLM | None = None,
        retriever: Retriever | None = None,
        *,
        max_tokens: int = 220,
        use_tools: bool = True,
    ) -> None:
        self.store = store
        self.llm = llm or get_llm()
        self.retriever = retriever or Retriever(store, get_embedder())
        self.max_tokens = max_tokens
        self.use_tools = use_tools
        self._tools = tool_specs()
        self._handlers: dict[str, Callable[[_Turn, dict[str, Any]], Awaitable[tuple[Any, int]]]] = {
            "search_meetings": self._t_search,
            "list_action_items": self._t_list_items,
            "complete_action_item": self._t_complete_item,
            "list_decisions": self._t_list_decisions,
            "who_knows": self._t_who_knows,
            "summarize_meeting": self._t_summarize,
            "seek_to": self._t_seek,
            "draft_follow_up": self._t_draft,
        }

    # ------------------------------------------------------------------ main flow
    async def ask(self, session_id: str, question: str, meeting_id: str | None = None) -> AskResult:
        t0 = time.perf_counter()
        question = " ".join((question or "").split())
        if not question:
            raise ValueError("question is empty")
        turn = _Turn(session_id=session_id, meeting_id=meeting_id)
        history = (
            await self._store(turn, self.store.get_turns, session_id, HISTORY_TURNS, record=False, soft=True)
            or []
        )

        query, carried = self._contextualise(question, history)
        t_ret = time.perf_counter()
        hits, cypher = await asyncio.to_thread(self.retriever.retrieve, query, meeting_id, EVIDENCE_K)
        turn.cypher.extend(cypher)
        ranked = [_as_hit(h, via=h.get("via", "fulltext")) for h in hits]
        previous = await self._fetch_utterances(turn, carried) if carried else []
        turn.add_evidence(previous)
        turn.add_evidence(ranked)
        context = await self._context_windows(turn, [*previous[:1], *ranked[:2]])
        turn.add_evidence(context)
        linked = await self._linked_records(turn)
        retrieve_ms = (time.perf_counter() - t_ret) * 1000

        messages = [
            {"role": "system", "content": await self._system_prompt(turn)},
            *self._history_messages(history),
            {"role": "user", "content": self._evidence_prompt(question, previous, ranked, context, linked)},
        ]
        answer = await self._converse(turn, messages)
        if not turn.trace and self.use_tools:
            answer = await self._complete_if_requested(turn, question) or answer

        answer = strip_tool_text(answer, set(self._handlers))
        clean, cited = resolve_citations(answer, turn.evidence)
        if not clean:
            clean = "Sorry, I couldn't put an answer together."
        if not cited and turn.evidence and not is_dont_know(clean) and not clean.endswith("?"):
            top = turn.tool_evidence[0] if turn.tool_evidence else next(iter(turn.evidence))
            cited = [top]
        citations = [turn.evidence[uid] for uid in cited]

        await self._remember(turn, question, clean, cited)
        last = turn.last
        return AskResult(
            answer=clean,
            citations=citations,
            actions=turn.actions,
            tool_trace=turn.trace,
            cypher=list(dict.fromkeys(turn.cypher)),
            model=last.model if last else self.llm.model,
            provider=last.provider if last else self.llm.provider,
            session_id=session_id,
            timings={
                "retrieve": round(retrieve_ms, 1),
                "llm": round(turn.llm_ms, 1),
                "tools": round(sum(t.get("ms", 0.0) for t in turn.trace), 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

    async def _complete_if_requested(self, turn: _Turn, question: str) -> str | None:
        """Safety net for a small model that answers "which item?" (or claims success without acting) on an
        explicit "mark it done": run complete_action_item on the item the conversation points at. Only used
        when the model called no tool; the trace entry is tagged fallback="intent"."""
        description = completion_request(question)
        if description is None:
            return None
        output = await self._run_tool(turn, ToolCall("complete_action_item", {"description": description}))
        turn.trace[-1]["fallback"] = "intent"
        result = parse_json_object(output) or {}
        item = result.get("item")
        if turn.trace[-1]["result_count"] == 0 or not isinstance(item, dict):
            return None
        cite = f" [{item['evidence']}]" if item.get("evidence") else ""
        if result.get("already_done"):
            return f'"{item.get("text")}" was already marked done.{cite}'
        return f'Done. I marked "{item.get("text")}" as done.{cite}'

    async def _converse(self, turn: _Turn, messages: list[dict[str, Any]]) -> str:
        tools = self._tools if self.use_tools else None
        names = {t["name"] for t in self._tools}
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            allow_tools = tools if round_no < MAX_TOOL_ROUNDS else None
            try:
                result = await self._chat(turn, messages, tools=allow_tools)
            except LLMError as exc:
                log.warning("agent model call failed: %s", exc)
                turn.trace.append(
                    {"tool": "llm", "args": {}, "result_count": 0, "ms": 0.0, "error": str(exc)}
                )
                return self._extractive_answer(turn)
            calls = result.tool_calls
            if not calls and allow_tools:
                fallback = tool_call_from_text(result.text, names)
                calls = [fallback] if fallback else []
            if not calls or not allow_tools:
                if result.text.strip():
                    return result.text
                if round_no == 0 or not allow_tools:
                    break
                messages.append(
                    {"role": "user", "content": "Now answer the question in at most 3 sentences."}
                )
                tools = None
                continue
            message = assistant_tool_message(result.text, calls)
            messages.append(message)
            for call, wire in zip(calls, message["tool_calls"]):
                messages.append(tool_result_message(wire, await self._run_tool(turn, call)))
        try:
            result = await self._chat(turn, messages, tools=None)
            return result.text
        except LLMError as exc:
            log.warning("agent model call failed: %s", exc)
            return self._extractive_answer(turn)

    async def _chat(
        self,
        turn: _Turn,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        result = await self.llm.chat(
            messages, tools=tools, temperature=0.2, max_tokens=max_tokens or self.max_tokens
        )
        turn.llm_ms += result.latency_ms
        turn.last = result
        return result

    def _extractive_answer(self, turn: _Turn) -> str:
        if not turn.evidence:
            return "I couldn't reach the language model, and I found nothing in the meetings about that."
        uid, hit = next(iter(turn.evidence.items()))
        return (
            f"I couldn't reach the language model, but the closest moment is {_who(hit)} saying: "
            f'"{hit.get("text", "")}" [u:{uid}]'
        )

    # ------------------------------------------------------------------ context
    def _contextualise(self, question: str, history: list[dict[str, Any]]) -> tuple[str, list[str]]:
        """Follow-ups ("and when is that due?") borrow the previous question and its citations."""
        if not history or not is_follow_up(question):
            return question, []
        prev_question = next((t.get("text", "") for t in reversed(history) if t.get("role") == "user"), "")
        prev_cites = next(
            (t.get("citations") or [] for t in reversed(history) if t.get("role") == "assistant"), []
        )
        query = f"{prev_question} {question}".strip()
        return query, [c for c in prev_cites if isinstance(c, str)][:4]

    @staticmethod
    def _history_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": t["role"], "content": t.get("text", "")}
            for t in history[-HISTORY_TURNS:]
            if t.get("role") in ("user", "assistant") and t.get("text")
        ]

    async def _system_prompt(self, turn: _Turn) -> str:
        lines = [
            (
                f"You are the voice of {product_name()}, a memory of the team's meetings. "
                "You answer spoken questions."
            ),
            "- Answer in at most 3 short spoken sentences. Plain conversational words: no lists, no markdown.",
            (
                "- Cite every claim with the marker of the utterance that supports it, like "
                "[u:<utterance_id>]. Use ONLY ids shown in the evidence or in tool results; never invent one."
            ),
            '- Say who said it, and when if it helps ("Priya said at 03:12 that ...").',
            (
                "- If the evidence does not contain the answer, say plainly that the meetings don't cover "
                "it and cite nothing."
            ),
            '- Use the conversation so far to resolve words like "she", "that" or "it".',
            (
                "- When the user asks you to do something, call the matching tool right away instead of "
                "asking for ids: complete_action_item to mark an item done (describe it in a few words "
                "from the conversation), seek_to to play a moment, draft_follow_up to draft a message. "
                "Call search_meetings only when the evidence is not enough; otherwise answer from the "
                "evidence."
            ),
        ]
        meetings = await self._store(turn, self.store.list_meetings, record=False, soft=True) or []
        if turn.meeting_id:
            current = next((m for m in meetings if m.get("id") == turn.meeting_id), None)
            title = current.get("title") if current else turn.meeting_id
            lines.append(f'Current meeting: "{title}" (id {turn.meeting_id}).')
        if meetings:
            listed = "; ".join(f'{m["id"]} "{m.get("title", "")}"' for m in meetings[:6])
            lines.append(f"Meetings (newest first): {listed}.")
        return "\n".join(lines)

    @staticmethod
    def _evidence_prompt(
        question: str,
        previous: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
        context: list[dict[str, Any]],
        linked: str = "",
    ) -> str:
        shown: set[str] = set()

        def section(title: str, hits: list[dict[str, Any]]) -> str:
            rows = []
            for hit in hits:
                if hit["utterance_id"] not in shown:
                    shown.add(hit["utterance_id"])
                    rows.append(evidence_line(hit))
            return f"{title}:\n" + "\n".join(rows) if rows else ""

        parts = [
            section("Sources of your previous answer", previous),
            section("Evidence (most relevant first)", ranked),
            section("Surrounding context", context),
            linked,
            (
                'This is a follow-up: words like "it", "that", "she" or "he" refer to the subject of your '
                "previous answer."
                if previous
                else ""
            ),
        ]
        body = "\n\n".join(p for p in parts if p) or "Evidence: none found in the meetings."
        return f"{body}\n\nQuestion: {question}"

    async def _linked_records(self, turn: _Turn) -> str:
        """Action items and decisions EVIDENCED_BY the evidence utterances, so the model can act on "it"."""
        items = await self._store(turn, self.store.list_action_items, soft=True) or []
        decisions = await self._store(turn, self.store.list_decisions, soft=True) or []
        open_items = self._conversation_items(items, turn)
        done_items = self._conversation_items(items, turn, status="done")
        items = [*open_items, *done_items][:5]  # open first: "mark it done" should land on an open item
        decisions = self._conversation_items(decisions, turn, status=None)[:4]
        lines = []
        if items:
            lines.append("Action items linked to this evidence:")
            for i in items:
                extras = [
                    f"owner {i['owner']}" if i.get("owner") else "",
                    f"due {i['due']}" if i.get("due") else "",
                    "open" if i.get("status") != "done" else "already done",
                ]
                lines.append(
                    f"- item {i.get('id')}: {i.get('text')} ({', '.join(x for x in extras if x)}) "
                    f"[u:{self._pool_id(i, turn)}]"
                )
        if decisions:
            lines.append("Decisions linked to this evidence:")
            lines += [
                f"- {d.get('text')} ({d.get('status') or 'active'}) [u:{self._pool_id(d, turn)}]"
                for d in decisions
            ]
        return "\n".join(lines)

    @staticmethod
    def _pool_id(record: dict[str, Any], turn: _Turn) -> str:
        """The record's evidence id to show the model: one already in this turn's evidence (the one that
        linked it), so citing it survives resolve_citations; the store's first id otherwise."""
        evidence = record.get("evidence") or []
        return next((e for e in evidence if e in turn.evidence), evidence[0] if evidence else "")

    async def _context_windows(self, turn: _Turn, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for anchor in anchors:
            rows = (
                await self._store(turn, self.store.context_window, anchor["utterance_id"], 2, 2, soft=True)
                or []
            )
            out += [_as_hit(r, anchor.get("meeting_title")) for r in rows]
        return sorted(
            {h["utterance_id"]: h for h in out}.values(),
            key=lambda h: (str(h.get("meeting_id")), float(h.get("t_start") or 0)),
        )

    async def _fetch_utterances(self, turn: _Turn, ids: list[str]) -> list[dict[str, Any]]:
        ids = [i for i in dict.fromkeys(ids) if i]
        if not ids:
            return []
        rows = await self._store(turn, self.store.get_utterances, ids, soft=True) or []
        return [_as_hit(r) for r in rows]

    async def _remember(self, turn: _Turn, question: str, answer: str, cited: list[str]) -> None:
        now = int(time.time() * 1000)
        try:
            await asyncio.to_thread(self.store.save_turn, turn.session_id, "user", question, now)
            await asyncio.to_thread(
                self.store.save_turn, turn.session_id, "assistant", answer, now + 1, cited
            )
        except Exception as exc:  # noqa: BLE001 - answer stays valid; only memory is lost
            log.warning("could not save conversation turns for %s: %s", turn.session_id, exc)

    async def _store(
        self,
        turn: _Turn,
        fn: Callable[..., Any],
        *args: Any,
        record: bool = True,
        soft: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a sync GraphStore call off the event loop and collect the Cypher it ran. `soft` calls feed
        optional context, so a failure is logged and yields None instead of failing the question."""

        def run() -> tuple[Any, list[str]]:
            value = fn(*args, **kwargs)
            return value, list(getattr(self.store, "last_cypher", None) or [])

        try:
            value, cypher = await asyncio.to_thread(run)
        except Exception as exc:
            if not soft:
                raise
            log.warning("graph read %s failed: %s", getattr(fn, "__name__", fn), exc)
            return None
        if record:
            turn.cypher.extend(cypher)
        return value

    # ------------------------------------------------------------------ tools
    async def _run_tool(self, turn: _Turn, call: ToolCall) -> str:
        t0 = time.perf_counter()
        handler = self._handlers.get(call.name)
        entry: dict[str, Any] = {"tool": call.name, "args": call.arguments, "result_count": 0}
        if handler is None:
            result: Any = {"error": f"unknown tool {call.name!r}"}
            entry["error"] = result["error"]
        else:
            try:
                result, entry["result_count"] = await handler(turn, call.arguments or {})
            except Exception as exc:  # noqa: BLE001 - reported to the model and the trace
                log.warning("tool %s failed: %s", call.name, exc)
                result = {"error": f"{type(exc).__name__}: {exc}"}
                entry["error"] = result["error"]
        entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        turn.trace.append(entry)
        return _dump(result)

    async def _t_search(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        query = _arg(args, "query")
        if not query:
            return {"error": "query is required"}, 0
        meeting_id = await self._resolve_meeting(turn, _arg(args, "meeting_id"))
        hits, cypher = await asyncio.to_thread(self.retriever.retrieve, query, meeting_id, 6)
        turn.cypher.extend(cypher)
        hits = [_as_hit(h, via=h.get("via", "fulltext")) for h in hits]
        turn.add_evidence(hits, from_tool=True)
        return {"results": [_compact_hit(h) for h in hits]}, len(hits)

    async def _t_list_items(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        owner, status = _arg(args, "owner"), _arg(args, "status")
        status = status.lower() if status and status.lower() in ("open", "done") else None
        items = await self._store(turn, self.store.list_action_items, status=status, owner=owner) or []
        if not items and owner:  # the store's owner match may be exact; widen to a case-insensitive contains
            everything = await self._store(turn, self.store.list_action_items, status=status) or []
            needle = owner.lower()
            items = [
                i
                for i in everything
                if needle in f"{i.get('owner') or ''} {i.get('owner_track') or ''}".lower()
            ]
        items = items[:10]
        await self._join_item_evidence(turn, items)
        return {"items": [self._compact_item(i) for i in items]}, len(items)

    async def _t_complete_item(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        item_id, description = _arg(args, "item_id"), _arg(args, "description")
        items = await self._store(turn, self.store.list_action_items) or []
        target = next((i for i in items if item_id and i.get("id") == item_id), None)
        candidates: list[dict[str, Any]] = []
        if target is None:
            target, candidates = self._match_item(items, description or item_id, turn)
        if target is None and candidates:
            return {
                "error": "several action items match equally well; nothing changed. Ask the user which one",
                "candidates": [self._compact_item(i) for i in candidates[:4]],
            }, 0
        if target is None:
            open_items = [self._compact_item(i) for i in items if i.get("status") == "open"][:6]
            return {"error": "no open action item matches", "open_items": open_items}, 0
        if target.get("status") == "done":
            await self._join_item_evidence(turn, [target])
            others = [self._compact_item(i) for i in self._conversation_items(items, turn) if i is not target]
            return {
                "error": f"item {target.get('id')} was already done before this request; nothing changed",
                "already_done": True,
                "item": self._compact_item(target),
                "open_items_in_this_conversation": others[:4],
            }, 1
        updated = await self._store(turn, self.store.set_action_item_status, target["id"], "done")
        if updated is None:
            return {"error": f"action item {target['id']} could not be updated"}, 0
        await self._join_item_evidence(turn, [updated])
        turn.add_action({"type": "refresh", "what": "action_items"})
        return {"item": self._compact_item(updated), "status": "done"}, 1

    async def _t_list_decisions(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        meeting_id = await self._resolve_meeting(turn, _arg(args, "meeting_id")) or turn.meeting_id
        decisions = (await self._store(turn, self.store.list_decisions, meeting_id) or [])[:10]
        # Fetch every id the result shows, so the model can cite any of them.
        await self._join_evidence_ids(turn, [d.get("evidence") or [] for d in decisions], per_record=2)
        rows = [
            {
                "id": d.get("id"),
                "text": d.get("text"),
                "status": d.get("status"),
                "meeting_id": d.get("meeting_id"),
                "superseded_by": d.get("superseded_by"),
                "evidence": [f"u:{e}" for e in (d.get("evidence") or [])[:2]],
            }
            for d in decisions
        ]
        return {"decisions": rows}, len(rows)

    async def _t_who_knows(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        topic = _arg(args, "topic")
        if not topic:
            return {"error": "topic is required"}, 0
        rows = await self._store(turn, self.store.who_knows, topic, 5) or []
        await self._join_evidence_ids(turn, [[(r.get("sample") or {}).get("id")] for r in rows])
        people = []
        for r in rows:
            sample_id = (r.get("sample") or {}).get("id")
            people.append(
                {
                    "name": r.get("name"),
                    "hits": r.get("hits"),
                    "meeting_id": r.get("meeting_id"),
                    "sample": f"u:{sample_id}" if sample_id else None,
                }
            )
        return {"people": people}, len(people)

    async def _t_summarize(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        meeting_id = await self._resolve_meeting(
            turn, _arg(args, "meeting_id")
        ) or await self._default_meeting(turn)
        if not meeting_id:
            return {"error": "no meeting to summarise"}, 0
        detail = await self._store(turn, self.store.get_meeting, meeting_id)
        if not detail:
            return {"error": f"meeting {meeting_id} not found"}, 0
        meeting = detail.get("meeting") or {}
        decisions = [d for d in detail.get("decisions", []) if d.get("status") != "superseded"][:5]
        items = detail.get("action_items", [])[:6]
        # Show an evidence id with every record and fetch exactly those, so each one shown can be cited.
        await self._join_evidence_ids(
            turn, [d.get("evidence") or [] for d in decisions] + [i.get("evidence") or [] for i in items]
        )
        result: dict[str, Any] = {
            "meeting": meeting.get("title"),
            "meeting_id": meeting_id,
            "decisions": [
                {"text": d.get("text"), "evidence": f"u:{d['evidence'][0]}" if d.get("evidence") else None}
                for d in decisions
            ],
            "action_items": [self._compact_item(i) for i in items],
        }
        if meeting.get("summary"):
            result["summary"] = meeting["summary"]
        else:
            recent = await self._store(turn, self.store.recent_utterances, meeting_id, 12) or []
            hits = [_as_hit(h, meeting.get("title"), via="recent") for h in recent]
            turn.add_evidence(hits, from_tool=True)
            result["recent"] = [_compact_hit(h) for h in hits]
        return result, 1

    async def _t_seek(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        utterance_id = _arg(args, "utterance_id")
        if utterance_id and utterance_id.lower().startswith("u:"):
            utterance_id = utterance_id[2:]
        meeting_id, t = None, parse_time(args.get("t"))
        if utterance_id:
            utterance_id = self._full_utterance_id(turn, utterance_id, _arg(args, "meeting_id"))
            hit = turn.evidence.get(utterance_id)
            if hit is None:
                fetched = await self._fetch_utterances(turn, [utterance_id])
                hit = fetched[0] if fetched else None
            if hit is None:
                return {"error": f"utterance {utterance_id} not found"}, 0
            turn.add_evidence([hit], from_tool=True)
            meeting_id, t = hit.get("meeting_id"), float(hit.get("t_start") or 0.0)
        else:
            meeting_id = await self._resolve_meeting(turn, _arg(args, "meeting_id")) or turn.meeting_id
        if not meeting_id or t is None:
            return {"error": "seek_to needs an utterance_id, or a meeting_id and a time t"}, 0
        turn.add_action({"type": "seek", "meeting_id": meeting_id, "t": round(t, 2)})
        return {"ok": True, "meeting_id": meeting_id, "t": round(t, 2)}, 1

    async def _t_draft(self, turn: _Turn, args: dict[str, Any]) -> tuple[Any, int]:
        meeting_id = await self._resolve_meeting(
            turn, _arg(args, "meeting_id")
        ) or await self._default_meeting(turn)
        if not meeting_id:
            return {"error": "no meeting to follow up on"}, 0
        detail = await self._store(turn, self.store.get_meeting, meeting_id)
        if not detail:
            return {"error": f"meeting {meeting_id} not found"}, 0
        title = (detail.get("meeting") or {}).get("title") or meeting_id
        items = detail.get("action_items", [])
        decisions = [d for d in detail.get("decisions", []) if d.get("status") != "superseded"]
        body = await self._draft_body(turn, title, items, decisions, _arg(args, "recipient"))
        turn.add_action({"type": "draft", "title": f"Follow-up: {title}", "body": body})
        return {
            "drafted": True,
            "title": f"Follow-up: {title}",
            "action_items": len(items),
            "note": "the draft is shown to the user; it is never sent",
        }, 1

    # ------------------------------------------------------------------ tool helpers
    @staticmethod
    def _conversation_items(
        records: list[dict[str, Any]], turn: _Turn, status: str | None = "open"
    ) -> list[dict[str, Any]]:
        """Records (action items or decisions) evidenced by this turn's evidence, in evidence order; the
        previous answer's sources come first, so "it" resolves to what was just discussed."""
        order = {uid: i for i, uid in enumerate(turn.evidence)}

        def rank(record: dict[str, Any]) -> int:
            return min((order[e] for e in record.get("evidence") or [] if e in order), default=-1)

        kept = [r for r in records if rank(r) >= 0 and (status is None or r.get("status", "open") == status)]
        return sorted(kept, key=rank)

    @staticmethod
    def _full_utterance_id(turn: _Turn, raw: str, meeting_id: str | None) -> str:
        """Models sometimes drop the meeting prefix ("u00002"); recover it from the evidence or meeting_id."""
        if raw in turn.evidence or ":" in raw:
            return raw
        matches = [uid for uid in turn.evidence if uid.endswith(f":{raw}")]
        if len(matches) == 1:
            return matches[0]
        return f"{meeting_id}:{raw}" if meeting_id else raw

    @staticmethod
    def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
        evidence = item.get("evidence") or []
        return {
            "id": item.get("id"),
            "text": item.get("text"),
            "owner": item.get("owner"),
            "due": item.get("due"),
            "status": item.get("status"),
            "meeting": item.get("meeting_title"),
            "evidence": f"u:{evidence[0]}" if evidence else None,
        }

    def _match_item(
        self, items: list[dict[str, Any]], description: str | None, turn: _Turn
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The item a completion request describes: (item, []), (None, []) when nothing fits, or (None,
        candidates) when several fit equally well. Done items are candidates too, so an item that is
        already done is reported as such instead of a weaker open match being completed. Ties go to the
        item this conversation is about, then to the current meeting, then to an open item."""
        wanted = content_words(description or "") - _ITEM_FILLER
        if not wanted:
            # "mark it done" with no description: the open item evidenced by this conversation's evidence
            linked = self._conversation_items([i for i in items if i.get("status") == "open"], turn)
            return (linked[0] if linked else None), []
        # One or two words must all match ("pricing deck" never lands on "pricing page"); longer
        # descriptions need half their words.
        needed = 1.0 if len(wanted) <= 2 else 0.5
        linked_rank = {
            r.get("id"): n for n, r in enumerate(self._conversation_items(items, turn, status=None))
        }
        scored: list[tuple[tuple[float, int, bool, bool], dict[str, Any]]] = []
        for item in items:
            words = content_words(f"{item.get('text', '')} {item.get('owner') or ''}")
            score = len(wanted & words) / len(wanted)
            if score >= needed:
                key = (
                    score,
                    -linked_rank.get(item.get("id"), len(items) + 1),
                    bool(turn.meeting_id) and item.get("meeting_id") == turn.meeting_id,
                    item.get("status") != "done",
                )
                scored.append((key, item))
        if not scored:
            return None, []
        scored.sort(key=lambda s: s[0], reverse=True)
        best_key = scored[0][0]
        tied = [item for key, item in scored if key == best_key]
        return (tied[0], []) if len(tied) == 1 else (None, tied)

    async def _join_item_evidence(self, turn: _Turn, items: list[dict[str, Any]]) -> None:
        await self._join_evidence_ids(turn, [i.get("evidence") or [] for i in items])

    async def _join_evidence_ids(
        self, turn: _Turn, groups: list[list[str | None]], per_record: int = 1
    ) -> None:
        """Fetch the first `per_record` evidence ids of each record (the ids the tool result shows)."""
        ids = [uid for g in groups for uid in g[:per_record] if uid]
        hits = await self._fetch_utterances(turn, ids)
        turn.add_evidence(hits, from_tool=True)

    async def _resolve_meeting(self, turn: _Turn, raw: str | None) -> str | None:
        if not raw:
            return None
        meetings = await self._store(turn, self.store.list_meetings, record=False) or []
        ids = {m.get("id") for m in meetings}
        if raw in ids:
            return raw
        wanted = content_words(raw)
        scored = [(len(wanted & content_words(m.get("title", ""))), m.get("id")) for m in meetings]
        best = max(scored, default=(0, None))
        return best[1] if best[0] > 0 else None

    async def _default_meeting(self, turn: _Turn) -> str | None:
        if turn.meeting_id:
            return turn.meeting_id
        for hit in turn.evidence.values():
            if hit.get("meeting_id"):
                return hit["meeting_id"]
        meetings = await self._store(turn, self.store.list_meetings, record=False) or []
        return meetings[0]["id"] if meetings else None

    async def _draft_body(
        self,
        turn: _Turn,
        title: str,
        items: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        recipient: str | None,
    ) -> str:
        template = draft_template(title, items, decisions, recipient)
        if self.llm.provider == "fake":
            return template
        facts = template.split("\n", 2)[-1]
        messages = [
            {
                "role": "system",
                "content": "You write short, friendly follow-up emails after meetings. Plain text, "
                "no markdown, under 120 words. Use only the facts given; never invent "
                "tasks, owners or dates.",
            },
            {
                "role": "user",
                "content": f"Meeting: {title}\nRecipient: {recipient or 'the attendees'}\n{facts}\n\n"
                "Write the email body only, starting with a greeting.",
            },
        ]
        try:
            result = await self._chat(turn, messages, tools=None, max_tokens=260)
        except LLMError as exc:
            log.warning("draft generation failed, using the template: %s", exc)
            return template
        return result.text.strip() or template


def draft_template(
    title: str, items: list[dict[str, Any]], decisions: list[dict[str, Any]], recipient: str | None
) -> str:
    lines = [f"Hi {recipient or 'all'},", "", f'Thanks for joining "{title}". A quick recap:']
    if decisions:
        lines.append("Decisions:")
        lines += [f"- {d.get('text')}" for d in decisions[:6]]
    if items:
        lines.append("Action items:")
        for i in items[:10]:
            extras = ", ".join(
                x
                for x in (
                    i.get("owner"),
                    f"due {i['due']}" if i.get("due") else None,
                    "done" if i.get("status") == "done" else None,
                )
                if x
            )
            lines.append(f"- {i.get('text')}" + (f" ({extras})" if extras else ""))
    if not decisions and not items:
        lines.append("No decisions or action items were recorded.")
    lines += ["", "Thanks!"]
    return "\n".join(lines)
