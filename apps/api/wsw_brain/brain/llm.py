"""Gemma 4 adapters behind contracts.LLM: Ollama (local, audio), Gemini API, OpenAI-compatible, and a fake.

Message format (extends the contract's): every message is {"role", "content"}; an assistant message may
also carry "tool_calls": [{"id", "name", "arguments"}], and a tool message carries "name" (tool name) and
"tool_call_id". Build those two with `assistant_tool_message` and `tool_result_message` so every adapter
can round-trip a function-calling exchange.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from functools import lru_cache
from typing import Any

import httpx

from ..config import Settings, settings
from ..contracts import LLM, LLMResult, ToolCall, Transcript

log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=60.0, pool=5.0)
PROBE_TIMEOUT_S = 2.0

_THOUGHT_BLOCK = re.compile(r"^\s*(?:<\|channel>thought.*?<channel\|>|<think>.*?</think>)", re.DOTALL)
_EVIDENCE_LINE = re.compile(r"^\[u:([^\]]+)\]\s*(.+)$", re.MULTILINE)


class LLMError(RuntimeError):
    """A model call failed (network, HTTP status, or an unusable response)."""


# ---------------------------------------------------------------------------
# Helpers shared by the adapters
# ---------------------------------------------------------------------------


def strip_thoughts(text: str) -> str:
    """Remove leading Gemma `<|channel>thought ... <channel|>` or `<think>...</think>` blocks."""
    previous = None
    while previous != text:
        previous = text
        text = _THOUGHT_BLOCK.sub("", text, count=1)
    return text.strip()


def assistant_tool_message(text: str, calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": f"call_{uuid.uuid4().hex[:10]}", "name": c.name, "arguments": c.arguments} for c in calls
        ],
    }


def tool_result_message(call: dict[str, Any], content: str) -> dict[str, Any]:
    """`call` is one entry of an assistant message's "tool_calls"."""
    return {"role": "tool", "name": call["name"], "tool_call_id": call["id"], "content": content}


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return value if isinstance(value, dict) else {"_value": value}
    return {}


def ollama_model_hears_audio(model: str) -> bool:
    """Gemma 4 takes audio only on E2B, E4B and 12B; bare `gemma4` / `gemma4:latest` is E4B."""
    name = model.lower().strip()
    if name in ("gemma4", "gemma4:latest"):
        return True
    return "gemma4" in name and any(size in name for size in ("e2b", "e4b", "12b"))


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# ---------------------------------------------------------------------------
# Ollama (laptop: Gemma 4 E4B with native audio)
# ---------------------------------------------------------------------------


TRANSCRIBE_PROMPT = (
    "Transcribe this audio clip of a spoken question exactly, in English. "
    "Output only the transcript on a single line, with no preamble, quotes or commentary. "
    "Write numbers as digits."
)


def _clean_transcript(text: str) -> str:
    text = strip_thoughts(text).strip()
    text = re.sub(r"^(?:transcript(?:ion)?|question)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


class OllamaLLM:
    provider = "ollama"

    def __init__(
        self, base_url: str, model: str, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.supports_audio = ollama_model_hears_audio(model)
        self._transport = transport

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(m) for m in messages],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": self._function(t)} for t in tools]
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        data = await self._post_chat(payload)
        message = data.get("message") or {}
        calls = [
            ToolCall(name=fn.get("name", ""), arguments=_parse_arguments(fn.get("arguments")))
            for fn in ((c.get("function") or {}) for c in message.get("tool_calls") or [])
            if fn.get("name")
        ]
        return LLMResult(
            text=strip_thoughts(message.get("content") or ""),
            tool_calls=calls,
            model=self.model,
            provider=self.provider,
            latency_ms=_ms_since(t0),
        )

    async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
        if not self.supports_audio:
            raise NotImplementedError(
                f"Ollama model {self.model!r} has no audio input (needs Gemma 4 E2B/E4B/12B)"
            )
        prompt = TRANSCRIBE_PROMPT
        if hint:
            prompt += f" Names and terms that may occur: {hint}."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": [base64.b64encode(wav_bytes).decode("ascii")]}
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 160},
        }
        t0 = time.perf_counter()
        data = await self._post_chat(payload)
        text = _clean_transcript((data.get("message") or {}).get("content") or "")
        if not text:
            raise LLMError(f"Ollama {self.model} returned an empty transcript")
        return Transcript(
            text=text, model=self.model, provider="gemma4-audio:ollama", latency_ms=_ms_since(t0)
        )

    @staticmethod
    def _function(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }

    @staticmethod
    def _message(m: dict[str, Any]) -> dict[str, Any]:
        role = m["role"]
        out: dict[str, Any] = {"role": role, "content": m.get("content") or ""}
        if role == "assistant" and m.get("tool_calls"):
            out["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                for c in m["tool_calls"]
            ]
        elif role == "tool":
            out["tool_name"] = m.get("name", "")
        if m.get("images"):
            out["images"] = m["images"]
        return out

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT, transport=self._transport) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 400 and "think" in response.text.lower() and "think" in payload:
                    # Models without a thinking switch reject the field; retry without it.
                    payload = {k: v for k, v in payload.items() if k != "think"}
                    response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request to {url} failed: {exc!r}") from exc
        if response.status_code != 200:
            raise LLMError(f"Ollama {url} returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned non-JSON: {response.text[:200]}") from exc
        if data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")
        return data


# ---------------------------------------------------------------------------
# Gemini API (hosted Gemma 4 26B/31B: text only)
# ---------------------------------------------------------------------------


class GeminiLLM:
    provider = "gemini"
    supports_audio = False  # the Gemini API serves only 26B A4B / 31B, which take no audio

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        if not api_key:
            raise ValueError("GeminiLLM needs an API key (GEMINI_API_KEY)")
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._compat = False  # set when the served model rejects system instructions or JSON mode

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        from google.genai import errors

        t0 = time.perf_counter()
        try:
            response = await self._generate(messages, tools, json_mode, temperature, max_tokens)
        except errors.ClientError as exc:
            if self._compat or not _is_feature_rejection(exc):
                raise LLMError(f"Gemini API error ({self.model}): {exc}") from exc
            log.info("Gemini %s rejected system_instruction/JSON mode; retrying in compat mode", self.model)
            self._compat = True
            try:
                response = await self._generate(messages, tools, json_mode, temperature, max_tokens)
            except (errors.APIError, httpx.HTTPError) as exc2:
                raise LLMError(f"Gemini API error ({self.model}): {exc2!r}") from exc2
        except errors.APIError as exc:
            raise LLMError(f"Gemini API error ({self.model}): {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini API request failed: {exc!r}") from exc
        text, calls = _read_gemini_response(response)
        return LLMResult(
            text=text, tool_calls=calls, model=self.model, provider=self.provider, latency_ms=_ms_since(t0)
        )

    async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
        raise NotImplementedError("Gemma 4 on the Gemini API (26B/31B) has no audio input")

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        from google.genai import types

        system, contents = to_gemini_contents(messages)
        if self._compat and system:
            contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text=system)]))
            contents = _merge_same_role(contents)
            system = ""
        if self._compat and json_mode:
            contents.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text="Reply with one JSON object only.")]
                )
            )
            contents = _merge_same_role(contents)
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"],
                            description=t.get("description", ""),
                            parameters_json_schema=t.get("parameters")
                            or {"type": "object", "properties": {}},
                        )
                        for t in tools
                    ]
                )
            ]
            if tools
            else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if tools else None,
            response_mime_type="application/json" if json_mode and not self._compat else None,
        )
        return await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )


def _is_feature_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("instruction", "json mode", "response_mime_type", "mime type"))


def to_gemini_contents(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Contract messages -> (system instruction, [types.Content]). Tool results become function responses."""
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []
    for m in messages:
        role, text = m["role"], m.get("content") or ""
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        elif role == "assistant":
            parts = [types.Part.from_text(text=text)] if text else []
            parts += [
                types.Part.from_function_call(name=c["name"], args=c.get("arguments") or {})
                for c in m.get("tool_calls") or []
            ]
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            try:
                result: Any = json.loads(text)
            except json.JSONDecodeError:
                result = text
            response = result if isinstance(result, dict) else {"result": result}
            part = types.Part.from_function_response(name=m.get("name", "tool"), response=response)
            contents.append(types.Content(role="user", parts=[part]))
        else:
            raise ValueError(f"unknown message role {role!r}")
    return "\n\n".join(system_parts), _merge_same_role(contents)


def _merge_same_role(contents: list[Any]) -> list[Any]:
    """Gemini wants alternating turns; consecutive same-role contents (e.g. several tool results) merge."""
    from google.genai import types

    merged: list[Any] = []
    for c in contents:
        if merged and merged[-1].role == c.role:
            merged[-1] = types.Content(role=c.role, parts=[*(merged[-1].parts or []), *(c.parts or [])])
        else:
            merged.append(c)
    return merged


def _read_gemini_response(response: Any) -> tuple[str, list[ToolCall]]:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for candidate in (response.candidates or [])[:1]:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if getattr(part, "thought", False):
                continue
            if part.function_call is not None:
                calls.append(
                    ToolCall(
                        name=part.function_call.name or "", arguments=dict(part.function_call.args or {})
                    )
                )
            elif part.text:
                texts.append(part.text)
    return strip_thoughts("".join(texts)), [c for c in calls if c.name]


# ---------------------------------------------------------------------------
# OpenAI-compatible (DigitalOcean, OpenRouter, llama.cpp server)
# ---------------------------------------------------------------------------


class OpenAICompatLLM:
    provider = "openai"
    supports_audio = False

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = f"{base if base.endswith('/v1') else base + '/v1'}/chat/completions"
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._transport = transport

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": OllamaLLM._function(t)} for t in tools]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT, transport=self._transport) as client:
                response = await client.post(self.url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"request to {self.url} failed: {exc!r}") from exc
        if response.status_code != 200:
            raise LLMError(f"{self.url} returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat completion body: {response.text[:300]}") from exc
        calls = [
            ToolCall(name=fn.get("name", ""), arguments=_parse_arguments(fn.get("arguments")))
            for fn in ((c.get("function") or {}) for c in message.get("tool_calls") or [])
            if fn.get("name")
        ]
        return LLMResult(
            text=strip_thoughts(message.get("content") or ""),
            tool_calls=calls,
            model=self.model,
            provider=self.provider,
            latency_ms=_ms_since(t0),
        )

    async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
        raise NotImplementedError("audio questions are not routed through the OpenAI-compatible adapter")

    @staticmethod
    def _message(m: dict[str, Any]) -> dict[str, Any]:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            return {
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {
                        "id": c.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})},
                    }
                    for i, c in enumerate(m["tool_calls"])
                ],
            }
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or m.get("name", ""),
                "content": m.get("content") or "",
            }
        return {"role": role, "content": m.get("content") or ""}


# ---------------------------------------------------------------------------
# Fake (tests, offline)
# ---------------------------------------------------------------------------


class FakeLLM:
    """Deterministic, no network. Answers by quoting the first evidence line it sees; never calls tools."""

    provider = "fake"
    model = "fake"
    supports_audio = False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        t0 = time.perf_counter()
        text = "{}" if json_mode else self._answer(messages)
        return LLMResult(
            text=text, tool_calls=[], model=self.model, provider=self.provider, latency_ms=_ms_since(t0)
        )

    async def transcribe(self, wav_bytes: bytes, *, hint: str | None = None) -> Transcript:
        raise NotImplementedError("FakeLLM cannot hear audio")

    @staticmethod
    def _answer(messages: list[dict[str, Any]]) -> str:
        for m in messages:
            match = _EVIDENCE_LINE.search(m.get("content") or "")
            if match:
                uid, rest = match.group(1), match.group(2)
                who, _, said = rest.partition(": ")
                quote = said.strip() or rest.strip()
                return f'{who.strip()} said: "{quote}" [u:{uid}]'
        return "I couldn't find that in the meetings."


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def probe_ollama_model(base_url: str, model: str, timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any] | None:
    """The /api/tags entry for `model` if Ollama is reachable and has it, else None. Never raises."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models") or []
    except (httpx.HTTPError, ValueError) as exc:
        log.info("Ollama at %s not usable: %r", base_url, exc)
        return None
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    for entry in models:
        if entry.get("name") in wanted or entry.get("model") in wanted:
            return entry
    log.info("Ollama at %s does not have model %s", base_url, model)
    return None


def select_llm(s: Settings, probe: Any = probe_ollama_model) -> LLM:
    provider = s.llm_provider
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini" or (provider == "auto" and s.gemini_api_key):
        if s.gemini_api_key:
            return GeminiLLM(s.gemini_api_key, s.gemini_model)
        log.warning("WSW_LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; using the fake LLM")
        return FakeLLM()
    if provider == "openai":
        if s.openai_base_url:
            return OpenAICompatLLM(s.openai_base_url, s.openai_api_key, s.openai_model)
        log.warning("WSW_LLM_PROVIDER=openai but WSW_OPENAI_BASE_URL is not set; using the fake LLM")
        return FakeLLM()
    if provider == "ollama":
        return OllamaLLM(s.ollama_url, s.ollama_model)
    if probe(s.ollama_url, s.ollama_model) is not None:
        return OllamaLLM(s.ollama_url, s.ollama_model)
    log.warning(
        "no Gemma 4 runtime available (no GEMINI_API_KEY, Ollama model %s not found); using the fake LLM",
        s.ollama_model,
    )
    return FakeLLM()


def select_audio_llm(s: Settings, text_llm: LLM, probe: Any = probe_ollama_model) -> OllamaLLM | None:
    if s.question_audio not in ("auto", "ollama") or not ollama_model_hears_audio(s.ollama_model):
        return None
    entry = probe(s.ollama_url, s.ollama_model)
    if entry is None:
        return None
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, list) and capabilities and "audio" not in capabilities:
        log.info("Ollama model %s does not list the audio capability", s.ollama_model)
        return None
    if isinstance(text_llm, OllamaLLM) and text_llm.model == s.ollama_model:
        return text_llm
    return OllamaLLM(s.ollama_url, s.ollama_model)


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    llm = select_llm(settings())
    log.info("LLM: %s %s", llm.provider, llm.model)
    return llm


@lru_cache(maxsize=1)
def get_audio_llm() -> OllamaLLM | None:
    return select_audio_llm(settings(), get_llm())


def get_extraction_llm() -> LLM:
    return get_llm()


def reset_llm_cache() -> None:
    get_llm.cache_clear()
    get_audio_llm.cache_clear()


def llm_info() -> dict[str, Any]:
    llm = get_llm()
    return {"provider": llm.provider, "model": llm.model, "audio": get_audio_llm() is not None}
