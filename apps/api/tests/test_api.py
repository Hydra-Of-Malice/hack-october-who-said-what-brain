"""The API end to end: the real FastAPI app on a throwaway FalkorDB graph, fed by the real listener
(Silero VAD, TitaNet, faster-whisper) with the two-speaker fixture, answering with the fake LLM.

The environment is set before the app is built and restored afterwards (settings and LLM caches are
cleared both times): WSW_GRAPH=wsw_test_api_<hex>, WSW_LLM_PROVIDER=fake, WSW_QUESTION_AUDIO=whisper,
WSW_SEED_DEMO=false, WSW_DATA_DIR=<tmp>.

Skipped when FalkorDB is unreachable, when faster-whisper-small is not in the HF cache, or with
WSW_SKIP_SLOW=1. WSW_LIVE_API=1 adds one live check against Ollama gemma4:e4b (extraction + an answer).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from wsw_brain.api import create_app
from wsw_brain.api.capture import CaptureClosed, CaptureError, CaptureRegistry
from wsw_brain.api.demo import demo_meeting_id, find_demo_meetings, name_tracks, parse_date_hint, plan_seed
from wsw_brain.brain.llm import probe_ollama_model, reset_llm_cache
from wsw_brain.config import settings
from wsw_brain.contracts import SAMPLE_RATE, ActionItemDraft, DecisionDraft, Insights, Utterance
from wsw_brain.graph import GraphStore
from wsw_brain.pipeline import _MeetingRecords, _same_owner

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_WAV = FIXTURES / "two_speakers.wav"
TRUTH = json.loads((FIXTURES / "two_speakers.truth.json").read_text(encoding="utf-8"))["turns"]
FALKOR_URL = os.environ.get("WSW_FALKORDB_URL", "falkor://localhost:6379")
FRAME_BYTES = 3200  # 100 ms of 16 kHz s16le
MEETING_ID = re.compile(r"^m_[0-9a-f]{10}$")
MEETING_KEYS = {
    "id",
    "title",
    "status",
    "source",
    "started_at",
    "ended_at",
    "duration_s",
    "utterance_count",
    "speaker_count",
    "has_audio",
    "summary",
}
UTTERANCE_KEYS = {
    "id",
    "meeting_id",
    "seq",
    "speaker",
    "speaker_name",
    "t_start",
    "t_end",
    "text",
    "confidence",
}
HIT_KEYS = {
    "utterance_id",
    "meeting_id",
    "meeting_title",
    "speaker",
    "speaker_name",
    "t_start",
    "t_end",
    "text",
    "score",
    "via",
}
ASK_KEYS = {
    "answer",
    "citations",
    "actions",
    "tool_trace",
    "cypher",
    "model",
    "provider",
    "session_id",
    "timings",
    "transcript",
    "transcript_provider",
}


# ---------------------------------------------------------------------------
# environment and app fixtures
# ---------------------------------------------------------------------------


def _skip_unless_ready() -> None:
    if os.environ.get("WSW_SKIP_SLOW") == "1":
        pytest.skip("WSW_SKIP_SLOW=1: the API tests run the Whisper listener")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache("Systran/faster-whisper-small", "model.bin"), str):
        pytest.skip("faster-whisper-small is not in the HF cache")
    _skip_without_falkordb()


def _skip_without_falkordb() -> None:
    store = GraphStore(FALKOR_URL, "wsw_test_api_probe")
    try:
        if not store.ping()["ok"]:
            pytest.skip(f"FalkorDB not reachable at {FALKOR_URL}")
    finally:
        store.close()


@contextlib.contextmanager
def environment(values: dict[str, str]) -> Iterator[None]:
    """Set env vars and rebuild the cached settings/LLM routing; restore both afterwards."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    settings.cache_clear()
    reset_llm_cache()
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        settings.cache_clear()
        reset_llm_cache()


@contextlib.contextmanager
def running_app(tmp: Path, graph: str | None = None, **env: str) -> Iterator[tuple[TestClient, str]]:
    graph = graph or "wsw_test_api_" + uuid.uuid4().hex[:8]
    values = {
        "WSW_GRAPH": graph,
        "WSW_LLM_PROVIDER": "fake",
        "WSW_QUESTION_AUDIO": "whisper",
        "WSW_SEED_DEMO": "false",
        "WSW_DATA_DIR": str(tmp),
        **env,
    }
    with environment(values):
        app = create_app(disconnect_grace_s=1.0)
        try:
            with TestClient(app) as client:
                yield client, graph
        finally:
            store = GraphStore(FALKOR_URL, graph)
            store.reset()
            store.close()


@pytest.fixture(scope="class")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    _skip_unless_ready()
    with running_app(tmp_path_factory.mktemp("api_data")) as (c, _graph):
        yield c


@pytest.fixture(scope="class")
def state() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fixture_samples(start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    samples, rate = sf.read(str(FIXTURE_WAV), dtype="float32")
    assert rate == SAMPLE_RATE
    lo = int(start_s * SAMPLE_RATE)
    hi = int(end_s * SAMPLE_RATE) if end_s is not None else samples.size
    return samples[lo:hi]


def pcm_frames(samples: np.ndarray) -> list[bytes]:
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return [data[i : i + FRAME_BYTES] for i in range(0, len(data), FRAME_BYTES)]


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def wait_for(fetch: Callable[[], Any], done: Callable[[Any], bool], timeout: float, what: str) -> Any:
    deadline = time.monotonic() + timeout
    value = fetch()
    while not done(value):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"timed out after {timeout:.0f} s waiting for {what}; last value: {value!r:.400}"
            )
        time.sleep(0.5)
        value = fetch()
    return value


def wait_ended(client: TestClient, meeting_id: str, timeout: float = 300.0) -> dict[str, Any]:
    return wait_for(
        lambda: client.get(f"/api/meetings/{meeting_id}").json(),
        lambda d: d.get("meeting", {}).get("status") == "ended",
        timeout,
        f"meeting {meeting_id} to end",
    )


def parse_sse(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key] = value.strip()
        if "data" in fields:
            events.append(
                {"id": fields.get("id"), "event": fields.get("event"), "data": json.loads(fields["data"])}
            )
    return events


def read_sse(
    client: TestClient, query: str, until: Callable[[list[dict[str, Any]]], bool], timeout: float = 10.0
) -> tuple[dict[str, Any], str]:
    """GET /api/events through the ASGI app on the test client's loop; disconnect once `until` holds
    (TestClient itself buffers a whole response, which never ends for SSE)."""

    async def run() -> tuple[dict[str, Any], str]:
        start: dict[str, Any] = {}
        body: list[str] = []
        stop = asyncio.Event()

        async def receive() -> dict[str, Any]:
            await stop.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                start["status"] = message["status"]
                start["headers"] = {k.decode(): v.decode() for k, v in message["headers"]}
            elif message["type"] == "http.response.body":
                body.append(message.get("body", b"").decode())
                if until(parse_sse("".join(body))):
                    stop.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/events",
            "raw_path": b"/api/events",
            "root_path": "",
            "query_string": query.encode(),
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        loop = asyncio.get_running_loop()
        timer = loop.call_later(timeout, stop.set)
        try:
            await client.app(scope, receive, send)
        finally:
            timer.cancel()
        return start, "".join(body)

    return client.portal.call(run)


def stream_meeting_audio(
    client: TestClient, meeting_id: str, samples: np.ndarray, stop: dict[str, Any] | None
) -> dict[str, Any]:
    """Send PCM frames over the socket, then a stop frame (or just disconnect when stop is None).
    Returns the first status frame."""
    with client.websocket_connect(f"/ws/meetings/{meeting_id}/audio") as ws:
        first = ws.receive_json()
        for frame in pcm_frames(samples):
            ws.send_bytes(frame)
        if stop is not None:
            ws.send_text(json.dumps(stop))
            receive_until_closed(ws)  # leaving the block cancels the server task: let it take the stop first
    return first


def receive_until_closed(ws: Any, limit: int = 1000) -> list[dict[str, Any]]:
    frames = []
    with contextlib.suppress(WebSocketDisconnect):
        for _ in range(limit):
            frames.append(ws.receive_json())
    return frames


# ---------------------------------------------------------------------------
# the app with the fake LLM
# ---------------------------------------------------------------------------


class TestApi:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        health = response.json()
        assert health["ok"] is True
        assert health["falkordb"]["ok"] is True and health["falkordb"]["graph"].startswith("wsw_test_api_")
        assert health["llm"] == {"provider": "fake", "model": "fake", "audio": False}
        assert health["question_audio"] == "whisper"
        assert health["embedder"]["dim"] == settings().embed_dim and health["embedder"]["name"]
        assert health["asr"]["name"] == "faster-whisper-small"
        assert health["brand"]["name"] == client.get("/api/brand").json()["name"]

    def test_live_meeting_over_websocket(self, client: TestClient, state: dict[str, Any]) -> None:
        response = client.post("/api/meetings", json={"title": "  Pricing   sync "})
        assert response.status_code == 200
        meeting = response.json()
        assert set(meeting) == MEETING_KEYS
        assert MEETING_ID.match(meeting["id"]) and meeting["title"] == "Pricing sync"
        assert meeting["status"] == "live" and meeting["source"] == "mic" and meeting["ended_at"] is None
        assert abs(meeting["started_at"] - time.time() * 1000) < 60_000
        mid = state["mid"] = meeting["id"]

        first = stream_meeting_audio(client, mid, fixture_samples(), {"type": "stop"})
        assert first["type"] == "status" and first["state"] in ("listening", "processing")
        assert isinstance(first["lag_s"], float | int)

        detail = wait_ended(client, mid)
        assert set(detail) == {"meeting", "speakers", "utterances", "decisions", "action_items", "topics"}
        summary = detail["meeting"]
        assert summary["ended_at"] >= summary["started_at"] and summary["has_audio"] is True
        assert summary["duration_s"] >= 49.0
        utterances = detail["utterances"]
        assert len(utterances) >= 4, utterances
        assert all(set(u) == UTTERANCE_KEYS and u["meeting_id"] == mid for u in utterances)
        assert [u["seq"] for u in utterances] == list(range(len(utterances)))
        assert all(0.0 <= u["t_start"] < u["t_end"] <= summary["duration_s"] + 0.5 for u in utterances)
        tracks = {u["speaker"] for u in utterances}
        assert len(tracks) >= 2 and {s["track"] for s in detail["speakers"]} == tracks
        text = " ".join(u["text"].lower() for u in utterances)
        assert "pricing" in text and "monthly" in text
        assert detail["action_items"], "the heuristic extractor should find 'I will update the pricing page'"
        state["utterance_ids"] = [u["id"] for u in utterances]
        state["tracks"] = sorted(tracks)

    def test_recording_is_served_with_ranges(self, client: TestClient, state: dict[str, Any]) -> None:
        mid = state["mid"]
        audio = client.get(f"/api/meetings/{mid}/audio")
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        assert audio.headers.get("accept-ranges") == "bytes"
        info = sf.info(io.BytesIO(audio.content))
        assert info.samplerate == SAMPLE_RATE and info.channels == 1 and info.duration >= 49.0
        part = client.get(f"/api/meetings/{mid}/audio", headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100
        assert client.get("/api/meetings/m_0000000000/audio").status_code == 404
        assert client.get("/api/meetings/..%2F..%2Fsecret/audio").status_code == 404

    def test_sse_replays_the_meeting(self, client: TestClient, state: dict[str, Any]) -> None:
        mid = state["mid"]

        def complete(events: list[dict[str, Any]]) -> bool:
            kinds = {e["event"] for e in events}
            return {"meeting.started", "utterance.created", "graph.write", "meeting.ended"} <= kinds

        start, raw = read_sse(client, f"since_id=0&meeting_id={mid}", complete)
        assert start["status"] == 200 and start["headers"]["content-type"].startswith("text/event-stream")
        assert raw.startswith("retry: 2000\n: connected\n\n")
        events = parse_sse(raw)
        assert complete(events), sorted({e["event"] for e in events})
        for event in events:
            assert event["data"]["type"] == event["event"] and str(event["data"]["id"]) == event["id"]
            assert event["data"]["meeting_id"] == mid
        created = [e["data"]["data"] for e in events if e["event"] == "utterance.created"]
        assert {u["id"] for u in created} == set(state["utterance_ids"])
        assert all(set(u) == UTTERANCE_KEYS for u in created)
        writes = [e["data"]["data"] for e in events if e["event"] == "graph.write"]
        assert sum(w["utterances"] for w in writes) == len(state["utterance_ids"])
        assert any("totals" in w and w["totals"]["Utterance"] >= 1 for w in writes)
        assert any(
            e["event"] == "insight.created" and e["data"]["data"]["kind"] == "action_item" for e in events
        )
        assert any(e["event"] == "listener.status" for e in events)
        last_id = max(int(e["id"]) for e in events)
        _, later = read_sse(client, f"since_id={last_id}&meeting_id={mid}", lambda ev: False, timeout=0.5)
        assert all(int(e["id"]) > last_id for e in parse_sse(later))

    def test_reads_across_meetings(self, client: TestClient, state: dict[str, Any]) -> None:
        mid = state["mid"]
        meetings = client.get("/api/meetings").json()
        assert meetings[0]["id"] == mid and set(meetings[0]) == MEETING_KEYS
        after = client.get(f"/api/meetings/{mid}/utterances", params={"after_seq": 1}).json()
        assert [u["seq"] for u in after] == list(range(2, len(state["utterance_ids"])))
        assert client.get("/api/meetings/m_ffffffffff/utterances").status_code == 404
        items = client.get("/api/action-items", params={"meeting_id": mid}).json()
        assert items and all(i["meeting_id"] == mid and i["status"] == "open" for i in items)
        assert client.get("/api/action-items", params={"status": "all"}).status_code == 200
        assert client.get("/api/action-items", params={"status": "later"}).status_code == 400
        assert isinstance(client.get("/api/decisions", params={"meeting_id": mid}).json(), list)
        stats = client.get("/api/stats").json()
        assert stats["Meeting"] >= 1 and stats["Utterance"] == len(state["utterance_ids"])
        assert stats["relationships"] > 0
        hits = client.get("/api/search", params={"q": "monthly plan", "meeting_id": mid}).json()
        assert hits and set(hits[0]) == HIT_KEYS and hits[0]["utterance_id"] in state["utterance_ids"]
        assert client.get("/api/search").status_code == 422
        graph = client.get("/api/graph", params={"meeting_id": mid}).json()
        kinds = {n["kind"] for n in graph["nodes"]}
        assert {"Meeting", "Speaker", "Utterance"} <= kinds and graph["links"]
        two = ",".join(state["utterance_ids"][:2])
        focused = client.get("/api/graph", params={"utterance_ids": two}).json()
        assert {n["id"] for n in focused["nodes"] if n["kind"] == "Utterance"} == set(
            state["utterance_ids"][:2]
        )
        knows = client.get("/api/who-knows", params={"topic": "pricing"}).json()
        assert knows and knows[0]["hits"] >= 1 and set(knows[0]["sample"]) == UTTERANCE_KEYS

    def test_name_speakers(self, client: TestClient, state: dict[str, Any]) -> None:
        mid, track = state["mid"], state["tracks"][0]
        response = client.patch(f"/api/meetings/{mid}/speakers/{track}", json={"name": "  David  "})
        assert response.status_code == 200, response.text
        speaker = response.json()
        assert speaker["track"] == track and speaker["name"] == "David" and speaker["person_id"] == "david"
        detail = client.get(f"/api/meetings/{mid}").json()
        assert all(u["speaker_name"] == "David" for u in detail["utterances"] if u["speaker"] == track)
        assert any(p["name"] == "David" for p in client.get("/api/people").json())
        missing = client.patch(f"/api/meetings/{mid}/speakers/S9", json={"name": "Nobody"})
        assert missing.status_code == 404 and "S9" in missing.json()["detail"]
        blank = client.patch(f"/api/meetings/{mid}/speakers/{track}", json={"name": "   "})
        assert blank.status_code == 422 and isinstance(blank.json()["detail"], str)
        _, raw = read_sse(
            client, f"since_id=0&meeting_id={mid}", lambda ev: any(e["event"] == "speaker.named" for e in ev)
        )
        named = [e["data"]["data"] for e in parse_sse(raw) if e["event"] == "speaker.named"]
        assert named[-1] == {"track": track, "name": "David", "person_id": "david"}

    def test_action_item_status(self, client: TestClient, state: dict[str, Any]) -> None:
        mid = state["mid"]
        item = client.get("/api/action-items", params={"meeting_id": mid}).json()[0]
        done = client.patch(f"/api/action-items/{item['id']}", json={"status": "done"})
        assert done.status_code == 200
        assert done.json()["status"] == "done" and done.json()["done_at"] >= done.json()["created_at"]
        assert item["id"] in {
            i["id"] for i in client.get("/api/action-items", params={"status": "done"}).json()
        }
        _, raw = read_sse(
            client, f"since_id=0&meeting_id={mid}", lambda ev: any(e["event"] == "action.updated" for e in ev)
        )
        updated = [e["data"]["data"] for e in parse_sse(raw) if e["event"] == "action.updated"]
        assert updated[-1]["id"] == item["id"] and updated[-1]["status"] == "done"
        assert client.patch("/api/action-items/m_nothing:a-1", json={"status": "done"}).status_code == 404
        assert client.patch(f"/api/action-items/{item['id']}", json={"status": "later"}).status_code == 422
        reopened = client.patch(f"/api/action-items/{item['id']}", json={"status": "open"}).json()
        assert reopened["status"] == "open" and reopened["done_at"] is None
        state["item"] = reopened

    def test_ask_cites_real_utterances(self, client: TestClient, state: dict[str, Any]) -> None:
        session = "test-" + uuid.uuid4().hex[:8]
        response = client.post(
            "/api/ask",
            json={"session_id": session, "question": "What did they decide about the monthly plan?"},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert set(result) == ASK_KEYS and result["provider"] == "fake" and result["answer"].strip()
        assert result["citations"], result
        assert {c["utterance_id"] for c in result["citations"]} <= set(state["utterance_ids"])
        assert all(set(c) == HIT_KEYS for c in result["citations"])
        assert result["cypher"] and result["timings"]["total"] > 0
        assert result["transcript"] is None and result["transcript_provider"] is None
        turns = client.get(f"/api/sessions/{session}").json()
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert set(turns[1]["citations"]) <= set(state["utterance_ids"])
        blank = client.post("/api/ask", json={"session_id": session, "question": "   "})
        assert blank.status_code == 422 and "question" in blank.json()["detail"]

    def test_ask_completes_an_action_item(self, client: TestClient, state: dict[str, Any]) -> None:
        item = state["item"]
        words = [w for w in re.findall(r"[a-z]+", item["text"].lower()) if len(w) >= 5][:3]
        question = f"Mark the {' '.join(words)} item as done."
        result = client.post("/api/ask", json={"session_id": "test-done", "question": question}).json()
        assert {"type": "refresh", "what": "action_items"} in result["actions"], result
        assert any(t["tool"] == "complete_action_item" for t in result["tool_trace"])
        items = {i["id"]: i for i in client.get("/api/action-items").json()}
        assert items[item["id"]]["status"] == "done"
        _, raw = read_sse(
            client,
            f"since_id=0&meeting_id={state['mid']}",
            lambda ev: sum(e["event"] == "action.updated" for e in ev) >= 3,
        )
        updated = [e["data"]["data"] for e in parse_sse(raw) if e["event"] == "action.updated"]
        assert updated[-1]["id"] == item["id"] and updated[-1]["status"] == "done"

    def test_ask_by_voice_with_whisper(self, client: TestClient, state: dict[str, Any]) -> None:
        question = wav_bytes(fixture_samples(13.0, 19.4))  # "...launch the monthly plan first..."
        response = client.post(
            "/api/ask/voice",
            files={"audio": ("question.wav", question, "audio/wav")},
            data={"session_id": "test-voice", "meeting_id": state["mid"]},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["transcript_provider"] == "whisper"
        assert "monthly" in result["transcript"].lower(), result["transcript"]
        assert result["answer"].strip() and result["timings"]["transcribe"] > 0
        assert {c["utterance_id"] for c in result["citations"]} <= set(state["utterance_ids"])
        silent = client.post(
            "/api/ask/voice",
            files={"audio": ("q.wav", wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32)), "audio/wav")},
            data={"session_id": "test-voice"},
        )
        assert silent.status_code == 400 and silent.json()["detail"].startswith("No speech heard")
        too_long = client.post(
            "/api/ask/voice",
            files={
                "audio": ("q.wav", wav_bytes(np.full(31 * SAMPLE_RATE, 0.1, dtype=np.float32)), "audio/wav")
            },
            data={"session_id": "test-voice"},
        )
        assert too_long.status_code == 413
        garbage = client.post(
            "/api/ask/voice", files={"audio": ("q.wav", b"not audio", "audio/wav")}, data={"session_id": "s"}
        )
        assert garbage.status_code == 400

    def test_upload_runs_the_listener(
        self, client: TestClient, state: dict[str, Any], tmp_path: Path
    ) -> None:
        clip = tmp_path / "Weekly_pricing-call.wav"
        sf.write(str(clip), fixture_samples(0.0, 25.6), SAMPLE_RATE, subtype="PCM_16")
        with clip.open("rb") as handle:
            response = client.post(
                "/api/meetings/upload",
                files={"file": (clip.name, handle, "audio/wav")},
                data={"pace": "fast"},
            )
        assert response.status_code == 200, response.text
        meeting = response.json()
        assert meeting["source"] == "upload" and meeting["status"] == "live"
        assert meeting["title"] == "Weekly pricing call"
        detail = wait_ended(client, meeting["id"])
        assert detail["meeting"]["has_audio"] is True and detail["meeting"]["duration_s"] >= 25.0
        assert len(detail["utterances"]) >= 3 and len({u["speaker"] for u in detail["utterances"]}) >= 2
        assert client.get(f"/api/meetings/{meeting['id']}/audio").status_code == 200
        state["upload_id"] = meeting["id"]

        bad_type = client.post("/api/meetings/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert bad_type.status_code == 400 and "unsupported" in bad_type.json()["detail"]
        not_audio = client.post(
            "/api/meetings/upload", files={"file": ("fake.wav", b"RIFF0000junk", "audio/wav")}
        )
        assert not_audio.status_code == 400
        bad_pace = client.post(
            "/api/meetings/upload", files={"file": (clip.name, b"x", "audio/wav")}, data={"pace": "slow"}
        )
        assert bad_pace.status_code == 422

    def test_pause_and_resume_keeps_one_timeline(self, client: TestClient) -> None:
        mid = client.post("/api/meetings", json={"title": "Resume test"}).json()["id"]
        stream_meeting_audio(client, mid, fixture_samples(0.0, 13.0), {"type": "stop", "end": False})
        time.sleep(1.5)  # longer than the test's disconnect grace: an explicit end=false must not end it
        assert client.get(f"/api/meetings/{mid}").json()["meeting"]["status"] == "live"
        stream_meeting_audio(client, mid, fixture_samples(13.0, 26.0), {"type": "stop", "end": False})
        ended = client.post(f"/api/meetings/{mid}/end")
        assert ended.status_code == 200 and ended.json()["status"] == "ended"
        again = client.post(f"/api/meetings/{mid}/end")
        assert again.status_code == 200 and again.json()["ended_at"] == ended.json()["ended_at"]
        utterances = client.get(f"/api/meetings/{mid}").json()["utterances"]
        assert [u["seq"] for u in utterances] == list(range(len(utterances)))
        starts = [u["t_start"] for u in utterances]
        assert starts == sorted(starts) and starts[-1] > 13.0, starts  # the second stream continues the clock
        info = sf.info(str(settings().audio_dir / f"{mid}.wav"))
        assert info.duration >= 26.0
        assert client.post("/api/meetings/m_ffffffffff/end").status_code == 404

    def test_resume_keeps_each_voice_on_its_track(self, client: TestClient) -> None:
        mid = client.post("/api/meetings", json={"title": "Resume speakers"}).json()["id"]
        stream_meeting_audio(client, mid, fixture_samples(0.0, 13.0), {"type": "stop", "end": False})
        # The second stream opens with Zira, whom a fresh speaker clusterer would number S1 (David's track).
        stream_meeting_audio(client, mid, fixture_samples(19.5, 31.0), {"type": "stop", "end": False})
        assert client.post(f"/api/meetings/{mid}/end").status_code == 200
        utterances = client.get(f"/api/meetings/{mid}").json()["utterances"]

        def track_of(words: str) -> str:
            tracks = [u["speaker"] for u in utterances if words in u["text"].lower()]
            assert tracks, (words, [(u["speaker"], u["text"]) for u in utterances])
            return tracks[0]

        david, zira = track_of("morning"), track_of("survey")
        assert david != zira
        assert track_of("pricing page") == zira and track_of("mobile app") == david, utterances
        speakers = settings().audio_dir / f"{mid}.speakers.json"
        assert speakers.is_file()  # where the tracks wait between sessions (and across a restart)
        assert client.delete(f"/api/meetings/{mid}").json() == {"deleted": True}
        assert not speakers.exists()

    def test_dropped_socket_ends_the_meeting(self, client: TestClient) -> None:
        mid = client.post("/api/meetings", json={"title": "Dropped"}).json()["id"]
        stream_meeting_audio(client, mid, fixture_samples(0.0, 7.0), stop=None)
        detail = wait_ended(client, mid, timeout=60)
        assert detail["meeting"]["status"] == "ended"

    def test_socket_refusals(self, client: TestClient, state: dict[str, Any]) -> None:
        for meeting_id, needle in (
            (state["mid"], "has ended"),
            ("m_ffffffffff", "not found"),
            (state["upload_id"], "has ended"),
        ):
            with client.websocket_connect(f"/ws/meetings/{meeting_id}/audio") as ws:
                message = ws.receive_json()
            assert message["type"] == "error" and needle in message["detail"], message
        mid = client.post("/api/meetings", json={"title": "Busy"}).json()["id"]
        with client.websocket_connect(f"/ws/meetings/{mid}/audio") as first:
            assert first.receive_json()["type"] == "status"
            with client.websocket_connect(f"/ws/meetings/{mid}/audio") as second:
                message = second.receive_json()
            assert message["type"] == "error" and "already being captured" in message["detail"]
            first.send_bytes(b"\x00\x01\x02")  # odd byte count: not s16le PCM
            errors = [f for f in receive_until_closed(first) if f["type"] == "error"]
            assert errors and "bad audio frame" in errors[0]["detail"], errors
        client.post(f"/api/meetings/{mid}/end")

    def test_errors_are_json_with_a_detail(self, client: TestClient) -> None:
        missing = client.get("/api/meetings/m_ffffffffff")
        assert missing.status_code == 404 and missing.json() == {"detail": "meeting m_ffffffffff not found"}
        assert client.post("/api/meetings", json={"title": "x" * 300}).status_code == 422
        assert client.get("/api/nope").status_code == 404

    def test_web_ui_is_served(self, client: TestClient) -> None:
        index = client.get("/")
        if index.status_code == 404:
            pytest.skip("apps/web/dist is not built")
        assert index.status_code == 200 and "text/html" in index.headers["content-type"]
        deep = client.get("/m/m_0123456789")
        assert deep.status_code == 200 and deep.text == index.text
        assert client.get("/api/does-not-exist").json()["detail"].startswith("no such endpoint")

    def test_delete_meeting(self, client: TestClient, state: dict[str, Any]) -> None:
        mid = state["upload_id"]
        recording = settings().audio_dir / f"{mid}.wav"
        assert recording.is_file()
        assert client.delete(f"/api/meetings/{mid}").json() == {"deleted": True}
        assert client.get(f"/api/meetings/{mid}").status_code == 404
        assert client.get(f"/api/meetings/{mid}/audio").status_code == 404
        assert not recording.exists()
        assert client.delete(f"/api/meetings/{mid}").json() == {"deleted": False}
        assert mid not in {m["id"] for m in client.get("/api/meetings").json()}


# ---------------------------------------------------------------------------
# no database, demo seeding helpers (no listener models needed)
# ---------------------------------------------------------------------------


def test_overlapping_windows_do_not_repeat_insights() -> None:
    held = _MeetingRecords.from_detail(
        {
            "speakers": [{"track": "S3", "label": "Speaker 3", "name": "Meera"}],
            "decisions": [
                {"text": "The annual discount will be 15% and reviewed after 90 days of real data.",
                 "status": "active", "evidence": ["m:u00001"]},
            ],
            "action_items": [
                {"text": "Meera will send the revised deck to Saffron Basket by Thursday.", "owner": "Meera",
                 "owner_track": "S3", "evidence": ["m:u00003"]},
            ],
        }
    )  # fmt: skip
    insights = Insights(
        decisions=[
            DecisionDraft("The annual discount is 15%.", ["m:u00009"]),  # restates a held decision
            DecisionDraft("Starter costs 4,999 rupees a month.", ["m:u00002"]),  # missed before: kept
            DecisionDraft("Self-serve launches on Monday, November 2.", ["m:u00010"]),
            DecisionDraft("Self-serve launches Monday November 2!", ["m:u00011"]),  # repeats the one above
        ],
        action_items=[
            ActionItemDraft("Send the revised deck to Saffron Basket by Thursday.", ["m:u00010"], owner="S3"),
            ActionItemDraft("Meera sends the deck over to Saffron", ["m:u00003"], owner="Meera"),  # context
            ActionItemDraft("Send the revised deck to Saffron Basket by Thursday.", ["m:u00010"], owner="S2"),
            ActionItemDraft("Rahul sets up the Razorpay plans.", [], owner="S2", due="Wednesday"),
        ],
        topics=["pricing"],
    )
    kept, dropped = held.admit(insights, {"m:u00009", "m:u00010", "m:u00011"})
    assert [d.text[:12] for d in kept.decisions] == ["Starter cost", "Self-serve l"]
    assert [(i.owner, i.text[:8]) for i in kept.action_items] == [("S2", "Send the"), ("S2", "Rahul se")]
    assert dropped == 4 and kept.topics == ["pricing"]

    # While speakers are unnamed, "Priya" and "Speaker 1" may be the same person: compared, and repeats.
    unnamed = _MeetingRecords.from_detail(
        {
            "speakers": [{"track": "S1", "label": "Speaker 1", "name": None}],
            "decisions": [],
            "action_items": [
                {"text": "Speaker 1 will draft the pricing page and the FAQ by Monday.", "owner": "Speaker 1",
                 "owner_track": "S1", "evidence": ["m:u00001"]},
            ],
        }
    )  # fmt: skip
    later = Insights(
        action_items=[ActionItemDraft("Priya has the pricing page by Monday.", ["m:u00009"], "Priya")]
    )
    assert unnamed.admit(later, {"m:u00009"})[1] == 1
    assert _same_owner("S2", {"owner": "Meera", "owner_track": "S3"}) is False
    assert _same_owner("Meera", {"owner": "Rahul", "owner_track": "S2"}) is False
    assert _same_owner("Priya", {"owner": "Priya Sharma", "owner_track": None}) is True


def test_starts_and_reports_without_falkordb(tmp_path: Path) -> None:
    values = {
        "WSW_FALKORDB_URL": "falkor://127.0.0.1:1",
        "WSW_GRAPH": "wsw_test_api_down",
        "WSW_LLM_PROVIDER": "fake",
        "WSW_QUESTION_AUDIO": "whisper",
        "WSW_SEED_DEMO": "true",  # the seed must notice the missing database and skip itself
        "WSW_DATA_DIR": str(tmp_path),
    }
    with environment(values), TestClient(create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is False and body["falkordb"]["ok"] is False and body["falkordb"]["error"]
        assert body["llm"]["provider"] == "fake"
        meetings = client.get("/api/meetings")
        assert meetings.status_code == 503 and "unavailable" in meetings.json()["detail"]


def test_demo_plan_and_naming() -> None:
    demos = find_demo_meetings()
    if not demos:
        pytest.skip("no demo meetings with audio")
    assert all(MEETING_ID.match(d.meeting_id) for d in demos)
    assert demo_meeting_id("01-pricing-review") == demo_meeting_id("01-pricing-review")
    assert [d for d, _ in plan_seed([], demos)] == demos
    ended = [{"id": d.meeting_id, "status": "ended"} for d in demos]
    assert plan_seed(ended, demos) == []
    assert plan_seed([{"id": "m_0123456789", "status": "live"}], demos) == []
    interrupted = [{"id": demos[0].meeting_id, "status": "live"}]
    assert plan_seed(interrupted, demos) == [(demos[0], True), *[(d, False) for d in demos[1:]]]
    assert parse_date_hint("Tuesday 13 October 2026, 11:00 IST") == 1791869400000
    assert parse_date_hint("no date") is None
    turns = [
        {"speaker": "Priya", "t_start": 0.0, "t_end": 5.0},
        {"speaker": "Rahul", "t_start": 5.5, "t_end": 9.0},
    ]
    utterances = [
        Utterance("m", 0, "S1", 0.1, 4.9, "a"),
        Utterance("m", 1, "S2", 5.6, 8.8, "b"),
        Utterance("m", 2, "S1", 6.0, 6.5, "c"),
    ]
    assert name_tracks(utterances, turns) == {"S1": "Priya", "S2": "Rahul"}


# ---------------------------------------------------------------------------
# who may change what (FalkorDB, no listener models)
# ---------------------------------------------------------------------------


def test_open_server_protects_demo_meetings_and_limits_new_ones(tmp_path: Path) -> None:
    _skip_without_falkordb()
    with running_app(tmp_path, WSW_CREATE_LIMIT="2") as (client, _graph):
        demo = client.app.state.services.new_meeting("Pricing review", "demo")
        assert client.get("/api/auth").json() == {"required": False, "operator": False}
        for response in (
            client.delete(f"/api/meetings/{demo['id']}"),
            client.post(f"/api/meetings/{demo['id']}/end"),
            client.patch(f"/api/meetings/{demo['id']}/speakers/S1", json={"name": "Mallory"}),
        ):
            assert response.status_code == 403 and "demo" in response.json()["detail"], response.text
        assert client.get(f"/api/meetings/{demo['id']}").json()["meeting"]["status"] == "live"

        first = client.post("/api/meetings", json={"title": "One"})
        second = client.post("/api/meetings", json={"title": "Two"})
        third = client.post("/api/meetings", json={"title": "Three"})
        assert first.status_code == second.status_code == 200
        assert third.status_code == 429 and int(third.headers["retry-after"]) > 0, third.text
        assert client.delete(f"/api/meetings/{first.json()['id']}").json() == {"deleted": True}

        preflight = client.options(
            "/api/meetings",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"},
        )
        assert "access-control-allow-origin" not in preflight.headers  # same origin only by default
        live = f"/ws/meetings/{second.json()['id']}/audio"
        evil = {"origin": "https://evil.example"}  # a page on another site cannot stream into a meeting
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(live, headers=evil):
            pass
        with client.websocket_connect(
            "/ws/meetings/m_ffffffffff/audio", headers={"origin": "http://testserver"}
        ) as ws:
            assert "not found" in ws.receive_json()["detail"]


def test_operator_token_guards_every_change(tmp_path: Path) -> None:
    _skip_without_falkordb()
    token = "op-" + uuid.uuid4().hex
    with running_app(tmp_path, WSW_ADMIN_TOKEN=token) as (client, _graph):
        demo = client.app.state.services.new_meeting("Pricing review", "demo")
        assert client.get("/api/meetings").status_code == 200  # reading stays open
        assert client.get("/api/auth").json() == {"required": True, "operator": False}
        refused = [
            client.post("/api/meetings", json={"title": "x"}),
            client.post("/api/meetings", json={"title": "x"}, headers={"Authorization": "Bearer wrong"}),
            client.post("/api/meetings/upload", files={"file": ("a.wav", b"x", "audio/wav")}),
            client.post(f"/api/meetings/{demo['id']}/end"),
            client.delete(f"/api/meetings/{demo['id']}"),
            client.patch(f"/api/meetings/{demo['id']}/speakers/S1", json={"name": "Mallory"}),
            client.patch("/api/action-items/m_x:a-1", json={"status": "done"}),
        ]
        assert [r.status_code for r in refused] == [401] * len(refused), [r.text for r in refused]
        with client.websocket_connect("/ws/meetings/m_ffffffffff/audio") as ws:
            message = ws.receive_json()
        assert message["type"] == "error" and "token" in message["detail"], message

        made = client.post("/api/meetings", json={"title": "Operator"}, headers={"X-WSW-Token": token})
        assert made.status_code == 200, made.text
        assert client.post("/api/auth", json={"token": "wrong"}).status_code == 401
        assert client.post("/api/auth", json={"token": token}).json() == {"required": True, "operator": True}
        # From here the cookie rides along on every request and on the socket, as it does for the web UI.
        assert client.get("/api/auth").json()["operator"] is True
        assert client.post(f"/api/meetings/{made.json()['id']}/end").json()["status"] == "ended"
        assert client.delete(f"/api/meetings/{demo['id']}").json() == {"deleted": True}
        with client.websocket_connect("/ws/meetings/m_ffffffffff/audio") as ws:
            assert "not found" in ws.receive_json()["detail"]


def test_interrupted_upload_is_ended_at_startup(tmp_path: Path) -> None:
    _skip_without_falkordb()
    graph = "wsw_test_api_" + uuid.uuid4().hex[:8]
    store = GraphStore(FALKOR_URL, graph)
    try:
        now = int(time.time() * 1000)
        store.create_meeting("m_0123456789", "Cut short", "upload", now)
        store.write_utterances(
            [Utterance("m_0123456789", 0, "S1", 0.5, 4.0, "The pricing page ships on Friday.")], None
        )
        store.create_meeting("m_9876543210", "Paused", "mic", now)
        with running_app(tmp_path, graph) as (client, _graph):
            upload = client.get("/api/meetings/m_0123456789").json()["meeting"]
            assert upload["status"] == "ended" and upload["duration_s"] >= 4.0, upload
            # a microphone meeting can still reconnect after a restart: it stays live
            assert client.get("/api/meetings/m_9876543210").json()["meeting"]["status"] == "live"
    finally:
        store.reset()
        store.close()


# ---------------------------------------------------------------------------
# live capture bookkeeping (a stand-in listener, no models, no database)
# ---------------------------------------------------------------------------


class _FakeListener:
    """Stands in for LiveSession. close() lets the worker run `flush_s` more seconds and, like the real
    one, raises TimeoutError when that is longer than the timeout (the worker keeps going)."""

    made: list[_FakeListener]
    flush_s = 0.0

    def __init__(self, meeting_id: str, asr: Any, embedder: Any, callbacks: Any, **options: Any) -> None:
        self.meeting_id = meeting_id
        self.options = options
        self.closed = False
        self.lag_s = 0.0
        self._stopped = threading.Event()
        self.others_running = [s for s in self.made if s.running]
        self.made.append(self)

    @property
    def running(self) -> bool:
        return not self._stopped.is_set()

    def feed(self, samples: np.ndarray) -> None:
        assert not self.closed

    def close(self, timeout: float | None = 60.0) -> None:
        if not self.closed:
            self.closed = True
            threading.Timer(self.flush_s, self._stopped.set).start()
        if not self._stopped.wait(timeout):
            raise TimeoutError(f"listener for meeting {self.meeting_id} still processing")


class _FakeServices:
    """What CaptureRegistry uses of Services, for a live mic meeting that started just now."""

    def __init__(self, tmp: Path, *, models_s: float = 0.0) -> None:
        self.tmp = tmp
        self.models_s = models_s
        self.started_at = int(time.time() * 1000)
        self.settings = SimpleNamespace(speaker_threshold=0.55, max_utterance_s=8.0)
        self.pipeline = SimpleNamespace(
            flush=lambda: True, next_seq=lambda mid: 0, callbacks=lambda mid, time_offset_s=0.0: None
        )
        self.store = SimpleNamespace(set_meeting_has_audio=lambda mid, flag: None)

    def require_graph(self) -> None:
        pass

    def meeting_summary(self, meeting_id: str) -> dict[str, Any]:
        return {"id": meeting_id, "status": "live", "source": "mic", "started_at": self.started_at}

    def recording_path(self, meeting_id: str) -> Path:
        return self.tmp / f"{meeting_id}.wav"

    def speaker_state_path(self, meeting_id: str) -> Path:
        return self.tmp / f"{meeting_id}.speakers.json"

    def listener_models(self) -> tuple[None, None]:
        time.sleep(self.models_s)
        return None, None

    def spawn(self, coro: Any) -> asyncio.Future[Any]:
        return asyncio.ensure_future(coro)


@pytest.fixture
def fake_listener(monkeypatch: pytest.MonkeyPatch) -> type[_FakeListener]:
    listener = type("FakeListener", (_FakeListener,), {"made": [], "flush_s": 0.0})
    monkeypatch.setattr("wsw_brain.api.capture.LiveSession", listener)
    return listener


async def test_concurrent_releases_share_the_listener_flush(
    tmp_path: Path, fake_listener: type[_FakeListener]
) -> None:
    fake_listener.flush_s = 0.5
    registry = CaptureRegistry(_FakeServices(tmp_path), disconnect_grace_s=30.0)  # type: ignore[arg-type]
    capture = await registry.attach("m_0000000001")
    capture.feed(b"\x00\x00" * SAMPLE_RATE)  # 1 s
    first, second = await asyncio.gather(registry.release("m_0000000001"), registry.release("m_0000000001"))
    assert first == second == pytest.approx(1.0, abs=0.01)  # the second end/delete waited for the same flush
    assert all(s.closed and not s.running for s in fake_listener.made)


async def test_end_while_a_socket_attaches_closes_its_listener(
    tmp_path: Path, fake_listener: type[_FakeListener]
) -> None:
    registry = CaptureRegistry(_FakeServices(tmp_path, models_s=0.5), disconnect_grace_s=30.0)  # type: ignore[arg-type]
    attaching = asyncio.ensure_future(registry.attach("m_0000000002"))
    await asyncio.sleep(0.2)  # attach() is loading the listener models
    await registry.release("m_0000000002")
    with pytest.raises(CaptureClosed):
        await attaching
    assert fake_listener.made and all(s.closed for s in fake_listener.made)  # nothing left running
    with pytest.raises(CaptureError) as refused:
        await registry.attach("m_0000000002")
    assert refused.value.code == 4409


async def test_resume_waits_for_the_old_listener_and_keeps_its_speakers(
    tmp_path: Path, fake_listener: type[_FakeListener], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wsw_brain.api.capture.CLOSE_TIMEOUT_S", 0.05)
    fake_listener.flush_s = 0.5  # the old listener keeps emitting after close() times out
    services = _FakeServices(tmp_path)
    registry = CaptureRegistry(services, disconnect_grace_s=30.0)  # type: ignore[arg-type]
    first = await registry.attach("m_0000000003")
    old = first.session
    registry.detach(first, end=False)  # paused: the meeting stays live
    again = await registry.attach("m_0000000003")
    new = again.session
    assert again is first and isinstance(old, _FakeListener) and isinstance(new, _FakeListener)
    assert new is not old and not old.running
    assert new.others_running == []  # never two listeners (two seq counters) for one meeting at once
    # both continue the meeting's speaker tracks, which the old one saved when it finished
    state = services.speaker_state_path("m_0000000003")
    assert old.options["speaker_state_path"] == new.options["speaker_state_path"] == state
    await registry.release("m_0000000003")


# ---------------------------------------------------------------------------
# live: Ollama gemma4:e4b (WSW_LIVE_API=1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def live_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    if os.environ.get("WSW_LIVE_API") != "1":
        pytest.skip("set WSW_LIVE_API=1 to run the live Ollama check")
    _skip_unless_ready()
    if probe_ollama_model("http://127.0.0.1:11434", "gemma4:e4b") is None:
        pytest.skip("Ollama gemma4:e4b is not available")
    with running_app(tmp_path_factory.mktemp("api_live"), WSW_LLM_PROVIDER="ollama") as (c, _graph):
        yield c


@pytest.mark.live
class TestLiveOllama:
    def test_live_extraction_and_answer(self, live_client: TestClient) -> None:
        client = live_client
        assert client.get("/api/health").json()["llm"]["provider"] == "ollama"
        mid = client.post("/api/meetings", json={"title": "Live Gemma check"}).json()["id"]
        t0 = time.perf_counter()
        stream_meeting_audio(client, mid, fixture_samples(), {"type": "stop", "end": False})
        ended = client.post(f"/api/meetings/{mid}/end").json()
        listened_s = time.perf_counter() - t0
        assert ended["status"] == "ended"

        def insights() -> list[dict[str, Any]]:
            _, raw = read_sse(
                client,
                f"since_id=0&meeting_id={mid}",
                lambda ev: sum(e["event"] == "meeting.ended" for e in ev) >= 2,
                timeout=240,
            )
            return parse_sse(raw)

        events = insights()
        finalized_s = time.perf_counter() - t0
        found = [e["data"]["data"] for e in events if e["event"] == "insight.created"]
        by = [f["by"] for f in found]
        summary = [e["data"]["data"]["summary"] for e in events if e["event"] == "meeting.ended"][-1]
        assert found and any(b["provider"] == "ollama" for b in by), by
        assert summary and len(summary.split()) <= 80

        session = "live-" + uuid.uuid4().hex[:6]
        t1 = time.perf_counter()
        result = client.post(
            "/api/ask",
            json={"session_id": session, "question": "Who will update the pricing page, and by when?"},
        ).json()
        ask_s = time.perf_counter() - t1
        assert result["provider"] == "ollama" and result["answer"].strip() and result["citations"]
        detail = client.get(f"/api/meetings/{mid}").json()
        assert {c["utterance_id"] for c in result["citations"]} <= {u["id"] for u in detail["utterances"]}
        print(
            f"\nLIVE: listened {listened_s:.1f} s, final extraction+summary done {finalized_s - listened_s:.1f} s later; "
            f"{len(found)} insights {sorted({f['kind'] for f in found})}; extraction calls "
            f"{[round(b['latency_ms']) for b in by if b.get('latency_ms')]} ms; ask {ask_s:.1f} s "
            f"(timings {result['timings']}); answer: {result['answer']!r}; summary: {summary!r}"
        )
