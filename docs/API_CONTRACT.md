# API contract (backend ↔ web UI)

Base: same origin. In development the Vite dev server (port 5173) proxies `/api` and `/ws` to FastAPI on `:8000`. In production FastAPI serves the built UI from `apps/web/dist`.

All JSON shapes (`MeetingSummary`, `Speaker`, `UtteranceOut`, `Decision`, `ActionItem`, `Topic`, `Hit`, `Subgraph`, `AskResult`) are defined in [`apps/api/wsw_brain/contracts.py`](../apps/api/wsw_brain/contracts.py).

## REST

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ok, brand, falkordb:{ok, version, graph}, llm:{provider, model, audio: bool}, question_audio: "gemma4-audio:ollama"\|"whisper", embedder:{name, dim}, asr:{name}}` |
| GET | `/api/brand` | — | contents of `/brand.json` |
| GET | `/api/stats` | — | `{"Meeting": n, "Speaker": n, "Utterance": n, "Decision": n, "ActionItem": n, "Topic": n, "Person": n, "relationships": n}` |
| GET | `/api/meetings` | — | `MeetingSummary[]` (newest first) |
| POST | `/api/meetings` | `{title}` | `MeetingSummary` (status `live`, source `mic`). Open the audio WebSocket next |
| GET | `/api/meetings/{id}` | — | `{meeting, speakers[], utterances[], decisions[], action_items[], topics[]}` |
| POST | `/api/meetings/{id}/end` | — | `MeetingSummary`. Flushes the listener, runs a final extraction and a summary |
| DELETE | `/api/meetings/{id}` | — | `{deleted: bool}` |
| POST | `/api/meetings/upload` | multipart `file` (wav/mp3/m4a/webm/ogg), `title`, `pace` (`"fast"` default \| `"realtime"`) | `MeetingSummary` immediately; processing streams events |
| GET | `/api/meetings/{id}/audio` | — | `audio/wav` of the meeting (404 if none) |
| GET | `/api/meetings/{id}/utterances?after_seq=-1` | — | `UtteranceOut[]` |
| PATCH | `/api/meetings/{id}/speakers/{track}` | `{name}` | `Speaker` |
| GET | `/api/action-items` | `?status=open\|done&owner=&meeting_id=` | `ActionItem[]` |
| PATCH | `/api/action-items/{id}` | `{status: "open"\|"done"}` | `ActionItem` |
| GET | `/api/decisions` | `?meeting_id=` | `Decision[]` |
| GET | `/api/people` | — | `[{id, name, meetings, utterances, talk_time_s, open_items}]` |
| GET | `/api/who-knows` | `?topic=` | `[{name, person_id, meeting_id, hits, sample}]` |
| GET | `/api/graph` | `?meeting_id=&utterance_ids=a,b` | `Subgraph` |
| GET | `/api/search` | `?q=&meeting_id=` | `Hit[]` (hybrid) |
| POST | `/api/ask` | `{session_id, question, meeting_id?}` | `AskResult` |
| POST | `/api/ask/voice` | multipart `audio` (16 kHz mono WAV ≤30 s), `session_id`, `meeting_id?` | `AskResult` with `transcript` + `transcript_provider` |
| GET | `/api/sessions/{session_id}` | — | `[{role, text, at, citations}]` |

Errors: `{"detail": str}` with a 4xx/5xx status.

## Server-Sent Events

`GET /api/events?since_id=0&meeting_id=` (`text/event-stream`). Each message:

```
id: 42
event: utterance.created
data: {"id":42,"type":"utterance.created","meeting_id":"m_...","ts":1760500000.1,"data":{...UtteranceOut}}
```

The event types and payloads are listed in [`apps/api/wsw_brain/events.py`](../apps/api/wsw_brain/events.py). A heartbeat comment (`: ping`) is sent every 15 s. On reconnect, the client sends `since_id` = the last id it saw, or it re-fetches over REST.

## Live audio WebSocket

`WS /ws/meetings/{id}/audio`

- **Client → server:** binary frames of **16 kHz mono signed 16-bit little-endian PCM**, about 100 ms each (3,200 bytes). Text frame `{"type":"stop"}` ends the stream; the meeting ends too unless `{"type":"stop","end":false}` is sent.
- **Server → client:** JSON text frames `{"type":"status","state":"listening"|"processing","lag_s":float}` about every 2 s, plus `{"type":"error","detail":str}`.
- Utterances, speakers and insights arrive over **SSE**, not this socket, so every open tab sees them.

## Client actions in `AskResult.actions`

| type | fields | UI behaviour |
|---|---|---|
| `seek` | `meeting_id, t` | open the meeting and play the audio from `t` seconds (or highlight the utterance if there is no audio) |
| `open_meeting` | `meeting_id` | navigate to the meeting |
| `refresh` | `what: "action_items"\|"decisions"\|"meeting"` | refetch that data |
| `draft` | `title, body` | show a copyable draft (never sent anywhere) |
