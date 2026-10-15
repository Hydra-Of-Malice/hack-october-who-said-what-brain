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

## Access (public URL safety)

Reading, searching and asking are always open. Changes are limited as follows (see [`api/access.py`](../apps/api/wsw_brain/api/access.py)).

| Setting | Effect |
|---|---|
| no `WSW_ADMIN_TOKEN` (default) | Anyone may create meetings, capture, upload, rename speakers and toggle action items. **Demo meetings can't be ended, deleted or have their speakers renamed** |
| `WSW_ADMIN_TOKEN=<secret>` | Creating, uploading, ending, deleting, renaming and the audio socket need the operator token. Send it in one of three ways: `Authorization: Bearer <token>`, `X-WSW-Token`, or the HttpOnly `wsw_token` cookie, which `POST /api/auth {"token": ...}` sets. `GET /api/auth` returns `{required, operator}` |
| `WSW_CREATE_LIMIT` (default 30) | New meetings allowed per client address per 10 minutes. The operator is exempt |
| `WSW_ALLOWED_HOSTS` | Comma-separated Host allowlist (empty means any) |

**Status codes:**
- `401`: operator token missing or wrong.
- `403`: a protected demo meeting.
- `429`: create limit reached, or Gemma is busy with other requests.

The audio socket refuses cross-site origins (close code 1008). When the listener falls too far behind real time, it closes with 1013 ("reconnect in a moment").

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
