import type {
  ActionItem,
  AskResult,
  Decision,
  Health,
  Hit,
  MeetingDetail,
  MeetingSummary,
  Person,
  Speaker,
  Stats,
  Subgraph,
  Turn,
  Utterance,
  WhoKnows,
} from "./types";

// Same origin by default; set VITE_API_BASE for a separately hosted API.
export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "Can't reach the server. Check that the API is running.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  stats: () => req<Stats>("/api/stats"),
  meetings: () => req<MeetingSummary[]>("/api/meetings"),
  createMeeting: (title: string) => req<MeetingSummary>("/api/meetings", json("POST", { title })),
  meeting: (id: string) => req<MeetingDetail>(`/api/meetings/${encodeURIComponent(id)}`),
  endMeeting: (id: string) => req<MeetingSummary>(`/api/meetings/${encodeURIComponent(id)}/end`, { method: "POST" }),
  deleteMeeting: (id: string) => req<{ deleted: boolean }>(`/api/meetings/${encodeURIComponent(id)}`, { method: "DELETE" }),
  utterancesAfter: (id: string, afterSeq: number) =>
    req<Utterance[]>(`/api/meetings/${encodeURIComponent(id)}/utterances?after_seq=${afterSeq}`),
  uploadMeeting: (file: File, title: string, pace: "fast" | "realtime" = "fast") => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("title", title);
    fd.append("pace", pace);
    return req<MeetingSummary>("/api/meetings/upload", { method: "POST", body: fd });
  },
  audioUrl: (id: string) => `${API_BASE}/api/meetings/${encodeURIComponent(id)}/audio`,
  nameSpeaker: (meetingId: string, track: string, name: string) =>
    req<Speaker>(`/api/meetings/${encodeURIComponent(meetingId)}/speakers/${encodeURIComponent(track)}`, json("PATCH", { name })),
  actionItems: (q: { status?: "open" | "done"; owner?: string; meeting_id?: string } = {}) =>
    req<ActionItem[]>(`/api/action-items?${new URLSearchParams(q as Record<string, string>)}`),
  setActionItem: (id: string, status: "open" | "done") =>
    req<ActionItem>(`/api/action-items/${encodeURIComponent(id)}`, json("PATCH", { status })),
  decisions: (meetingId?: string) =>
    req<Decision[]>(`/api/decisions${meetingId ? `?meeting_id=${encodeURIComponent(meetingId)}` : ""}`),
  people: () => req<Person[]>("/api/people"),
  whoKnows: (topic: string) => req<WhoKnows[]>(`/api/who-knows?topic=${encodeURIComponent(topic)}`),
  graph: (q: { meeting_id?: string; utterance_ids?: string[] } = {}) => {
    const p = new URLSearchParams();
    if (q.meeting_id) p.set("meeting_id", q.meeting_id);
    if (q.utterance_ids?.length) p.set("utterance_ids", q.utterance_ids.join(","));
    return req<Subgraph>(`/api/graph?${p}`);
  },
  search: (q: string, meetingId?: string) =>
    req<Hit[]>(`/api/search?q=${encodeURIComponent(q)}${meetingId ? `&meeting_id=${encodeURIComponent(meetingId)}` : ""}`),
  ask: (session_id: string, question: string, meeting_id?: string, signal?: AbortSignal) =>
    req<AskResult>("/api/ask", { ...json("POST", { session_id, question, meeting_id }), signal }),
  askVoice: (session_id: string, wav: Blob, meeting_id?: string, signal?: AbortSignal) => {
    const fd = new FormData();
    fd.append("audio", wav, "question.wav");
    fd.append("session_id", session_id);
    if (meeting_id) fd.append("meeting_id", meeting_id);
    return req<AskResult>("/api/ask/voice", { method: "POST", body: fd, signal });
  },
  session: (id: string) => req<Turn[]>(`/api/sessions/${encodeURIComponent(id)}`),
};

/** ws(s)://host/ws/meetings/{id}/audio */
export function audioSocketUrl(meetingId: string): string {
  const base = API_BASE || window.location.origin;
  const u = new URL(`/ws/meetings/${encodeURIComponent(meetingId)}/audio`, base);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}
