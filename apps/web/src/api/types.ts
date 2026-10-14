// Mirrors apps/api/wsw_brain/contracts.py and docs/API_CONTRACT.md. Keep in sync.

export type MeetingStatus = "live" | "ended";
export type MeetingSource = "mic" | "upload" | "demo";

export interface MeetingSummary {
  id: string;
  title: string;
  status: MeetingStatus;
  source: MeetingSource;
  started_at: number; // epoch ms
  ended_at: number | null;
  duration_s: number;
  utterance_count: number;
  speaker_count: number;
  has_audio: boolean;
  summary: string | null;
}

export interface Speaker {
  track: string; // "S1"
  label: string; // "Speaker 1"
  name: string | null;
  person_id: string | null;
  color_idx: number;
  utterance_count: number;
  talk_time_s: number;
}

export interface Utterance {
  id: string;
  meeting_id: string;
  seq: number;
  speaker: string; // track
  speaker_name: string | null;
  t_start: number; // seconds from meeting start
  t_end: number;
  text: string;
  confidence: number;
}

export interface Decision {
  id: string;
  meeting_id: string;
  text: string;
  status: "active" | "superseded";
  made_at: number;
  proposed_by: string | null;
  evidence: string[];
  superseded_by: string | null;
  supersedes: string | null;
}

export interface ActionItem {
  id: string;
  meeting_id: string;
  meeting_title: string;
  text: string;
  owner: string | null;
  owner_track: string | null;
  due: string | null;
  status: "open" | "done";
  created_at: number;
  done_at: number | null;
  evidence: string[];
}

export interface Topic {
  id: string;
  name: string;
  mentions: number;
}

export interface MeetingDetail {
  meeting: MeetingSummary;
  speakers: Speaker[];
  utterances: Utterance[];
  decisions: Decision[];
  action_items: ActionItem[];
  topics: Topic[];
}

export interface Hit {
  utterance_id: string;
  meeting_id: string;
  meeting_title: string;
  speaker: string;
  speaker_name: string | null;
  t_start: number;
  t_end: number;
  text: string;
  score: number;
  via: "fulltext" | "vector" | "graph" | "recent";
}

export interface GraphNode {
  id: string;
  label: string; // display text
  kind: "Meeting" | "Speaker" | "Person" | "Utterance" | "Decision" | "ActionItem" | "Topic" | "Entity" | string;
  [k: string]: unknown;
}
export interface GraphLink {
  source: string;
  target: string;
  type: string;
}
export interface Subgraph {
  nodes: GraphNode[];
  links: GraphLink[];
}

export type ClientAction =
  | { type: "seek"; meeting_id: string; t: number }
  | { type: "open_meeting"; meeting_id: string }
  | { type: "refresh"; what: "action_items" | "decisions" | "meeting" }
  | { type: "draft"; title: string; body: string };

export interface AskResult {
  answer: string;
  citations: Hit[];
  actions: ClientAction[];
  tool_trace: { tool: string; args: Record<string, unknown>; result_count: number; ms: number }[];
  cypher: string[];
  model: string;
  provider: string;
  session_id: string;
  timings: Record<string, number>;
  transcript: string | null;
  transcript_provider: string | null;
}

export interface Health {
  ok: boolean;
  brand: Brand;
  falkordb: { ok: boolean; version: string | null; graph: string };
  llm: { provider: string; model: string; audio: boolean };
  question_audio: string;
  embedder: { name: string; dim: number };
  asr: { name: string };
}

export interface Brand {
  name: string;
  short: string;
  tagline: string;
  event: string;
  track: string;
  problem_statement: string;
}

export interface Person {
  id: string;
  name: string;
  meetings: number;
  utterances: number;
  talk_time_s: number;
  open_items: number;
}

export interface WhoKnows {
  name: string;
  person_id: string | null;
  meeting_id: string;
  hits: number;
  sample: Utterance;
}

export type Stats = Record<string, number>;

export interface BusEvent<T = unknown> {
  id: number;
  type:
    | "meeting.started"
    | "meeting.ended"
    | "utterance.created"
    | "speaker.merged"
    | "speaker.named"
    | "insight.created"
    | "action.updated"
    | "graph.write"
    | "listener.status";
  meeting_id: string | null;
  ts: number;
  data: T;
}

export interface Turn {
  role: "user" | "assistant";
  text: string;
  at: number;
  citations: string[];
}
