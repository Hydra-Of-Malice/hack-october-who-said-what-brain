// Meeting room state: REST snapshot merged with live SSE events. Pure reducer, no I/O.
import type { ActionItem, Decision, MeetingDetail, MeetingSummary, Speaker, Topic, Utterance } from "../../api/types";
import { speakerIndex } from "../../lib/format";

export interface LoadError {
  status: number;
  message: string;
}

export interface WriteStat {
  utterances: number;
  ms: number;
  at: number;
}

export interface ListenerStat {
  state: string;
  lagS: number | null;
  detail: string;
  at: number;
}

export interface RoomState {
  phase: "loading" | "ready" | "error";
  error: LoadError | null;
  meeting: MeetingSummary | null;
  speakers: Speaker[];
  utterances: Utterance[];
  decisions: Decision[];
  actionItems: ActionItem[];
  topics: Topic[];
  /** Ids that arrived live after the snapshot (lane blocks grow in, insights flash). */
  fresh: ReadonlySet<string>;
  lastWrite: WriteStat | null;
  writeBatches: number;
  listener: ListenerStat | null;
}

export type RoomAction =
  | { type: "reset" }
  | { type: "loaded"; detail: Partial<MeetingDetail> }
  | { type: "load-failed"; error: LoadError }
  | { type: "utterances"; list: Utterance[]; live: boolean }
  | { type: "speaker-merged"; from: string; to: string }
  | { type: "speaker-named"; track: string; name: string | null; personId?: string | null }
  | { type: "speaker"; speaker: Speaker }
  | { type: "insight"; kind: string; item: unknown; recent?: boolean }
  | { type: "action-item"; item: ActionItem }
  | { type: "graph-write"; utterances: number; ms: number }
  | { type: "listener"; state: string; lagS: number | null; detail: string }
  | { type: "meeting"; meeting: MeetingSummary };

export const initialRoom: RoomState = {
  phase: "loading",
  error: null,
  meeting: null,
  speakers: [],
  utterances: [],
  decisions: [],
  actionItems: [],
  topics: [],
  fresh: new Set(),
  lastWrite: null,
  writeBatches: 0,
  listener: null,
};

const trackNum = (track: string): number => {
  const n = parseInt(track.replace(/\D/g, ""), 10);
  return Number.isFinite(n) ? n : Number.MAX_SAFE_INTEGER;
};

const byTrack = (a: Speaker, b: Speaker) => trackNum(a.track) - trackNum(b.track) || a.track.localeCompare(b.track);

const bySeq = (a: Utterance, b: Utterance) => a.seq - b.seq || a.t_start - b.t_start;

function normUtterance(u: Utterance): Utterance {
  return {
    ...u,
    seq: Number.isFinite(u.seq) ? u.seq : 0,
    speaker: u.speaker ?? "S0",
    speaker_name: u.speaker_name ?? null,
    t_start: Number(u.t_start) || 0,
    t_end: Math.max(Number(u.t_end) || 0, Number(u.t_start) || 0),
    text: u.text ?? "",
    confidence: typeof u.confidence === "number" ? u.confidence : 1,
  };
}

function syntheticSpeaker(track: string, name: string | null): Speaker {
  return {
    track,
    label: `Speaker ${track.replace(/^S/, "")}`,
    name,
    person_id: null,
    color_idx: speakerIndex(track),
    utterance_count: 0,
    talk_time_s: 0,
  };
}

/** Adds speakers for tracks that only appear on utterances (live meetings announce speakers implicitly). */
function ensureSpeakers(speakers: Speaker[], utterances: Utterance[]): Speaker[] {
  const known = new Set(speakers.map((s) => s.track));
  const added: Speaker[] = [];
  for (const u of utterances) {
    if (!known.has(u.speaker)) {
      known.add(u.speaker);
      added.push(syntheticSpeaker(u.speaker, u.speaker_name));
    }
  }
  return added.length ? [...speakers, ...added].sort(byTrack) : speakers;
}

function mergeUtterances(current: Utterance[], incoming: Utterance[]): { list: Utterance[]; added: string[] } {
  if (!incoming.length) return { list: current, added: [] };
  const index = new Map(current.map((u, i) => [u.id, i]));
  const next = current.slice();
  const added: string[] = [];
  let ordered = true;
  for (const raw of incoming) {
    if (!raw || typeof raw.id !== "string") continue;
    const u = normUtterance(raw);
    const at = index.get(u.id);
    if (at !== undefined) {
      next[at] = u;
      continue;
    }
    const last = next[next.length - 1];
    if (last && bySeq(last, u) > 0) ordered = false;
    index.set(u.id, next.length);
    next.push(u);
    added.push(u.id);
  }
  if (!ordered) next.sort(bySeq);
  return { list: next, added };
}

function upsertById<T extends { id: string }>(list: T[], item: T): T[] {
  const i = list.findIndex((x) => x.id === item.id);
  if (i === -1) return [...list, item];
  const next = list.slice();
  next[i] = { ...list[i], ...item };
  return next;
}

function unionById<T extends { id: string }>(snapshot: T[], live: T[]): T[] {
  const ids = new Set(snapshot.map((x) => x.id));
  return [...snapshot, ...live.filter((x) => !ids.has(x.id))];
}

function withFresh(fresh: ReadonlySet<string>, ids: string[]): ReadonlySet<string> {
  if (!ids.length) return fresh;
  const next = new Set(fresh);
  ids.forEach((id) => next.add(id));
  return next;
}

function asDecision(item: unknown): Decision | null {
  const d = item as Partial<Decision> | null;
  if (!d || typeof d.id !== "string" || typeof d.text !== "string") return null;
  return {
    id: d.id,
    meeting_id: d.meeting_id ?? "",
    text: d.text,
    status: d.status === "superseded" ? "superseded" : "active",
    made_at: Number(d.made_at) || 0,
    proposed_by: d.proposed_by ?? null,
    evidence: Array.isArray(d.evidence) ? d.evidence : [],
    superseded_by: d.superseded_by ?? null,
    supersedes: d.supersedes ?? null,
  };
}

export function asActionItem(item: unknown): ActionItem | null {
  const a = item as Partial<ActionItem> | null;
  if (!a || typeof a.id !== "string" || typeof a.text !== "string") return null;
  return {
    id: a.id,
    meeting_id: a.meeting_id ?? "",
    meeting_title: a.meeting_title ?? "",
    text: a.text,
    owner: a.owner ?? null,
    owner_track: a.owner_track ?? null,
    due: a.due ?? null,
    status: a.status === "done" ? "done" : "open",
    created_at: Number(a.created_at) || 0,
    done_at: a.done_at ?? null,
    evidence: Array.isArray(a.evidence) ? a.evidence : [],
  };
}

function asTopic(item: unknown): Topic | null {
  const t = item as Partial<Topic> | null;
  if (!t || typeof t.name !== "string") return null;
  return { id: typeof t.id === "string" ? t.id : t.name.toLowerCase(), name: t.name, mentions: Number(t.mentions) || 1 };
}

function applyDecision(list: Decision[], d: Decision): Decision[] {
  let next = upsertById(list, d);
  if (d.supersedes) {
    next = next.map((x) => (x.id === d.supersedes ? { ...x, status: "superseded" as const, superseded_by: d.id } : x));
  }
  return next.sort((a, b) => a.made_at - b.made_at);
}

function applyInsight(state: RoomState, kind: string, item: unknown): RoomState {
  if (kind === "decision") {
    const d = asDecision(item);
    if (!d) return state;
    return { ...state, decisions: applyDecision(state.decisions, d), fresh: withFresh(state.fresh, [d.id]) };
  }
  if (kind === "action_item") {
    const a = asActionItem(item);
    if (!a) return state;
    return { ...state, actionItems: upsertById(state.actionItems, a), fresh: withFresh(state.fresh, [a.id]) };
  }
  if (kind === "topic") {
    const t = asTopic(item);
    if (!t) return state;
    const existing = state.topics.find((x) => x.id === t.id);
    const topic = existing ? { ...existing, mentions: Math.max(existing.mentions + 1, t.mentions) } : t;
    return { ...state, topics: upsertById(state.topics, topic), fresh: withFresh(state.fresh, [t.id]) };
  }
  return state;
}

function relabel(state: RoomState, from: string, to: string): RoomState {
  if (from === to) return state;
  const target = state.speakers.find((s) => s.track === to);
  const source = state.speakers.find((s) => s.track === from);
  const name = target?.name ?? source?.name ?? null;
  const utterances = state.utterances.map((u) => (u.speaker === from ? { ...u, speaker: to, speaker_name: name } : u));
  let speakers = state.speakers.filter((s) => s.track !== from);
  if (!target) speakers = [...speakers, { ...(source ?? syntheticSpeaker(to, name)), track: to, label: `Speaker ${to.replace(/^S/, "")}` }].sort(byTrack);
  const actionItems = state.actionItems.map((a) => (a.owner_track === from ? { ...a, owner_track: to } : a));
  const decisions = state.decisions.map((d) => (d.proposed_by === from ? { ...d, proposed_by: to } : d));
  return { ...state, utterances, speakers, actionItems, decisions };
}

function rename(state: RoomState, track: string, name: string | null, personId?: string | null): RoomState {
  const speakers = state.speakers.some((s) => s.track === track)
    ? state.speakers.map((s) => (s.track === track ? { ...s, name, person_id: personId === undefined ? s.person_id : personId } : s))
    : [...state.speakers, { ...syntheticSpeaker(track, name), person_id: personId ?? null }].sort(byTrack);
  const utterances = state.utterances.map((u) => (u.speaker === track ? { ...u, speaker_name: name } : u));
  return { ...state, speakers, utterances };
}

export function roomReducer(state: RoomState, action: RoomAction): RoomState {
  switch (action.type) {
    case "reset":
      return initialRoom;
    case "loaded": {
      const d = action.detail;
      if (!d.meeting) return { ...state, phase: "error", error: { status: 0, message: "The server returned a meeting without its details." } };
      // Events may have landed while the snapshot was in flight: keep them (their copies win on equal ids).
      const merged = mergeUtterances((d.utterances ?? []).map(normUtterance).sort(bySeq), state.utterances);
      const speakers = ensureSpeakers([...(d.speakers ?? [])].sort(byTrack), merged.list);
      return {
        ...state,
        phase: "ready",
        error: null,
        meeting: { ...d.meeting },
        speakers,
        utterances: merged.list,
        decisions: unionById((d.decisions ?? []).map(asDecision).filter((x): x is Decision => !!x), state.decisions).sort(
          (a, b) => a.made_at - b.made_at,
        ),
        actionItems: unionById((d.action_items ?? []).map(asActionItem).filter((x): x is ActionItem => !!x), state.actionItems),
        topics: unionById((d.topics ?? []).map(asTopic).filter((x): x is Topic => !!x), state.topics),
      };
    }
    case "load-failed":
      return { ...state, phase: "error", error: action.error };
    case "utterances": {
      const merged = mergeUtterances(state.utterances, action.list);
      if (merged.list === state.utterances) return state;
      return {
        ...state,
        utterances: merged.list,
        speakers: ensureSpeakers(state.speakers, merged.list),
        fresh: action.live ? withFresh(state.fresh, merged.added) : state.fresh,
      };
    }
    case "speaker-merged":
      return relabel(state, action.from, action.to);
    case "speaker-named":
      return rename(state, action.track, action.name, action.personId);
    case "speaker": {
      const next = rename(state, action.speaker.track, action.speaker.name, action.speaker.person_id);
      return { ...next, speakers: next.speakers.map((s) => (s.track === action.speaker.track ? { ...s, ...action.speaker } : s)) };
    }
    case "insight": {
      const next = applyInsight(state, action.kind, action.item);
      return action.recent === false ? { ...next, fresh: state.fresh } : next;
    }
    case "action-item":
      return { ...state, actionItems: upsertById(state.actionItems, action.item) };
    case "graph-write":
      return {
        ...state,
        lastWrite: { utterances: action.utterances, ms: action.ms, at: Date.now() },
        writeBatches: state.writeBatches + 1,
      };
    case "listener":
      return { ...state, listener: { state: action.state, lagS: action.lagS, detail: action.detail, at: Date.now() } };
    case "meeting":
      return { ...state, meeting: state.meeting ? { ...state.meeting, ...action.meeting } : action.meeting };
  }
}

/** Palette index per track: the server's colour when known, else derived from the track number. */
export function colourMap(speakers: Speaker[]): Map<string, number> {
  return new Map(speakers.map((s) => [s.track, ((Number.isFinite(s.color_idx) ? s.color_idx : speakerIndex(s.track)) % 8 + 8) % 8]));
}

/** Index of the utterance being heard at time t (last one that started at or before t), or -1. */
export function utteranceIndexAt(utterances: Utterance[], t: number): number {
  let lo = 0;
  let hi = utterances.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (utterances[mid].t_start <= t) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}
