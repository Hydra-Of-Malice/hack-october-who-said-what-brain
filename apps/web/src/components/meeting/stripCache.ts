// Lazily fetched speaker segments for the tape strips on the meetings index (one detail fetch per visible row).
import { useCallback, useSyncExternalStore } from "react";
import { api } from "../../api/client";
import type { MeetingDetail, Utterance } from "../../api/types";
import { speakerIndex, speakerName } from "../../lib/format";

export interface StripSpeaker {
  track: string;
  name: string;
  named: boolean;
  colour: number;
}

export interface StripSegment {
  track: string;
  t0: number;
  t1: number;
}

export interface StripData {
  speakers: StripSpeaker[];
  segments: StripSegment[];
  lastEnd: number;
}

export interface StripEntry {
  status: "idle" | "loading" | "ready" | "error";
  data?: StripData;
  error?: string;
}

const IDLE: StripEntry = { status: "idle" };
const MAX_CONCURRENT = 3;

const entries = new Map<string, StripEntry>();
const listeners = new Map<string, Set<() => void>>();
const queue: string[] = [];
let active = 0;

function put(id: string, entry: StripEntry | null) {
  if (entry) entries.set(id, entry);
  else entries.delete(id);
  listeners.get(id)?.forEach((fn) => fn());
}

const trackNum = (t: string) => parseInt(t.replace(/\D/g, ""), 10) || 0;

function speakerOf(track: string, name: string | null, colourIdx?: number): StripSpeaker {
  return {
    track,
    name: speakerName({ name, track }),
    named: !!name,
    colour: (((Number.isFinite(colourIdx) ? (colourIdx as number) : speakerIndex(track)) % 8) + 8) % 8,
  };
}

function toStrip(d: Partial<MeetingDetail>): StripData {
  const speakers = (d.speakers ?? []).map((s) => speakerOf(s.track, s.name, s.color_idx));
  const known = new Set(speakers.map((s) => s.track));
  const segments: StripSegment[] = [];
  let lastEnd = 0;
  for (const u of d.utterances ?? []) {
    const t0 = Number(u.t_start) || 0;
    const t1 = Math.max(t0, Number(u.t_end) || 0);
    segments.push({ track: u.speaker, t0, t1 });
    lastEnd = Math.max(lastEnd, t1);
    if (!known.has(u.speaker)) {
      known.add(u.speaker);
      speakers.push(speakerOf(u.speaker, u.speaker_name));
    }
  }
  speakers.sort((a, b) => trackNum(a.track) - trackNum(b.track));
  return { speakers, segments, lastEnd };
}

function pump() {
  while (active < MAX_CONCURRENT && queue.length) {
    const id = queue.shift() as string;
    active++;
    api
      .meeting(id)
      .then((d) => put(id, { status: "ready", data: toStrip(d) }))
      .catch((e: unknown) => put(id, { status: "error", error: e instanceof Error ? e.message : String(e) }))
      .finally(() => {
        active--;
        pump();
      });
  }
}

export function requestStrip(id: string): void {
  if ((entries.get(id) ?? IDLE).status !== "idle") return;
  put(id, { status: "loading" });
  queue.push(id);
  pump();
}

/** Drop a cached strip (e.g. the meeting just ended) so the next visible render refetches it. */
export function invalidateStrip(id: string): void {
  if (entries.has(id) && entries.get(id)?.status !== "loading") put(id, null);
}

/** Live meetings: extend an already-loaded strip without refetching. */
export function appendToStrip(u: Utterance): void {
  const e = entries.get(u.meeting_id);
  if (!e || e.status !== "ready" || !e.data || typeof u.speaker !== "string") return;
  const t0 = Number(u.t_start) || 0;
  const t1 = Math.max(t0, Number(u.t_end) || 0);
  const speakers = e.data.speakers.some((s) => s.track === u.speaker)
    ? e.data.speakers
    : [...e.data.speakers, speakerOf(u.speaker, u.speaker_name)].sort((a, b) => trackNum(a.track) - trackNum(b.track));
  put(u.meeting_id, {
    status: "ready",
    data: { speakers, segments: [...e.data.segments, { track: u.speaker, t0, t1 }], lastEnd: Math.max(e.data.lastEnd, t1) },
  });
}

export function useStrip(id: string): StripEntry {
  const subscribe = useCallback(
    (fn: () => void) => {
      let set = listeners.get(id);
      if (!set) {
        set = new Set();
        listeners.set(id, set);
      }
      set.add(fn);
      return () => {
        set.delete(fn);
      };
    },
    [id],
  );
  return useSyncExternalStore(subscribe, () => entries.get(id) ?? IDLE);
}
