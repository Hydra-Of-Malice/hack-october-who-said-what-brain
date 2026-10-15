// Small shared caches for meeting lists and meeting details, used by the ask drawer, palette and pages.
import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../../api/client";
import { useBusEvents } from "../../api/events";
import type { MeetingDetail, MeetingSummary, Utterance } from "../../api/types";

const LIST_TTL_MS = 30_000;
const DETAIL_TTL_MS = 30_000;

let meetingsEntry: { at: number; promise: Promise<MeetingSummary[]> } | null = null;
const detailEntries = new Map<string, { at: number; promise: Promise<MeetingDetail> }>();

export function loadMeetings(force = false): Promise<MeetingSummary[]> {
  if (!force && meetingsEntry && Date.now() - meetingsEntry.at < LIST_TTL_MS) return meetingsEntry.promise;
  const promise = api.meetings();
  const entry = { at: Date.now(), promise };
  meetingsEntry = entry;
  promise.catch(() => {
    if (meetingsEntry === entry) meetingsEntry = null;
  });
  return promise;
}

export function loadMeetingDetail(id: string, force = false): Promise<MeetingDetail> {
  const hit = detailEntries.get(id);
  if (!force && hit && Date.now() - hit.at < DETAIL_TTL_MS) return hit.promise;
  const promise = api.meeting(id);
  const entry = { at: Date.now(), promise };
  detailEntries.set(id, entry);
  promise.catch(() => {
    if (detailEntries.get(id) === entry) detailEntries.delete(id);
  });
  return promise;
}

export function forgetMeetingDetail(id: string): void {
  detailEntries.delete(id);
}

/** Details for a set of meetings, keyed by id (meetings that fail to load are skipped). */
export async function loadDetails(meetingIds: Iterable<string>, limit = 12, force = false): Promise<Map<string, MeetingDetail>> {
  const ids = Array.from(new Set(meetingIds)).slice(0, limit);
  const settled = await Promise.allSettled(ids.map((id) => loadMeetingDetail(id, force)));
  const out = new Map<string, MeetingDetail>();
  settled.forEach((d, i) => {
    if (d.status === "fulfilled") out.set(ids[i], d.value);
  });
  return out;
}

/** Utterances by id across loaded meeting details. */
export function utteranceIndex(details: Map<string, MeetingDetail>): Map<string, Utterance> {
  const out = new Map<string, Utterance>();
  for (const d of details.values()) for (const u of d.utterances) out.set(u.id, u);
  return out;
}

/** Display name of a meeting-scoped speaker track ("S2"), using the meeting's speaker list when loaded. */
export function trackName(details: Map<string, MeetingDetail>, meetingId: string, track: string): string {
  const s = details.get(meetingId)?.speakers.find((x) => x.track === track);
  if (s) return s.name || s.label;
  return /^S\d+$/.test(track) ? `Speaker ${track.slice(1)}` : track;
}

/** Start time of an utterance, fetching its meeting if needed. Null when it can't be found. */
export async function utteranceStart(meetingId: string, utteranceId: string): Promise<number | null> {
  try {
    const detail = await loadMeetingDetail(meetingId);
    const u = detail.utterances.find((x) => x.id === utteranceId);
    return u ? u.t_start : null;
  } catch {
    return null;
  }
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export interface MeetingsState {
  meetings: MeetingSummary[] | null;
  error: string | null;
  reload: () => void;
}

/** The meeting list, refreshed when meetings start or end. Pass enabled=false to defer the fetch. */
export function useMeetings(enabled = true): MeetingsState {
  const [meetings, setMeetings] = useState<MeetingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let live = true;
    loadMeetings(nonce > 0)
      .then((m) => {
        if (!live) return;
        setMeetings(m);
        setError(null);
      })
      .catch((e) => {
        if (live) setError(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [enabled, nonce]);

  useBusEvents(
    () => setNonce((n) => n + 1),
    (e) => e.type === "meeting.started" || e.type === "meeting.ended",
  );

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { meetings, error, reload };
}

export function titleMap(meetings: MeetingSummary[] | null): Map<string, string> {
  return new Map((meetings ?? []).map((m) => [m.id, m.title]));
}
