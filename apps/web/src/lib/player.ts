// Tiny app-wide channel so the Ask drawer (or anything) can ask the meeting room to seek/highlight.
// The meeting room listens with onSeek(); producers call requestSeek().
import { useEffect, useRef } from "react";

export interface SeekRequest {
  meeting_id: string;
  t: number; // seconds
  utterance_id?: string;
}

type Fn = (r: SeekRequest) => void;
const listeners = new Set<Fn>();
let pending: SeekRequest | null = null;

export function requestSeek(r: SeekRequest): void {
  pending = r;
  listeners.forEach((fn) => fn(r));
}

/** Consume the last seek request (e.g. after navigating to the meeting page). */
export function takePendingSeek(meetingId: string): SeekRequest | null {
  if (pending && pending.meeting_id === meetingId) {
    const p = pending;
    pending = null;
    return p;
  }
  return null;
}

export function useSeekRequests(fn: Fn): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    const h: Fn = (r) => ref.current(r);
    listeners.add(h);
    return () => {
      listeners.delete(h);
    };
  }, []);
}
