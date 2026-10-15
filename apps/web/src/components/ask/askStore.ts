// Conversation state for the ask drawer, kept per browser tab (sessionStorage) so a reload keeps the thread.
import type { AskResult, Turn } from "../../api/types";
import { newSessionId } from "../../lib/format";

export type ExchangeStatus = "pending" | "answered" | "error" | "stopped";

export interface Exchange {
  id: string;
  mode: "voice" | "text";
  /** Typed text, or the transcript of a spoken question once it is known. */
  question: string | null;
  /** AskResult.transcript_provider for spoken questions. */
  heardBy: string | null;
  /** Length of the recorded clip, for spoken questions. */
  seconds: number | null;
  status: ExchangeStatus;
  result: AskResult | null;
  /** Answer text restored from the server's session log (no citations). */
  restoredAnswer?: string | null;
  error: string | null;
  /** Meeting the question was scoped to. */
  scope: string | null;
  at: number;
}

const SESSION_KEY = "wsw.ask.session";
const logKey = (sessionId: string) => `wsw.ask.log.${sessionId}`;
const MAX_KEPT = 30;

function read(storage: () => Storage, key: string): string | null {
  try {
    return storage().getItem(key);
  } catch {
    return null;
  }
}

function write(storage: () => Storage, key: string, value: string | null): void {
  try {
    if (value === null) storage().removeItem(key);
    else storage().setItem(key, value);
  } catch {
    /* storage unavailable (private mode, quota): the drawer still works for this page view */
  }
}

const session = () => window.sessionStorage;
const local = () => window.localStorage;

/** The current session id, and whether it was carried over from earlier in this tab. */
export function currentSession(): { id: string; resumed: boolean } {
  const existing = read(session, SESSION_KEY);
  if (existing) return { id: existing, resumed: true };
  const id = newSessionId();
  write(session, SESSION_KEY, id);
  return { id, resumed: false };
}

export function startNewSession(previous: string | null): string {
  if (previous) write(session, logKey(previous), null);
  const id = newSessionId();
  write(session, SESSION_KEY, id);
  return id;
}

export function loadExchanges(sessionId: string): Exchange[] {
  const raw = read(session, logKey(sessionId));
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as Exchange[];
    return Array.isArray(parsed) ? parsed.filter((x) => x && typeof x.id === "string") : [];
  } catch {
    return [];
  }
}

export function saveExchanges(sessionId: string, exchanges: Exchange[]): void {
  // A pending request cannot survive a reload, so it is not kept.
  const kept = exchanges.filter((x) => x.status !== "pending").slice(-MAX_KEPT);
  write(session, logKey(sessionId), JSON.stringify(kept));
}

/** Rebuild a text-only thread from the server's session log (used when this tab lost its cache). */
export function exchangesFromTurns(turns: Turn[]): Exchange[] {
  const out: Exchange[] = [];
  let open: Exchange | null = null;
  for (const t of turns) {
    if (t.role === "user") {
      open = {
        id: `r_${t.at}_${out.length}`,
        mode: "text",
        question: t.text,
        heardBy: null,
        seconds: null,
        status: "answered",
        result: null,
        restoredAnswer: null,
        error: null,
        scope: null,
        at: t.at,
      };
      out.push(open);
    } else if (open && !open.restoredAnswer) {
      open.restoredAnswer = t.text;
    }
  }
  return out;
}

export function readFlag(key: string, fallback: boolean): boolean {
  const v = read(local, key);
  return v === null ? fallback : v === "1";
}

export function writeFlag(key: string, value: boolean): void {
  write(local, key, value ? "1" : "0");
}

let seq = 0;
export function exchangeId(): string {
  seq += 1;
  return `x_${Date.now().toString(36)}_${seq}`;
}
