import { useEffect, useRef } from "react";
import { API_BASE } from "./client";
import type { BusEvent } from "./types";

type Handler = (e: BusEvent) => void;

/**
 * One shared EventSource for the whole app (SSE /api/events), auto-reconnecting with since_id.
 * Subscribe with useBusEvents(handler, filter?).
 */
class EventHub {
  private es: EventSource | null = null;
  private handlers = new Set<Handler>();
  private lastId = 0;
  private retry = 0;
  private timer: number | null = null;
  connected = false;
  private statusListeners = new Set<(c: boolean) => void>();

  subscribe(h: Handler): () => void {
    this.handlers.add(h);
    this.ensure();
    return () => {
      this.handlers.delete(h);
    };
  }

  onStatus(fn: (c: boolean) => void): () => void {
    this.statusListeners.add(fn);
    fn(this.connected);
    return () => {
      this.statusListeners.delete(fn);
    };
  }

  private setConnected(c: boolean) {
    this.connected = c;
    this.statusListeners.forEach((fn) => fn(c));
  }

  private ensure() {
    if (this.es) return;
    const url = `${API_BASE}/api/events?since_id=${this.lastId}`;
    const es = new EventSource(url);
    this.es = es;
    es.onopen = () => {
      this.retry = 0;
      this.setConnected(true);
    };
    es.onmessage = (m) => this.dispatch(m);
    const types = [
      "meeting.started",
      "meeting.ended",
      "utterance.created",
      "speaker.merged",
      "speaker.named",
      "insight.created",
      "action.updated",
      "graph.write",
      "listener.status",
    ];
    types.forEach((t) => es.addEventListener(t, (m) => this.dispatch(m as MessageEvent)));
    es.onerror = () => {
      this.setConnected(false);
      es.close();
      this.es = null;
      const delay = Math.min(10_000, 500 * 2 ** this.retry++);
      if (this.timer) window.clearTimeout(this.timer);
      this.timer = window.setTimeout(() => this.ensure(), delay);
    };
  }

  private dispatch(m: MessageEvent) {
    try {
      const e = JSON.parse(m.data) as BusEvent;
      if (typeof e.id === "number") this.lastId = Math.max(this.lastId, e.id);
      this.handlers.forEach((h) => h(e));
    } catch {
      /* ignore malformed */
    }
  }
}

export const hub = new EventHub();

export function useBusEvents(handler: Handler, filter?: (e: BusEvent) => boolean): void {
  const ref = useRef(handler);
  ref.current = handler;
  const fref = useRef(filter);
  fref.current = filter;
  useEffect(
    () =>
      hub.subscribe((e) => {
        if (!fref.current || fref.current(e)) ref.current(e);
      }),
    [],
  );
}
