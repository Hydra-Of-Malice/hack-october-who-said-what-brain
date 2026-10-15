import { useEffect, useRef } from "react";
import { useBusEvents } from "../../api/events";
import type { BusEvent } from "../../api/types";

interface LiveRefreshOptions {
  /** Bus events that should trigger a refetch. */
  events?: (e: BusEvent) => boolean;
  /** "wsw:refresh" details that should trigger a refetch ("action_items", "decisions", "meeting"). */
  refresh?: string[];
  delayMs?: number;
}

/** Refetch (debounced) when the server reports a change or the ask drawer asks for a refresh. */
export function useLiveRefresh(refetch: () => void, { events, refresh = [], delayMs = 700 }: LiveRefreshOptions): void {
  const fn = useRef(refetch);
  fn.current = refetch;
  const timer = useRef<number | null>(null);
  const refreshKey = refresh.join(",");

  const schedule = useRef(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      fn.current();
    }, delayMs);
  });

  useBusEvents(() => schedule.current(), events ?? (() => false));

  useEffect(() => {
    if (!refreshKey) return;
    const wanted = new Set(refreshKey.split(","));
    const on = (e: Event) => {
      const what = (e as CustomEvent<unknown>).detail;
      if (typeof what !== "string" || wanted.has(what)) schedule.current();
    };
    window.addEventListener("wsw:refresh", on);
    return () => window.removeEventListener("wsw:refresh", on);
  }, [refreshKey]);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
}
