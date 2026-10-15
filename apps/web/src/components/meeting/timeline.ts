// Time-axis helpers shared by the lanes and the tape strips.
import { useEffect, useState } from "react";

const LIVE_DOMAIN_STEPS_S = [60, 120, 180, 300, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200, 10800];
const TICK_STEPS_S = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

/** While live, the axis grows in steps (with headroom after the head) so blocks rescale rarely, not every second. */
export function liveDomain(head: number): number {
  const need = head * 1.12 + 4;
  return LIVE_DOMAIN_STEPS_S.find((s) => need <= s) ?? Math.ceil(need / 3600) * 3600;
}

export function endedDomain(durationS: number | null | undefined, lastEnd: number): number {
  return Math.max(1, Number(durationS) || 0, lastEnd);
}

export function rulerTicks(domain: number, maxTicks = 10): number[] {
  const step = TICK_STEPS_S.find((s) => domain / s <= maxTicks) ?? Math.ceil(domain / maxTicks / 3600) * 3600;
  const ticks: number[] = [];
  for (let t = 0; t < domain - step * 0.3; t += step) ticks.push(t);
  return ticks;
}

export function pct(t: number, domain: number): number {
  if (!(domain > 0)) return 0;
  return Math.min(100, Math.max(0, (t / domain) * 100));
}

/** Track width floor so long meetings scroll instead of crushing blocks to hairlines. */
export function minTrackPx(domain: number): number {
  return Math.max(560, Math.round((domain / 60) * 24));
}

/** Re-renders every `ms` while enabled; returns Date.now() at the last tick. */
export function useNow(ms: number, enabled = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return;
    const t = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(t);
  }, [ms, enabled]);
  return now;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}
