import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import { useBusEvents } from "../../api/events";
import type { Stats } from "../../api/types";

const ROWS: { key: string; label: string }[] = [
  { key: "Meeting", label: "Meetings" },
  { key: "Utterance", label: "Utterances" },
  { key: "Speaker", label: "Speakers" },
  { key: "Decision", label: "Decisions" },
  { key: "ActionItem", label: "Action items" },
  { key: "relationships", label: "Relationships" },
];

const fmt = new Intl.NumberFormat();
const REFETCH_DEBOUNCE_MS = 600;

/** Node and relationship counts in FalkorDB, kept current from graph.write events. */
export default function GraphTotals() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const load = useCallback(() => {
    api
      .stats()
      .then((s) => {
        setStats(s ?? {});
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    load();
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [load]);

  useBusEvents((e) => {
    if (e.type === "graph.write") {
      const totals = (e.data as { totals?: Record<string, unknown> } | null)?.totals;
      if (totals && typeof totals === "object") {
        const numeric = Object.fromEntries(Object.entries(totals).filter(([, v]) => typeof v === "number")) as Stats;
        setStats((s) => ({ ...(s ?? {}), ...numeric }));
        return;
      }
    }
    if (e.type === "graph.write" || e.type === "meeting.started" || e.type === "meeting.ended" || e.type === "insight.created") {
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(load, REFETCH_DEBOUNCE_MS);
    }
  });

  return (
    <section className="totals" aria-labelledby="totals-title">
      <h2 id="totals-title" className="totals-title">
        In the FalkorDB graph
      </h2>
      {error && !stats ? (
        <p className="totals-error">
          Graph totals aren't available: {error}.{" "}
          <button type="button" className="totals-retry" onClick={load}>
            Retry
          </button>
        </p>
      ) : (
        <dl className="totals-list" aria-busy={!stats}>
          {ROWS.map((r) => (
            <div key={r.key} className="totals-item">
              <dt>{r.label}</dt>
              <dd>{stats ? fmt.format(Number(stats[r.key]) || 0) : "–"}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
