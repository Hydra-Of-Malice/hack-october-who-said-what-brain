import { memo, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { MeetingSummary } from "../../api/types";
import { duration, when } from "../../lib/format";
import TapeStrip from "./TapeStrip";
import { requestStrip, useStrip, type StripData } from "./stripCache";

function peopleText(data: StripData | undefined, count: number): string {
  const n = data ? data.speakers.length : count;
  const named = data?.speakers.filter((s) => s.named).map((s) => s.name) ?? [];
  if (!named.length) return n === 0 ? "No speakers yet" : `${n} ${n === 1 ? "speaker" : "speakers"}`;
  const shown = named.slice(0, 3).join(", ");
  const rest = n - Math.min(3, named.length);
  return rest > 0 ? `${shown} and ${rest} more` : shown;
}

function sourceText(m: MeetingSummary): string {
  if (m.source === "upload") return "Uploaded";
  if (m.source === "demo") return "Demo";
  return "Recorded live";
}

/** Fires once when the element first comes near the viewport. */
function useSeenOnce<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "240px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

function MeetingRow({ m, now }: { m: MeetingSummary; now: number }) {
  const [ref, seen] = useSeenOnce<HTMLLIElement>();
  const entry = useStrip(m.id);
  useEffect(() => {
    if (seen && entry.status === "idle") requestStrip(m.id);
  }, [seen, entry.status, m.id]);

  const live = m.status === "live";
  const startedAt = Number(m.started_at) || 0;
  const elapsed = live && startedAt ? (now - startedAt) / 1000 : 0;
  const lastEnd = entry.data?.lastEnd ?? 0;
  const length = Math.max(Number(m.duration_s) || 0, live && m.source !== "upload" ? elapsed : 0, lastEnd);

  return (
    <li ref={ref} className={`mrow${live ? " is-live" : ""}`}>
      <div className="mrow-main">
        <div className="mrow-titleline">
          {live && (
            <span className="mrow-live">
              <span className="live-dot" aria-hidden="true" />
              Live
            </span>
          )}
          <Link to={`/m/${encodeURIComponent(m.id)}`} className="mrow-title">
            {m.title || "Untitled meeting"}
          </Link>
        </div>
        <div className="mrow-meta">
          {startedAt > 0 && <span>{when(startedAt)}</span>}
          <span>{length > 0 ? duration(length) : "Just started"}</span>
          <span>{peopleText(entry.data, m.speaker_count ?? 0)}</span>
          <span className="mrow-source">{sourceText(m)}</span>
        </div>
      </div>
      <div className="mrow-tape">
        <TapeStrip entry={entry} domain={length} />
      </div>
    </li>
  );
}

export default memo(MeetingRow);
