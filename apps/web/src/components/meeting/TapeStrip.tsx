import { memo } from "react";
import type { StripEntry } from "./stripCache";
import "./TapeStrip.css";

const ROW = 10; // viewBox units per speaker track
const GAP = 2;
const WIDTH = 1000;
const MIN_W = 1.6;

/** A meeting compressed to a strip of tape: one thin track per speaker, blocks where they spoke. */
function TapeStrip({ entry, domain }: { entry: StripEntry; domain: number }) {
  const data = entry.status === "ready" ? entry.data : undefined;
  if (!data || data.speakers.length === 0) {
    const label =
      entry.status === "error"
        ? "Speaker timeline unavailable"
        : entry.status === "ready"
          ? "No speech recorded"
          : "Loading speaker timeline";
    return (
      <div className={`tape tape--blank${entry.status === "loading" ? " is-loading" : ""}`} role="img" aria-label={label} title={entry.error ?? label} />
    );
  }

  const d = Math.max(domain, data.lastEnd, 1);
  const rows = data.speakers.length;
  const rowOf = new Map(data.speakers.map((s, i) => [s.track, i]));
  const talk = new Map<string, number>();
  for (const s of data.segments) talk.set(s.track, (talk.get(s.track) ?? 0) + (s.t1 - s.t0));
  const total = [...talk.values()].reduce((a, b) => a + b, 0) || 1;
  const summary = data.speakers
    .map((s) => `${s.name} ${Math.round(((talk.get(s.track) ?? 0) / total) * 100)}%`)
    .join(", ");

  return (
    <svg
      className="tape"
      viewBox={`0 0 ${WIDTH} ${rows * ROW}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Share of talk time: ${summary}`}
      shapeRendering="crispEdges"
    >
      {data.speakers.map((s, i) => (
        <rect key={s.track} className="tape-track" x={0} y={i * ROW + GAP / 2} width={WIDTH} height={ROW - GAP} />
      ))}
      {data.segments.map((seg, i) => {
        const row = rowOf.get(seg.track) ?? 0;
        const colour = data.speakers[row]?.colour ?? 7;
        const x = (seg.t0 / d) * WIDTH;
        return (
          <rect
            key={i}
            x={x}
            y={row * ROW + GAP / 2}
            width={Math.max(MIN_W, ((seg.t1 - seg.t0) / d) * WIDTH)}
            height={ROW - GAP}
            style={{ fill: `var(--spk-${colour})` }}
          />
        );
      })}
    </svg>
  );
}

export default memo(TapeStrip);
