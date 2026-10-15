import { useSyncExternalStore } from "react";
import type { LevelStore } from "./useCapture";

const SEGMENTS = 12;
const HOT_FROM = 10; // the last two segments mark clipping territory
const FLOOR_DB = -54;

/** Segmented input meter, like a channel strip: RMS mapped to dBFS. */
export default function LevelMeter({ store }: { store: LevelStore }) {
  const rms = useSyncExternalStore(store.subscribe, store.get);
  const db = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
  const level = Math.max(0, Math.min(1, (db - FLOOR_DB) / -FLOOR_DB));
  const lit = Math.round(level * SEGMENTS);
  return (
    <span className="meter" role="meter" aria-label="Microphone input level" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(level * 100)}>
      {Array.from({ length: SEGMENTS }, (_, i) => (
        <span key={i} className={`meter-seg${i < lit ? " is-on" : ""}${i >= HOT_FROM ? " is-hot" : ""}`} />
      ))}
    </span>
  );
}
