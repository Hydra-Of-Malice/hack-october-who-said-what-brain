import type { CSSProperties } from "react";
import { isHollow, kindStyle, toneVar } from "./graphStyle";

/** The node shape for a kind, drawn the same way as on the canvas (for legends and panels). */
export default function KindSwatch({ kind, spk = 0, size = 14 }: { kind: string; spk?: number; size?: number }) {
  const s = kindStyle(kind);
  const color = toneVar(s.tone, spk);
  // CSS variables only resolve in style, not in SVG presentation attributes.
  const paint: CSSProperties = isHollow(s.shape)
    ? { fill: "var(--surface)", stroke: color, strokeWidth: 1.6 }
    : { fill: color, stroke: "none" };
  let shape;
  switch (s.shape) {
    case "circle":
    case "ring":
      shape = <circle cx={7} cy={7} r={5} style={paint} />;
      break;
    case "square":
    case "hollowSquare":
      shape = <rect x={2.2} y={2.2} width={9.6} height={9.6} style={paint} />;
      break;
    case "diamond":
      shape = <polygon points="7,1.2 12.8,7 7,12.8 1.2,7" style={paint} />;
      break;
    case "triangle":
      shape = <polygon points="7,1.6 12.6,11.6 1.4,11.6" style={paint} />;
      break;
  }
  return (
    <svg className="kind-swatch" width={size} height={size} viewBox="0 0 14 14" aria-hidden="true" focusable="false">
      {shape}
    </svg>
  );
}
