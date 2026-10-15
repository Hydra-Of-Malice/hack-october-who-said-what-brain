// Node kinds, their shapes and tones, and canvas drawing for the graph views.
// Colours come from the design tokens at runtime (getComputedStyle), never from literals here.
import type { GraphNode } from "../../api/types";
import { speakerIndex } from "../../lib/format";

export type Shape = "circle" | "ring" | "square" | "hollowSquare" | "diamond" | "triangle";
export type Tone = "speaker" | "ink" | "ink2" | "ink3" | "lineStrong";

export interface KindStyle {
  shape: Shape;
  r: number;
  tone: Tone;
  label: string;
  plural: string;
  /** Label shown at moderate zoom (big, few nodes); others only when zoomed in or hovered. */
  prominent: boolean;
}

export const KINDS: Record<string, KindStyle> = {
  Meeting: { shape: "square", r: 7, tone: "ink", label: "Meeting", plural: "Meetings", prominent: true },
  Speaker: { shape: "circle", r: 6, tone: "speaker", label: "Speaker", plural: "Speakers", prominent: true },
  Person: { shape: "ring", r: 6.5, tone: "ink", label: "Person", plural: "People", prominent: true },
  Utterance: { shape: "circle", r: 2.8, tone: "lineStrong", label: "Utterance", plural: "Utterances", prominent: false },
  Decision: { shape: "diamond", r: 5.5, tone: "ink2", label: "Decision", plural: "Decisions", prominent: true },
  ActionItem: { shape: "hollowSquare", r: 5, tone: "ink2", label: "Action item", plural: "Action items", prominent: false },
  Topic: { shape: "triangle", r: 5.5, tone: "ink3", label: "Topic", plural: "Topics", prominent: true },
  Entity: { shape: "ring", r: 4.5, tone: "ink3", label: "Entity", plural: "Entities", prominent: false },
};

const OTHER: KindStyle = { shape: "circle", r: 4, tone: "ink3", label: "Other", plural: "Other", prominent: false };

export const KIND_ORDER = ["Meeting", "Speaker", "Person", "Utterance", "Decision", "ActionItem", "Topic", "Entity"];

export function kindStyle(kind: string): KindStyle {
  return KINDS[kind] ?? OTHER;
}

/** CSS colour for a tone, for SVG swatches (reads the same tokens as the canvas). */
export function toneVar(tone: Tone, spk = 0): string {
  switch (tone) {
    case "speaker":
      return `var(--spk-${spk})`;
    case "ink":
      return "var(--ink)";
    case "ink2":
      return "var(--ink-2)";
    case "ink3":
      return "var(--ink-3)";
    case "lineStrong":
      return "var(--line-strong)";
  }
}

/** Palette index of a Speaker node: color_idx, else its track ("S2" or "m1:S2"). */
export function nodeSpeakerIndex(node: GraphNode): number {
  if (typeof node.color_idx === "number" && Number.isFinite(node.color_idx)) return Math.abs(node.color_idx) % 8;
  if (typeof node.track === "string") return speakerIndex(node.track);
  if (typeof node.speaker === "string") return speakerIndex(node.speaker);
  return speakerIndex(String(node.id).split(":").pop());
}

export interface Palette {
  spk: string[];
  ink: string;
  ink2: string;
  ink3: string;
  line: string;
  lineStrong: string;
  surface: string;
  focus: string;
  font: string;
}

export function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    spk: Array.from({ length: 8 }, (_, i) => v(`--spk-${i}`, "gray")),
    ink: v("--ink", "black"),
    ink2: v("--ink-2", "dimgray"),
    ink3: v("--ink-3", "gray"),
    line: v("--line", "silver"),
    lineStrong: v("--line-strong", "darkgray"),
    surface: v("--surface", "white"),
    focus: v("--focus", "blue"),
    font: v("--font-ui", "sans-serif"),
  };
}

export function toneColor(p: Palette, tone: Tone, spk: number): string {
  switch (tone) {
    case "speaker":
      return p.spk[spk] ?? p.ink2;
    case "ink":
      return p.ink;
    case "ink2":
      return p.ink2;
    case "ink3":
      return p.ink3;
    case "lineStrong":
      return p.lineStrong;
  }
}

/** Trace the shape's outline at (x, y); the caller fills or strokes. */
export function traceShape(ctx: CanvasRenderingContext2D, shape: Shape, x: number, y: number, r: number): void {
  ctx.beginPath();
  switch (shape) {
    case "circle":
    case "ring":
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    case "square":
    case "hollowSquare":
      ctx.rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8);
      break;
    case "diamond":
      ctx.moveTo(x, y - r * 1.2);
      ctx.lineTo(x + r * 1.2, y);
      ctx.lineTo(x, y + r * 1.2);
      ctx.lineTo(x - r * 1.2, y);
      ctx.closePath();
      break;
    case "triangle":
      ctx.moveTo(x, y - r * 1.1);
      ctx.lineTo(x + r, y + r * 0.75);
      ctx.lineTo(x - r, y + r * 0.75);
      ctx.closePath();
      break;
  }
}

export function isHollow(shape: Shape): boolean {
  return shape === "ring" || shape === "hollowSquare" || shape === "triangle";
}

export function truncate(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}
