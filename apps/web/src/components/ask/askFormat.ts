// Wording for the ask drawer: provenance labels, retrieval paths, timings, answer blocks, errors.
import { ApiError } from "../../api/client";
import type { ClientAction, Hit } from "../../api/types";
import { timecode } from "../../lib/format";

export const NOT_CAUGHT = "I didn't catch that — try again closer to the mic.";

export function heardByLabel(provider: string | null | undefined): string | null {
  if (!provider) return null;
  return provider.startsWith("gemma4-audio") ? "Heard by Gemma 4 (audio)" : "Transcribed by Whisper";
}

const PROVIDERS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "the Gemini API",
  openai: "an OpenAI-compatible server",
  fake: "the offline test model",
};

export function answeredBy(model: string, provider: string): string {
  const via = PROVIDERS[provider.toLowerCase()] ?? provider;
  return model ? `Answered by ${model} via ${via}` : `Answered via ${via}`;
}

const VIA: Record<Hit["via"], string> = {
  fulltext: "Full-text match",
  vector: "Meaning match (EmbeddingGemma)",
  graph: "Graph lookup",
  recent: "Recent in the meeting",
};

export function viaLabel(via: string): string {
  return VIA[via as Hit["via"]] ?? via;
}

const TIMINGS: Record<string, string> = {
  transcribe: "Hearing the question",
  hear: "Hearing the question",
  retrieve: "Searching the graph",
  embed: "Embedding the question",
  llm: "Gemma",
  agent: "Gemma",
  tools: "Tool calls",
  total: "Total",
};

export function timingLabel(key: string): string {
  return TIMINGS[key] ?? key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function ms(value: number): string {
  if (!Number.isFinite(value)) return "";
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

export function formatArgs(args: Record<string, unknown>): string {
  const parts = Object.entries(args ?? {}).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  const s = parts.join(", ");
  return s.length > 160 ? `${s.slice(0, 159)}…` : s;
}

export function describeAction(a: ClientAction, titles: Map<string, string>): string | null {
  switch (a.type) {
    case "seek":
      return `Jumped to ${titles.get(a.meeting_id) ?? "the meeting"} at ${timecode(a.t)}`;
    case "open_meeting":
      return `Opened ${titles.get(a.meeting_id) ?? "the meeting"}`;
    case "refresh":
      return a.what === "action_items"
        ? "Updated the commitments"
        : a.what === "decisions"
          ? "Updated the decisions"
          : "Updated the meeting";
    case "draft":
      return null;
  }
}

export type AnswerBlock = { kind: "p"; text: string } | { kind: "ul" | "ol"; items: string[] };

/** Light formatting for model answers: paragraphs and simple bullet or numbered lists. */
export function answerBlocks(answer: string): AnswerBlock[] {
  const clean = answer.replace(/\*\*(.+?)\*\*/g, "$1").replace(/__(.+?)__/g, "$1");
  const blocks: AnswerBlock[] = [];
  for (const chunk of clean.split(/\n\s*\n/)) {
    const lines = chunk.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    const bullets = lines.every((l) => /^[-*•]\s+/.test(l));
    const numbered = lines.every((l) => /^\d+[.)]\s+/.test(l));
    if (bullets && lines.length > 0) blocks.push({ kind: "ul", items: lines.map((l) => l.replace(/^[-*•]\s+/, "")) });
    else if (numbered) blocks.push({ kind: "ol", items: lines.map((l) => l.replace(/^\d+[.)]\s+/, "")) });
    else blocks.push({ kind: "p", text: lines.join(" ") });
  }
  return blocks;
}

export function askErrorText(e: unknown, mode: "voice" | "text"): string {
  if (e instanceof ApiError) {
    if (e.status === 0) return "Can't reach the server. Check that the API is running, then try again.";
    if (mode === "voice" && /empty|no speech|silence|didn.t hear|could not (transcribe|hear)/i.test(e.message)) return NOT_CAUGHT;
    if (e.status === 413) return "That recording is too long to send. Keep questions under 30 seconds.";
    if (e.status >= 500) return `The server hit an error while answering: ${e.message}`;
    return e.message;
  }
  return e instanceof Error ? e.message : String(e);
}
