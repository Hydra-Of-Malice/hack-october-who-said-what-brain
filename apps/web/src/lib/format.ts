/** 192.4 -> "03:12"; >= 1h -> "1:03:12" */
export function timecode(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** 412 -> "6 min 52 s"; 45 -> "45 s"; 3720 -> "1 h 2 min" */
export function duration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h} h ${m} min`;
  return `${m} min${s % 60 ? ` ${s % 60} s` : ""}`;
}

const dateFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
export function when(ms: number | null | undefined): string {
  return ms ? dateFmt.format(new Date(ms)) : "";
}

/** Speaker display name: the given name, else "Speaker 2". */
export function speakerName(s: { name?: string | null; speaker_name?: string | null; label?: string; track?: string; speaker?: string }): string {
  return s.name || s.speaker_name || s.label || (s.track || s.speaker ? `Speaker ${(s.track || s.speaker || "").replace(/^S/, "")}` : "Unknown");
}

/** "S3" -> 2 (palette index), stable per meeting. */
export function speakerIndex(track: string | null | undefined): number {
  const n = parseInt(String(track ?? "").replace(/\D/g, ""), 10);
  return Number.isFinite(n) && n > 0 ? (n - 1) % 8 : 7;
}

export function newSessionId(): string {
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
