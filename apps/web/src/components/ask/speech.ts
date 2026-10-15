// Spoken answers via the browser's speechSynthesis. Cancel at any time for barge-in.

export function ttsSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined";
}

let voice: SpeechSynthesisVoice | null | undefined;
let listening = false;

function pickVoice(): SpeechSynthesisVoice | null {
  if (voice !== undefined) return voice;
  if (!listening) {
    listening = true;
    // Voices load asynchronously in Chrome; re-pick when the list arrives.
    window.speechSynthesis.addEventListener("voiceschanged", () => {
      voice = undefined;
    });
  }
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null; // not loaded yet: try again next time
  const byLang = (prefix: string) =>
    voices.find((v) => v.lang.replace("_", "-").toLowerCase().startsWith(prefix) && v.localService) ??
    voices.find((v) => v.lang.replace("_", "-").toLowerCase().startsWith(prefix));
  voice = byLang("en-in") ?? byLang("en-gb") ?? voices.find((v) => v.default) ?? null;
  return voice;
}

/** Strip citation markers, markup and URLs so the answer reads naturally aloud. */
export function speakableText(answer: string): string {
  return answer
    .replace(/\[[^\]]{0,80}\]/g, " ")
    .replace(/https?:\/\/\S+/g, " a link ")
    .replace(/[*_`#>|~]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Short chunks: some engines stop speaking long utterances after about 15 seconds. */
function chunks(text: string, max = 220): string[] {
  const sentences = text.match(/[^.!?;]+[.!?;]*\s*/g) ?? [text];
  const out: string[] = [];
  let cur = "";
  for (const s of sentences) {
    if ((cur + s).length > max && cur) {
      out.push(cur.trim());
      cur = "";
    }
    cur += s;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

let generation = 0;

export interface SpeakHandlers {
  onEnd: () => void;
  /** The audio has started (engines can take a moment after speak()). */
  onStart?: () => void;
}

/** Speak `text`, replacing anything already being spoken. onStart and onEnd fire once each, unless stopSpeaking() runs first. */
export function speak(text: string, handlers: SpeakHandlers): boolean {
  if (!ttsSupported()) return false;
  const synth = window.speechSynthesis;
  const parts = chunks(speakableText(text));
  if (!parts.length) return false;
  const gen = ++generation;
  synth.cancel();
  const v = pickVoice();
  let ended = false;
  const finish = () => {
    if (ended || gen !== generation) return;
    ended = true;
    handlers.onEnd();
  };
  parts.forEach((p, i) => {
    const u = new SpeechSynthesisUtterance(p);
    if (v) {
      u.voice = v;
      u.lang = v.lang;
    }
    u.rate = 1.03;
    if (i === 0) {
      u.onstart = () => {
        if (!ended && gen === generation) handlers.onStart?.();
      };
    }
    if (i === parts.length - 1) u.onend = finish;
    u.onerror = finish;
    synth.speak(u);
  });
  return true;
}

/** Stop speaking now (barge-in). The pending onEnd is dropped. */
export function stopSpeaking(): void {
  generation++;
  if (ttsSupported()) window.speechSynthesis.cancel();
}
