// Talkback behaviour for the ask drawer:
// - press and hold (mouse, touch or Space), release to send;
// - a quick click starts listening, the next press sends;
// - hands-free: the mic stays open and an energy VAD sends each question when the speaker pauses.
import { useEffect, useRef, useState, type RefObject } from "react";
import { MicError, QUESTION_RATE, QuestionRecorder, encodeWav } from "../../audio/question";
import { EnergyVad } from "../../audio/vad";

export type MicState = "off" | "opening" | "armed" | "recording";
export type CaptureKind = "hold" | "toggle" | "vad";
export type SendReason = "released" | "clicked" | "paused" | "limit";

export interface VoiceQuestionOptions {
  onQuestion: (wav: Blob, seconds: number, reason: SendReason) => void;
  /** The user pressed talkback, or started talking in hands-free mode. */
  onBargeIn: (source: "press" | "voice") => void;
  onLevel: (level: number) => void;
  onNotice: (message: string) => void;
  /** True while an answer is being spoken (raises the VAD threshold so the speakers don't trigger it). */
  speakingRef: RefObject<boolean>;
  /** True once that answer's audio has actually started (the VAD then learns how loud it is at the mic). */
  audibleRef: RefObject<boolean>;
}

export interface VoiceSnapshot {
  mic: MicState;
  capture: CaptureKind | null;
  captureStartedAt: number | null;
  handsFree: boolean;
  error: string | null;
}

const INITIAL: VoiceSnapshot = { mic: "off", capture: null, captureStartedAt: null, handsFree: false, error: null };
const HOLD_MS = 350; // a press shorter than this is a click (toggle)
const MIN_CLIP_S = 0.35;
const VAD_PREROLL_MS = 450;
const SPEAKING_BOOST_DB = 14;

class VoiceController {
  snapshot: VoiceSnapshot = INITIAL;
  private readonly options: () => VoiceQuestionOptions;
  private readonly emit: (s: VoiceSnapshot) => void;
  private readonly rec: QuestionRecorder;
  private vad: EnergyVad | null = null;
  private press: { captureAt: number | null; released: boolean } | null = null;

  constructor(options: () => VoiceQuestionOptions, emit: (s: VoiceSnapshot) => void) {
    this.options = options;
    this.emit = emit;
    this.rec = new QuestionRecorder({
      onLevel: (l) => this.options().onLevel(l),
      onFrame: (f) => this.onFrame(f),
      onLimit: (clip) => this.onLimit(clip),
    });
  }

  private set(patch: Partial<VoiceSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    this.emit(this.snapshot);
  }

  clearError(): void {
    if (this.snapshot.error) this.set({ error: null });
  }

  pressStart(): void {
    this.options().onBargeIn("press");
    if (this.snapshot.capture) {
      // Already listening (after a click, or hands-free): this press sends.
      this.press = null;
      this.finish("clicked");
      return;
    }
    if (this.press) return;
    const p = { captureAt: null as number | null, released: false };
    this.press = p;
    void this.ensureOpen().then((ok) => {
      if (this.press !== p) {
        // Cancelled while the mic was starting.
        if (ok && !this.snapshot.handsFree && !this.rec.capturing) this.releaseMic();
        return;
      }
      if (!ok) {
        this.press = null;
        return;
      }
      if (p.released) {
        // Released before the mic was ready (a click, or a permission prompt): keep listening.
        this.press = null;
        this.startCapture("toggle");
      } else {
        p.captureAt = performance.now();
        this.startCapture("hold");
      }
    });
  }

  pressEnd(): void {
    const p = this.press;
    if (!p) return;
    if (p.captureAt === null) {
      p.released = true;
      return;
    }
    this.press = null;
    if (performance.now() - p.captureAt >= HOLD_MS) this.finish("released");
    else this.set({ capture: "toggle" });
  }

  /** Throw away the current recording without sending it. */
  cancel(): void {
    this.press = null;
    if (this.rec.capturing) this.rec.end();
    this.afterCapture();
  }

  async setHandsFree(on: boolean): Promise<void> {
    if (on === this.snapshot.handsFree) return;
    if (!on) {
      this.vad = null;
      this.set({ handsFree: false });
      if (this.snapshot.capture === "vad") this.cancel();
      else if (!this.snapshot.capture && !this.press) this.releaseMic();
      return;
    }
    this.set({ handsFree: true });
    const ok = await this.ensureOpen();
    if (!ok || !this.snapshot.handsFree) {
      this.set({ handsFree: false });
      return;
    }
    this.vad = new EnergyVad(
      { sampleRate: QUESTION_RATE },
      {
        onSpeechStart: () => this.onVadStart(),
        onSpeechEnd: () => {
          if (this.snapshot.capture === "vad") this.finish("paused");
        },
        onMisfire: () => {
          if (this.snapshot.capture === "vad") this.cancel();
        },
      },
    );
    if (!this.snapshot.capture) this.set({ mic: "armed" });
  }

  /** Stop everything and release the microphone. */
  shutdown(): void {
    this.press = null;
    this.vad = null;
    this.rec.close();
    this.set({ mic: "off", capture: null, captureStartedAt: null, handsFree: false });
  }

  private async ensureOpen(): Promise<boolean> {
    if (this.rec.isOpen) return true;
    this.set({ mic: "opening", error: null });
    try {
      await this.rec.open();
    } catch (e) {
      const message = e instanceof MicError ? e.message : `The microphone didn't start. (${String(e)})`;
      this.set({ mic: "off", error: message });
      return false;
    }
    return this.rec.isOpen;
  }

  private startCapture(kind: CaptureKind): void {
    this.rec.begin(kind === "vad" ? VAD_PREROLL_MS : 0);
    this.set({ mic: "recording", capture: kind, captureStartedAt: performance.now() });
  }

  private finish(reason: SendReason): void {
    const clip = this.rec.capturing ? this.rec.end() : new Float32Array(0);
    this.afterCapture();
    this.deliver(clip, reason);
  }

  private afterCapture(): void {
    if (this.snapshot.handsFree && this.rec.isOpen) {
      this.vad?.reset();
      this.set({ mic: "armed", capture: null, captureStartedAt: null });
    } else {
      this.releaseMic();
    }
  }

  private releaseMic(): void {
    this.rec.close();
    this.set({ mic: "off", capture: null, captureStartedAt: null });
  }

  private deliver(clip: Float32Array, reason: SendReason): void {
    const seconds = clip.length / QUESTION_RATE;
    if (seconds < MIN_CLIP_S) {
      if (reason !== "paused") this.options().onNotice("That was too short to hear. Hold the button while you speak.");
      return;
    }
    this.options().onQuestion(encodeWav(clip), seconds, reason);
  }

  private onFrame(frame: Float32Array): void {
    const vad = this.vad;
    if (!vad || !this.snapshot.handsFree) return;
    const { speakingRef, audibleRef } = this.options();
    const speaking = !!speakingRef.current;
    vad.boostDb = speaking ? SPEAKING_BOOST_DB : 0;
    // Spoken answers bypass the browser's echo canceller; the VAD learns their level so they can't barge in on themselves.
    vad.setEcho(!speaking ? "off" : audibleRef.current ? "playing" : "pending");
    vad.process(frame);
  }

  private onVadStart(): void {
    if (this.snapshot.capture || this.press) return; // a manual capture owns the mic
    this.options().onBargeIn("voice");
    this.startCapture("vad");
  }

  private onLimit(clip: Float32Array): void {
    this.press = null;
    this.options().onNotice("Stopped at 30 seconds, the longest question Gemma can hear. Sending what you said.");
    this.afterCapture();
    this.deliver(clip, "limit");
  }
}

export interface VoiceQuestion extends VoiceSnapshot {
  pressStart: () => void;
  pressEnd: () => void;
  cancel: () => void;
  setHandsFree: (on: boolean) => void;
  clearError: () => void;
  shutdown: () => void;
}

export function useVoiceQuestion(options: VoiceQuestionOptions): VoiceQuestion {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const [snapshot, setSnapshot] = useState<VoiceSnapshot>(INITIAL);
  const [controller] = useState(() => new VoiceController(() => optionsRef.current, setSnapshot));

  useEffect(() => () => controller.shutdown(), [controller]);

  return {
    ...snapshot,
    pressStart: () => controller.pressStart(),
    pressEnd: () => controller.pressEnd(),
    cancel: () => controller.cancel(),
    setHandsFree: (on: boolean) => void controller.setHandsFree(on),
    clearError: () => controller.clearError(),
    shutdown: () => controller.shutdown(),
  };
}
