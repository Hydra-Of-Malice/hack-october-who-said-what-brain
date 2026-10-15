// Spoken-question capture for the Ask drawer.
// Mic at the device rate -> mono -> anti-alias low-pass -> 16 kHz -> WAV (PCM16), capped at 30 s.
// The AudioWorklet tap is inlined as a Blob module; ScriptProcessor is the fallback.

export const QUESTION_RATE = 16_000;
export const MAX_QUESTION_S = 30;

export type MicErrorCode = "insecure" | "unsupported" | "denied" | "not-found" | "busy" | "failed";

export class MicError extends Error {
  readonly code: MicErrorCode;
  constructor(code: MicErrorCode, message: string) {
    super(message);
    this.name = "MicError";
    this.code = code;
  }
}

const MIC_MESSAGES: Record<MicErrorCode, string> = {
  insecure: "Voice questions need a secure page (https or localhost). You can still type your question.",
  unsupported: "This browser can't record audio. You can still type your question.",
  denied: "Microphone access is blocked. Allow it in the browser's site settings, then try again.",
  "not-found": "No microphone found. Connect one, or type your question.",
  busy: "Another app is using the microphone. Close it, then try again.",
  failed: "The microphone didn't start.",
};

export function micError(code: MicErrorCode, detail?: string): MicError {
  return new MicError(code, detail ? `${MIC_MESSAGES[code]} (${detail})` : MIC_MESSAGES[code]);
}

function toMicError(e: unknown): MicError {
  if (e instanceof MicError) return e;
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return micError("denied");
  if (name === "NotFoundError" || name === "OverconstrainedError") return micError("not-found");
  if (name === "NotReadableError" || name === "AbortError") return micError("busy");
  return micError("failed", e instanceof Error ? e.message : String(e));
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/** RBJ biquad low-pass, transposed direct form II. */
class Biquad {
  private readonly b0: number;
  private readonly b1: number;
  private readonly b2: number;
  private readonly a1: number;
  private readonly a2: number;
  private z1 = 0;
  private z2 = 0;

  constructor(sampleRate: number, cutoff: number, q: number) {
    const w0 = (2 * Math.PI * cutoff) / sampleRate;
    const cos = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    this.b0 = (1 - cos) / 2 / a0;
    this.b1 = (1 - cos) / a0;
    this.b2 = (1 - cos) / 2 / a0;
    this.a1 = (-2 * cos) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  run(x: number): number {
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

/** Streaming resampler: 4th-order Butterworth low-pass (when downsampling), then linear interpolation. */
export class Resampler {
  private readonly ratio: number; // input samples per output sample
  private readonly filters: Biquad[];
  private pos = 0; // read position of the next output sample, relative to the current chunk (-1 = prev)
  private prev = 0;

  constructor(inRate: number, outRate: number = QUESTION_RATE) {
    if (!(inRate > 0 && outRate > 0)) throw new Error(`Invalid sample rates ${inRate} -> ${outRate}`);
    this.ratio = inRate / outRate;
    const cutoff = Math.min(outRate * 0.42, inRate * 0.45);
    this.filters = this.ratio > 1 ? [new Biquad(inRate, cutoff, 0.5412), new Biquad(inRate, cutoff, 1.3066)] : [];
  }

  process(input: Float32Array): Float32Array {
    const n = input.length;
    if (n === 0) return new Float32Array(0);
    if (this.ratio === 1) return input.slice();

    const x = this.filters.length ? new Float32Array(n) : input;
    if (this.filters.length) {
      for (let i = 0; i < n; i++) {
        let v = input[i];
        for (const f of this.filters) v = f.run(v);
        x[i] = v;
      }
    }

    const out = new Float32Array(Math.ceil((n + 1) / this.ratio) + 1);
    let m = 0;
    let t = this.pos;
    while (t < n - 1) {
      const i = Math.floor(t);
      const a = i < 0 ? this.prev : x[i];
      const b = x[i + 1];
      out[m++] = a + (b - a) * (t - i);
      t += this.ratio;
    }
    this.pos = t - n;
    this.prev = x[n - 1];
    return out.subarray(0, m);
  }
}

// ---------------------------------------------------------------------------
// Clip buffer and WAV encoding
// ---------------------------------------------------------------------------

/** Collects 16 kHz frames into one clip, with a short rolling pre-roll so VAD-started clips keep their first syllable. */
export class ClipBuffer {
  private readonly maxSamples: number;
  private readonly ring: Float32Array;
  private ringPos = 0;
  private ringFilled = 0;
  private chunks: Float32Array[] = [];
  private length = 0;
  capturing = false;

  constructor(maxSeconds: number = MAX_QUESTION_S, prerollSeconds = 0.6) {
    this.maxSamples = Math.round(maxSeconds * QUESTION_RATE);
    this.ring = new Float32Array(Math.max(1, Math.round(prerollSeconds * QUESTION_RATE)));
  }

  get samples(): number {
    return this.length;
  }

  get seconds(): number {
    return this.length / QUESTION_RATE;
  }

  /** Start capturing, seeded with up to `prerollMs` of audio heard just before. */
  begin(prerollMs = 0): void {
    this.chunks = [];
    this.length = 0;
    this.capturing = true;
    const want = Math.min(this.ringFilled, Math.round((prerollMs / 1000) * QUESTION_RATE));
    if (want > 0) {
      const pre = new Float32Array(want);
      const start = (this.ringPos - want + this.ring.length) % this.ring.length;
      for (let i = 0; i < want; i++) pre[i] = this.ring[(start + i) % this.ring.length];
      this.chunks.push(pre);
      this.length = want;
    }
  }

  /** Returns "full" when this frame filled the clip to its maximum length (capture stops itself). */
  push(frame: Float32Array): "ok" | "full" {
    for (let i = 0; i < frame.length; i++) {
      this.ring[this.ringPos] = frame[i];
      this.ringPos = (this.ringPos + 1) % this.ring.length;
    }
    this.ringFilled = Math.min(this.ring.length, this.ringFilled + frame.length);
    if (!this.capturing) return "ok";
    const room = this.maxSamples - this.length;
    const take = Math.min(room, frame.length);
    if (take > 0) {
      this.chunks.push(frame.slice(0, take));
      this.length += take;
    }
    if (this.length >= this.maxSamples) {
      this.capturing = false;
      return "full";
    }
    return "ok";
  }

  /** Stop capturing and return the clip. */
  end(): Float32Array {
    const out = new Float32Array(this.length);
    let o = 0;
    for (const c of this.chunks) {
      out.set(c, o);
      o += c.length;
    }
    this.chunks = [];
    this.length = 0;
    this.capturing = false;
    return out;
  }
}

/** Mono Float32 [-1, 1] -> 16-bit PCM WAV bytes. */
export function wavBytes(samples: Float32Array, sampleRate: number = QUESTION_RATE): ArrayBuffer {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const ascii = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  v.setUint32(40, dataBytes, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

export function encodeWav(samples: Float32Array, sampleRate: number = QUESTION_RATE): Blob {
  return new Blob([wavBytes(samples, sampleRate)], { type: "audio/wav" });
}

/** dBFS -> 0..1 for a level meter (-55 dB is empty, -10 dB is full). */
export function meterLevel(block: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
  const db = 20 * Math.log10(Math.sqrt(sum / Math.max(1, block.length)) + 1e-10);
  return Math.min(1, Math.max(0, (db + 55) / 45));
}

// ---------------------------------------------------------------------------
// Microphone tap
// ---------------------------------------------------------------------------

const TAP_NAME = "wsw-question-tap";
const TAP_SOURCE = `
class WswQuestionTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1024); this.n = 0; }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const chs = input.length, len = input[0].length;
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let c = 0; c < chs; c++) s += input[c][i];
        this.buf[this.n++] = s / chs;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(1024);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("${TAP_NAME}", WswQuestionTap);
`;

let tapUrl: string | null = null;
function tapModuleUrl(): string {
  tapUrl ??= URL.createObjectURL(new Blob([TAP_SOURCE], { type: "text/javascript" }));
  return tapUrl;
}

export interface MicHandlers {
  /** 16 kHz mono frames, roughly every 20-60 ms. */
  onFrame: (frame: Float32Array) => void;
}

/** An open microphone delivering 16 kHz mono frames. Close it to release the device (and the browser's mic indicator). */
export class QuestionMic {
  readonly deviceRate: number;
  private readonly ctx: AudioContext;
  private readonly stream: MediaStream;
  private readonly nodes: AudioNode[];
  private closed = false;

  private constructor(ctx: AudioContext, stream: MediaStream, nodes: AudioNode[]) {
    this.ctx = ctx;
    this.stream = stream;
    this.nodes = nodes;
    this.deviceRate = ctx.sampleRate;
  }

  static async open(handlers: MicHandlers): Promise<QuestionMic> {
    if (typeof window === "undefined" || !window.isSecureContext) throw micError("insecure");
    if (!navigator.mediaDevices?.getUserMedia) throw micError("unsupported");
    const AudioCtx: typeof AudioContext | undefined =
      window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) throw micError("unsupported");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e) {
      throw toMicError(e);
    }

    const ctx = new AudioCtx();
    try {
      if (ctx.state === "suspended") await ctx.resume();
      const resampler = new Resampler(ctx.sampleRate, QUESTION_RATE);
      const deliver = (block: Float32Array) => {
        const frame = resampler.process(block);
        if (frame.length) handlers.onFrame(frame);
      };
      const source = ctx.createMediaStreamSource(stream);
      const sink = ctx.createGain();
      sink.gain.value = 0; // keeps the graph pulled without playing the mic back
      sink.connect(ctx.destination);

      const tap = await createTap(ctx, deliver);
      source.connect(tap);
      tap.connect(sink);
      return new QuestionMic(ctx, stream, [source, tap, sink]);
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close().catch(() => undefined);
      throw toMicError(e);
    }
  }

  get isOpen(): boolean {
    return !this.closed;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const n of this.nodes) {
      try {
        n.disconnect();
      } catch {
        /* already disconnected */
      }
      if (typeof AudioWorkletNode !== "undefined" && n instanceof AudioWorkletNode) n.port.onmessage = null;
      if (typeof ScriptProcessorNode !== "undefined" && n instanceof ScriptProcessorNode) n.onaudioprocess = null;
    }
    this.stream.getTracks().forEach((t) => t.stop());
    void this.ctx.close().catch(() => undefined);
  }
}

async function createTap(ctx: AudioContext, deliver: (block: Float32Array) => void): Promise<AudioNode> {
  if (ctx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
    try {
      await ctx.audioWorklet.addModule(tapModuleUrl());
      const node = new AudioWorkletNode(ctx, TAP_NAME, { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
      node.port.onmessage = (m: MessageEvent<Float32Array>) => deliver(m.data);
      return node;
    } catch {
      // Blob modules can be blocked by a strict CSP; fall through to ScriptProcessor.
    }
  }
  const proc = ctx.createScriptProcessor(2048, 1, 1);
  proc.onaudioprocess = (ev) => {
    const ib = ev.inputBuffer;
    const n = ib.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < ib.numberOfChannels; c++) {
      const ch = ib.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += ch[i] / ib.numberOfChannels;
    }
    deliver(mono);
  };
  return proc;
}

// ---------------------------------------------------------------------------
// Recorder: mic + clip + level, the object the Ask drawer drives
// ---------------------------------------------------------------------------

export interface QuestionRecorderEvents {
  /** 0..1 input level, for the talkback ring. Fires for every frame while the mic is open. */
  onLevel?: (level: number) => void;
  /** Every 16 kHz frame after it is buffered (feed a VAD from here). */
  onFrame?: (frame: Float32Array) => void;
  /** The clip reached MAX_QUESTION_S and stopped itself; this is the clip. */
  onLimit?: (clip: Float32Array) => void;
}

export class QuestionRecorder {
  private mic: QuestionMic | null = null;
  private opening: Promise<void> | null = null;
  private generation = 0; // bumped by close(), so a mic that finishes opening after close() is released at once
  private readonly clip = new ClipBuffer(MAX_QUESTION_S);
  private readonly events: QuestionRecorderEvents;

  constructor(events: QuestionRecorderEvents) {
    this.events = events;
  }

  get isOpen(): boolean {
    return !!this.mic?.isOpen;
  }

  get capturing(): boolean {
    return this.clip.capturing;
  }

  get capturedSeconds(): number {
    return this.clip.seconds;
  }

  /** Acquire the microphone (prompts for permission the first time). Throws MicError.
   * Resolves without opening if close() was called meanwhile; check isOpen afterwards. */
  open(): Promise<void> {
    if (this.mic?.isOpen) return Promise.resolve();
    const gen = this.generation;
    this.opening ??= QuestionMic.open({ onFrame: (f) => this.onFrame(f) })
      .then((mic) => {
        if (gen !== this.generation) mic.close();
        else this.mic = mic;
      })
      .finally(() => {
        this.opening = null;
      });
    return this.opening;
  }

  begin(prerollMs = 0): void {
    this.clip.begin(prerollMs);
  }

  /** Stop capturing; returns the clip (possibly empty). The mic stays open. */
  end(): Float32Array {
    return this.clip.end();
  }

  close(): void {
    this.generation++;
    if (this.clip.capturing) this.clip.end();
    this.mic?.close();
    this.mic = null;
    this.events.onLevel?.(0);
  }

  private onFrame(frame: Float32Array): void {
    const wasCapturing = this.clip.capturing;
    const state = this.clip.push(frame);
    this.events.onLevel?.(meterLevel(frame));
    if (wasCapturing && state === "full") {
      this.events.onLimit?.(this.clip.end());
      return;
    }
    this.events.onFrame?.(frame);
  }
}
