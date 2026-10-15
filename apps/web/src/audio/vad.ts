// Energy voice-activity detection with an adaptive noise floor. Pure TypeScript: no wasm, no model.
// Feed it mono Float32 audio in any chunk size; it analyses fixed frames and reports speech start/end.

export interface VadOptions {
  sampleRate: number;
  /** Analysis frame length. */
  frameMs?: number;
  /** Time above threshold before speech is declared. */
  startMs?: number;
  /** Time below threshold before speech is declared over. */
  endMs?: number;
  /** Speech shorter than this is reported as a misfire, not a question. */
  minSpeechMs?: number;
  /** How far above the noise floor a frame must be to count as speech. */
  marginDb?: number;
  /** Frames quieter than this never count as speech, however quiet the room. */
  absoluteMinDb?: number;
  /** Speech is force-ended after this long (the noise floor may have jumped). */
  maxSpeechMs?: number;
}

export interface VadHandlers {
  onSpeechStart?: () => void;
  /** voicedMs excludes the trailing silence that ended the speech. */
  onSpeechEnd?: (voicedMs: number) => void;
  onMisfire?: (voicedMs: number) => void;
}

const DEFAULTS = {
  frameMs: 20,
  startMs: 150,
  endMs: 800,
  minSpeechMs: 400,
  marginDb: 9,
  absoluteMinDb: -58,
  maxSpeechMs: 30_000,
};

const HYSTERESIS_DB = 3;
const CALIBRATION_FRAMES = 12;
const FLOOR_MIN_DB = -90;
const FLOOR_MAX_DB = -20;

/** RMS level of a block in dBFS (silence is about -100). */
export function levelDb(block: Float32Array, from = 0, to = block.length): number {
  let sum = 0;
  for (let i = from; i < to; i++) sum += block[i] * block[i];
  const n = Math.max(1, to - from);
  return 20 * Math.log10(Math.sqrt(sum / n) + 1e-10);
}

export class EnergyVad {
  private readonly opts: Required<Omit<VadOptions, "sampleRate">>;
  private readonly handlers: VadHandlers;
  private readonly frame: Float32Array;
  private fill = 0;
  private calibrated = 0;
  private floorDb = -60;
  private inSpeech = false;
  private aboveMs = 0;
  private belowMs = 0;
  private speechMs = 0;
  /** Extra dB added to the threshold, e.g. while the answer is being spoken aloud (self-trigger guard). */
  boostDb = 0;

  constructor(options: VadOptions, handlers: VadHandlers) {
    const { sampleRate, ...rest } = options;
    this.opts = { ...DEFAULTS, ...stripUndefined(rest) };
    this.handlers = handlers;
    this.frame = new Float32Array(Math.max(1, Math.round((sampleRate * this.opts.frameMs) / 1000)));
  }

  get speaking(): boolean {
    return this.inSpeech;
  }

  get noiseFloorDb(): number {
    return this.floorDb;
  }

  /** Current speech threshold in dBFS. */
  get thresholdDb(): number {
    return Math.max(this.floorDb + this.opts.marginDb, this.opts.absoluteMinDb) + this.boostDb;
  }

  process(samples: Float32Array): void {
    let i = 0;
    while (i < samples.length) {
      const take = Math.min(this.frame.length - this.fill, samples.length - i);
      this.frame.set(samples.subarray(i, i + take), this.fill);
      this.fill += take;
      i += take;
      if (this.fill === this.frame.length) {
        this.step(levelDb(this.frame));
        this.fill = 0;
      }
    }
  }

  /** Forget the current speech state (keeps the learned noise floor). */
  reset(): void {
    this.inSpeech = false;
    this.aboveMs = 0;
    this.belowMs = 0;
    this.speechMs = 0;
    this.fill = 0;
  }

  private step(db: number): void {
    const { frameMs, startMs, endMs, minSpeechMs, maxSpeechMs } = this.opts;
    if (this.calibrated < CALIBRATION_FRAMES) {
      // Learn the room before listening: average the first frames into the floor.
      this.floorDb = this.calibrated === 0 ? db : this.floorDb + (db - this.floorDb) / (this.calibrated + 1);
      this.floorDb = clamp(this.floorDb, FLOOR_MIN_DB, FLOOR_MAX_DB);
      this.calibrated++;
      return;
    }
    const threshold = this.thresholdDb;

    if (!this.inSpeech) {
      if (db > threshold) {
        this.aboveMs += frameMs;
        if (this.aboveMs >= startMs) {
          this.inSpeech = true;
          this.speechMs = this.aboveMs;
          this.belowMs = 0;
          this.handlers.onSpeechStart?.();
        }
      } else {
        this.aboveMs = 0;
        this.adaptFloor(db, false);
      }
      return;
    }

    this.speechMs += frameMs;
    // A slow upward drift lets a sudden steady noise (a fan, a projector) eventually end "speech".
    this.adaptFloor(db, true);
    if (db < threshold - HYSTERESIS_DB) {
      this.belowMs += frameMs;
    } else {
      this.belowMs = 0;
    }
    if (this.belowMs >= endMs || this.speechMs >= maxSpeechMs) {
      const voiced = Math.max(0, this.speechMs - this.belowMs);
      this.inSpeech = false;
      this.aboveMs = 0;
      this.belowMs = 0;
      this.speechMs = 0;
      if (voiced >= minSpeechMs) this.handlers.onSpeechEnd?.(voiced);
      else this.handlers.onMisfire?.(voiced);
    }
  }

  private adaptFloor(db: number, duringSpeech: boolean): void {
    // Fall fast toward quieter frames, rise slowly toward louder ones.
    const rate = db < this.floorDb ? 0.25 : duringSpeech ? 0.004 : 0.02;
    this.floorDb = clamp(this.floorDb + (db - this.floorDb) * rate, FLOOR_MIN_DB, FLOOR_MAX_DB);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function stripUndefined<T extends object>(o: T): Partial<T> {
  const out: Partial<T> = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) (out as Record<string, unknown>)[k] = v;
  return out;
}
