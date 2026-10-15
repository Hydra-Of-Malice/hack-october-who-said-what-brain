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

/** An answer being spoken aloud: "pending" is queued but not heard yet, "playing" has started. */
export type EchoState = "off" | "pending" | "playing";

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

// Spoken answers play through the OS speech engine, so the browser's echo canceller never removes them.
// While one plays, learn how loud it is at the mic and only accept speech that stands clear of it.
const ECHO_LEARN_MS = 400; // listen to the answer this long after it starts before anything can interrupt it
const ECHO_PENDING_MAX_MS = 2000; // the engine never said it started: assume it has by now
const ECHO_MARGIN_DB = 6; // interrupting speech must be this much louder than the answer's peaks
const ECHO_START_MS = 200; // and last this long, while the answer is audible
const ECHO_PERCENTILE = 0.9;

interface EchoGuard {
  phase: "pending" | "learning" | "guarding";
  ms: number;
  levels: number[];
  /** The answer's level at the mic (a high percentile, so its peaks don't count as speech). */
  db: number;
}

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
  private echoState: EchoState = "off";
  private echo: EchoGuard | null = null;
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
    const base = this.baseThresholdDb;
    return this.echo?.phase === "guarding" ? Math.max(base, this.echo.db + ECHO_MARGIN_DB) : base;
  }

  private get baseThresholdDb(): number {
    return Math.max(this.floorDb + this.opts.marginDb, this.opts.absoluteMinDb) + this.boostDb;
  }

  /** Tell the VAD whether an answer is being spoken aloud (call as often as you like; changes restart the guard). */
  setEcho(state: EchoState): void {
    if (state === this.echoState) return;
    this.echoState = state;
    this.echo = state === "off" ? null : { phase: state === "pending" ? "pending" : "learning", ms: 0, levels: [], db: this.floorDb };
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
    const echo = this.echo;
    if (echo) this.stepEcho(echo, db);
    if (!this.inSpeech && echo && echo.phase !== "guarding") {
      // The answer is about to play or has only just started: nothing interrupts it until we know how loud it is here.
      this.aboveMs = 0;
      return;
    }
    const threshold = this.thresholdDb;

    if (!this.inSpeech) {
      const audibleEcho = !!echo && echo.db + ECHO_MARGIN_DB > this.baseThresholdDb;
      if (db > threshold) {
        this.aboveMs += frameMs;
        if (this.aboveMs >= (audibleEcho ? Math.max(startMs, ECHO_START_MS) : startMs)) {
          this.inSpeech = true;
          this.speechMs = this.aboveMs;
          this.belowMs = 0;
          this.handlers.onSpeechStart?.();
        }
      } else {
        this.aboveMs = 0;
        // Track the answer's level while it plays; keep the room's floor for when it stops.
        if (echo) echo.db += (db - echo.db) * (db > echo.db ? 0.1 : 0.002);
        else this.adaptFloor(db, false);
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

  private stepEcho(echo: EchoGuard, db: number): void {
    echo.ms += this.opts.frameMs;
    if (echo.phase === "pending" && echo.ms >= ECHO_PENDING_MAX_MS) {
      echo.phase = "learning";
      echo.ms = this.opts.frameMs;
    }
    if (echo.phase !== "learning") return;
    if (echo.ms <= ECHO_LEARN_MS) {
      echo.levels.push(db);
      return;
    }
    const sorted = echo.levels.sort((a, b) => a - b);
    if (sorted.length) echo.db = sorted[Math.floor(ECHO_PERCENTILE * (sorted.length - 1))];
    echo.levels = [];
    echo.phase = "guarding";
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
