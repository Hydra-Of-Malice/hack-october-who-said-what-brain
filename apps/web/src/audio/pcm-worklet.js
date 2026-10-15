// AudioWorklet: mono input at the context rate -> 16 kHz signed 16-bit PCM frames of 100 ms.
// Runs in AudioWorkletGlobalScope (`sampleRate`, `registerProcessor`, `AudioWorkletProcessor` are globals there).

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 1600; // 100 ms at 16 kHz = 3200 bytes
const CUTOFF_HZ = 7200;

/** Symmetric FIR low-pass (Blackman-windowed sinc), unity gain at DC. fc is cycles per input sample. */
function lowpassTaps(fc, n) {
  const taps = new Float32Array(n);
  const mid = (n - 1) / 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - mid;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1));
    taps[i] = sinc * w;
    sum += taps[i];
  }
  for (let i = 0; i < n; i++) taps[i] /= sum;
  return taps;
}

class Pcm16kDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    // Windowed-sinc low-pass below the new Nyquist (8 kHz), so higher content does not alias into speech.
    this.taps = this.ratio > 1 ? lowpassTaps(CUTOFF_HZ / sampleRate, 2 * Math.ceil(this.ratio * 8) + 1) : new Float32Array([1]);
    this.hist = new Float32Array(this.taps.length * 2); // doubled ring: a contiguous window without modulo
    this.histPos = 0;
    // Linear interpolation between filtered samples at fractional output positions.
    this.prev = 0;
    this.t = 1;
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.fill = 0;
    this.sumSq = 0;
    this.running = true;
    this.port.onmessage = (e) => {
      if (e.data === "stop") this.running = false;
    };
  }

  process(inputs) {
    const channels = inputs[0];
    if (!this.running) return false;
    if (!channels || channels.length === 0) return true;
    const mono = channels[0];
    for (let i = 0; i < mono.length; i++) {
      const y = this.lowpass(mono[i]);
      while (this.t <= 1) {
        this.emit(this.prev + (y - this.prev) * this.t);
        this.t += this.ratio;
      }
      this.t -= 1;
      this.prev = y;
    }
    return true;
  }

  lowpass(x) {
    const taps = this.taps;
    const n = taps.length;
    const h = this.hist;
    h[this.histPos] = x;
    h[this.histPos + n] = x;
    this.histPos = (this.histPos + 1) % n;
    // h[histPos .. histPos+n-1] holds the last n samples, oldest first.
    let acc = 0;
    for (let k = 0, j = this.histPos; k < n; k++, j++) acc += taps[k] * h[j];
    return acc;
  }

  emit(sample) {
    const s = sample > 1 ? 1 : sample < -1 ? -1 : sample;
    this.sumSq += s * s;
    this.frame[this.fill++] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
    if (this.fill === FRAME_SAMPLES) {
      const rms = Math.sqrt(this.sumSq / FRAME_SAMPLES);
      const pcm = this.frame.buffer;
      this.port.postMessage({ pcm, rms }, [pcm]);
      this.frame = new Int16Array(FRAME_SAMPLES);
      this.fill = 0;
      this.sumSq = 0;
    }
  }
}

registerProcessor("wsw-pcm-16k", Pcm16kDownsampler);
