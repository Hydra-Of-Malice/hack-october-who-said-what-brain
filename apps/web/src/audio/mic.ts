// Live meeting capture: microphone -> AudioWorklet (16 kHz Int16 PCM, 100 ms frames) -> WebSocket.
// Utterances come back over SSE, not this socket; the socket only carries audio up and status down.
import { audioSocketUrl } from "../api/client";
import workletUrl from "./pcm-worklet.js?url&no-inline";

export type CaptureErrorKind = "insecure" | "unsupported" | "denied" | "no-device" | "device-busy" | "socket" | "server" | "audio";

export class CaptureError extends Error {
  kind: CaptureErrorKind;
  constructor(kind: CaptureErrorKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

export interface CaptureStatus {
  state: "connecting" | "listening" | "processing" | "stopped";
  lagS: number | null;
}

export interface CaptureCallbacks {
  /** RMS of each 100 ms frame, 0..1. */
  onLevel?: (rms: number) => void;
  /** Fatal problems after start. The capture has already torn itself down when this fires. */
  onError?: (err: CaptureError) => void;
  onStatus?: (status: CaptureStatus) => void;
}

export interface CaptureHandle {
  /** Stops the mic and closes the stream. endMeeting=false keeps the meeting live on the server. */
  stop(endMeeting: boolean): Promise<void>;
}

const SOCKET_OPEN_TIMEOUT_MS = 8000;
const MAX_BUFFERED_BYTES = 4 * 1024 * 1024; // ~2 min of audio; beyond this the link is effectively down
const DRAIN_WAIT_MS = 1500;

function micError(err: unknown): CaptureError {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return new CaptureError("denied", "Microphone access is blocked. Allow it in the browser's site settings, then start capturing again.");
    case "NotFoundError":
    case "OverconstrainedError":
      return new CaptureError("no-device", "No microphone was found. Connect one, then start capturing again.");
    case "NotReadableError":
    case "AbortError":
      return new CaptureError("device-busy", "The microphone is in use by another app or couldn't start. Close the other app, then try again.");
    default:
      return new CaptureError("audio", `The microphone couldn't start: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function openSocket(meetingId: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    let ws: WebSocket;
    try {
      ws = new WebSocket(audioSocketUrl(meetingId));
    } catch (e) {
      reject(new CaptureError("socket", `Couldn't open the audio stream: ${e instanceof Error ? e.message : String(e)}`));
      return;
    }
    ws.binaryType = "arraybuffer";
    const timer = window.setTimeout(() => {
      ws.close();
      reject(new CaptureError("socket", "The server didn't accept the audio stream in time. Check that the API is running, then try again."));
    }, SOCKET_OPEN_TIMEOUT_MS);
    ws.onopen = () => {
      window.clearTimeout(timer);
      resolve(ws);
    };
    ws.onerror = () => {
      window.clearTimeout(timer);
      reject(new CaptureError("socket", "Couldn't connect the audio stream to the server. Check that the API is running, then try again."));
    };
  });
}

function parseServerFrame(raw: unknown): { type: string; state?: string; lag_s?: number; detail?: string } | null {
  if (typeof raw !== "string") return null;
  try {
    const msg = JSON.parse(raw);
    return msg && typeof msg.type === "string" ? msg : null;
  } catch {
    return null;
  }
}

async function waitForDrain(ws: WebSocket): Promise<void> {
  const until = performance.now() + DRAIN_WAIT_MS;
  while (ws.readyState === WebSocket.OPEN && ws.bufferedAmount > 0 && performance.now() < until) {
    await new Promise((r) => window.setTimeout(r, 50));
  }
}

export async function startCapture(meetingId: string, cb: CaptureCallbacks = {}): Promise<CaptureHandle> {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new CaptureError("insecure", "Microphone needs HTTPS or localhost. Open the app on https:// or http://localhost.");
  }
  if (typeof AudioWorkletNode === "undefined") {
    throw new CaptureError("unsupported", "This browser can't process live audio (no AudioWorklet). Use a current Chrome, Edge, Firefox or Safari.");
  }

  cb.onStatus?.({ state: "connecting", lagS: null });

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (e) {
    throw micError(e);
  }

  let ctx: AudioContext | null = null;
  let ws: WebSocket | null = null;
  let node: AudioWorkletNode | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let sink: GainNode | null = null;
  let finished = false;

  const releaseAudio = () => {
    node?.port.postMessage("stop");
    if (node) node.port.onmessage = null;
    source?.disconnect();
    node?.disconnect();
    sink?.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    if (ctx && ctx.state !== "closed") void ctx.close().catch(() => undefined);
  };

  const fail = (err: CaptureError) => {
    if (finished) return;
    finished = true;
    releaseAudio();
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
    cb.onStatus?.({ state: "stopped", lagS: null });
    cb.onError?.(err);
  };

  try {
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule(workletUrl);
    ws = await openSocket(meetingId);
  } catch (e) {
    finished = true;
    releaseAudio();
    if (ws) ws.close();
    throw e instanceof CaptureError ? e : new CaptureError("audio", `Audio processing couldn't start: ${e instanceof Error ? e.message : String(e)}`);
  }

  const socket = ws;
  socket.onmessage = (m) => {
    const msg = parseServerFrame(m.data);
    if (!msg) return;
    if (msg.type === "status" && (msg.state === "listening" || msg.state === "processing")) {
      cb.onStatus?.({ state: msg.state, lagS: typeof msg.lag_s === "number" ? msg.lag_s : null });
    } else if (msg.type === "error") {
      fail(new CaptureError("server", `The server stopped the audio stream: ${msg.detail ?? "unknown error"}`));
    }
  };
  socket.onclose = () => {
    fail(new CaptureError("socket", "The audio connection closed. Reconnect to keep capturing."));
  };
  socket.onerror = null;

  source = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, "wsw-pcm-16k", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1,
    channelCountMode: "explicit",
    channelInterpretation: "speakers",
  });
  // A silent sink keeps the graph pulled in every browser without playing the mic back.
  sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(node);
  node.connect(sink);
  sink.connect(ctx.destination);

  node.port.onmessage = (e: MessageEvent<{ pcm: ArrayBuffer; rms: number }>) => {
    if (finished || socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
      fail(new CaptureError("socket", "The connection to the server is too slow to keep up with live audio. Reconnect to keep capturing."));
      return;
    }
    socket.send(e.data.pcm);
    cb.onLevel?.(Math.min(1, e.data.rms));
  };

  stream.getAudioTracks().forEach((track) => {
    track.onended = () => fail(new CaptureError("audio", "The microphone stopped (unplugged, or taken by another app). Reconnect to keep capturing."));
  });

  if (ctx.state === "suspended") await ctx.resume().catch(() => undefined);
  if (ctx.state !== "running") {
    finished = true;
    releaseAudio();
    socket.onclose = null;
    socket.close();
    throw new CaptureError("audio", "The browser kept audio paused until you interact with the page. Click Start capturing to begin.");
  }

  cb.onStatus?.({ state: "listening", lagS: null });

  return {
    async stop(endMeeting: boolean) {
      if (finished) return;
      finished = true;
      releaseAudio();
      socket.onclose = null;
      socket.onmessage = null;
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(endMeeting ? { type: "stop" } : { type: "stop", end: false }));
        await waitForDrain(socket);
        socket.close(1000, "stopped");
      }
      cb.onStatus?.({ state: "stopped", lagS: null });
    },
  };
}
