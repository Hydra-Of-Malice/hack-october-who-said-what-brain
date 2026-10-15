// Mic capture lifecycle for one meeting, safe under StrictMode double mounts and unmounts mid-start.
import { useCallback, useEffect, useRef, useState } from "react";
import { CaptureError, startCapture, type CaptureHandle, type CaptureStatus } from "../../audio/mic";

export type CapturePhase = "idle" | "starting" | "capturing" | "stopping" | "error";

/** Tiny observable so only the level meter re-renders ten times a second. */
export interface LevelStore {
  get: () => number;
  set: (v: number) => void;
  subscribe: (fn: () => void) => () => void;
}

function createLevelStore(): LevelStore {
  let value = 0;
  const subs = new Set<() => void>();
  return {
    get: () => value,
    set: (v) => {
      value = v;
      subs.forEach((fn) => fn());
    },
    subscribe: (fn) => {
      subs.add(fn);
      return () => {
        subs.delete(fn);
      };
    },
  };
}

export interface CaptureControl {
  phase: CapturePhase;
  error: CaptureError | null;
  status: CaptureStatus | null;
  level: LevelStore;
  start: () => void;
  stop: (endMeeting: boolean) => Promise<void>;
}

export function useCapture(meetingId: string): CaptureControl {
  const [phase, setPhase] = useState<CapturePhase>("idle");
  const [error, setError] = useState<CaptureError | null>(null);
  const [status, setStatus] = useState<CaptureStatus | null>(null);
  const [level] = useState(createLevelStore);
  const handle = useRef<CaptureHandle | null>(null);
  const attempt = useRef(0);
  const mounted = useRef(true);
  const inFlight = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      attempt.current++;
      // Deliberately the live handle (not a DOM ref): whatever capture is running at unmount must stop.
      const h = handle.current;
      handle.current = null;
      void h?.stop(false);
    };
  }, [meetingId]);

  const start = useCallback(() => {
    if (handle.current || inFlight.current) return;
    inFlight.current = true;
    const mine = ++attempt.current;
    setPhase("starting");
    setError(null);
    startCapture(meetingId, {
      onLevel: (rms) => level.set(rms),
      onStatus: (s) => {
        if (mounted.current && attempt.current === mine) setStatus(s);
      },
      onError: (err) => {
        if (attempt.current !== mine) return;
        handle.current = null;
        level.set(0);
        if (mounted.current) {
          setError(err);
          setPhase("error");
        }
      },
    })
      .then((h) => {
        inFlight.current = false;
        if (attempt.current !== mine || !mounted.current) {
          void h.stop(false);
          return;
        }
        handle.current = h;
        setPhase("capturing");
      })
      .catch((e: unknown) => {
        inFlight.current = false;
        if (attempt.current !== mine || !mounted.current) return;
        setError(e instanceof CaptureError ? e : new CaptureError("audio", e instanceof Error ? e.message : String(e)));
        setPhase("error");
      });
  }, [meetingId, level]);

  const stop = useCallback(
    async (endMeeting: boolean) => {
      attempt.current++;
      const h = handle.current;
      handle.current = null;
      level.set(0);
      if (!h) {
        setPhase("idle");
        return;
      }
      setPhase("stopping");
      try {
        await h.stop(endMeeting);
      } finally {
        if (mounted.current) setPhase("idle");
      }
    },
    [level],
  );

  return { phase, error, status, level, start, stop };
}
