// Loads one meeting and keeps it current from the shared SSE stream.
import { useCallback, useEffect, useLayoutEffect, useReducer, useRef } from "react";
import { ApiError, api } from "../../api/client";
import { hub, useBusEvents } from "../../api/events";
import type { BusEvent, MeetingSummary, Speaker, Utterance } from "../../api/types";
import { asActionItem, initialRoom, roomReducer, type RoomAction, type RoomState } from "./meetingState";

type Data = Record<string, unknown>;

// The event stream replays recent history when the page connects; only events from after the page opened
// get the "just extracted" highlight (5 s slack for clock drift between server and browser).
const PAGE_OPENED_S = Date.now() / 1000 - 5;

function toAction(e: BusEvent): RoomAction | null {
  const d = (e.data ?? {}) as Data;
  switch (e.type) {
    case "utterance.created":
      return { type: "utterances", list: [d as unknown as Utterance], live: true };
    case "speaker.merged":
      return typeof d.from === "string" && typeof d.to === "string" ? { type: "speaker-merged", from: d.from, to: d.to } : null;
    case "speaker.named":
      return typeof d.track === "string"
        ? { type: "speaker-named", track: d.track, name: typeof d.name === "string" ? d.name : null, personId: (d.person_id as string) ?? null }
        : null;
    case "insight.created":
      return typeof d.kind === "string"
        ? { type: "insight", kind: d.kind, item: d.item, recent: e.ts >= PAGE_OPENED_S }
        : null;
    case "action.updated": {
      const item = asActionItem(d);
      return item ? { type: "action-item", item } : null;
    }
    case "graph.write":
      return { type: "graph-write", utterances: Number(d.utterances) || 0, ms: Number(d.ms) || 0 };
    case "listener.status":
      return {
        type: "listener",
        state: typeof d.state === "string" ? d.state : "idle",
        lagS: typeof d.lag_s === "number" ? d.lag_s : null,
        detail: typeof d.detail === "string" ? d.detail : "",
      };
    case "meeting.started":
    case "meeting.ended":
      return d && typeof d.id === "string" ? { type: "meeting", meeting: d as unknown as MeetingSummary } : null;
    default:
      return null;
  }
}

export interface MeetingRoomApi {
  state: RoomState;
  dispatch: (a: RoomAction) => void;
  reload: () => void;
  /** Optimistic rename; resolves when the server confirms, rejects (and reverts) on failure. */
  renameSpeaker: (track: string, name: string) => Promise<Speaker>;
}

export function useMeetingRoom(id: string): MeetingRoomApi {
  const [state, dispatch] = useReducer(roomReducer, initialRoom);
  const stateRef = useRef(state);
  useLayoutEffect(() => {
    stateRef.current = state;
  }, [state]);

  const load = useCallback(
    (signal: { cancelled: boolean }) => {
      api
        .meeting(id)
        .then((detail) => {
          if (!signal.cancelled) dispatch({ type: "loaded", detail });
        })
        .catch((e: unknown) => {
          if (signal.cancelled) return;
          const status = e instanceof ApiError ? e.status : 0;
          const message = e instanceof Error ? e.message : String(e);
          dispatch({ type: "load-failed", error: { status, message } });
        });
    },
    [id],
  );

  const reloadSignal = useRef({ cancelled: false });
  useEffect(() => {
    const signal = { cancelled: false };
    reloadSignal.current = signal;
    dispatch({ type: "reset" });
    load(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [load]);

  const reload = useCallback(() => {
    dispatch({ type: "reset" });
    load(reloadSignal.current);
  }, [load]);

  useBusEvents(
    (e) => {
      const action = toAction(e);
      if (action) dispatch(action);
    },
    (e) => e.meeting_id === id || (e.type.startsWith("meeting.") && (e.data as Data | null)?.id === id),
  );

  // SSE dropped and came back: fetch anything said in the gap (the bus replay may not reach back far enough).
  // The first connect needs no catch-up: the snapshot plus the bus replay (since_id) cover it.
  useEffect(() => {
    let seenUp = false;
    let wasDown = false;
    return hub.onStatus((connected) => {
      if (!connected) {
        wasDown = seenUp;
        return;
      }
      seenUp = true;
      if (!wasDown) return;
      wasDown = false;
      const s = stateRef.current;
      if (s.phase !== "ready") return;
      const afterSeq = s.utterances.length ? s.utterances[s.utterances.length - 1].seq : -1;
      api
        .utterancesAfter(id, afterSeq)
        .then((list) => dispatch({ type: "utterances", list, live: true }))
        .catch(() => undefined); // next reconnect retries; the stream itself is live again
    });
  }, [id]);

  const renameSpeaker = useCallback(
    async (track: string, name: string) => {
      const before = stateRef.current.speakers.find((s) => s.track === track);
      dispatch({ type: "speaker-named", track, name });
      try {
        const speaker = await api.nameSpeaker(id, track, name);
        dispatch({ type: "speaker", speaker: { ...speaker, track: speaker.track ?? track } });
        return speaker;
      } catch (e) {
        dispatch({ type: "speaker-named", track, name: before?.name ?? null });
        throw e;
      }
    },
    [id],
  );

  return { state, dispatch, reload, renameSpeaker };
}
