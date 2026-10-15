import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ArrowLeft, CircleAlert, LoaderCircle, Mic, Square } from "lucide-react";
import { api } from "../api/client";
import type { ActionItem, MeetingSummary, Utterance } from "../api/types";
import { brand } from "../lib/brand";
import { duration, speakerName, timecode, when } from "../lib/format";
import { takePendingSeek, useSeekRequests } from "../lib/player";
import AudioTransport from "../components/meeting/AudioTransport";
import InsightsRail from "../components/meeting/InsightsRail";
import LevelMeter from "../components/meeting/LevelMeter";
import SpeakerLanes, { type HeadMode } from "../components/meeting/SpeakerLanes";
import Transcript, { type SeekTarget } from "../components/meeting/Transcript";
import { asActionItem, colourMap, utteranceIndexAt, type RoomState } from "../components/meeting/meetingState";
import { endedDomain, liveDomain, useNow } from "../components/meeting/timeline";
import { useCapture, type CaptureControl } from "../components/meeting/useCapture";
import { useAudioElement } from "../components/meeting/useAudioElement";
import { useMeetingRoom, type MeetingRoomApi } from "../components/meeting/useMeetingRoom";
import "./MeetingRoom.css";

const LISTENER_FRESH_MS = 10_000;

export default function MeetingRoom() {
  const { id = "" } = useParams();
  return <Room key={id} id={id} />;
}

function Room({ id }: { id: string }) {
  const room = useMeetingRoom(id);
  const { phase, error, meeting } = room.state;

  if (phase === "loading") return <RoomLoading />;
  if (phase === "error" || !meeting) {
    const notFound = error?.status === 404;
    return (
      <div className="room room--message">
        <Link to="/" className="room-back">
          <ArrowLeft aria-hidden="true" />
          Meetings
        </Link>
        <div className="room-message panel" role="alert">
          <h1 className="room-message-title">{notFound ? "This meeting doesn't exist" : "Couldn't load this meeting"}</h1>
          <p>
            {notFound
              ? "It may have been deleted, or the link is wrong. Pick a meeting from the list instead."
              : `${error?.message ?? "Unknown error"}. Check that the API is running, then try again.`}
          </p>
          <div className="room-message-actions">
            {!notFound && (
              <button type="button" className="btn btn--primary" onClick={room.reload}>
                Try again
              </button>
            )}
            <Link to="/" className="btn">
              Back to meetings
            </Link>
          </div>
        </div>
      </div>
    );
  }
  return <RoomReady id={id} room={room} meeting={meeting} />;
}

function RoomLoading() {
  return (
    <div className="room" aria-busy="true">
      <div className="room-head">
        <span className="room-back room-back--muted">Meetings</span>
        <p className="room-loading" role="status">
          <LoaderCircle className="spin" aria-hidden="true" />
          Loading the meeting
        </p>
      </div>
      <div className="lanes-skeleton panel" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span key={i} />
        ))}
      </div>
    </div>
  );
}

function RoomReady({ id, room, meeting }: { id: string; room: MeetingRoomApi; meeting: MeetingSummary }) {
  const { state, dispatch, renameSpeaker } = room;
  const [params] = useSearchParams();
  const live = meeting.status === "live";
  const isUpload = meeting.source === "upload";
  const canCapture = live && !isUpload && meeting.source !== "demo";
  const capture = useCapture(id);
  const capturing = capture.phase === "capturing";
  const audio = useAudioElement();
  const hasPlayer = !live && meeting.has_audio === true;
  const canPlay = hasPlayer && !audio.error;

  useEffect(() => {
    document.title = `${meeting.title || "Untitled meeting"} - ${brand.short}`;
  }, [meeting.title]);

  const now = useNow(1000, live);
  const startedAt = Number(meeting.started_at) || 0;
  const lastEnd = useMemo(() => state.utterances.reduce((m, u) => Math.max(m, u.t_end), 0), [state.utterances]);
  const elapsed = live && startedAt ? Math.max(0, (now - startedAt) / 1000) : 0;
  const liveHead = isUpload || !startedAt ? lastEnd : Math.max(lastEnd, elapsed);
  const domain = live ? liveDomain(liveHead) : endedDomain(meeting.duration_s, lastEnd);

  const colours = useMemo(() => colourMap(state.speakers), [state.speakers]);
  const speakerByTrack = useMemo(() => new Map(state.speakers.map((s) => [s.track, s])), [state.speakers]);
  const nameOf = useCallback(
    (track: string) => {
      const s = speakerByTrack.get(track);
      if (s) return speakerName(s);
      // proposed_by / owner may already be a display name ("Priya") rather than a track ("S2")
      return /^S\d+$/.test(track) ? speakerName({ track }) : track;
    },
    [speakerByTrack],
  );
  const utterancesById = useMemo(() => new Map(state.utterances.map((u) => [u.id, u])), [state.utterances]);

  // The playhead reads these every animation frame and callbacks read them later; refs keep those stable.
  const headRef = useRef({ live, isUpload, startedAt, lastEnd });
  const stateRef = useRef<RoomState>(state);
  const canPlayRef = useRef(canPlay);
  useLayoutEffect(() => {
    headRef.current = { live, isUpload, startedAt, lastEnd };
    stateRef.current = state;
    canPlayRef.current = canPlay;
  });

  const { readTime, seek } = audio;
  const readHead = useCallback((): number | null => {
    const h = headRef.current;
    if (h.live) return h.isUpload || !h.startedAt ? h.lastEnd : Math.max(h.lastEnd, (Date.now() - h.startedAt) / 1000);
    return readTime();
  }, [readTime]);
  const headMode: HeadMode = live ? "live" : hasPlayer ? "playback" : "none";

  const [target, setTarget] = useState<SeekTarget | null>(null);

  const seekTo = useCallback(
    (t: number, utteranceId?: string, play = true) => {
      if (canPlayRef.current) seek(t, play);
      const list = stateRef.current.utterances;
      const known = utteranceId && list.some((u) => u.id === utteranceId) ? utteranceId : undefined;
      const uid = known ?? list[utteranceIndexAt(list, t + 0.05)]?.id;
      if (uid) setTarget({ id: uid, n: Date.now() });
    },
    [seek],
  );

  const onPick = useCallback((u: Utterance) => seekTo(u.t_start, u.id), [seekTo]);
  const onScrub = useCallback((t: number) => void seek(t, false), [seek]);

  // Seeks from the Ask drawer: one queued before we navigated here, and any that arrive while open.
  useSeekRequests((r) => {
    if (r.meeting_id !== id) return;
    takePendingSeek(id);
    seekTo(r.t, r.utterance_id);
  });
  useEffect(() => {
    const pending = takePendingSeek(id);
    if (pending) seekTo(pending.t, pending.utterance_id);
  }, [id, seekTo]);

  // ?live=1: the person just pressed "Start a live meeting", so begin capturing. The timer skips StrictMode's throwaway mount.
  const wantCapture = params.get("live") === "1";
  const autoTried = useRef(false);
  const startCapture = capture.start;
  useEffect(() => {
    if (!wantCapture || !canCapture || autoTried.current) return;
    const t = window.setTimeout(() => {
      autoTried.current = true;
      startCapture();
    }, 0);
    return () => window.clearTimeout(t);
  }, [wantCapture, canCapture, startCapture]);

  // Ended elsewhere (another tab, or the server): release the mic.
  const stopCapture = capture.stop;
  useEffect(() => {
    if (!live && capture.phase === "capturing") void stopCapture(false);
  }, [live, capture.phase, stopCapture]);

  useEffect(() => {
    if (!capturing) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [capturing]);

  const [ending, setEnding] = useState(false);
  const [endError, setEndError] = useState<string | null>(null);
  const endMeeting = async () => {
    setEnding(true);
    setEndError(null);
    await capture.stop(false);
    try {
      const m = await api.endMeeting(id);
      dispatch({ type: "meeting", meeting: m });
    } catch (e) {
      setEndError(`Couldn't end the meeting: ${e instanceof Error ? e.message : String(e)}. Everything said so far is saved; try again.`);
    } finally {
      setEnding(false);
    }
  };

  const toggleAction = useCallback(
    async (item: ActionItem, done: boolean) => {
      const status = done ? "done" : "open";
      dispatch({ type: "action-item", item: { ...item, status, done_at: done ? Date.now() : null } });
      try {
        const saved = asActionItem(await api.setActionItem(item.id, status));
        if (saved) dispatch({ type: "action-item", item: saved });
      } catch (e) {
        dispatch({ type: "action-item", item });
        throw e;
      }
    },
    [dispatch],
  );

  const heardId = useMemo(() => {
    if (!hasPlayer || !audio.engaged) return null;
    return state.utterances[utteranceIndexAt(state.utterances, audio.time)]?.id ?? null;
  }, [hasPlayer, audio.engaged, audio.time, state.utterances]);

  const listenerFresh = !!state.listener && now - state.listener.at < LISTENER_FRESH_MS && state.listener.state !== "idle" && state.listener.state !== "error";
  const audioFlowing = live && (capturing || (listenerFresh && !isUpload));
  const shownDuration = live ? (isUpload ? lastEnd : elapsed) : Number(meeting.duration_s) || lastEnd;

  return (
    <div className="room">
      <header className="room-head">
        <Link to="/" className="room-back">
          <ArrowLeft aria-hidden="true" />
          Meetings
        </Link>
        <div className="room-titlebar">
          <h1 className="room-title">{meeting.title || "Untitled meeting"}</h1>
          <RoomActions
            live={live}
            canCapture={canCapture}
            capture={capture}
            ending={ending}
            onStart={capture.start}
            onEnd={endMeeting}
          />
        </div>
        <div className="room-status">
          {live && !isUpload && <RecIndicator flowing={audioFlowing} elapsed={elapsed} capture={capturing ? capture : null} />}
          {live && isUpload && (
            <span className="room-state">
              <LoaderCircle className="spin" aria-hidden="true" />
              Processing the recording, now at <span className="tc">{timecode(lastEnd)}</span>
            </span>
          )}
          {!live && <span className="room-state">Ended</span>}
          {startedAt > 0 && <span>Started {when(startedAt)}</span>}
          {!live && <span>{duration(shownDuration)} long</span>}
          <span>
            {state.speakers.length} {state.speakers.length === 1 ? "speaker" : "speakers"}
          </span>
          <span>{meeting.source === "upload" ? "Uploaded recording" : meeting.source === "demo" ? "Demo meeting" : "Recorded live"}</span>
        </div>
        {capture.error && (
          <div className="room-alert" role="alert">
            <CircleAlert aria-hidden="true" />
            <p>{capture.error.message}</p>
            {live && capture.error.kind !== "insecure" && capture.error.kind !== "unsupported" && (
              <button type="button" className="btn" onClick={capture.start} disabled={capture.phase === "starting"}>
                {capture.error.kind === "socket" ? "Reconnect" : "Try again"}
              </button>
            )}
          </div>
        )}
        {endError && (
          <div className="room-alert" role="alert">
            <CircleAlert aria-hidden="true" />
            <p>{endError}</p>
          </div>
        )}
        {hasPlayer && <AudioTransport src={api.audioUrl(id)} title={meeting.title} audio={audio} fallbackDuration={shownDuration} />}
      </header>

      <SpeakerLanes
        speakers={state.speakers}
        utterances={state.utterances}
        colours={colours}
        domain={domain}
        readHead={readHead}
        headMode={headMode}
        follow={live || audio.playing}
        fresh={state.fresh}
        activeId={heardId ?? target?.id ?? null}
        onPick={onPick}
        onScrub={canPlay ? onScrub : undefined}
        onRename={renameSpeaker}
        meta={<GraphWrites state={state} live={live} captureLag={capture.status?.lagS ?? null} listenerFresh={listenerFresh} />}
        emptyText={
          live
            ? isUpload
              ? "Lanes fill in as the recording is processed."
              : "Lanes appear as soon as someone speaks."
            : "No speech was recorded in this meeting."
        }
      />

      <div className="room-body">
        <Transcript
          utterances={state.utterances}
          colours={colours}
          nameOf={nameOf}
          heardId={heardId}
          target={target}
          live={live}
          canPlay={canPlay}
          onSeek={onPick}
          emptyText={
            live
              ? "Nothing said yet. Each line appears a moment after the speaker finishes."
              : isUpload
                ? "No speech was found in this recording. If it does contain speech, processing may have failed; try uploading it again."
                : "No speech was captured in this meeting."
          }
        />
        <InsightsRail
          decisions={state.decisions}
          actionItems={state.actionItems}
          topics={state.topics}
          utterancesById={utterancesById}
          colours={colours}
          nameOf={nameOf}
          fresh={state.fresh}
          live={live}
          onEvidence={onPick}
          onToggleAction={toggleAction}
        />
      </div>
    </div>
  );
}

interface RoomActionsProps {
  live: boolean;
  canCapture: boolean;
  capture: CaptureControl;
  ending: boolean;
  onStart: () => void;
  onEnd: () => void;
}

function RoomActions({ live, canCapture, capture, ending, onStart, onEnd }: RoomActionsProps) {
  if (!live) return null;
  if (ending || capture.phase === "stopping") {
    return (
      <div className="room-actions">
        <button type="button" className="btn btn--primary" disabled>
          <LoaderCircle className="spin" aria-hidden="true" />
          Saving the meeting
        </button>
      </div>
    );
  }
  if (capture.phase === "capturing") {
    return (
      <div className="room-actions">
        <button type="button" className="btn btn--primary" onClick={onEnd}>
          <Square aria-hidden="true" />
          Stop and save
        </button>
      </div>
    );
  }
  return (
    <div className="room-actions">
      {canCapture && (
        <button type="button" className="btn btn--primary" onClick={onStart} disabled={capture.phase === "starting"}>
          {capture.phase === "starting" ? <LoaderCircle className="spin" aria-hidden="true" /> : <Mic aria-hidden="true" />}
          {capture.phase === "starting" ? "Starting the microphone" : "Start capturing"}
        </button>
      )}
      {canCapture && (
        <button type="button" className="btn" onClick={onEnd}>
          End meeting
        </button>
      )}
    </div>
  );
}

function RecIndicator({ flowing, elapsed, capture }: { flowing: boolean; elapsed: number; capture: CaptureControl | null }) {
  // Screen readers hear recording start and stop once; the clock is a timer, which isn't read out on every tick.
  const [wasFlowing, setWasFlowing] = useState(flowing);
  const [announcement, setAnnouncement] = useState("");
  if (flowing !== wasFlowing) {
    setWasFlowing(flowing);
    setAnnouncement(flowing ? "Recording started" : "Recording stopped");
  }
  return (
    <>
      <span className="visually-hidden" role="status">
        {announcement}
      </span>
      {flowing ? (
        <span className="rec">
          <span className="live-dot" aria-hidden="true" />
          <span className="rec-label">REC</span>
          <span className="tc rec-time" role="timer" aria-label={`Recording for ${duration(elapsed)}`}>
            {timecode(elapsed)}
          </span>
          {capture && <LevelMeter store={capture.level} />}
        </span>
      ) : (
        <span className="room-state">
          <span className="room-idle-dot" aria-hidden="true" />
          Live, not capturing audio in this tab
        </span>
      )}
    </>
  );
}

function listenerText(state: RoomState, captureLag: number | null, listenerFresh: boolean): string | null {
  const l = state.listener;
  if (l && listenerFresh && l.state === "error") return l.detail || "Error";
  const lag = (listenerFresh ? l?.lagS : null) ?? captureLag;
  if (l && listenerFresh && l.state === "processing" && (lag ?? 0) < 0.5) return "Processing";
  if (lag === null || lag === undefined) return listenerFresh ? "Listening" : null;
  return lag < 0.5 ? "Keeping up" : `${lag.toFixed(1)} s behind`;
}

function GraphWrites({ state, live, captureLag, listenerFresh }: { state: RoomState; live: boolean; captureLag: number | null; listenerFresh: boolean }) {
  const lag = live ? listenerText(state, captureLag, listenerFresh) : null;
  const n = state.utterances.length;
  return (
    <dl className="writes">
      <div className="writes-item">
        <dt>Written to FalkorDB</dt>
        <dd>
          {n.toLocaleString()} {n === 1 ? "utterance" : "utterances"}
        </dd>
      </div>
      {state.lastWrite && (
        <div className="writes-item">
          <dt>Last batch</dt>
          <dd>
            {Math.round(state.lastWrite.ms)} ms
            {state.lastWrite.utterances > 0 && (
              <span className="writes-sub">
                {" "}
                for {state.lastWrite.utterances} {state.lastWrite.utterances === 1 ? "utterance" : "utterances"}
              </span>
            )}
          </dd>
        </div>
      )}
      {lag && (
        <div className="writes-item">
          <dt>Listener</dt>
          <dd>{lag}</dd>
        </div>
      )}
    </dl>
  );
}
