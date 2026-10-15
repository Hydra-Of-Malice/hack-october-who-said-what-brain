// The voice-first ask drawer. Opens on "wsw:ask-open", "/" or a "wsw:ask" {question} event.
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { matchPath, useLocation } from "react-router-dom";
import { Plus, Send, X } from "lucide-react";
import { api } from "../../api/client";
import type { AskResult, ClientAction, Health } from "../../api/types";
import { timecode } from "../../lib/format";
import { prefersReducedMotion } from "../graph/useReducedMotion";
import AskExchange from "./AskExchange";
import { askErrorText, NOT_CAUGHT } from "./askFormat";
import {
  currentSession,
  exchangeId,
  exchangesFromTurns,
  loadExchanges,
  readFlag,
  saveExchanges,
  startNewSession,
  writeFlag,
  type Exchange,
} from "./askStore";
import { titleMap, useMeetings } from "./meetingData";
import { speak, stopSpeaking, ttsSupported } from "./speech";
import Suggestions from "./Suggestions";
import Switch from "./Switch";
import Talkback, { type TalkbackVisual } from "./Talkback";
import { isActivatable, isTypingTarget, useFocusTrap } from "./useFocusTrap";
import { useJump, useOpenMeeting } from "./useJump";
import { useVoiceQuestion } from "./useVoiceQuestion";
import "./ask.css";

const OPEN_EVENT = "wsw:ask-open";
const ASK_EVENT = "wsw:ask";
const REFRESH_EVENT = "wsw:refresh";
const SPEAK_PREF = "wsw.ask.speak";
const HEARING_STAGE_MS = 3000;
const PHONE = "(max-width: 767px)";

interface Pending {
  id: string;
  mode: "voice" | "text";
  startedAt: number;
}

export default function AskDrawer() {
  const location = useLocation();
  const jump = useJump();
  const openMeeting = useOpenMeeting();
  const routeMeetingId = matchPath("/m/:id", location.pathname)?.params.id ?? null;

  const [open, setOpen] = useState(false);
  const [session, setSession] = useState(currentSession);
  const [exchanges, setExchanges] = useState<Exchange[]>(() => loadExchanges(session.id));
  const [scopeAll, setScopeAll] = useState(true);
  const [pending, setPending] = useState<Pending | null>(null);
  const [now, setNow] = useState(() => performance.now());
  const [speakOn, setSpeakOn] = useState(() => readFlag(SPEAK_PREF, true));
  const [speaking, setSpeaking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const { meetings } = useMeetings(open);
  const titles = useMemo(() => titleMap(meetings), [meetings]);

  const drawerRef = useRef<HTMLElement>(null);
  const talkRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const lastRef = useRef<HTMLLIElement>(null);
  const levelRef = useRef(0);
  const openRef = useRef(open);
  const speakOnRef = useRef(speakOn);
  const speakingRef = useRef(false);
  const audibleRef = useRef(false);
  const sessionRef = useRef(session.id);
  const inflight = useRef<AbortController | null>(null);
  const clips = useRef(new Map<string, Blob>());
  const spaceHeld = useRef(false);
  const restored = useRef(false);
  openRef.current = open;
  speakOnRef.current = speakOn;
  sessionRef.current = session.id;

  const scopeId = !scopeAll && routeMeetingId ? routeMeetingId : null;

  // ---- persistence ----------------------------------------------------------------------
  useEffect(() => saveExchanges(session.id, exchanges), [session.id, exchanges]);

  useEffect(() => {
    // This tab lost its cached thread but the server still has the session: show it as text.
    if (!open || restored.current) return;
    restored.current = true;
    if (!session.resumed || exchanges.length) return;
    api
      .session(session.id)
      .then((turns) => {
        if (turns.length && sessionRef.current === session.id) setExchanges((xs) => (xs.length ? xs : exchangesFromTurns(turns)));
      })
      .catch(() => undefined);
  }, [open, session, exchanges.length]);

  useEffect(() => {
    if (!open || health) return;
    api
      .health()
      .then(setHealth)
      .catch(() => undefined);
  }, [open, health]);

  // ---- speech -----------------------------------------------------------------------------
  const stopSpeech = useCallback(() => {
    stopSpeaking();
    speakingRef.current = false;
    audibleRef.current = false;
    setSpeaking(false);
  }, []);

  const speakAnswer = useCallback((text: string) => {
    audibleRef.current = false;
    const started = speak(text, {
      onStart: () => {
        audibleRef.current = true;
      },
      onEnd: () => {
        speakingRef.current = false;
        audibleRef.current = false;
        setSpeaking(false);
      },
    });
    speakingRef.current = started;
    setSpeaking(started);
  }, []);

  // ---- requests ---------------------------------------------------------------------------
  const update = useCallback((id: string, patch: Partial<Exchange>) => {
    setExchanges((xs) => xs.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }, []);

  const abortInflight = useCallback(() => {
    const c = inflight.current;
    if (!c) return;
    inflight.current = null;
    c.abort();
    setPending(null);
  }, []);

  const runActions = useCallback(
    (actions: ClientAction[]) => {
      for (const a of actions ?? []) {
        if (a.type === "seek") jump({ meeting_id: a.meeting_id, t: a.t });
        else if (a.type === "open_meeting") openMeeting(a.meeting_id);
        else if (a.type === "refresh") window.dispatchEvent(new CustomEvent(REFRESH_EVENT, { detail: a.what }));
      }
    },
    [jump, openMeeting],
  );

  const perform = useCallback(
    async (x: Exchange, call: (signal: AbortSignal) => Promise<AskResult>) => {
      stopSpeech();
      abortInflight();
      const ctrl = new AbortController();
      inflight.current = ctrl;
      setExchanges((xs) => (xs.some((e) => e.id === x.id) ? xs.map((e) => (e.id === x.id ? x : e)) : [...xs, x]));
      setPending({ id: x.id, mode: x.mode, startedAt: performance.now() });
      setNotice(null);
      try {
        const result = await call(ctrl.signal);
        if (ctrl.signal.aborted) return;
        if (x.mode === "voice" && typeof result.transcript === "string" && !result.transcript.trim()) {
          clips.current.delete(x.id);
          update(x.id, { status: "error", error: NOT_CAUGHT, heardBy: result.transcript_provider });
          return;
        }
        clips.current.delete(x.id);
        update(x.id, {
          status: "answered",
          result,
          error: null,
          question: x.mode === "voice" ? result.transcript?.trim() || null : x.question,
          heardBy: result.transcript_provider ?? null,
        });
        runActions(result.actions);
        if (speakOnRef.current && openRef.current && result.answer.trim()) speakAnswer(result.answer);
      } catch (e) {
        // The client turns an aborted fetch into a "can't reach" error; trust our own signal instead.
        if (ctrl.signal.aborted) update(x.id, { status: "stopped" });
        else update(x.id, { status: "error", error: askErrorText(e, x.mode) });
      } finally {
        if (inflight.current === ctrl) {
          inflight.current = null;
          setPending(null);
        }
      }
    },
    [abortInflight, runActions, speakAnswer, stopSpeech, update],
  );

  const askText = useCallback(
    (text: string) => {
      const question = text.trim();
      if (!question) return;
      const x: Exchange = {
        id: exchangeId(),
        mode: "text",
        question,
        heardBy: null,
        seconds: null,
        status: "pending",
        result: null,
        error: null,
        scope: scopeId,
        at: Date.now(),
      };
      void perform(x, (signal) => api.ask(sessionRef.current, question, scopeId ?? undefined, signal));
    },
    [perform, scopeId],
  );

  const askVoice = useCallback(
    (wav: Blob, seconds: number, retryOf?: Exchange) => {
      const x: Exchange = retryOf
        ? { ...retryOf, status: "pending", error: null, result: null }
        : {
            id: exchangeId(),
            mode: "voice",
            question: null,
            heardBy: null,
            seconds,
            status: "pending",
            result: null,
            error: null,
            scope: scopeId,
            at: Date.now(),
          };
      clips.current.set(x.id, wav);
      const scope = x.scope ?? undefined;
      void perform(x, (signal) => api.askVoice(sessionRef.current, wav, scope, signal));
    },
    [perform, scopeId],
  );

  const retry = useCallback(
    (x: Exchange) => {
      if (x.mode === "text" && x.question) {
        const question = x.question;
        const scope = x.scope ?? undefined;
        void perform({ ...x, status: "pending", error: null, result: null }, (signal) =>
          api.ask(sessionRef.current, question, scope, signal),
        );
        return;
      }
      const wav = clips.current.get(x.id);
      if (wav) askVoice(wav, x.seconds ?? 0, x);
    },
    [askVoice, perform],
  );

  // ---- voice ------------------------------------------------------------------------------
  const voice = useVoiceQuestion({
    onQuestion: (wav, seconds) => askVoice(wav, seconds),
    onBargeIn: (source) => {
      setNotice(null);
      if (source === "press" || speakingRef.current) {
        stopSpeech();
        abortInflight();
      }
    },
    onLevel: (level) => {
      levelRef.current = level;
    },
    onNotice: setNotice,
    speakingRef,
    audibleRef,
  });
  const { shutdown: shutdownVoice } = voice;

  // ---- open / close ----------------------------------------------------------------------------
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (open) return;
    // Closing releases the microphone and stops speech; a request in flight still lands in the thread.
    shutdownVoice();
    stopSpeech();
    spaceHeld.current = false;
    // shutdownVoice wraps a stable controller but is re-created each render; only "open" matters here.
  }, [open, stopSpeech]);

  const askTextRef = useRef(askText);
  askTextRef.current = askText;

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      if (!openRef.current) setOpen(true);
      else inputRef.current?.focus();
    };
    const onOpen = () => setOpen(true);
    const onAsk = (e: Event) => {
      const question = (e as CustomEvent<{ question?: unknown } | null>).detail?.question;
      setOpen(true);
      if (typeof question === "string" && question.trim()) askTextRef.current(question);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_EVENT, onOpen);
    window.addEventListener(ASK_EVENT, onAsk);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_EVENT, onOpen);
      window.removeEventListener(ASK_EVENT, onAsk);
    };
  }, []);

  useFocusTrap(drawerRef, open, talkRef);

  // Tick the stage label and timers while something is happening.
  const busy = !!pending || voice.mic === "recording";
  useEffect(() => {
    if (!busy) return;
    setNow(performance.now());
    const t = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(t);
  }, [busy]);

  // Bring the newest exchange into view when one is added or the drawer opens.
  useEffect(() => {
    if (!open) return;
    lastRef.current?.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }, [open, exchanges.length]);

  const newConversation = () => {
    voice.cancel();
    stopSpeech();
    abortInflight();
    clips.current.clear();
    const id = startNewSession(session.id);
    setSession({ id, resumed: false });
    setExchanges([]);
    setNotice(null);
    talkRef.current?.focus();
  };

  const setSpeak = (on: boolean) => {
    setSpeakOn(on);
    writeFlag(SPEAK_PREF, on);
    if (!on) stopSpeech();
  };

  const onJumped = useCallback(() => {
    if (window.matchMedia?.(PHONE).matches) close();
  }, [close]);

  const onDrawerKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      if (voice.capture) {
        voice.cancel();
        setNotice("Recording discarded.");
      } else {
        close();
      }
      return;
    }
    if (e.key === " " && !e.repeat && !isActivatable(e.target) && !isTypingTarget(e.target)) {
      e.preventDefault();
      spaceHeld.current = true;
      voice.pressStart();
    }
  };

  const onDrawerKeyUp = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key === " " && spaceHeld.current) {
      e.preventDefault();
      spaceHeld.current = false;
      voice.pressEnd();
    }
  };

  // ---- status -----------------------------------------------------------------------------
  const visual: TalkbackVisual =
    voice.mic === "recording"
      ? "recording"
      : voice.mic === "opening"
        ? "opening"
        : pending
          ? "waiting"
          : speaking
            ? "speaking"
            : voice.mic === "armed"
              ? "armed"
              : "idle";

  const stage = pending
    ? pending.mode === "voice" && now - pending.startedAt < HEARING_STAGE_MS
      ? "Hearing your question…"
      : "Searching the meetings…"
    : null;

  const status = statusText(visual, voice.capture, voice.handsFree, stage);
  const timer =
    visual === "recording" && voice.captureStartedAt !== null
      ? `${timecode((now - voice.captureStartedAt) / 1000)} of 00:30`
      : visual === "waiting" && pending
        ? timecode((now - pending.startedAt) / 1000)
        : null;
  const talkLabel =
    visual === "recording"
      ? voice.capture === "hold"
        ? "Listening. Release to send."
        : "Listening. Press to send."
      : "Ask by voice. Hold to talk, or press once to start.";

  const meetingTitle = routeMeetingId ? (titles.get(routeMeetingId) ?? "This meeting") : "This meeting";
  const canSpeak = ttsSupported();
  const audioPath = health?.question_audio ?? null;

  if (!open) return null;

  return (
    <>
      <div className="ask-scrim" onClick={close} aria-hidden="true" />
      <section
        ref={drawerRef}
        className="ask-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ask-title"
        tabIndex={-1}
        onKeyDown={onDrawerKeyDown}
        onKeyUp={onDrawerKeyUp}
      >
        <header className="ask-head">
          <h2 id="ask-title">Ask</h2>
          <div className="ask-head__actions">
            <button type="button" className="btn btn--ghost" onClick={newConversation} disabled={!exchanges.length && !pending}>
              <Plus aria-hidden="true" />
              New conversation
            </button>
            <button type="button" className="btn btn--ghost ask-icon-btn" aria-label="Close the ask panel" onClick={close}>
              <X aria-hidden="true" />
            </button>
          </div>
        </header>

        <fieldset className="ask-scope">
          <legend className="visually-hidden">Search in</legend>
          <label className="ask-seg">
            <input type="radio" name="ask-scope" checked={!scopeId} onChange={() => setScopeAll(true)} />
            <span>All meetings</span>
          </label>
          <label className="ask-seg" title={routeMeetingId ? meetingTitle : "Open a meeting to ask about it alone"}>
            <input
              type="radio"
              name="ask-scope"
              checked={!!scopeId}
              disabled={!routeMeetingId}
              onChange={() => setScopeAll(false)}
            />
            <span>{meetingTitle}</span>
          </label>
        </fieldset>

        <div className="ask-log">
          {exchanges.length ? (
            <ol className="ask-list" role="log" aria-label="Conversation" aria-live="polite">
              {exchanges.map((x, i) => (
                <AskExchange
                  key={x.id}
                  ref={i === exchanges.length - 1 ? lastRef : undefined}
                  x={x}
                  titles={titles}
                  waiting={
                    pending?.id === x.id && stage ? { stage, elapsed: (now - pending.startedAt) / 1000 } : null
                  }
                  canSpeak={canSpeak}
                  onSpeak={speakAnswer}
                  onRetry={
                    x.status === "error" && x.error !== NOT_CAUGHT && (x.mode === "text" || clips.current.has(x.id))
                      ? () => retry(x)
                      : null
                  }
                  onJumped={onJumped}
                />
              ))}
            </ol>
          ) : (
            <div className="ask-empty">
              <h3>Ask about your meetings</h3>
              <p>
                Hold the button and ask out loud, or type below. Answers quote who said it, with a timecode you can
                play.
              </p>
              {audioPath ? (
                <p className="ask-prov">
                  {audioPath.startsWith("gemma4-audio")
                    ? "Your voice goes straight to Gemma 4, which hears the question itself."
                    : "Your voice is transcribed by Whisper, then Gemma 4 answers."}
                </p>
              ) : null}
              <Suggestions onPick={askText} />
            </div>
          )}
        </div>

        <footer className="ask-dock">
          <div className="ask-dock__main">
            <Talkback
              visual={visual}
              levelRef={levelRef}
              onPressStart={voice.pressStart}
              onPressEnd={voice.pressEnd}
              label={talkLabel}
              describedBy="ask-status-hint"
              buttonRef={talkRef}
            />
            <div className="ask-status">
              <div className="ask-status__row">
                <p className="ask-status__title" aria-live="polite">
                  {visual === "armed" ? <span className="live-dot" aria-hidden="true" /> : null}
                  {status.title}
                </p>
                {timer ? (
                  <span className="tc ask-status__timer" aria-hidden="true">
                    {timer}
                  </span>
                ) : null}
              </div>
              <p className="ask-status__hint" id="ask-status-hint">
                {status.hint}
              </p>
            </div>
          </div>

          {voice.error ? (
            <p className="ask-note ask-note--err" role="alert">
              {voice.error}
            </p>
          ) : notice ? (
            <p className="ask-note" role="status">
              {notice}
            </p>
          ) : null}

          <div className="ask-dock__switches">
            <Switch checked={voice.handsFree} onChange={voice.setHandsFree} label="Hands-free" />
            <Switch
              checked={speakOn && canSpeak}
              onChange={setSpeak}
              label="Speak answers"
              disabled={!canSpeak}
              title={canSpeak ? undefined : "This browser can't speak answers aloud"}
            />
          </div>

          <form
            className="ask-type"
            onSubmit={(e) => {
              e.preventDefault();
              if (!draft.trim()) return;
              askText(draft);
              setDraft("");
            }}
          >
            <label className="visually-hidden" htmlFor="ask-input">
              Type a question
            </label>
            <input
              id="ask-input"
              ref={inputRef}
              className="field"
              placeholder="Type a question"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoComplete="off"
              enterKeyHint="send"
            />
            <button type="submit" className="btn btn--primary" disabled={!draft.trim()}>
              <Send aria-hidden="true" />
              Ask
            </button>
          </form>
        </footer>
      </section>
    </>
  );
}

function statusText(
  visual: TalkbackVisual,
  capture: "hold" | "toggle" | "vad" | null,
  handsFree: boolean,
  stage: string | null,
): { title: string; hint: string } {
  switch (visual) {
    case "recording":
      return {
        title: "Listening…",
        hint:
          capture === "hold"
            ? "Release to send. Esc discards."
            : capture === "vad"
              ? "Pause when you're done and it sends itself."
              : "Press again to send. Esc discards.",
      };
    case "opening":
      return { title: "Starting the microphone…", hint: "If the browser asks, allow microphone access." };
    case "waiting":
      return { title: stage ?? "Searching the meetings…", hint: "Press the button to stop and ask something else." };
    case "speaking":
      return {
        title: "Speaking the answer",
        hint: handsFree ? "Start talking or press the button to interrupt." : "Press the button to interrupt.",
      };
    case "armed":
      return { title: "Hands-free is listening", hint: "Ask your question. It sends when you pause." };
    case "idle":
      return { title: "Hold to talk", hint: "Or press once to start and again to send. Space works too." };
  }
}
