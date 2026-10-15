import { forwardRef, useState } from "react";
import { CircleAlert, Ear, LoaderCircle, Play, RotateCcw, Volume2, Workflow } from "lucide-react";
import type { AskResult, ClientAction } from "../../api/types";
import { speakerName, timecode } from "../../lib/format";
import { answerBlocks, answeredBy, describeAction, heardByLabel } from "./askFormat";
import type { Exchange } from "./askStore";
import DraftCard from "./DraftCard";
import EvidenceQuote from "./EvidenceQuote";
import { useJump } from "./useJump";
import WhyPanel from "./WhyPanel";

const SHOWN_CITATIONS = 3;

interface AskExchangeProps {
  x: Exchange;
  titles: Map<string, string>;
  /** Present while this exchange is the one in flight. */
  waiting: { stage: string; elapsed: number } | null;
  canSpeak: boolean;
  onSpeak: (text: string) => void;
  onRetry: (() => void) | null;
  onJumped: () => void;
}

const AskExchange = forwardRef<HTMLLIElement, AskExchangeProps>(function AskExchange(
  { x, titles, waiting, canSpeak, onSpeak, onRetry, onJumped },
  ref,
) {
  const heard = heardByLabel(x.heardBy);
  const question =
    x.question ?? (x.mode === "voice" ? (x.status === "pending" ? "Your spoken question" : "Spoken question") : "");
  return (
    <li className="ask-x" ref={ref}>
      <div className={`ask-q${x.question ? "" : " ask-q--placeholder"}`}>
        <p className="ask-q__text">
          <span className="visually-hidden">You asked: </span>
          {question}
        </p>
        {x.mode === "voice" ? (
          <p className="ask-prov">
            <Ear aria-hidden="true" />
            <span>{heard ?? "Recorded question"}</span>
            {x.seconds ? <span className="tc">{x.seconds.toFixed(1)} s</span> : null}
          </p>
        ) : null}
      </div>

      <div className="ask-a">
        <span className="visually-hidden">Answer: </span>
        {x.status === "pending" ? (
          <p className="ask-a__wait">
            <LoaderCircle aria-hidden="true" className="ask-spin" />
            <span>{waiting?.stage ?? "Working…"}</span>
            {waiting ? <span className="tc">{timecode(waiting.elapsed)}</span> : null}
          </p>
        ) : null}

        {x.status === "stopped" ? <p className="ask-a__muted">Stopped before the answer arrived.</p> : null}

        {x.status === "error" ? (
          <div className="ask-err" role="alert">
            <CircleAlert aria-hidden="true" />
            <p>{x.error}</p>
            {onRetry ? (
              <button type="button" className="btn btn--ghost ask-mini" onClick={onRetry}>
                <RotateCcw aria-hidden="true" />
                Try again
              </button>
            ) : null}
          </div>
        ) : null}

        {x.status === "answered" && x.result ? (
          <Answer result={x.result} titles={titles} canSpeak={canSpeak} onSpeak={onSpeak} onJumped={onJumped} />
        ) : null}

        {x.status === "answered" && !x.result && x.restoredAnswer ? (
          <>
            <AnswerText text={x.restoredAnswer} />
            <p className="ask-prov">Restored from this conversation. Sources are shown only for new answers.</p>
          </>
        ) : null}
      </div>
    </li>
  );
});

export default AskExchange;

function AnswerText({ text }: { text: string }) {
  return (
    <div className="ask-answer__text">
      {answerBlocks(text).map((b, i) =>
        b.kind === "p" ? (
          <p key={i}>{b.text}</p>
        ) : b.kind === "ul" ? (
          <ul key={i}>
            {b.items.map((it, j) => (
              <li key={j}>{it}</li>
            ))}
          </ul>
        ) : (
          <ol key={i}>
            {b.items.map((it, j) => (
              <li key={j}>{it}</li>
            ))}
          </ol>
        ),
      )}
    </div>
  );
}

interface AnswerProps {
  result: AskResult;
  titles: Map<string, string>;
  canSpeak: boolean;
  onSpeak: (text: string) => void;
  onJumped: () => void;
}

function Answer({ result, titles, canSpeak, onSpeak, onJumped }: AnswerProps) {
  const [showAll, setShowAll] = useState(false);
  const citations = result.citations ?? [];
  const shown = showAll ? citations : citations.slice(0, SHOWN_CITATIONS);
  const actions = result.actions ?? [];
  const drafts = actions.filter((a): a is Extract<ClientAction, { type: "draft" }> => a.type === "draft");
  const done = actions.filter((a) => a.type !== "draft");

  return (
    <div className="ask-answer">
      {result.answer.trim() ? <AnswerText text={result.answer} /> : <p className="ask-a__muted">Gemma returned an empty answer.</p>}

      <div className="ask-answer__by">
        <p className="ask-prov">{answeredBy(result.model, result.provider)}</p>
        {canSpeak && result.answer.trim() ? (
          <button type="button" className="btn btn--ghost ask-mini" onClick={() => onSpeak(result.answer)}>
            <Volume2 aria-hidden="true" />
            Read aloud
          </button>
        ) : null}
      </div>

      {done.length ? (
        <ul className="ask-acts">
          {done.map((a, i) => (
            <ActionNote key={i} action={a} titles={titles} onJumped={onJumped} />
          ))}
        </ul>
      ) : null}

      {drafts.map((d, i) => (
        <DraftCard key={i} title={d.title} body={d.body} />
      ))}

      {citations.length ? (
        <section className="ask-cites" aria-label="Cited moments">
          <h4 className="ask-cites__h">{citations.length === 1 ? "1 cited moment" : `${citations.length} cited moments`}</h4>
          <ul>
            {shown.map((c) => (
              <li key={c.utterance_id}>
                <EvidenceQuote
                  size="sm"
                  text={c.text}
                  track={c.speaker}
                  speakerName={speakerName(c)}
                  meetingId={c.meeting_id}
                  meetingTitle={c.meeting_title}
                  t={c.t_start}
                  utteranceId={c.utterance_id}
                  onJumped={onJumped}
                />
              </li>
            ))}
          </ul>
          {citations.length > SHOWN_CITATIONS ? (
            <button type="button" className="btn btn--ghost ask-mini" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show fewer" : `Show ${citations.length - SHOWN_CITATIONS} more`}
            </button>
          ) : null}
        </section>
      ) : null}

      <WhyPanel result={result} />
    </div>
  );
}

function ActionNote({ action, titles, onJumped }: { action: ClientAction; titles: Map<string, string>; onJumped: () => void }) {
  const jump = useJump();
  const text = describeAction(action, titles);
  if (!text) return null;
  return (
    <li className="ask-act">
      <Workflow aria-hidden="true" />
      <span>{text}</span>
      {action.type === "seek" ? (
        <button
          type="button"
          className="btn btn--ghost ask-mini"
          onClick={() => {
            jump({ meeting_id: action.meeting_id, t: action.t });
            onJumped();
          }}
        >
          <Play aria-hidden="true" />
          Play again
        </button>
      ) : null}
    </li>
  );
}
