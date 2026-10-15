import { Play } from "lucide-react";
import { speakerIndex, timecode } from "../../lib/format";
import { useJump } from "./useJump";
import "./evidence.css";

export interface EvidenceQuoteProps {
  text: string;
  /** Speaker track ("S2"), for the colour of the rule. */
  track: string | null | undefined;
  speakerName: string;
  meetingId: string;
  meetingTitle?: string | null;
  t: number;
  utteranceId?: string;
  /** Called after the jump, e.g. to close a full-screen drawer on a phone. */
  onJumped?: () => void;
  /** Extra line under the meta row (retrieval path, score). */
  note?: string;
  size?: "md" | "sm";
}

/** Quoted speech as evidence: speaker colour rule, who, where, and a timecode that plays the moment. */
export default function EvidenceQuote({
  text,
  track,
  speakerName,
  meetingId,
  meetingTitle,
  t,
  utteranceId,
  onJumped,
  note,
  size = "md",
}: EvidenceQuoteProps) {
  const jump = useJump();
  const where = meetingTitle ? ` in ${meetingTitle}` : "";
  return (
    <figure className={`evq evq--${size} spk spk-${speakerIndex(track)}`}>
      <figcaption className="evq__meta">
        <span className="evq__who">{speakerName}</span>
        {meetingTitle ? <span className="evq__where">{meetingTitle}</span> : null}
        <button
          type="button"
          className="evq__play"
          onClick={() => {
            jump({ meeting_id: meetingId, t, utterance_id: utteranceId });
            onJumped?.();
          }}
          aria-label={`Play ${speakerName} at ${timecode(t)}${where}`}
        >
          <Play aria-hidden="true" />
          <span className="tc">{timecode(t)}</span>
        </button>
      </figcaption>
      <blockquote className="quote evq__text">{text}</blockquote>
      {note ? <p className="evq__note">{note}</p> : null}
    </figure>
  );
}
