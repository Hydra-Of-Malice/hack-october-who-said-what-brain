import { memo, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import type { Utterance } from "../../api/types";
import { timecode } from "../../lib/format";
import { prefersReducedMotion } from "./timeline";
import "./Transcript.css";

export interface SeekTarget {
  id: string;
  n: number;
}

interface Props {
  utterances: Utterance[];
  colours: Map<string, number>;
  nameOf: (track: string) => string;
  /** Line being heard right now (audio playback). */
  heardId: string | null;
  /** Line someone asked to see (block click, citation, insight evidence). */
  target: SeekTarget | null;
  live: boolean;
  canPlay: boolean;
  onSeek: (u: Utterance) => void;
  emptyText: string;
}

const NEAR_BOTTOM_PX = 56;
const FLASH_MS = 1800;

export default function Transcript({ utterances, colours, nameOf, heardId, target, live, canPlay, onSeek, emptyText }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const [unseen, setUnseen] = useState(0);
  const [flashId, setFlashId] = useState<string | null>(null);
  const prevCount = useRef(utterances.length);
  const followingRef = useRef(following);
  useLayoutEffect(() => {
    followingRef.current = following;
  }, [following]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    if (atBottom !== followingRef.current) setFollowing(atBottom);
    if (atBottom) setUnseen(0);
  };

  // Live: stick to the newest line unless the reader scrolled up.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const added = utterances.length - prevCount.current;
    prevCount.current = utterances.length;
    if (!el || !live || added <= 0) return;
    if (followingRef.current) el.scrollTop = el.scrollHeight;
    else setUnseen((n) => n + added);
  }, [utterances.length, live]);

  useEffect(() => {
    if (!target) return;
    const el = scrollRef.current?.querySelector<HTMLElement>(`[data-uid="${CSS.escape(target.id)}"]`);
    if (!el) return;
    el.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    setFlashId(target.id);
    const t = window.setTimeout(() => setFlashId(null), FLASH_MS);
    return () => window.clearTimeout(t);
  }, [target]);

  // Playback: keep the line being heard in view.
  useEffect(() => {
    if (!heardId) return;
    const box = scrollRef.current;
    const el = box?.querySelector<HTMLElement>(`[data-uid="${CSS.escape(heardId)}"]`);
    if (!box || !el) return;
    const b = box.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (r.top < b.top || r.bottom > b.bottom) box.scrollTop += r.top - b.top - b.height / 3;
  }, [heardId]);

  const jumpToLive = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: prefersReducedMotion() ? "auto" : "smooth" });
    setFollowing(true);
    setUnseen(0);
  };

  return (
    <section className="tx panel" aria-labelledby="tx-title">
      <div className="tx-headbar">
        <h2 id="tx-title" className="tx-title">
          Transcript <span className="tx-count">{utterances.length}</span>
        </h2>
        {live && <span className="tx-hint">{following ? "Following live" : "Paused while you read"}</span>}
      </div>
      <div className="tx-scroll" ref={scrollRef} onScroll={onScroll} tabIndex={0} aria-label="Transcript lines">
        {utterances.length === 0 ? (
          <p className="tx-empty">{emptyText}</p>
        ) : (
          <ol className="tx-list">
            {utterances.map((u) => (
              <Line
                key={u.id}
                u={u}
                colour={colours.get(u.speaker) ?? 7}
                name={nameOf(u.speaker)}
                heard={u.id === heardId}
                flash={u.id === flashId}
                canPlay={canPlay}
                onSeek={onSeek}
              />
            ))}
          </ol>
        )}
        {live && !following && (
          <button type="button" className="btn btn--primary tx-jump" onClick={jumpToLive}>
            <ArrowDown aria-hidden="true" />
            Jump to live{unseen > 0 ? ` (${unseen} new)` : ""}
          </button>
        )}
      </div>
    </section>
  );
}

interface LineProps {
  u: Utterance;
  colour: number;
  name: string;
  heard: boolean;
  flash: boolean;
  canPlay: boolean;
  onSeek: (u: Utterance) => void;
}

const Line = memo(function Line({ u, colour, name, heard, flash, canPlay, onSeek }: LineProps) {
  const unsure = u.confidence < 0.5;
  const confPct = Math.round(Math.max(0, Math.min(1, u.confidence)) * 100);
  const cls = ["tx-line", "spk", `spk-${colour}`, heard && "is-heard", flash && "is-flash"].filter(Boolean).join(" ");
  return (
    <li className={cls} data-uid={u.id} aria-current={heard ? "true" : undefined}>
      <div className="tx-meta">
        <span className="tx-name">{name}</span>
        {unsure && (
          <span className="tx-unsure" title={`Speaker uncertain: ${confPct}% confidence this was ${name}`}>
            <span aria-hidden="true">?</span>
            <span className="visually-hidden">Speaker uncertain, {confPct}% confidence</span>
          </span>
        )}
        <button
          type="button"
          className="tx-time tc"
          onClick={() => onSeek(u)}
          aria-label={canPlay ? `Play from ${timecode(u.t_start)}` : `Show ${timecode(u.t_start)} in the lanes`}
        >
          {timecode(u.t_start)}
        </button>
      </div>
      <p className="tx-text quote">{u.text}</p>
    </li>
  );
});
