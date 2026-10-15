import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";
import { Pencil } from "lucide-react";
import type { Speaker, Utterance } from "../../api/types";
import { duration, speakerName, timecode } from "../../lib/format";
import { minTrackPx, pct, rulerTicks } from "./timeline";
import "./SpeakerLanes.css";

export type HeadMode = "live" | "playback" | "none";

interface Props {
  speakers: Speaker[];
  utterances: Utterance[];
  colours: Map<string, number>;
  domain: number;
  /** Read once per animation frame; null hides the playhead. */
  readHead: () => number | null;
  headMode: HeadMode;
  /** Keep the playhead in view while it moves (live or playing). */
  follow: boolean;
  fresh: ReadonlySet<string>;
  activeId: string | null;
  onPick: (u: Utterance) => void;
  /** Click on empty track or ruler; omitted when there is nothing to scrub. */
  onScrub?: (t: number) => void;
  onRename: (track: string, name: string) => Promise<unknown>;
  meta?: ReactNode;
  emptyText: string;
}

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s);

function nearestByStart(list: Utterance[], t: number): Utterance | undefined {
  let best: Utterance | undefined;
  for (const u of list) if (!best || Math.abs(u.t_start - t) < Math.abs(best.t_start - t)) best = u;
  return best;
}

export default function SpeakerLanes(props: Props) {
  const { speakers, utterances, colours, domain, fresh, activeId, onPick, onScrub, onRename, meta, emptyText } = props;
  const scrollRef = useRef<HTMLDivElement>(null);
  const tracksRef = useRef<HTMLDivElement>(null);
  const blockEls = useRef(new Map<string, HTMLButtonElement>());
  const [roving, setRoving] = useState<Record<string, string>>({});
  const [renameError, setRenameError] = useState<string | null>(null);

  const lanes = useMemo(() => {
    const byTrack = new Map<string, Utterance[]>(speakers.map((s) => [s.track, []]));
    for (const u of utterances) byTrack.get(u.speaker)?.push(u);
    return speakers.map((s) => {
      const list = (byTrack.get(s.track) ?? []).slice().sort((a, b) => a.t_start - b.t_start);
      const talk = list.reduce((sum, u) => sum + Math.max(0, u.t_end - u.t_start), 0);
      return { speaker: s, list, talk, colour: colours.get(s.track) ?? 7, name: speakerName(s) };
    });
  }, [speakers, utterances, colours]);

  const ticks = useMemo(() => rulerTicks(domain), [domain]);

  const timeAt = useCallback(
    (clientX: number): number | null => {
      const el = tracksRef.current;
      if (!el) return null;
      const r = el.getBoundingClientRect();
      if (r.width <= 0) return null;
      return Math.max(0, Math.min(domain, ((clientX - r.left) / r.width) * domain));
    },
    [domain],
  );

  const scrub = (e: MouseEvent) => {
    if (!onScrub) return;
    const t = timeAt(e.clientX);
    if (t !== null) onScrub(t);
  };

  const lanesRef = useRef(lanes);
  useLayoutEffect(() => {
    lanesRef.current = lanes;
  }, [lanes]);

  const onBlockKey = useCallback((e: KeyboardEvent<HTMLButtonElement>, li: number, bi: number) => {
    const all = lanesRef.current;
    const list = all[li]?.list ?? [];
    let target: Utterance | undefined;
    switch (e.key) {
      case "ArrowRight":
        target = list[bi + 1];
        break;
      case "ArrowLeft":
        target = list[bi - 1];
        break;
      case "Home":
        target = list[0];
        break;
      case "End":
        target = list[list.length - 1];
        break;
      case "ArrowUp":
      case "ArrowDown": {
        const dir = e.key === "ArrowUp" ? -1 : 1;
        for (let j = li + dir; j >= 0 && j < all.length; j += dir) {
          if (all[j].list.length) {
            target = nearestByStart(all[j].list, list[bi].t_start);
            break;
          }
        }
        break;
      }
      default:
        return;
    }
    e.preventDefault();
    if (target) blockEls.current.get(target.id)?.focus();
  }, []);

  const blocks = useMemo(
    () =>
      lanes.map((lane, li) => {
        const tabId = lane.list.some((u) => u.id === roving[lane.speaker.track]) ? roving[lane.speaker.track] : lane.list[0]?.id;
        return lane.list.map((u, bi) => {
          const unsure = u.confidence < 0.5;
          const cls = ["lane-block", fresh.has(u.id) && "is-fresh", u.id === activeId && "is-active", unsure && "is-unsure"]
            .filter(Boolean)
            .join(" ");
          return (
            <button
              key={u.id}
              type="button"
              ref={(el) => {
                if (el) blockEls.current.set(u.id, el);
                else blockEls.current.delete(u.id);
              }}
              className={cls}
              style={{ left: `${pct(u.t_start, domain)}%`, width: `${pct(u.t_end - u.t_start, domain)}%` }}
              tabIndex={u.id === tabId ? 0 : -1}
              aria-label={`${lane.name}, ${timecode(u.t_start)} to ${timecode(u.t_end)}${unsure ? ", speaker uncertain" : ""}: ${clip(u.text, 90)}`}
              title={`${timecode(u.t_start)}  ${clip(u.text, 140)}`}
              onClick={(e) => {
                e.stopPropagation();
                onPick(u);
              }}
              onFocus={() => setRoving((r) => (r[lane.speaker.track] === u.id ? r : { ...r, [lane.speaker.track]: u.id }))}
              onKeyDown={(e) => onBlockKey(e, li, bi)}
            />
          );
        });
      }),
    [lanes, domain, fresh, activeId, roving, onPick, onBlockKey],
  );

  const innerStyle = { "--track-min": `${minTrackPx(domain)}px` } as CSSProperties;

  return (
    <section className="lanes panel" aria-labelledby="lanes-title">
      <div className="lanes-head">
        <h2 id="lanes-title" className="lanes-title">
          Speakers <span className="lanes-count">{speakers.length}</span>
        </h2>
        {meta}
      </div>
      {renameError && (
        <p className="lanes-error" role="alert">
          {renameError}
        </p>
      )}
      {lanes.length === 0 ? (
        <p className="lanes-empty">{emptyText}</p>
      ) : (
        <div className="lanes-scroll" ref={scrollRef}>
          <div className="lanes-inner" style={innerStyle}>
            <div className="lanes-names">
              <div className="lanes-corner" aria-hidden="true" />
              {lanes.map((lane) => (
                <LaneName
                  key={lane.speaker.track}
                  speaker={lane.speaker}
                  colour={lane.colour}
                  talkS={lane.talk}
                  onRename={onRename}
                  onError={setRenameError}
                />
              ))}
            </div>
            <div className={`lanes-tracks${onScrub ? " can-scrub" : ""}`} ref={tracksRef}>
              <div className="lanes-ruler" onClick={scrub} aria-hidden="true">
                {ticks.map((t) => (
                  <span key={t} className="lanes-tick tc" style={{ left: `${pct(t, domain)}%` }}>
                    {timecode(t)}
                  </span>
                ))}
              </div>
              <div className="lanes-gridlines" aria-hidden="true">
                {ticks.map((t) => (
                  <span key={t} style={{ left: `${pct(t, domain)}%` }} />
                ))}
              </div>
              <Playhead
                readHead={props.readHead}
                mode={props.headMode}
                domain={domain}
                follow={props.follow}
                scrollRef={scrollRef}
                tracksRef={tracksRef}
              />
              {lanes.map((lane, li) => (
                <div
                  key={lane.speaker.track}
                  className={`lane-track spk spk-${lane.colour}`}
                  role="group"
                  aria-label={`${lane.name}: ${lane.list.length} ${lane.list.length === 1 ? "turn" : "turns"}. Arrow keys move between turns.`}
                  onClick={scrub}
                >
                  {blocks[li]}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

interface LaneNameProps {
  speaker: Speaker;
  colour: number;
  talkS: number;
  onRename: (track: string, name: string) => Promise<unknown>;
  onError: (msg: string | null) => void;
}

function LaneName({ speaker, colour, talkS, onRename, onError }: LaneNameProps) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const settled = useRef(false);
  const name = speakerName(speaker);

  const finish = async (value: string | null) => {
    if (settled.current) return;
    settled.current = true;
    setEditing(false);
    const v = value?.trim() ?? "";
    if (!v || v === (speaker.name ?? "")) return;
    setSaving(true);
    try {
      await onRename(speaker.track, v);
      onError(null);
    } catch (e) {
      onError(`Couldn't rename ${speaker.label || name}: ${e instanceof Error ? e.message : String(e)}. Click the name to try again.`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={`lane-name spk spk-${colour}`}>
      <div className="lane-name-row">
        <span className="lane-swatch" aria-hidden="true" />
        {editing ? (
          <input
            className="field lane-input"
            defaultValue={speaker.name ?? ""}
            placeholder={speaker.label || name}
            aria-label={`Name for ${speaker.label || name}`}
            maxLength={60}
            autoFocus
            onFocus={(e) => e.currentTarget.select()}
            onKeyDown={(e) => {
              if (e.key === "Enter") void finish(e.currentTarget.value);
              if (e.key === "Escape") void finish(null);
            }}
            onBlur={(e) => void finish(e.currentTarget.value)}
          />
        ) : (
          <button
            type="button"
            className="lane-name-btn"
            title="Rename this speaker"
            aria-label={`${name}. Rename`}
            disabled={saving}
            onClick={() => {
              settled.current = false;
              setEditing(true);
            }}
          >
            <span className="lane-name-text">{name}</span>
            <Pencil aria-hidden="true" />
          </button>
        )}
      </div>
      <span className="lane-talk">
        {saving ? (
          "Saving name…"
        ) : (
          <>
            <span className="lane-talk-verb">Spoke </span>
            {duration(talkS)}
          </>
        )}
      </span>
    </div>
  );
}

interface PlayheadProps {
  readHead: () => number | null;
  mode: HeadMode;
  domain: number;
  follow: boolean;
  scrollRef: React.RefObject<HTMLDivElement | null>;
  tracksRef: React.RefObject<HTMLDivElement | null>;
}

const USER_SCROLL_GRACE_MS = 4000;

function Playhead({ readHead, mode, domain, follow, scrollRef, tracksRef }: PlayheadProps) {
  const lineRef = useRef<HTMLDivElement>(null);
  const flagRef = useRef<HTMLSpanElement>(null);
  const domainRef = useRef(domain);
  const followRef = useRef(follow);
  useLayoutEffect(() => {
    domainRef.current = domain;
    followRef.current = follow;
  }, [domain, follow]);

  // Don't fight someone who is scrolling the lanes by hand.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const mark = () => {
      el.dataset.userScrollAt = String(Date.now());
    };
    el.addEventListener("wheel", mark, { passive: true });
    el.addEventListener("touchstart", mark, { passive: true });
    el.addEventListener("pointerdown", mark, { passive: true });
    return () => {
      el.removeEventListener("wheel", mark);
      el.removeEventListener("touchstart", mark);
      el.removeEventListener("pointerdown", mark);
    };
  }, [scrollRef]);

  useEffect(() => {
    const line = lineRef.current;
    if (!line) return;
    if (mode === "none") {
      line.hidden = true;
      return;
    }
    let raf = 0;
    let lastPct = -1;
    let lastSec = -1;
    const frame = () => {
      const t = readHead();
      if (t === null || !Number.isFinite(t)) {
        line.hidden = true;
      } else {
        line.hidden = false;
        const p = pct(t, domainRef.current);
        if (Math.abs(p - lastPct) > 0.005) {
          line.style.left = `${p}%`;
          lastPct = p;
          if (followRef.current) keepInView(p);
        }
        const sec = Math.floor(t);
        if (sec !== lastSec && flagRef.current) {
          flagRef.current.textContent = timecode(t);
          lastSec = sec;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    const keepInView = (p: number) => {
      const sc = scrollRef.current;
      const tracks = tracksRef.current;
      if (!sc || !tracks || sc.scrollWidth <= sc.clientWidth + 1) return;
      if (Date.now() - Number(sc.dataset.userScrollAt ?? 0) < USER_SCROLL_GRACE_MS) return;
      const namesW = tracks.offsetLeft;
      const x = namesW + (tracks.offsetWidth * p) / 100;
      const viewStart = sc.scrollLeft + namesW;
      const viewEnd = sc.scrollLeft + sc.clientWidth;
      if (x < viewStart + 16 || x > viewEnd - 48) {
        sc.scrollLeft = Math.max(0, x - namesW - (sc.clientWidth - namesW) * 0.3);
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [mode, readHead, scrollRef, tracksRef]);

  return (
    <div ref={lineRef} className={`lanes-playhead is-${mode}`} aria-hidden="true" hidden>
      <span ref={flagRef} className="lanes-playhead-flag tc" />
    </div>
  );
}
