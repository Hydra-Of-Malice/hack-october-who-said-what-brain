import { useState, type MouseEvent } from "react";
import { Play } from "lucide-react";
import { Link } from "react-router-dom";
import type { ActionItem, Decision, Topic, Utterance } from "../../api/types";
import { timecode } from "../../lib/format";
import { prefersReducedMotion } from "./timeline";
import "./InsightsRail.css";

interface Props {
  decisions: Decision[];
  actionItems: ActionItem[];
  topics: Topic[];
  utterancesById: Map<string, Utterance>;
  colours: Map<string, number>;
  nameOf: (track: string) => string;
  fresh: ReadonlySet<string>;
  live: boolean;
  onEvidence: (u: Utterance) => void;
  onToggleAction: (item: ActionItem, done: boolean) => Promise<void>;
}

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s);

export default function InsightsRail(props: Props) {
  const { decisions, actionItems, topics, live } = props;
  const openCount = actionItems.filter((a) => a.status !== "done").length;
  return (
    <aside className="ins panel" aria-labelledby="ins-title">
      <div className="ins-headbar">
        <h2 id="ins-title" className="ins-title">
          Extracted by Gemma 4
        </h2>
        {live && <span className="ins-hint">Updates as people talk</span>}
      </div>
      <div className="ins-scroll">
        <section className="ins-section" aria-labelledby="ins-dec">
          <h3 id="ins-dec" className="ins-h">
            Decisions <span className="ins-n">{decisions.length}</span>
          </h3>
          {decisions.length === 0 ? (
            <p className="ins-empty">{live ? "No decisions yet. They appear here as they're said." : "No decisions were recorded in this meeting."}</p>
          ) : (
            <ol className="ins-list">
              {decisions.map((d) => (
                <DecisionRow key={d.id} d={d} all={decisions} {...props} />
              ))}
            </ol>
          )}
        </section>

        <section className="ins-section" aria-labelledby="ins-act">
          <h3 id="ins-act" className="ins-h">
            Action items <span className="ins-n">{actionItems.length ? `${openCount} open of ${actionItems.length}` : 0}</span>
          </h3>
          {actionItems.length === 0 ? (
            <p className="ins-empty">
              {live ? "No action items yet. When someone commits to doing something, it lands here." : "Nobody took on an action item in this meeting."}
            </p>
          ) : (
            <ul className="ins-list">
              {actionItems.map((a) => (
                <ActionRow key={a.id} a={a} {...props} />
              ))}
            </ul>
          )}
        </section>

        <section className="ins-section" aria-labelledby="ins-top">
          <h3 id="ins-top" className="ins-h">
            Topics <span className="ins-n">{topics.length}</span>
          </h3>
          {topics.length === 0 ? (
            <p className="ins-empty">{live ? "No topics yet. They're picked out as the conversation moves." : "No topics were picked out."}</p>
          ) : (
            <ul className="ins-topics">
              {[...topics]
                .sort((a, b) => b.mentions - a.mentions)
                .map((t) => (
                  <li key={t.id} className={`ins-chip${props.fresh.has(t.id) ? " is-fresh" : ""}`} title={`Mentioned ${t.mentions} ${t.mentions === 1 ? "time" : "times"}`}>
                    {t.name}
                    <span className="ins-chip-n">
                      <span className="visually-hidden">, mentioned </span>
                      {t.mentions}
                      <span className="visually-hidden"> {t.mentions === 1 ? "time" : "times"}</span>
                    </span>
                  </li>
                ))}
            </ul>
          )}
        </section>
      </div>
    </aside>
  );
}

function focusDecision(e: MouseEvent<HTMLAnchorElement>, id: string) {
  const el = document.getElementById(`dec-${id}`);
  if (!el) return;
  e.preventDefault();
  el.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  el.focus({ preventScroll: true });
  el.classList.remove("is-fresh");
  void el.offsetWidth; // restart the highlight
  el.classList.add("is-fresh");
}

function DecisionRow({ d, all, fresh, nameOf, ...rest }: Props & { d: Decision; all: Decision[] }) {
  const superseded = d.status === "superseded";
  const replacement = d.superseded_by ? all.find((x) => x.id === d.superseded_by) : undefined;
  const cls = ["ins-item", fresh.has(d.id) && "is-fresh", superseded && "is-superseded"].filter(Boolean).join(" ");
  return (
    <li id={`dec-${d.id}`} className={cls} tabIndex={-1}>
      <p className="ins-text">
        {superseded ? (
          <>
            <s>{d.text}</s>
            <span className="visually-hidden"> (replaced)</span>
          </>
        ) : (
          d.text
        )}
      </p>
      {(d.proposed_by || superseded || d.supersedes) && (
        <div className="ins-meta">
          {d.proposed_by && <span>Proposed by {nameOf(d.proposed_by)}</span>}
          {superseded &&
            (replacement ? (
              <span>
                Replaced by{" "}
                <a href={`#dec-${replacement.id}`} onClick={(e) => focusDecision(e, replacement.id)}>
                  {clip(replacement.text, 56)}
                </a>
              </span>
            ) : (
              <span>
                Replaced by <Link to="/decisions">a later decision</Link>
              </span>
            ))}
          {!superseded && d.supersedes && <span>Replaces an earlier decision</span>}
        </div>
      )}
      <Evidence ids={d.evidence} nameOf={nameOf} {...rest} />
    </li>
  );
}

function ActionRow({ a, fresh, nameOf, onToggleAction, ...rest }: Props & { a: ActionItem }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const done = a.status === "done";
  const owner = a.owner_track ? nameOf(a.owner_track) : a.owner;

  const toggle = async (next: boolean) => {
    setPending(true);
    setError(null);
    try {
      await onToggleAction(a, next);
    } catch (e) {
      setError(`Couldn't update this item: ${e instanceof Error ? e.message : String(e)}. Try again.`);
    } finally {
      setPending(false);
    }
  };

  const cls = ["ins-item", "ins-action", fresh.has(a.id) && "is-fresh", done && "is-done"].filter(Boolean).join(" ");
  return (
    <li className={cls}>
      <label className="ins-check">
        <input type="checkbox" checked={done} disabled={pending} onChange={(e) => void toggle(e.currentTarget.checked)} />
        <span className="ins-text">{a.text}</span>
      </label>
      <div className="ins-meta ins-meta--indent">
        <span>
          Owner <strong>{owner || "not named yet"}</strong>
        </span>
        {a.due && (
          <span>
            Due <strong>{a.due}</strong>
          </span>
        )}
      </div>
      {error && (
        <p className="ins-error" role="alert">
          {error}
        </p>
      )}
      <div className="ins-indent">
        <Evidence ids={a.evidence} nameOf={nameOf} {...rest} />
      </div>
    </li>
  );
}

interface EvidenceProps {
  ids: string[];
  utterancesById: Map<string, Utterance>;
  colours: Map<string, number>;
  nameOf: (track: string) => string;
  onEvidence: (u: Utterance) => void;
}

function Evidence({ ids, utterancesById, colours, nameOf, onEvidence }: EvidenceProps) {
  const u = (ids ?? []).map((id) => utterancesById.get(id)).find((x): x is Utterance => !!x);
  if (!u) return null;
  const name = nameOf(u.speaker);
  return (
    <figure className={`ins-ev spk spk-${colours.get(u.speaker) ?? 7}`}>
      <blockquote className="ins-ev-quote quote">{clip(u.text, 180)}</blockquote>
      <figcaption>
        <button type="button" className="ins-ev-play" onClick={() => onEvidence(u)} aria-label={`Play ${name} at ${timecode(u.t_start)}`}>
          <Play aria-hidden="true" />
          <span className="ins-ev-name">{name}</span>
          <span className="tc">{timecode(u.t_start)}</span>
        </button>
      </figcaption>
    </figure>
  );
}
