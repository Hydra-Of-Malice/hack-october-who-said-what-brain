import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Play } from "lucide-react";
import { api } from "../api/client";
import type { Decision, MeetingDetail } from "../api/types";
import { speakerName, timecode } from "../lib/format";
import EvidenceQuote from "../components/ask/EvidenceQuote";
import { errorText, loadDetails, titleMap, trackName, useMeetings, utteranceIndex } from "../components/ask/meetingData";
import { meetingPath, useJump } from "../components/ask/useJump";
import { useLiveRefresh } from "../components/ask/useLiveRefresh";
import "./memoryPages.css";
import "./DecisionsPage.css";

const SHOWN_EVIDENCE = 2;

/** Newest decision first, followed by the decisions it replaced (oldest last). */
function buildChains(decisions: Decision[]): Decision[][] {
  const byId = new Map(decisions.map((d) => [d.id, d]));
  const heads = decisions.filter((d) => !d.superseded_by || !byId.has(d.superseded_by));
  return heads.map((head) => {
    const chain = [head];
    const seen = new Set([head.id]);
    let cur = head;
    while (cur.supersedes && byId.has(cur.supersedes) && !seen.has(cur.supersedes)) {
      cur = byId.get(cur.supersedes)!;
      seen.add(cur.id);
      chain.push(cur);
    }
    return chain;
  });
}

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState<Decision[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [details, setDetails] = useState<Map<string, MeetingDetail>>(new Map());
  const { meetings } = useMeetings();
  const titles = useMemo(() => titleMap(meetings), [meetings]);

  useEffect(() => {
    let live = true;
    api
      .decisions()
      .then((list) => {
        if (!live) return;
        setDecisions(list);
        setError(null);
      })
      .catch((e) => {
        if (live) setError(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [nonce]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  useLiveRefresh(refetch, {
    events: (e) => e.type === "insight.created" && (e.data as { kind?: string } | null)?.kind === "decision",
    refresh: ["decisions"],
  });

  const meetingKey = useMemo(
    () => Array.from(new Set((decisions ?? []).map((d) => d.meeting_id))).sort().join(","),
    [decisions],
  );
  useEffect(() => {
    if (!meetingKey) return;
    let live = true;
    loadDetails(meetingKey.split(","))
      .then((d) => {
        if (live) setDetails(d);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [meetingKey]);

  const chains = useMemo(() => buildChains(decisions ?? []), [decisions]);
  const titleOf = (meetingId: string) => titles.get(meetingId) ?? details.get(meetingId)?.meeting.title ?? "a meeting";
  const replacedCount = (decisions ?? []).filter((d) => d.status === "superseded").length;

  return (
    <div className="mp">
      <header className="mp-head">
        <div>
          <h1>Decisions</h1>
          <p className="mp-sub">
            What was decided, newest first. When a later meeting changes a decision, the earlier version stays underneath
            as history.
          </p>
          {decisions ? (
            <p className="mp-count">
              {chains.length === 1 ? "1 decision" : `${chains.length} decisions`}
              {replacedCount ? `, ${replacedCount} replaced along the way` : ""}
            </p>
          ) : null}
        </div>
      </header>

      {error ? (
        <div className="mp-alert" role="alert">
          <p>Decisions didn't load. {error}</p>
          <button type="button" className="btn mp-mini" onClick={refetch}>
            Try again
          </button>
        </div>
      ) : null}

      {!decisions && !error ? <p className="mp-state">Loading decisions…</p> : null}

      {decisions && !chains.length ? (
        <div className="panel mp-empty">
          <h2>No decisions yet</h2>
          <p>
            When a meeting settles something ("we ship on the 3rd"), Gemma 4 records it here with the words that settled
            it. If a later meeting changes it, both versions are kept.
          </p>
        </div>
      ) : null}

      <ol className="dc-list">
        {chains.map((chain) => (
          <li key={chain[0].id} className="panel dc">
            <DecisionEntry decision={chain[0]} title={titleOf(chain[0].meeting_id)} details={details} head />
            {chain.length > 1 ? (
              <ol className="dc-history" aria-label="Earlier versions">
                {chain.slice(1).map((d, i) => (
                  <li key={d.id} className="dc-history__item">
                    <p className="dc-replaced">Replaced on {titleOf(chain[i].meeting_id)}</p>
                    <DecisionEntry decision={d} title={titleOf(d.meeting_id)} details={details} />
                  </li>
                ))}
              </ol>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

interface EntryProps {
  decision: Decision;
  title: string;
  details: Map<string, MeetingDetail>;
  head?: boolean;
}

function DecisionEntry({ decision: d, title, details, head }: EntryProps) {
  const jump = useJump();
  const [showAll, setShowAll] = useState(false);
  const utterances = useMemo(() => utteranceIndex(details), [details]);
  const evidence = d.evidence.map((id) => utterances.get(id)).filter((u) => !!u);
  const shown = showAll ? evidence : evidence.slice(0, SHOWN_EVIDENCE);
  const proposer = d.proposed_by ? trackName(details, d.meeting_id, d.proposed_by) : null;
  const current = d.status === "active";

  return (
    <article className={`dc-entry${head ? " dc-entry--head" : ""}`}>
      {head ? <p className={`dc-status${current ? " dc-status--current" : ""}`}>{current ? "Current" : "Replaced"}</p> : null}
      <p className="dc-text">{d.text}</p>
      <div className="mp-meta dc-meta">
        <Link to={meetingPath(d.meeting_id)}>{title}</Link>
        {proposer ? <span>Proposed by {proposer}</span> : null}
        <button
          type="button"
          className="btn btn--ghost mp-mini"
          onClick={() => jump({ meeting_id: d.meeting_id, t: d.made_at, utterance_id: d.evidence[0] })}
          aria-label={`Play the moment this was decided, ${timecode(d.made_at)} in ${title}`}
        >
          <Play aria-hidden="true" />
          <span className="tc">{timecode(d.made_at)}</span>
        </button>
      </div>
      {shown.length ? (
        <ul className="dc-evidence" aria-label="Evidence">
          {shown.map((u) => (
            <li key={u.id}>
              <EvidenceQuote
                size="sm"
                text={u.text}
                track={u.speaker}
                speakerName={u.speaker_name || trackName(details, u.meeting_id, u.speaker) || speakerName(u)}
                meetingId={u.meeting_id}
                t={u.t_start}
                utteranceId={u.id}
              />
            </li>
          ))}
        </ul>
      ) : null}
      {evidence.length > SHOWN_EVIDENCE ? (
        <button type="button" className="btn btn--ghost mp-mini" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show fewer quotes" : `Show ${evidence.length - SHOWN_EVIDENCE} more quotes`}
        </button>
      ) : null}
    </article>
  );
}
