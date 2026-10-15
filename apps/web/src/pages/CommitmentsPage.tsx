import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Play } from "lucide-react";
import { api } from "../api/client";
import { useBusEvents } from "../api/events";
import type { ActionItem, Utterance } from "../api/types";
import { speakerIndex, timecode, when } from "../lib/format";
import { errorText, loadDetails, utteranceIndex, utteranceStart } from "../components/ask/meetingData";
import { useJump, meetingPath } from "../components/ask/useJump";
import { useLiveRefresh } from "../components/ask/useLiveRefresh";
import "./memoryPages.css";
import "./CommitmentsPage.css";

type Filter = "open" | "done" | "all";
const FILTERS: { value: Filter; label: string }[] = [
  { value: "open", label: "Open" },
  { value: "done", label: "Done" },
  { value: "all", label: "All" },
];

interface Group {
  owner: string | null;
  items: ActionItem[];
}

function groupByOwner(items: ActionItem[]): Group[] {
  const by = new Map<string, ActionItem[]>();
  for (const it of items) {
    const key = it.owner?.trim() ?? "";
    by.set(key, [...(by.get(key) ?? []), it]);
  }
  const order = (a: ActionItem, b: ActionItem) =>
    (a.status === b.status ? 0 : a.status === "open" ? -1 : 1) || b.created_at - a.created_at;
  const named = Array.from(by.entries())
    .filter(([k]) => k)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([owner, list]) => ({ owner, items: list.sort(order) }));
  const unassigned = by.get("");
  return unassigned ? [...named, { owner: null, items: unassigned.sort(order) }] : named;
}

export default function CommitmentsPage() {
  const [params, setParams] = useSearchParams();
  const filter: Filter = (["open", "done", "all"] as const).find((f) => f === params.get("status")) ?? "open";
  const ownerFilter = params.get("owner");
  const [items, setItems] = useState<ActionItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [utterances, setUtterances] = useState<Map<string, Utterance>>(new Map());
  const jump = useJump();

  useEffect(() => {
    let live = true;
    api
      .actionItems(filter === "all" ? {} : { status: filter })
      .then((list) => {
        if (!live) return;
        setItems(list);
        setError(null);
      })
      .catch((e) => {
        if (live) setError(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [filter, nonce]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  useLiveRefresh(refetch, {
    events: (e) => e.type === "insight.created" && (e.data as { kind?: string } | null)?.kind === "action_item",
    refresh: ["action_items"],
  });
  // Status changes from anywhere (including a spoken "mark it done") patch the row in place.
  useBusEvents(
    (e) => {
      const updated = e.data as ActionItem | null;
      if (!updated || typeof updated.id !== "string") return;
      setItems((xs) => (xs ? xs.map((x) => (x.id === updated.id ? { ...x, ...updated } : x)) : xs));
    },
    (e) => e.type === "action.updated",
  );

  const meetingKey = useMemo(() => Array.from(new Set((items ?? []).map((i) => i.meeting_id))).sort().join(","), [items]);
  useEffect(() => {
    if (!meetingKey) return;
    let live = true;
    loadDetails(meetingKey.split(","))
      .then((d) => {
        if (live) setUtterances(utteranceIndex(d));
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [meetingKey]);

  const visible = useMemo(() => {
    const list = items ?? [];
    if (!ownerFilter) return list;
    const want = ownerFilter.toLowerCase();
    return list.filter((i) => (i.owner ?? "").toLowerCase() === want);
  }, [items, ownerFilter]);
  const groups = useMemo(() => groupByOwner(visible), [visible]);
  const openCount = visible.filter((i) => i.status === "open").length;
  const doneCount = visible.length - openCount;

  const setFilter = (f: Filter) => {
    const next = new URLSearchParams(params);
    if (f === "open") next.delete("status");
    else next.set("status", f);
    setParams(next, { replace: true });
  };
  const clearOwner = () => {
    const next = new URLSearchParams(params);
    next.delete("owner");
    setParams(next, { replace: true });
  };

  const toggle = async (item: ActionItem) => {
    const status = item.status === "open" ? "done" : "open";
    setActionError(null);
    setItems((xs) =>
      (xs ?? []).map((x) => (x.id === item.id ? { ...x, status, done_at: status === "done" ? Date.now() : null } : x)),
    );
    try {
      const saved = await api.setActionItem(item.id, status);
      setItems((xs) => (xs ?? []).map((x) => (x.id === item.id ? saved : x)));
    } catch (e) {
      setItems((xs) => (xs ?? []).map((x) => (x.id === item.id ? item : x)));
      setActionError(`Couldn't mark "${item.text}" as ${status}. ${errorText(e)}`);
    }
  };

  const playEvidence = async (item: ActionItem) => {
    const uid = item.evidence[0];
    if (!uid) {
      jump({ meeting_id: item.meeting_id, t: 0 });
      return;
    }
    const known = utterances.get(uid);
    const t = known ? known.t_start : await utteranceStart(item.meeting_id, uid);
    jump({ meeting_id: item.meeting_id, t: t ?? 0, utterance_id: uid });
  };

  const countText =
    filter === "open"
      ? `${openCount} open`
      : filter === "done"
        ? `${doneCount} done`
        : `${openCount} open, ${doneCount} done`;

  return (
    <div className="mp">
      <header className="mp-head">
        <div>
          <h1>Commitments</h1>
          <p className="mp-sub">Action items Gemma 4 picked out of your meetings, grouped by who owns them.</p>
          {items ? <p className="mp-count">{countText}</p> : null}
        </div>
        <fieldset className="mp-seg">
          <legend className="visually-hidden">Show</legend>
          {FILTERS.map((f) => (
            <label key={f.value}>
              <input type="radio" name="commit-filter" checked={filter === f.value} onChange={() => setFilter(f.value)} />
              <span>{f.label}</span>
            </label>
          ))}
        </fieldset>
      </header>

      {ownerFilter ? (
        <p className="mp-filter">
          <span>
            Owned by <strong>{ownerFilter}</strong>
          </span>
          <button type="button" className="btn btn--ghost mp-mini" onClick={clearOwner}>
            Show everyone
          </button>
        </p>
      ) : null}

      <div aria-live="polite">
        {actionError ? (
          <div className="mp-alert" role="alert">
            <p>{actionError}</p>
            <button type="button" className="btn btn--ghost mp-mini" onClick={() => setActionError(null)}>
              Dismiss
            </button>
          </div>
        ) : null}
      </div>

      {error ? (
        <div className="mp-alert" role="alert">
          <p>Commitments didn't load. {error}</p>
          <button type="button" className="btn mp-mini" onClick={refetch}>
            Try again
          </button>
        </div>
      ) : null}

      {!items && !error ? <p className="mp-state">Loading commitments…</p> : null}

      {items && !groups.length ? (
        <div className="panel mp-empty">
          <h2>{emptyTitle(filter, ownerFilter)}</h2>
          <p>
            When someone says they'll do something in a meeting, Gemma 4 writes it here with the quote it came from.
            Mark items done here, or by asking: "mark the pricing deck done".
          </p>
        </div>
      ) : null}

      <div className="cm-groups">
        {groups.map((g) => (
          <section key={g.owner ?? "unassigned"} className="panel cm-group" aria-labelledby={`cm-${g.owner ?? "none"}`}>
            <h2 className="cm-owner" id={`cm-${g.owner ?? "none"}`}>
              {g.owner ?? "Unassigned"}
              <span className="cm-owner__count">
                {g.items.filter((i) => i.status === "open").length} open
              </span>
            </h2>
            <ul className="cm-list">
              {g.items.map((it) => (
                <CommitmentRow
                  key={it.id}
                  item={it}
                  evidence={it.evidence[0] ? (utterances.get(it.evidence[0]) ?? null) : null}
                  onToggle={() => void toggle(it)}
                  onPlay={() => void playEvidence(it)}
                />
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}

function emptyTitle(filter: Filter, owner: string | null): string {
  if (owner) return `No ${filter === "all" ? "" : `${filter} `}items owned by ${owner}`;
  if (filter === "done") return "Nothing marked done yet";
  if (filter === "all") return "No commitments yet";
  return "No open commitments";
}

interface RowProps {
  item: ActionItem;
  evidence: Utterance | null;
  onToggle: () => void;
  onPlay: () => void;
}

function CommitmentRow({ item, evidence, onToggle, onPlay }: RowProps) {
  const done = item.status === "done";
  const id = `cm-item-${item.id}`;
  return (
    <li className={`cm-item${done ? " is-done" : ""}`}>
      <input id={id} type="checkbox" className="cm-check" checked={done} onChange={onToggle} />
      <div className="cm-body">
        <label htmlFor={id} className="cm-text">
          {item.text}
        </label>
        <div className="mp-meta cm-meta">
          <Link to={meetingPath(item.meeting_id)}>{item.meeting_title || "Untitled meeting"}</Link>
          {item.due ? <span>Due {item.due}</span> : null}
          {done && item.done_at ? <span>Done {when(item.done_at)}</span> : null}
          <button type="button" className="btn btn--ghost mp-mini" onClick={onPlay}>
            <Play aria-hidden="true" />
            {item.evidence.length ? "Play evidence" : "Open meeting"}
            {evidence ? <span className="tc">{timecode(evidence.t_start)}</span> : null}
          </button>
        </div>
        {evidence ? (
          <blockquote className={`quote cm-quote spk spk-${speakerIndex(evidence.speaker)}`}>
            <span className="visually-hidden">{evidence.speaker_name ?? "Speaker"} said: </span>
            {evidence.text}
          </blockquote>
        ) : null}
      </div>
    </li>
  );
}
