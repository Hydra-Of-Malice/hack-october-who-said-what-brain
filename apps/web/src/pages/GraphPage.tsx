import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, Play } from "lucide-react";
import { api } from "../api/client";
import type { GraphNode, Subgraph } from "../api/types";
import { timecode, when } from "../lib/format";
import { errorText, titleMap, useMeetings } from "../components/ask/meetingData";
import { meetingPath, useJump } from "../components/ask/useJump";
import { useLiveRefresh } from "../components/ask/useLiveRefresh";
import GraphView from "../components/graph/GraphView";
import KindSwatch from "../components/graph/KindSwatch";
import { KIND_ORDER, kindStyle, nodeSpeakerIndex, truncate } from "../components/graph/graphStyle";
import "./memoryPages.css";
import "./GraphPage.css";

const ALL = "all";
const HIDDEN_PROPS = new Set(["id", "label", "kind", "x", "y", "vx", "vy", "fx", "fy", "index", "embedding", "centroid", "__indexColor"]);
const LIVE_EVENTS = new Set(["graph.write", "insight.created", "speaker.named", "speaker.merged", "utterance.created"]);

function kindsIn(graph: Subgraph | null): { kind: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const n of graph?.nodes ?? []) counts.set(String(n.kind), (counts.get(String(n.kind)) ?? 0) + 1);
  const known = KIND_ORDER.filter((k) => counts.has(k));
  const other = Array.from(counts.keys()).filter((k) => !KIND_ORDER.includes(k)).sort();
  return [...known, ...other].map((kind) => ({ kind, count: counts.get(kind) ?? 0 }));
}

function propValue(key: string, value: unknown): string {
  if (value === null || value === undefined || value === "") return "None";
  if (typeof value === "number") {
    if ((key === "t_start" || key === "t_end" || key === "made_at") && value < 1e7) return timecode(value);
    if (key.endsWith("_at") && value > 1e11) return when(value);
    if (key.endsWith("_s")) return `${Math.round(value)} s`;
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.length > 6 ? `${value.slice(0, 6).join(", ")} and ${value.length - 6} more` : value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function propLabel(key: string): string {
  // "duration_s" -> "Duration": the unit is shown with the value.
  return key
    .replace(/_s$/, "")
    .replace(/_/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase());
}

export default function GraphPage() {
  const [params, setParams] = useSearchParams();
  const { meetings, error: meetingsError } = useMeetings();
  const titles = useMemo(() => titleMap(meetings), [meetings]);
  const requested = params.get("meeting");
  const meetingKey = requested ?? (meetings ? (meetings[0]?.id ?? ALL) : null);

  const [graph, setGraph] = useState<Subgraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set());

  useEffect(() => {
    if (!meetingKey) return;
    let live = true;
    api
      .graph(meetingKey === ALL ? {} : { meeting_id: meetingKey })
      .then((g) => {
        if (!live) return;
        setGraph(g);
        setError(null);
        // Keep the selection across live refreshes when the node still exists.
        setSelected((s) => (s ? (g.nodes.find((n) => n.id === s.id) ?? null) : null));
      })
      .catch((e) => {
        if (live) setError(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [meetingKey, nonce]);

  useEffect(() => {
    setSelected(null);
    setGraph(null);
  }, [meetingKey]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  useLiveRefresh(refetch, {
    events: (e) => LIVE_EVENTS.has(e.type) && (meetingKey === ALL || e.meeting_id === meetingKey),
    refresh: ["meeting", "decisions", "action_items"],
    delayMs: 1500,
  });

  const kinds = useMemo(() => kindsIn(graph), [graph]);
  const nodesById = useMemo(() => new Map((graph?.nodes ?? []).map((n) => [String(n.id), n])), [graph]);

  const pickMeeting = (value: string) => {
    const next = new URLSearchParams(params);
    next.set("meeting", value);
    setParams(next, { replace: true });
  };

  const toggleKind = (kind: string) =>
    setHidden((h) => {
      const next = new Set(h);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });

  const selectById = (id: string) => setSelected(nodesById.get(id) ?? null);
  const title = meetingKey === ALL ? "All meetings" : (titles.get(meetingKey ?? "") ?? "This meeting");
  const noMeetings = meetings !== null && meetings.length === 0;

  return (
    <div className="mp mp--wide gp">
      <header className="mp-head">
        <div>
          <h1>Graph</h1>
          <p className="mp-sub">The FalkorDB graph behind your meetings: who spoke, what they said, and what it led to.</p>
        </div>
        <label className="gp-pick">
          <span>Meeting</span>
          <select
            className="field"
            value={meetingKey ?? ""}
            onChange={(e) => pickMeeting(e.target.value)}
            disabled={!meetings}
          >
            {!meetings ? <option value="">Loading meetings…</option> : null}
            {(meetings ?? []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.title}
                {m.status === "live" ? " (live)" : ""}
              </option>
            ))}
            {meetings ? <option value={ALL}>All meetings</option> : null}
          </select>
        </label>
      </header>

      {meetingsError || error ? (
        <div className="mp-alert" role="alert">
          <p>The graph didn't load. {error ?? meetingsError}</p>
          <button type="button" className="btn mp-mini" onClick={refetch}>
            Try again
          </button>
        </div>
      ) : null}

      {noMeetings ? (
        <div className="panel mp-empty">
          <h2>No graph yet</h2>
          <p>
            Record or upload a meeting. Each speaker, utterance, decision and action item is written to FalkorDB as it
            happens, and appears here.
          </p>
          <Link className="btn" to="/">
            Go to meetings
          </Link>
        </div>
      ) : (
        <>
          {kinds.length ? (
            <ul className="gp-legend" aria-label="Node kinds. Press to show or hide a kind.">
              {kinds.map(({ kind, count }) => (
                <li key={kind}>
                  <button
                    type="button"
                    className="gp-kind-btn"
                    aria-pressed={!hidden.has(kind)}
                    onClick={() => toggleKind(kind)}
                  >
                    <KindSwatch kind={kind} />
                    <span>{count === 1 ? kindStyle(kind).label : kindStyle(kind).plural}</span>
                    <span className="gp-num">{count}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          <div className="gp-body">
            <div className="panel gp-canvas">
              {!graph && !error ? <p className="gp-state">Loading the graph…</p> : null}
              {graph && !graph.nodes.length ? (
                <p className="gp-state">
                  Nothing in the graph for this meeting yet. Utterances appear as people speak.
                </p>
              ) : null}
              {graph && graph.nodes.length ? (
                <GraphView
                  data={graph}
                  hiddenKinds={hidden}
                  selectedId={selected ? String(selected.id) : null}
                  onSelect={setSelected}
                  fitKey={meetingKey ?? undefined}
                  label={`Knowledge graph of ${title}: ${graph.nodes.length} nodes and ${graph.links.length} relationships. Use the panel beside it to browse nodes with the keyboard.`}
                />
              ) : null}
            </div>

            <aside className="panel gp-side" aria-label="Selected node">
              {graph && graph.nodes.length ? (
                <NodePicker graph={graph} value={selected ? String(selected.id) : ""} onPick={selectById} />
              ) : null}
              {selected && graph ? (
                <NodeDetails node={selected} graph={graph} nodesById={nodesById} titles={titles} onPick={selectById} />
              ) : (
                <p className="gp-hint">Select a node in the graph, or pick one above, to see what it holds and what it connects to.</p>
              )}
            </aside>
          </div>
        </>
      )}
    </div>
  );
}

function NodePicker({ graph, value, onPick }: { graph: Subgraph; value: string; onPick: (id: string) => void }) {
  const groups = useMemo(() => {
    const by = new Map<string, GraphNode[]>();
    for (const n of graph.nodes) by.set(String(n.kind), [...(by.get(String(n.kind)) ?? []), n]);
    return kindsIn(graph).map(({ kind }) => ({ kind, nodes: by.get(kind) ?? [] }));
  }, [graph]);
  return (
    <label className="gp-picker">
      <span>Go to a node</span>
      <select className="field" value={value} onChange={(e) => e.target.value && onPick(e.target.value)}>
        <option value="">Choose…</option>
        {groups.map((g) => (
          <optgroup key={g.kind} label={kindStyle(g.kind).plural}>
            {g.nodes.map((n) => (
              <option key={String(n.id)} value={String(n.id)}>
                {truncate(String(n.label ?? n.id), 60)}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

interface DetailsProps {
  node: GraphNode;
  graph: Subgraph;
  nodesById: Map<string, GraphNode>;
  titles: Map<string, string>;
  onPick: (id: string) => void;
}

function NodeDetails({ node, graph, nodesById, titles, onPick }: DetailsProps) {
  const jump = useJump();
  const kind = String(node.kind);
  const id = String(node.id);
  const props = Object.entries(node).filter(([k, v]) => !HIDDEN_PROPS.has(k) && !k.startsWith("__") && typeof v !== "function");
  const text = typeof node.text === "string" ? node.text : null;
  const meetingId = typeof node.meeting_id === "string" ? node.meeting_id : kind === "Meeting" ? id : null;
  const t = typeof node.t_start === "number" ? node.t_start : null;

  const links = graph.links
    .map((l) => {
      const s = String(l.source);
      const tg = String(l.target);
      if (s === id) return { other: tg, type: l.type, out: true };
      if (tg === id) return { other: s, type: l.type, out: false };
      return null;
    })
    .filter((x): x is { other: string; type: string; out: boolean } => !!x && nodesById.has(x.other));

  return (
    <div className="gp-node">
      <p className="gp-node__kind">
        <KindSwatch kind={kind} spk={kind === "Speaker" ? nodeSpeakerIndex(node) : 0} />
        {kindStyle(kind).label}
      </p>
      <h2 className="gp-node__title">{String(node.label ?? id)}</h2>
      {text && text !== node.label ? <blockquote className="quote gp-node__quote">{text}</blockquote> : null}

      {meetingId ? (
        <div className="gp-node__actions">
          {t !== null ? (
            <button
              type="button"
              className="btn mp-mini"
              onClick={() => jump({ meeting_id: meetingId, t, utterance_id: kind === "Utterance" ? id : undefined })}
            >
              <Play aria-hidden="true" />
              Play <span className="tc">{timecode(t)}</span>
            </button>
          ) : null}
          <Link className="btn btn--ghost mp-mini" to={meetingPath(meetingId)}>
            Open {titles.get(meetingId) ?? "the meeting"}
          </Link>
        </div>
      ) : null}

      {props.length ? (
        <dl className="gp-props">
          {props.map(([k, v]) => (
            <div key={k}>
              <dt>{propLabel(k)}</dt>
              <dd>{propValue(k, v)}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      <h3 className="gp-node__h">{links.length === 1 ? "1 connection" : `${links.length} connections`}</h3>
      {links.length ? (
        <ul className="gp-links">
          {links.slice(0, 40).map((l, i) => {
            const other = nodesById.get(l.other)!;
            return (
              <li key={`${l.other}-${l.type}-${i}`}>
                <button type="button" className="gp-link" onClick={() => onPick(l.other)}>
                  <span className="gp-link__rel">
                    {l.out ? null : <ArrowLeft aria-hidden="true" />}
                    <span className="visually-hidden">{l.out ? "Outgoing" : "Incoming"} </span>
                    {l.type}
                    {l.out ? <ArrowRight aria-hidden="true" /> : null}
                  </span>
                  <KindSwatch kind={String(other.kind)} spk={other.kind === "Speaker" ? nodeSpeakerIndex(other) : 0} />
                  <span className="gp-link__label">{truncate(String(other.label ?? other.id), 48)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="gp-hint">No connections in this view.</p>
      )}
      {links.length > 40 ? <p className="gp-hint">Showing 40 of {links.length}.</p> : null}
    </div>
  );
}
