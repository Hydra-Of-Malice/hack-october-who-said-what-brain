import { useEffect, useState } from "react";
import { Check, ChevronRight, Copy } from "lucide-react";
import { api } from "../../api/client";
import type { AskResult, Subgraph } from "../../api/types";
import { speakerName, timecode } from "../../lib/format";
import GraphView from "../graph/GraphView";
import { formatArgs, ms, timingLabel, viaLabel } from "./askFormat";
import { useCopy } from "./clipboard";
import { errorText } from "./meetingData";

/** "Why this answer": evidence and how it was found, tool calls, Cypher, timings and the evidence subgraph. */
export default function WhyPanel({ result }: { result: AskResult }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="ask-why" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <ChevronRight aria-hidden="true" className="ask-why__chev" />
        Why this answer
      </summary>
      {open ? <WhyBody result={result} /> : null}
    </details>
  );
}

function WhyBody({ result }: { result: AskResult }) {
  const citations = result.citations ?? [];
  const tools = result.tool_trace ?? [];
  const cypher = (result.cypher ?? []).filter((c) => c.trim());
  const timings = Object.entries(result.timings ?? {}).filter(([, v]) => typeof v === "number" && Number.isFinite(v));
  const total = timings.find(([k]) => k === "total")?.[1] ?? Math.max(1, ...timings.map(([, v]) => v));

  return (
    <div className="ask-why__body">
      <section>
        <h4>Evidence and how it was found</h4>
        {citations.length ? (
          <ul className="ask-why__rows">
            {citations.map((c) => (
              <li key={c.utterance_id}>
                <span className="ask-why__who">{speakerName(c)}</span>
                <span className="tc">{timecode(c.t_start)}</span>
                <span className="ask-why__via">{viaLabel(c.via)}</span>
                <span className="ask-why__num">score {c.score.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="ask-why__empty">This answer cites no utterances.</p>
        )}
      </section>

      <section>
        <h4>Tools Gemma called</h4>
        {tools.length ? (
          <ol className="ask-why__tools">
            {tools.map((t, i) => (
              <li key={`${t.tool}-${i}`}>
                <code className="ask-why__tool">{t.tool}</code>
                <span className="ask-why__num">
                  {t.result_count === 1 ? "1 result" : `${t.result_count} results`} in {ms(t.ms)}
                </span>
                {Object.keys(t.args ?? {}).length ? <span className="ask-why__args">{formatArgs(t.args)}</span> : null}
              </li>
            ))}
          </ol>
        ) : (
          <p className="ask-why__empty">No tools were called; Gemma answered from the conversation so far.</p>
        )}
      </section>

      {cypher.length ? <CypherBlock statements={cypher} /> : null}

      {timings.length ? (
        <section>
          <h4>Timings</h4>
          <dl className="ask-why__timings">
            {timings.map(([k, v]) => (
              <div key={k} className="ask-why__timing">
                <dt>{timingLabel(k)}</dt>
                <dd>
                  <span className="ask-why__track" aria-hidden="true">
                    <span className="ask-why__bar" style={{ inlineSize: `${Math.max(2, Math.min(100, (v / total) * 100))}%` }} />
                  </span>
                  <span className="ask-why__num">{ms(v)}</span>
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}

      {citations.length ? <EvidenceGraph utteranceIds={citations.map((c) => c.utterance_id)} /> : null}
    </div>
  );
}

function CypherBlock({ statements }: { statements: string[] }) {
  const [state, copy] = useCopy();
  const text = statements.map((s) => s.trim().replace(/;?$/, ";")).join("\n\n");
  return (
    <section>
      <div className="ask-why__head">
        <h4>Cypher that ran on FalkorDB</h4>
        <button type="button" className="btn btn--ghost ask-mini" onClick={() => copy(text)}>
          {state === "copied" ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
          {state === "copied" ? "Copied" : state === "failed" ? "Copy blocked" : "Copy"}
        </button>
      </div>
      <pre className="ask-why__code" tabIndex={0} aria-label="Cypher statements">
        <code>{text}</code>
      </pre>
    </section>
  );
}

function EvidenceGraph({ utteranceIds }: { utteranceIds: string[] }) {
  const key = utteranceIds.join(",");
  const [graph, setGraph] = useState<Subgraph | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setGraph(null);
    setError(null);
    api
      .graph({ utterance_ids: key.split(",") })
      .then((g) => {
        if (live) setGraph(g);
      })
      .catch((e) => {
        if (live) setError(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [key]);

  return (
    <section>
      <h4>Evidence in the graph</h4>
      {error ? (
        <p className="ask-why__empty">The evidence graph didn't load: {error}</p>
      ) : !graph ? (
        <p className="ask-why__empty">Loading the evidence graph…</p>
      ) : graph.nodes.length ? (
        <div className="ask-why__graph">
          <GraphView
            data={graph}
            height={220}
            fitKey={key}
            label={`Evidence subgraph: ${graph.nodes.length} nodes and ${graph.links.length} links around the cited utterances`}
          />
        </div>
      ) : (
        <p className="ask-why__empty">The graph has no nodes for these utterances.</p>
      )}
    </section>
  );
}
