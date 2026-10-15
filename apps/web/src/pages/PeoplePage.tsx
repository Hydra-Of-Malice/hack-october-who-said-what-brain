import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import { api } from "../api/client";
import type { Person, WhoKnows } from "../api/types";
import { duration } from "../lib/format";
import EvidenceQuote from "../components/ask/EvidenceQuote";
import { errorText, titleMap, useMeetings } from "../components/ask/meetingData";
import { useLiveRefresh } from "../components/ask/useLiveRefresh";
import "./memoryPages.css";
import "./PeoplePage.css";

type SortKey = "name" | "meetings" | "utterances" | "talk_time_s" | "open_items";
const COLUMNS: { key: SortKey; label: string; numeric: boolean }[] = [
  { key: "name", label: "Name", numeric: false },
  { key: "meetings", label: "Meetings", numeric: true },
  { key: "utterances", label: "Utterances", numeric: true },
  { key: "talk_time_s", label: "Talk time", numeric: true },
  { key: "open_items", label: "Open items", numeric: true },
];

export default function PeoplePage() {
  const [people, setPeople] = useState<Person[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "talk_time_s", desc: true });

  useEffect(() => {
    let live = true;
    api
      .people()
      .then((list) => {
        if (!live) return;
        setPeople(list);
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
    events: (e) => e.type === "speaker.named" || e.type === "meeting.ended" || e.type === "action.updated",
    refresh: ["action_items"],
    delayMs: 1200,
  });

  const sorted = useMemo(() => {
    const list = [...(people ?? [])];
    const { key, desc } = sort;
    list.sort((a, b) => {
      const r = key === "name" ? a.name.localeCompare(b.name) : (a[key] as number) - (b[key] as number);
      return desc ? -r : r;
    });
    return list;
  }, [people, sort]);
  const maxTalk = Math.max(1, ...(people ?? []).map((p) => p.talk_time_s));

  const sortBy = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: key !== "name" }));

  return (
    <div className="mp">
      <header className="mp-head">
        <div>
          <h1>People</h1>
          <p className="mp-sub">
            Everyone the brain has heard, across meetings. Name a speaker in a meeting to link their voice to a person
            here.
          </p>
        </div>
      </header>

      <div className="pp-grid">
        <section className="panel pp-people" aria-labelledby="pp-people-h">
          <h2 id="pp-people-h" className="visually-hidden">
            People
          </h2>
          {error ? (
            <div className="mp-alert pp-pad" role="alert">
              <p>People didn't load. {error}</p>
              <button type="button" className="btn mp-mini" onClick={refetch}>
                Try again
              </button>
            </div>
          ) : null}
          {!people && !error ? <p className="mp-state pp-pad">Loading people…</p> : null}
          {people && !people.length ? (
            <div className="mp-empty">
              <h2>Nobody yet</h2>
              <p>People appear once a meeting has been recorded. Unnamed voices show as "Speaker 1", "Speaker 2" until you name them.</p>
            </div>
          ) : null}
          {sorted.length ? (
            <div className="pp-scroll">
              <table className="pp-table">
                <thead>
                  <tr>
                    {COLUMNS.map((c) => (
                      <th
                        key={c.key}
                        scope="col"
                        className={c.numeric ? "is-num" : undefined}
                        aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}
                      >
                        <button type="button" className="pp-sort" onClick={() => sortBy(c.key)}>
                          {c.label}
                          {sort.key === c.key ? (
                            sort.desc ? (
                              <ArrowDown aria-hidden="true" />
                            ) : (
                              <ArrowUp aria-hidden="true" />
                            )
                          ) : null}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((p) => (
                    <tr key={p.id}>
                      <th scope="row" className="pp-name">
                        {p.name}
                      </th>
                      <td className="is-num">{p.meetings}</td>
                      <td className="is-num">{p.utterances}</td>
                      <td className="is-num">
                        <span className="pp-talk">
                          <span className="pp-bar" aria-hidden="true">
                            <span style={{ inlineSize: `${Math.max(2, (p.talk_time_s / maxTalk) * 100)}%` }} />
                          </span>
                          {duration(p.talk_time_s)}
                        </span>
                      </td>
                      <td className="is-num">
                        {p.open_items ? (
                          <Link to={`/commitments?owner=${encodeURIComponent(p.name)}`}>{p.open_items} open</Link>
                        ) : (
                          <span className="pp-none">None</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>

        <WhoKnowsPanel />
      </div>
    </div>
  );
}

function WhoKnowsPanel() {
  const [topic, setTopic] = useState("");
  const [asked, setAsked] = useState<string | null>(null);
  const [results, setResults] = useState<WhoKnows[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { meetings } = useMeetings();
  const titles = useMemo(() => titleMap(meetings), [meetings]);

  const search = async (e: FormEvent) => {
    e.preventDefault();
    const q = topic.trim();
    if (!q) return;
    setAsked(q);
    setLoading(true);
    setError(null);
    try {
      setResults(await api.whoKnows(q));
    } catch (err) {
      setResults(null);
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel pp-who" aria-labelledby="pp-who-h">
      <h2 id="pp-who-h">Who knows about…</h2>
      <p className="pp-who__sub">Find who has talked most about a topic, with what they said.</p>
      <form className="pp-who__form" onSubmit={search} role="search">
        <label htmlFor="pp-topic" className="visually-hidden">
          Topic
        </label>
        <input
          id="pp-topic"
          className="field"
          placeholder="A topic, like pricing or the launch date"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          autoComplete="off"
        />
        <button type="submit" className="btn btn--primary" disabled={!topic.trim() || loading}>
          <Search aria-hidden="true" />
          Find people
        </button>
      </form>

      <div aria-live="polite" className="pp-who__out">
        {loading ? <p className="mp-state">Searching the meetings…</p> : null}
        {error ? (
          <div className="mp-alert" role="alert">
            <p>The search didn't run. {error}</p>
          </div>
        ) : null}
        {!loading && results && asked ? (
          results.length ? (
            <ol className="pp-results">
              {results.map((r) => (
                <li key={`${r.person_id ?? r.name}-${r.meeting_id}`} className="pp-result">
                  <p className="pp-result__head">
                    <span className="pp-result__name">{r.name}</span>
                    <span className="pp-result__hits">{r.hits === 1 ? "1 mention" : `${r.hits} mentions`}</span>
                  </p>
                  {r.sample ? (
                    <EvidenceQuote
                      size="sm"
                      text={r.sample.text}
                      track={r.sample.speaker}
                      speakerName={r.name}
                      meetingId={r.sample.meeting_id || r.meeting_id}
                      meetingTitle={titles.get(r.sample.meeting_id || r.meeting_id) ?? null}
                      t={r.sample.t_start}
                      utteranceId={r.sample.id}
                    />
                  ) : null}
                </li>
              ))}
            </ol>
          ) : (
            <p className="mp-state">Nobody has talked about "{asked}" yet. Try a broader word.</p>
          )
        ) : null}
      </div>
    </section>
  );
}
