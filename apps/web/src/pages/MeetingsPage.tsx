import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { useBusEvents } from "../api/events";
import type { MeetingSummary, Utterance } from "../api/types";
import { brand } from "../lib/brand";
import GraphTotals from "../components/meeting/GraphTotals";
import MeetingRow from "../components/meeting/MeetingRow";
import StartMeeting from "../components/meeting/StartMeeting";
import UploadRecording from "../components/meeting/UploadRecording";
import { appendToStrip, invalidateStrip } from "../components/meeting/stripCache";
import { useNow } from "../components/meeting/timeline";
import "./MeetingsPage.css";

const newestFirst = (a: MeetingSummary, b: MeetingSummary) => (Number(b.started_at) || 0) - (Number(a.started_at) || 0);

function upsert(list: MeetingSummary[], m: MeetingSummary): MeetingSummary[] {
  const i = list.findIndex((x) => x.id === m.id);
  if (i === -1) return [m, ...list].sort(newestFirst);
  const next = list.slice();
  next[i] = { ...list[i], ...m };
  return next;
}

export default function MeetingsPage() {
  const [meetings, setMeetings] = useState<MeetingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .meetings()
      .then((list) => {
        setMeetings(Array.isArray(list) ? list.filter((m) => m && typeof m.id === "string") : []);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    document.title = brand.name;
    load();
  }, [load]);

  useBusEvents((e) => {
    const data = e.data as Record<string, unknown> | null;
    if ((e.type === "meeting.started" || e.type === "meeting.ended") && data && typeof data.id === "string") {
      const m = data as unknown as MeetingSummary;
      setMeetings((list) => (list ? upsert(list, m) : list));
      if (e.type === "meeting.ended") invalidateStrip(m.id);
    } else if (e.type === "utterance.created" && data) {
      appendToStrip(data as unknown as Utterance);
    }
  });

  const anyLive = !!meetings?.some((m) => m.status === "live");
  const now = useNow(5000, anyLive);

  return (
    <div className="home">
      <header className="home-head">
        <h1 className="home-title">{brand.name}</h1>
        <p className="home-tagline">{brand.tagline}</p>
        <div className="home-actions">
          <StartMeeting />
          <UploadRecording />
        </div>
      </header>

      <GraphTotals />

      <section className="home-list" aria-labelledby="home-list-title">
        <div className="home-list-head">
          <h2 id="home-list-title" className="home-list-title">
            Meetings {meetings && meetings.length > 0 && <span className="home-count">{meetings.length}</span>}
          </h2>
          {meetings && meetings.length > 0 && <p className="home-list-hint">Each strip shows who spoke when. Open a meeting to hear it.</p>}
        </div>

        {error ? (
          <div className="home-message panel" role="alert">
            <p>Couldn't load meetings: {error}. Check that the API is running, then retry.</p>
            <button type="button" className="btn" onClick={load}>
              Retry
            </button>
          </div>
        ) : meetings === null ? (
          <ul className="mlist panel" aria-busy="true" aria-label="Loading meetings">
            {[0, 1, 2].map((i) => (
              <li key={i} className="mrow mrow--skeleton" aria-hidden="true">
                <div className="mrow-main">
                  <span className="skel skel--title" />
                  <span className="skel skel--meta" />
                </div>
                <div className="mrow-tape">
                  <span className="tape tape--blank" />
                </div>
              </li>
            ))}
          </ul>
        ) : meetings.length === 0 ? (
          <div className="home-message home-empty panel">
            <h3>No meetings yet</h3>
            <p>
              Start a live meeting to capture one from your microphone, or upload a recording. The demo meetings appear here once the
              database has been seeded.
            </p>
          </div>
        ) : (
          <ul className="mlist panel">
            {meetings.map((m) => (
              <MeetingRow key={m.id} m={m} now={now} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
