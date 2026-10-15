import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { loadMeetingDetail, loadMeetings } from "./meetingData";

const UNNAMED = /^speaker\s*\d+$/i;

interface SuggestionState {
  questions: string[];
  status: "ok" | "no-meetings" | "offline";
}

/** Starter questions built from what is actually in the graph. */
async function buildSuggestions(): Promise<SuggestionState> {
  const [peopleRes, meetingsRes] = await Promise.allSettled([api.people(), loadMeetings()]);
  if (meetingsRes.status === "rejected") return { questions: [], status: "offline" };
  const meetings = meetingsRes.value;
  if (!meetings.length) return { questions: [], status: "no-meetings" };

  const questions: string[] = [];
  const people = peopleRes.status === "fulfilled" ? peopleRes.value.filter((p) => p.name && !UNNAMED.test(p.name)) : [];
  people.sort((a, b) => b.open_items - a.open_items || b.talk_time_s - a.talk_time_s);
  if (people[0]) questions.push(`What did ${people[0].name} commit to?`);

  const latest = meetings.find((m) => m.utterance_count > 0) ?? meetings[0];
  try {
    const detail = await loadMeetingDetail(latest.id);
    const topic = [...detail.topics].sort((a, b) => b.mentions - a.mentions)[0];
    if (topic) questions.push(`What did we decide about ${topic.name}?`);
  } catch {
    /* the topic suggestion is optional */
  }
  questions.push("Summarise the latest meeting");
  if (people[1]) questions.push(`What has ${people[1].name} said about deadlines?`);
  if (questions.length < 3) questions.push("Which action items are still open?");
  return { questions: Array.from(new Set(questions)).slice(0, 4), status: "ok" };
}

export default function Suggestions({ onPick }: { onPick: (question: string) => void }) {
  const [state, setState] = useState<SuggestionState | null>(null);

  useEffect(() => {
    let live = true;
    buildSuggestions()
      .then((s) => {
        if (live) setState(s);
      })
      .catch(() => {
        if (live) setState({ questions: [], status: "offline" });
      });
    return () => {
      live = false;
    };
  }, []);

  if (!state) return <p className="ask-sugg__note">Finding questions to start with…</p>;
  if (state.status === "offline") {
    return <p className="ask-sugg__note">The server isn't answering, so there are no suggestions yet. Check that the API is running.</p>;
  }
  if (state.status === "no-meetings") {
    return <p className="ask-sugg__note">No meetings yet. Record or upload one, then ask about it here.</p>;
  }
  return (
    <ul className="ask-sugg" aria-label="Suggested questions">
      {state.questions.map((q) => (
        <li key={q}>
          <button type="button" className="ask-sugg__btn" onClick={() => onPick(q)}>
            {q}
          </button>
        </li>
      ))}
    </ul>
  );
}
