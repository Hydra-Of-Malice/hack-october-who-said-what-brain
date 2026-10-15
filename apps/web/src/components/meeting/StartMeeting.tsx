import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { LoaderCircle, Mic } from "lucide-react";
import { api } from "../../api/client";
import { when } from "../../lib/format";

/** "Start a live meeting": asks for a title inline, creates the meeting, then opens it capturing. */
export default function StartMeeting() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fallback, setFallback] = useState("");

  const openForm = () => {
    setFallback(`Meeting, ${when(Date.now())}`);
    setError(null);
    setOpen(true);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const m = await api.createMeeting(title.trim() || fallback);
      navigate(`/m/${encodeURIComponent(m.id)}?live=1`);
    } catch (err) {
      setError(`Couldn't start the meeting: ${err instanceof Error ? err.message : String(err)}`);
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button type="button" className="btn btn--primary" onClick={openForm}>
        <Mic aria-hidden="true" />
        Start a live meeting
      </button>
    );
  }

  return (
    <form
      className="start"
      onSubmit={submit}
      onKeyDown={(e) => {
        if (e.key === "Escape" && !busy) setOpen(false);
      }}
    >
      <label htmlFor="start-title" className="start-label">
        Meeting title
      </label>
      <div className="start-row">
        <input
          id="start-title"
          className="field start-input"
          value={title}
          placeholder={fallback}
          maxLength={120}
          autoFocus
          autoComplete="off"
          onChange={(e) => setTitle(e.target.value)}
        />
        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? <LoaderCircle className="spin" aria-hidden="true" /> : <Mic aria-hidden="true" />}
          {busy ? "Starting" : "Start meeting"}
        </button>
        <button type="button" className="btn btn--ghost" onClick={() => setOpen(false)} disabled={busy}>
          Cancel
        </button>
      </div>
      <p className="start-hint">The browser will ask to use your microphone. Leave the title empty to use the date.</p>
      {error && (
        <p className="home-error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
