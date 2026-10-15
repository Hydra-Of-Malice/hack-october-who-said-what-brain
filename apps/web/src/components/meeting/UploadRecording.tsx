import { useRef, useState, type ChangeEvent } from "react";
import { useNavigate } from "react-router-dom";
import { LoaderCircle, Upload } from "lucide-react";
import { api } from "../../api/client";

function titleFromFile(name: string): string {
  return name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim() || "Uploaded recording";
}

/** "Upload a recording": picks an audio file, uploads it, then opens the meeting while it is processed. */
export default function UploadRecording() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [busyName, setBusyName] = useState<string | null>(null);
  const [realtime, setRealtime] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const el = e.currentTarget;
    const file = el.files?.[0];
    if (!file) return;
    setBusyName(file.name);
    setError(null);
    try {
      const m = await api.uploadMeeting(file, titleFromFile(file.name), realtime ? "realtime" : "fast");
      navigate(`/m/${encodeURIComponent(m.id)}`);
    } catch (err) {
      setError(`Couldn't upload ${file.name}: ${err instanceof Error ? err.message : String(err)}. WAV, MP3, M4A, WebM and OGG files work.`);
      setBusyName(null);
    } finally {
      el.value = "";
    }
  };

  return (
    <div className="upload">
      <div className="upload-row">
        <button type="button" className="btn" onClick={() => input.current?.click()} disabled={!!busyName}>
          {busyName ? <LoaderCircle className="spin" aria-hidden="true" /> : <Upload aria-hidden="true" />}
          {busyName ? `Uploading ${busyName}` : "Upload a recording"}
        </button>
        <input ref={input} type="file" accept="audio/*" hidden onChange={(e) => void onFile(e)} />
        <label className="upload-pace" title="Lanes fill in as if the meeting were happening now. Off: processed as fast as possible.">
          <input type="checkbox" checked={realtime} onChange={(e) => setRealtime(e.currentTarget.checked)} disabled={!!busyName} />
          Replay at real speed
        </label>
      </div>
      {error && (
        <p className="home-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
