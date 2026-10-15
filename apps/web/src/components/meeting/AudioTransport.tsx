import { Download, Pause, Play } from "lucide-react";
import { timecode } from "../../lib/format";
import type { AudioElementState } from "./useAudioElement";

interface Props {
  src: string;
  title: string;
  audio: AudioElementState;
  fallbackDuration: number;
}

/** Transport for a recorded meeting: play/pause, position and a download. The speaker lanes are the scrubber. */
export default function AudioTransport({ src, title, audio, fallbackDuration }: Props) {
  const total = audio.duration || fallbackDuration;
  const fileName = `${title.replace(/[\\/:*?"<>|]+/g, " ").trim() || "meeting"}.wav`;
  return (
    <div className="transport">
      <div className="transport-row">
        <button
          type="button"
          className="btn btn--primary transport-play"
          onClick={audio.toggle}
          disabled={!!audio.error}
          aria-label={audio.playing ? "Pause recording" : "Play recording"}
        >
          {audio.playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
        </button>
        <span className="transport-time tc">
          {timecode(audio.time)} / {timecode(total)}
        </span>
        <a className="btn btn--ghost transport-dl" href={src} download={fileName}>
          <Download aria-hidden="true" />
          Download audio
        </a>
      </div>
      <audio ref={audio.attach} className="transport-audio" src={src} preload="metadata" aria-label="Meeting recording" />
      {audio.error && (
        <p className="transport-error" role="alert">
          {audio.error}
        </p>
      )}
    </div>
  );
}
