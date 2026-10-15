import { Check, Copy, FileText } from "lucide-react";
import { useCopy } from "./clipboard";

/** A follow-up drafted by the agent. It is only shown and copied, never sent. */
export default function DraftCard({ title, body }: { title: string; body: string }) {
  const [state, copy] = useCopy();
  return (
    <section className="ask-draft" aria-label={`Draft: ${title}`}>
      <h4 className="ask-draft__title">
        <FileText aria-hidden="true" />
        {title}
      </h4>
      <p className="ask-draft__body">{body}</p>
      <div className="ask-draft__foot">
        <button type="button" className="btn" onClick={() => copy(`${title}\n\n${body}`)}>
          {state === "copied" ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
          {state === "copied" ? "Copied" : "Copy"}
        </button>
        <span className="ask-draft__note">Drafted only. Nothing has been sent.</span>
      </div>
      {state === "failed" ? (
        <p className="ask-note ask-note--err" role="alert">
          The browser blocked copying. Select the text and copy it by hand.
        </p>
      ) : null}
    </section>
  );
}
