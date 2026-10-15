import { useCallback, useEffect, useRef, useState } from "react";

function copyWithSelection(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

/** Copy text: the async clipboard API first, then a hidden-textarea copy if that is unavailable or refused. */
export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      /* refused (no focus or permission): try the selection route below */
    }
  }
  if (!copyWithSelection(text)) throw new Error("The browser blocked copying. Select the text and copy it by hand.");
}

export type CopyState = "idle" | "copied" | "failed";

/** A copy action whose state resets after two seconds. */
export function useCopy(): [CopyState, (text: string) => void] {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current);
    },
    [],
  );
  const copy = useCallback((text: string) => {
    copyText(text)
      .then(() => setState("copied"))
      .catch(() => setState("failed"))
      .finally(() => {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setState("idle"), 2000);
      });
  }, []);
  return [state, copy];
}
