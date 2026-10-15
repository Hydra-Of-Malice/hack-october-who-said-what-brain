import { useEffect, type RefObject } from "react";

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type=hidden])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "summary",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export function focusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => el.getClientRects().length > 0 && !el.closest("[inert]"),
  );
}

// Stacked dialogs (the command palette over the ask drawer): only the topmost trap holds focus.
const stack: symbol[] = [];

/**
 * Keep keyboard focus inside `container` while `active`. Focus moves to `initial` (or the first
 * focusable) on activation and returns to the previously focused element on deactivation.
 */
export function useFocusTrap(
  container: RefObject<HTMLElement | null>,
  active: boolean,
  initial?: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!active) return;
    const id = Symbol("trap");
    stack.push(id);
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const isTop = () => stack[stack.length - 1] === id;

    const focusFirst = () => {
      const el = container.current;
      if (!el) return;
      (initial?.current ?? focusables(el)[0] ?? el).focus();
    };
    // The dialog is committed by the time effects run, so focus now (rAF would stall in a hidden tab).
    focusFirst();

    const onKey = (e: KeyboardEvent) => {
      const el = container.current;
      if (e.key !== "Tab" || !el || !isTop()) return;
      const items = focusables(el);
      if (!items.length) {
        e.preventDefault();
        el.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const current = document.activeElement;
      if (e.shiftKey && (current === first || !el.contains(current))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (current === last || !el.contains(current))) {
        e.preventDefault();
        first.focus();
      }
    };
    const onFocusIn = (e: FocusEvent) => {
      const el = container.current;
      if (!el || !isTop() || el.contains(e.target as Node)) return;
      focusFirst();
    };
    document.addEventListener("keydown", onKey, true);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.removeEventListener("focusin", onFocusIn);
      const at = stack.indexOf(id);
      if (at >= 0) stack.splice(at, 1);
      if (previous?.isConnected) previous.focus();
    };
    // The refs are stable; re-run only when the trap turns on or off.
  }, [active]);
}

/** True when a key press should be left to a text field (typing "/" or Space). */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "INPUT") {
    const type = (target as HTMLInputElement).type;
    return !["checkbox", "radio", "button", "submit", "reset", "range", "color", "file"].includes(type);
  }
  return false;
}

/** True when Space on this element already means "activate" (buttons, links, toggles). */
export function isActivatable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return !!target.closest("button, a[href], summary, select, input, textarea, [role=switch], [role=radio], [role=option]");
}
