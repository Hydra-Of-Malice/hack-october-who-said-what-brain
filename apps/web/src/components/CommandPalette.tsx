// Ctrl/Cmd+K: jump to a meeting or page, or ask a question (hands it to the ask drawer).
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { AudioLines, Gavel, ListChecks, MessageSquare, Network, Search, Users } from "lucide-react";
import { duration, when } from "../lib/format";
import { useMeetings } from "./ask/meetingData";
import { useFocusTrap } from "./ask/useFocusTrap";
import { meetingPath } from "./ask/useJump";
import "./CommandPalette.css";

const OPEN_EVENT = "wsw:palette-open";
const MAX_MEETINGS = 7;

interface Item {
  id: string;
  group: "Ask" | "Meetings" | "Go to";
  label: string;
  detail?: string;
  icon: ReactNode;
  live?: boolean;
  run: () => void;
}

const PAGES: { label: string; path: string; icon: ReactNode; words: string }[] = [
  { label: "Meetings", path: "/", icon: <AudioLines aria-hidden="true" />, words: "meetings home recordings" },
  { label: "Commitments", path: "/commitments", icon: <ListChecks aria-hidden="true" />, words: "commitments action items todo tasks" },
  { label: "Decisions", path: "/decisions", icon: <Gavel aria-hidden="true" />, words: "decisions decided history" },
  { label: "People", path: "/people", icon: <Users aria-hidden="true" />, words: "people speakers who knows" },
  { label: "Graph", path: "/graph", icon: <Network aria-hidden="true" />, words: "graph falkordb nodes knowledge" },
];

function meetingDetail(startedAt: number, seconds: number): string {
  const at = when(startedAt);
  return seconds ? `${at}, ${duration(seconds)} long` : at;
}

function matches(haystack: string, query: string): boolean {
  const h = haystack.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => h.includes(term));
}

export default function CommandPalette() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const dialogRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const { meetings, error } = useMeetings(open);

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_EVENT, onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_EVENT, onOpen);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActive(0);
  }, [open]);

  useFocusTrap(dialogRef, open, inputRef);

  const close = () => setOpen(false);

  const items = useMemo<Item[]>(() => {
    const q = query.trim();
    const out: Item[] = [];
    if (q) {
      out.push({
        id: "ask",
        group: "Ask",
        label: `Ask: ${q}`,
        detail: "Answered from your meetings, with quotes",
        icon: <MessageSquare aria-hidden="true" />,
        run: () => window.dispatchEvent(new CustomEvent("wsw:ask", { detail: { question: q } })),
      });
    }
    const found = (meetings ?? []).filter((m) => !q || matches(m.title, q)).slice(0, MAX_MEETINGS);
    for (const m of found) {
      out.push({
        id: `m-${m.id}`,
        group: "Meetings",
        label: m.title || "Untitled meeting",
        detail: m.status === "live" ? "Recording now" : meetingDetail(m.started_at, m.duration_s),
        icon: <AudioLines aria-hidden="true" />,
        live: m.status === "live",
        run: () => navigate(meetingPath(m.id)),
      });
    }
    for (const p of PAGES) {
      if (q && !matches(`${p.label} ${p.words}`, q)) continue;
      out.push({ id: `p-${p.path}`, group: "Go to", label: p.label, icon: p.icon, run: () => navigate(p.path) });
    }
    if (!q || matches("ask question voice talk", q)) {
      out.push({
        id: "open-ask",
        group: "Go to",
        label: "Ask by voice",
        detail: "Opens the ask panel",
        icon: <MessageSquare aria-hidden="true" />,
        run: () => window.dispatchEvent(new CustomEvent("wsw:ask-open")),
      });
    }
    return out;
  }, [query, meetings, navigate]);

  const safeActive = Math.min(active, Math.max(0, items.length - 1));
  const activeId = items[safeActive] ? `cp-opt-${items[safeActive].id}` : undefined;

  useEffect(() => {
    if (!activeId) return;
    document.getElementById(activeId)?.scrollIntoView({ block: "nearest" });
  }, [activeId]);

  const choose = (item: Item | undefined) => {
    if (!item) return;
    close();
    item.run();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setActive((a) => (items.length ? (Math.min(a, items.length - 1) + 1) % items.length : 0));
        break;
      case "ArrowUp":
        e.preventDefault();
        setActive((a) => (items.length ? (Math.min(a, items.length - 1) - 1 + items.length) % items.length : 0));
        break;
      case "Home":
        if (e.ctrlKey) {
          e.preventDefault();
          setActive(0);
        }
        break;
      case "End":
        if (e.ctrlKey) {
          e.preventDefault();
          setActive(Math.max(0, items.length - 1));
        }
        break;
      case "Enter":
        e.preventDefault();
        choose(items[safeActive]);
        break;
      case "Escape":
        e.preventDefault();
        e.stopPropagation();
        close();
        break;
    }
  };

  if (!open) return null;

  const groups = (["Ask", "Meetings", "Go to"] as const)
    .map((g) => ({ name: g, items: items.filter((i) => i.group === g) }))
    .filter((g) => g.items.length);

  return (
    <>
      <div className="cp-scrim" onClick={close} aria-hidden="true" />
      <div
        ref={dialogRef}
        className="cp"
        role="dialog"
        aria-modal="true"
        aria-label="Search and jump"
        onKeyDown={onKeyDown}
      >
        <div className="cp-input">
          <Search aria-hidden="true" />
          <input
            ref={inputRef}
            role="combobox"
            aria-expanded="true"
            aria-controls="cp-list"
            aria-activedescendant={activeId}
            aria-autocomplete="list"
            aria-label="Search meetings and pages, or type a question"
            placeholder="Search meetings, or type a question"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            autoComplete="off"
            spellCheck={false}
          />
          <kbd className="cp-kbd">Esc</kbd>
        </div>

        <div ref={listRef} id="cp-list" role="listbox" aria-label="Results" className="cp-list">
          {groups.map((g) => (
            <div key={g.name} role="group" aria-labelledby={`cp-g-${g.name.replace(/\s/g, "")}`} className="cp-group">
              <div id={`cp-g-${g.name.replace(/\s/g, "")}`} className="cp-group__name">
                {g.name}
              </div>
              {g.items.map((item) => {
                const index = items.indexOf(item);
                const selected = index === safeActive;
                return (
                  <div
                    key={item.id}
                    id={`cp-opt-${item.id}`}
                    role="option"
                    aria-selected={selected}
                    className={`cp-opt${selected ? " is-active" : ""}`}
                    onMouseDown={(e) => e.preventDefault()}
                    onMouseMove={() => {
                      if (!selected) setActive(index);
                    }}
                    onClick={() => choose(item)}
                  >
                    <span className="cp-opt__icon">{item.icon}</span>
                    <span className="cp-opt__text">
                      <span className="cp-opt__label">{item.label}</span>
                      {item.detail ? (
                        <span className="cp-opt__detail">
                          {item.live ? <span className="live-dot" aria-hidden="true" /> : null}
                          {item.detail}
                        </span>
                      ) : null}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
          {error && !meetings ? <p className="cp-empty">Meetings didn't load: {error}</p> : null}
        </div>

        <p className="cp-foot">
          <span>
            <kbd className="cp-kbd">↑</kbd> <kbd className="cp-kbd">↓</kbd> to move
          </span>
          <span>
            <kbd className="cp-kbd">Enter</kbd> to open
          </span>
          <span>
            <kbd className="cp-kbd">/</kbd> asks by voice from anywhere
          </span>
        </p>
      </div>
    </>
  );
}
