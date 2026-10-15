import { useEffect, useState, type MouseEvent, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { AudioLines, ListChecks, Mic, Network, Signpost, Users, type LucideIcon } from "lucide-react";
import { brand } from "../../lib/brand";
import ProductMark from "./ProductMark";
import { StatusDot, StatusLine } from "./SystemStatus";
import "./AppShell.css";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  match: (path: string) => boolean;
}

const NAV: NavItem[] = [
  { to: "/", label: "Meetings", icon: AudioLines, match: (p) => p === "/" || p.startsWith("/m/") },
  { to: "/commitments", label: "Commitments", icon: ListChecks, match: (p) => p.startsWith("/commitments") },
  { to: "/decisions", label: "Decisions", icon: Signpost, match: (p) => p.startsWith("/decisions") },
  { to: "/people", label: "People", icon: Users, match: (p) => p.startsWith("/people") },
  { to: "/graph", label: "Graph", icon: Network, match: (p) => p.startsWith("/graph") },
];

const MOBILE_QUERY = "(max-width: 767px)";

function useIsMobile(): boolean {
  const [mobile, setMobile] = useState(() => typeof window !== "undefined" && window.matchMedia(MOBILE_QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(MOBILE_QUERY);
    const on = () => setMobile(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return mobile;
}

function openAsk() {
  window.dispatchEvent(new CustomEvent("wsw:ask-open"));
}

function skipToMain(e: MouseEvent<HTMLAnchorElement>) {
  const main = document.getElementById("main");
  if (!main) return;
  e.preventDefault();
  main.focus();
  main.scrollIntoView({ block: "start" });
}

export default function AppShell({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const mobile = useIsMobile();

  return (
    <div className="shell">
      <a href="#main" className="shell-skip" onClick={skipToMain}>
        Skip to main content
      </a>
      <nav className="shell-rail" aria-label="Main">
        <Link to="/" className="shell-mark" aria-label={`${brand.short}, all meetings`}>
          <ProductMark />
          <span className="shell-mark-name" aria-hidden="true">
            {brand.short}
          </span>
        </Link>
        <ul className="shell-nav">
          {NAV.map(({ to, label, icon: Icon, match }) => {
            const active = match(pathname);
            return (
              <li key={to}>
                <Link to={to} className={`shell-link${active ? " is-active" : ""}`} aria-current={active ? "page" : undefined}>
                  <Icon aria-hidden="true" />
                  <span className="shell-link-label">{label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
        <div className="shell-rail-foot">
          <button type="button" className="shell-talk" onClick={openAsk} aria-label="Ask (press /)" aria-keyshortcuts="/" data-tip="Ask (press /)">
            <Mic aria-hidden="true" />
          </button>
          {mobile && <StatusDot />}
        </div>
      </nav>
      <div className="shell-main">
        <main id="main" className="shell-content" tabIndex={-1}>
          {children}
        </main>
        {!mobile && <StatusLine />}
      </div>
    </div>
  );
}
