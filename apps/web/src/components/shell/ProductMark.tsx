/** The multitrack glyph from the favicon: three lanes of speaker blocks on navy. */
export default function ProductMark({ size = 32 }: { size?: number }) {
  return (
    <svg className="mark" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect width="32" height="32" rx="6" style={{ fill: "var(--ink)" }} />
      <rect x="5" y="8" width="9" height="4" rx="1" style={{ fill: "var(--spk-0)" }} />
      <rect x="16" y="8" width="11" height="4" rx="1" style={{ fill: "var(--spk-0)" }} />
      <rect x="9" y="14" width="13" height="4" rx="1" style={{ fill: "var(--spk-1)" }} />
      <rect x="5" y="20" width="6" height="4" rx="1" style={{ fill: "var(--spk-2)" }} />
      <rect x="19" y="20" width="8" height="4" rx="1" style={{ fill: "var(--spk-2)" }} />
    </svg>
  );
}
