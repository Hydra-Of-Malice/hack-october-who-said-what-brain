import { useEffect, useRef, type KeyboardEvent, type PointerEvent, type RefObject } from "react";
import { Mic } from "lucide-react";

export type TalkbackVisual = "idle" | "opening" | "recording" | "armed" | "waiting" | "speaking";

interface TalkbackProps {
  visual: TalkbackVisual;
  /** Latest input level 0..1, read every animation frame while the mic is open. */
  levelRef: RefObject<number>;
  onPressStart: () => void;
  onPressEnd: () => void;
  label: string;
  describedBy?: string;
  buttonRef?: RefObject<HTMLButtonElement | null>;
}

/** The push-to-talk button: the only round control in the product. */
export default function Talkback({ visual, levelRef, onPressStart, onPressEnd, label, describedBy, buttonRef }: TalkbackProps) {
  const ringRef = useRef<HTMLSpanElement>(null);
  const spaceDown = useRef(false);
  const live = visual === "recording" || visual === "armed";

  useEffect(() => {
    const ring = ringRef.current;
    if (!ring) return;
    if (!live) {
      ring.style.setProperty("--lvl", "0");
      return;
    }
    let raf = 0;
    let smooth = 0;
    const tick = () => {
      const target = levelRef.current ?? 0;
      // Fast attack, slow release, like a VU meter.
      smooth += (target - smooth) * (target > smooth ? 0.5 : 0.1);
      ring.style.setProperty("--lvl", smooth.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [live, levelRef]);

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.focus();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer already gone */
    }
    onPressStart();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      if (!e.repeat && !spaceDown.current) {
        spaceDown.current = true;
        onPressStart();
      }
    } else if (e.key === "Enter") {
      // Enter behaves like a click: start listening, or send when already listening.
      e.preventDefault();
      e.stopPropagation();
      if (!e.repeat) {
        onPressStart();
        onPressEnd();
      }
    }
  };

  const onKeyUp = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== " ") return;
    e.preventDefault();
    e.stopPropagation();
    if (spaceDown.current) {
      spaceDown.current = false;
      onPressEnd();
    }
  };

  return (
    <div className="ask-talk" data-state={visual}>
      <span ref={ringRef} className="ask-talk__ring" aria-hidden="true" />
      <button
        ref={buttonRef}
        type="button"
        className="ask-talk__btn"
        aria-label={label}
        aria-describedby={describedBy}
        aria-pressed={visual === "recording"}
        onPointerDown={onPointerDown}
        onPointerUp={() => onPressEnd()}
        onPointerCancel={() => onPressEnd()}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onBlur={() => {
          if (spaceDown.current) {
            spaceDown.current = false;
            onPressEnd();
          }
        }}
        onClick={(e) => {
          // Pointer and keyboard presses are handled above. A click with detail 0 comes from
          // assistive technology (no pointer events): treat it as a toggle.
          if (e.detail === 0) {
            onPressStart();
            onPressEnd();
          }
        }}
        onContextMenu={(e) => e.preventDefault()}
      >
        <Mic aria-hidden="true" />
      </button>
    </div>
  );
}
