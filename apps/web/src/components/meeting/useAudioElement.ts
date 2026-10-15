import { useCallback, useEffect, useRef, useState } from "react";

export interface AudioElementState {
  /** Callback ref: pass as the <audio> element's ref. */
  attach: (el: HTMLAudioElement | null) => void;
  time: number;
  duration: number;
  playing: boolean;
  /** True once playback has started or someone seeked, so the "heard" line is meaningful. */
  engaged: boolean;
  error: string | null;
  /** Live playback position for animation frames; null until the element mounts. */
  readTime: () => number | null;
  /** Returns false when there is no element to seek. */
  seek: (t: number, play: boolean) => boolean;
  toggle: () => void;
}

export function useAudioElement(): AudioElementState {
  const elRef = useRef<HTMLAudioElement | null>(null);
  const [el, setEl] = useState<HTMLAudioElement | null>(null);
  const [time, setTime] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [engaged, setEngaged] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const attach = useCallback((node: HTMLAudioElement | null) => {
    elRef.current = node;
    setEl(node);
  }, []);

  const readTime = useCallback(() => (elRef.current ? elRef.current.currentTime : null), []);

  const seek = useCallback((t: number, play: boolean) => {
    const node = elRef.current;
    if (!node) return false;
    node.currentTime = Math.max(0, t);
    if (play) void node.play().catch(() => undefined); // autoplay may be refused; the position is still set
    return true;
  }, []);

  const toggle = useCallback(() => {
    const node = elRef.current;
    if (!node) return;
    if (node.paused) void node.play().catch(() => undefined);
    else node.pause();
  }, []);

  useEffect(() => {
    if (!el) return;
    const onTime = () => setTime(el.currentTime);
    const onMeta = () => setDur(Number.isFinite(el.duration) ? el.duration : 0);
    const onPlay = () => {
      setPlaying(true);
      setEngaged(true);
    };
    const onPause = () => setPlaying(false);
    const onSeeked = () => {
      setTime(el.currentTime);
      setEngaged(true);
    };
    const onError = () =>
      setError(
        el.error?.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
          ? "The recording isn't available for this meeting."
          : "The recording couldn't be loaded. Check the API, then reload the page.",
      );
    const events: [string, () => void][] = [
      ["timeupdate", onTime],
      ["loadedmetadata", onMeta],
      ["durationchange", onMeta],
      ["play", onPlay],
      ["pause", onPause],
      ["ended", onPause],
      ["seeked", onSeeked],
      ["error", onError],
    ];
    events.forEach(([name, fn]) => el.addEventListener(name, fn));
    if (el.readyState > 0) onMeta();
    return () => events.forEach(([name, fn]) => el.removeEventListener(name, fn));
  }, [el]);

  return { attach, time, duration: dur, playing, engaged, error, readTime, seek, toggle };
}
