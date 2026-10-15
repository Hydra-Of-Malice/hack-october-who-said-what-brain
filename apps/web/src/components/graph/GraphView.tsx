// Force-directed view of a FalkorDB subgraph. react-force-graph-2d is loaded on demand so it stays
// out of the main bundle.
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import type ForceGraph from "react-force-graph-2d";
import type { ForceGraphMethods, LinkObject, NodeObject } from "react-force-graph-2d";
import type { GraphNode, Subgraph } from "../../api/types";
import { isHollow, kindStyle, nodeSpeakerIndex, readPalette, toneColor, traceShape, truncate } from "./graphStyle";
import { useReducedMotion } from "./useReducedMotion";
import "./graph.css";

interface NodeData {
  id: string;
  kind: string;
  label: string;
  src: GraphNode;
}
interface LinkData {
  type: string;
}
type SimNode = NodeObject<NodeData>;
type SimLink = LinkObject<NodeData, LinkData>;
type TypedForceGraph = (props: Parameters<typeof ForceGraph<NodeData, LinkData>>[0]) => ReactElement;

// The library's default export is generic; pin it to our node/link shapes.
const ForceGraph2D = lazy(async () => {
  const mod = await import("react-force-graph-2d");
  return { default: mod.default as TypedForceGraph };
});

export interface GraphViewProps {
  data: Subgraph;
  /** Accessible summary of what the graph shows. */
  label: string;
  onSelect?: (node: GraphNode | null) => void;
  selectedId?: string | null;
  hiddenKinds?: ReadonlySet<string>;
  /** Fixed height in px; omit to fill the parent's height. */
  height?: number;
  /** Changing this re-fits the view (e.g. a different meeting). Live additions keep the camera still. */
  fitKey?: string;
}

const FIT_PADDING = 56;
const MAX_FIT_ZOOM = 3;
const EARLY_FIT_TICKS = new Set([1, 20, 50]);

function endId(end: SimLink["source"]): string {
  if (end && typeof end === "object") return String((end as SimNode).id);
  return String(end);
}

export default function GraphView({ data, label, onSelect, selectedId = null, hiddenKinds, height, fitKey }: GraphViewProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<SimNode, SimLink> | undefined>(undefined);
  const lastNodes = useRef<SimNode[]>([]);
  const fitStage = useRef<0 | 1 | 2>(0);
  const fitTicks = useRef(0);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const sizeRef = useRef(size);
  sizeRef.current = size;
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [palette] = useState(readPalette);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = Math.floor(entry.contentRect.width);
      const h = Math.floor(entry.contentRect.height);
      setSize((s) => (s && s.w === w && s.h === h ? s : { w, h }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    fitStage.current = 0;
    fitTicks.current = 0;
  }, [fitKey]);

  const graph = useMemo(() => {
    // Carry positions over from the previous layout so live additions don't reshuffle the picture.
    const known = new Map(lastNodes.current.map((n) => [n.id, n]));
    const nodes: SimNode[] = [];
    const ids = new Set<string>();
    for (const n of data.nodes ?? []) {
      if (!n || n.id === undefined || n.id === null) continue;
      const id = String(n.id);
      const kind = String(n.kind ?? "Other");
      if (ids.has(id) || hiddenKinds?.has(kind)) continue;
      ids.add(id);
      const prev = known.get(id);
      const node: SimNode = { id, kind, label: String(n.label ?? id), src: n };
      if (prev?.x !== undefined && prev.y !== undefined) {
        node.x = prev.x;
        node.y = prev.y;
      }
      nodes.push(node);
    }
    const links: SimLink[] = [];
    for (const l of data.links ?? []) {
      const s = endId(l.source as SimLink["source"]);
      const t = endId(l.target as SimLink["target"]);
      if (ids.has(s) && ids.has(t)) links.push({ source: s, target: t, type: String(l.type ?? "") });
    }
    lastNodes.current = nodes;
    return { nodes, links };
  }, [data, hiddenKinds]);

  const neighbours = useMemo(() => {
    const set = new Set<string>();
    if (!selectedId) return set;
    for (const l of graph.links) {
      const s = endId(l.source);
      const t = endId(l.target);
      if (s === selectedId) set.add(t);
      if (t === selectedId) set.add(s);
    }
    return set;
  }, [graph, selectedId]);

  const touchesSelected = useCallback(
    (l: SimLink) => !!selectedId && (endId(l.source) === selectedId || endId(l.target) === selectedId),
    [selectedId],
  );

  const drawNode = useCallback(
    (node: SimNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const s = kindStyle(node.kind);
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const selected = node.id === selectedId;
      const hovered = node.id === hoverId;
      const dim = !!selectedId && !selected && !neighbours.has(node.id);
      const color = toneColor(palette, s.tone, s.tone === "speaker" ? nodeSpeakerIndex(node.src) : 0);

      ctx.globalAlpha = dim ? 0.28 : 1;
      traceShape(ctx, s.shape, x, y, s.r);
      if (isHollow(s.shape)) {
        ctx.fillStyle = palette.surface;
        ctx.fill();
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = color;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
      }
      if (selected || hovered) {
        traceShape(ctx, s.shape, x, y, s.r + 2.5);
        ctx.lineWidth = 2 / scale;
        ctx.strokeStyle = selected ? palette.focus : palette.ink2;
        ctx.stroke();
      }

      const showLabel = selected || hovered || (s.prominent && scale >= 1.2) || scale >= 2.6;
      if (showLabel) {
        const emphasis = selected || hovered;
        const text = truncate(node.label, emphasis ? 64 : 28);
        const fs = 11.5 / scale;
        const ty = y + s.r * 1.25 + 2 / scale;
        ctx.font = `${emphasis ? 650 : 500} ${fs}px ${palette.font}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.lineJoin = "round";
        ctx.lineWidth = 3 / scale;
        ctx.strokeStyle = palette.surface;
        ctx.strokeText(text, x, ty);
        ctx.fillStyle = palette.ink;
        ctx.fillText(text, x, ty);
      }
      ctx.globalAlpha = 1;
    },
    [palette, selectedId, hoverId, neighbours],
  );

  const paintPointerArea = useCallback((node: SimNode, color: string, ctx: CanvasRenderingContext2D) => {
    const r = kindStyle(node.kind).r + 3;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, Math.PI * 2);
    ctx.fill();
  }, []);

  // Fit per fitKey: instantly while the pre-warmed layout expands, then gently once it settles.
  const fit = useCallback(
    (durationMs: number) => {
      const fg = fgRef.current;
      if (!fg || !graph.nodes.length) return;
      const box = sizeRef.current;
      const padding = box ? Math.round(Math.min(FIT_PADDING, Math.min(box.w, box.h) * 0.12)) : FIT_PADDING;
      fg.zoomToFit(durationMs, padding);
      // A handful of nodes would otherwise be blown up to fill the canvas. zoom() reports the
      // pre-animation value while a transition runs, so check once it has finished.
      const cap = () => {
        const current = fgRef.current;
        if (current && current.zoom() > MAX_FIT_ZOOM) current.zoom(MAX_FIT_ZOOM, durationMs ? 200 : 0);
      };
      if (durationMs) window.setTimeout(cap, durationMs + 30);
      else cap();
    },
    [graph],
  );

  const forcesSet = useRef(false);
  const onEngineTick = useCallback(() => {
    const fg = fgRef.current;
    if (fg && !forcesSet.current) {
      // Spread nodes a little more than the d3 defaults so labels have room.
      forcesSet.current = true;
      fg.d3Force("charge")?.strength?.(-70);
      fg.d3Force("link")?.distance?.(34);
      fg.d3ReheatSimulation();
    }
    // While the layout is still expanding, re-fit instantly a few times; the settled fit animates.
    if (fitStage.current === 2) return;
    fitTicks.current += 1;
    if (EARLY_FIT_TICKS.has(fitTicks.current)) {
      fitStage.current = 1;
      fit(0);
    }
  }, [fit]);

  const onEngineStop = useCallback(() => {
    if (fitStage.current === 2) return;
    fitStage.current = 2;
    fit(reduced ? 0 : 400);
  }, [fit, reduced]);

  // Never pass undefined to the kapsule: it would replace the library default rather than keep it.
  const timing = reduced ? { warmupTicks: 80, cooldownTicks: 12 } : { warmupTicks: 40, cooldownTime: 4000 };
  const style = height ? { height } : undefined;
  return (
    <div ref={boxRef} className={`gv${height ? "" : " gv--fill"}`} style={style} role="img" aria-label={label}>
      {size && size.w > 0 && size.h > 0 ? (
        <Suspense fallback={<p className="gv__loading">Loading the graph…</p>}>
          <ForceGraph2D
            ref={fgRef}
            graphData={graph}
            width={size.w}
            height={size.h}
            nodeId="id"
            nodeLabel={() => ""}
            nodeCanvasObject={drawNode}
            nodePointerAreaPaint={paintPointerArea}
            linkColor={(l: SimLink) => (touchesSelected(l) ? palette.ink2 : palette.line)}
            linkWidth={(l: SimLink) => (touchesSelected(l) ? 1.6 : 1)}
            onNodeHover={(n: SimNode | null) => setHoverId(n ? String(n.id) : null)}
            onNodeClick={(n: SimNode) => onSelect?.(n.src)}
            onBackgroundClick={() => onSelect?.(null)}
            onEngineTick={onEngineTick}
            onEngineStop={onEngineStop}
            minZoom={0.2}
            maxZoom={10}
            {...timing}
          />
        </Suspense>
      ) : null}
    </div>
  );
}
