import { useEffect, useMemo, useRef, useState } from 'react';
import { madeAt } from '../engine/graph';
import type { AssemblyPlan } from '../engine/plan';

/**
 * The assembly graph as a 3D skeleton: every part is a node at its center in the finished
 * furniture, every joint an edge. Drawn in SVG with a slowly turning isometric projection, so it
 * reads as the structure of the piece. Edges done are solid, the current step's edges glow and
 * flow, edges still to come are dashed. Drag to turn it; hover a node for the part.
 */

interface AssemblyGraphProps {
  plan: AssemblyPlan;
  /** Steps done: their joints are solid. */
  completed: number;
  /** The step in progress (1-based), whose joints glow. */
  active?: number | null;
  /** Only these joints exist yet (a graph being read or reconstructed); null: all of them. */
  known?: Set<string> | null;
  /** Joints the recovered graph got wrong, drawn in red. */
  wrong?: Set<string> | null;
  /** Edges a recovered graph has that the plan does not, drawn in red between these parts. */
  extra?: { id: string; parts: [string, string] }[];
  spin?: boolean;
  compact?: boolean;
}

interface Projected {
  x: number;
  y: number;
  depth: number;
}

const PITCH = 0.5;

export function AssemblyGraph({
  plan,
  completed,
  active = null,
  known = null,
  wrong = null,
  extra = [],
  spin = true,
  compact = false,
}: AssemblyGraphProps) {
  const [yaw, setYaw] = useState(-0.65);
  const [hover, setHover] = useState<string | null>(null);
  const drag = useRef<{ x: number; yaw: number } | null>(null);
  const spinning = spin && !drag.current;

  useEffect(() => {
    if (!spinning || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      setYaw((y) => y + ((now - last) / 1000) * 0.18);
      last = now;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [spinning]);

  const geometry = useMemo(() => {
    const centers = plan.parts.map((p) => p.assembled?.position ?? [0, 0, 0]);
    const lo = [0, 1, 2].map((i) => Math.min(...centers.map((c) => c[i])));
    const hi = [0, 1, 2].map((i) => Math.max(...centers.map((c) => c[i])));
    const mid = lo.map((v, i) => (v + hi[i]) / 2);
    const scale = 1 / Math.max(1, ...hi.map((v, i) => v - lo[i]));
    const nodes = plan.parts.map((p, i) => ({
      part: p,
      position: centers[i].map((v, k) => (v - mid[k]) * scale * 2) as [number, number, number],
      radius: 5 + 9 * Math.cbrt((p.size_mm[0] * p.size_mm[1] * p.size_mm[2]) / 1e8),
    }));
    return { nodes, index: new Map(nodes.map((n, i) => [n.part.id, i])) };
  }, [plan]);

  const made = useMemo(() => madeAt(plan), [plan]);
  const activeJoints = new Set(active ? (plan.steps[active - 1]?.joints ?? []) : []);
  const attached = new Set(
    plan.joints.filter((j) => (made.get(j.id) ?? Infinity) <= completed && (!known || known.has(j.id))).flatMap((j) => j.parts),
  );
  const activeParts = new Set(plan.joints.filter((j) => activeJoints.has(j.id)).flatMap((j) => j.parts));

  const project = ([x, y, z]: [number, number, number]): Projected => {
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const rx = x * cos - z * sin;
    const rz = x * sin + z * cos;
    return {
      x: rx,
      y: -(y * Math.cos(PITCH) - rz * Math.sin(PITCH)),
      depth: y * Math.sin(PITCH) + rz * Math.cos(PITCH),
    };
  };
  const points = geometry.nodes.map((n) => project(n.position));
  const size = 100;
  const toSvg = (p: Projected) => ({ x: 50 + p.x * 34, y: 52 + p.y * 34 });

  const floor = [-1.2, -0.6, 0, 0.6, 1.2].flatMap((v) => [
    [project([v, -1.05, -1.2]), project([v, -1.05, 1.2])],
    [project([-1.2, -1.05, v]), project([1.2, -1.05, v])],
  ]);

  const edges = [
    ...plan.joints
      .filter((j) => !known || known.has(j.id) || activeJoints.has(j.id))
      .map((j) => {
        const step = made.get(j.id) ?? Infinity;
        const state = wrong?.has(j.id) ? 'wrong' : activeJoints.has(j.id) ? 'active' : step <= completed ? 'done' : 'future';
        return {
          id: j.id,
          parts: j.parts,
          title: `${j.parts.join(' – ')} · ${j.kind}`,
          state,
        };
      }),
    ...extra.map((e) => ({
      id: e.id,
      parts: e.parts,
      title: `${e.parts.join(' – ')} · not in the plan`,
      state: 'wrong',
    })),
  ]
    .map((e) => {
      const [a, b] = e.parts.map((id) => geometry.index.get(id)!);
      return { ...e, a, b, depth: (points[a].depth + points[b].depth) / 2 };
    })
    .sort((e, f) => e.depth - f.depth);

  const order = geometry.nodes.map((_, i) => i).sort((a, b) => points[a].depth - points[b].depth);
  const hovered = hover ? geometry.nodes.find((n) => n.part.id === hover) : null;

  return (
    <div className={`graph${compact ? ' graph-compact' : ''}`}>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="graph-svg"
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, yaw };
          (e.target as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (drag.current) setYaw(drag.current.yaw + (e.clientX - drag.current.x) * 0.01);
        }}
        onPointerUp={() => (drag.current = null)}
        role="img"
        aria-label={`Assembly graph of ${plan.title}: ${plan.parts.length} parts, ${plan.joints.length} joints`}
      >
        <g className="graph-floor">
          {floor.map(([p, q], i) => {
            const [s, t] = [toSvg(p), toSvg(q)];
            return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} />;
          })}
        </g>
        {edges.map(({ id, title, a, b, state }) => {
          const [s, t] = [toSvg(points[a]), toSvg(points[b])];
          return (
            <line key={id} className={`graph-edge is-${state}`} x1={s.x} y1={s.y} x2={t.x} y2={t.y}>
              <title>{title}</title>
            </line>
          );
        })}
        {order.map((i) => {
          const node = geometry.nodes[i];
          const p = toSvg(points[i]);
          const state = activeParts.has(node.part.id) ? 'active' : attached.has(node.part.id) ? 'done' : 'future';
          const r = (node.radius / 10) * (compact ? 1.6 : 2);
          return (
            <g
              key={node.part.id}
              className={`graph-node is-${state}${hover === node.part.id ? ' is-hover' : ''}`}
              transform={`translate(${p.x} ${p.y})`}
              onPointerEnter={() => setHover(node.part.id)}
              onPointerLeave={() => setHover(null)}
            >
              {state === 'active' && <circle className="graph-halo" r={r * 1.9} />}
              <circle r={r} />
              <text dy="0.36em">{node.part.label}</text>
            </g>
          );
        })}
      </svg>
      {hovered && !compact && (
        <p className="graph-tip">
          <span className="letter">{hovered.part.label}</span> {hovered.part.name}{' '}
          <span className="faint">{hovered.part.size_mm.map((v) => Math.round(v)).join(' × ')} mm</span>
        </p>
      )}
    </div>
  );
}
