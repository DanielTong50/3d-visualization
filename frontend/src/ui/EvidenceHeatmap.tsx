import type { AssemblyPlan } from '../engine/plan';
import type { SessionResult } from '../engine/results';

/**
 * A decoded video at a glance: one row per step, one cell per sampled frame, brighter where the
 * model read the step as done; the decoded build history as a staircase through it; diamonds
 * where the tracker dates each step, green ticks where the truth does.
 */
export function EvidenceHeatmap({
  plan,
  result,
  placed,
  time = null,
  onSeek,
}: {
  plan: AssemblyPlan;
  result: SessionResult;
  /** When each step's parts were really in place (the truth), if known. */
  placed: Map<number, number>;
  /** The playhead, if any. */
  time?: number | null;
  onSeek?: (time: number) => void;
}) {
  const steps = plan.steps.length;
  const duration = result.duration_s;
  const W = 1000;
  const ROW = 22;
  const H = ROW * steps;
  const x = (t: number) => (t / duration) * W;
  const y = (k: number) => H - k * ROW; // the line sits above the k done rows (step 1 at the bottom)
  return (
    <svg
      viewBox={`-34 -6 ${W + 40} ${H + 26}`}
      className="heatmap"
      onPointerDown={(e) => {
        const box = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
        const px = ((e.clientX - box.left) / box.width) * (W + 40) - 34;
        onSeek?.((px / W) * duration);
      }}
      role="img"
      aria-label="Heatmap of the model's per-frame step readings with the decoded path"
    >
      {plan.steps.map((s) => (
        <text key={s.index} x={-10} y={y(s.index) + ROW / 2 + 4} className="hm-label">
          {s.index}
        </text>
      ))}
      {result.frames.map((f, i) => {
        const x0 = x(f.time_s - result.every_s);
        const w = x(result.every_s) - 1;
        return plan.steps.map((s) => {
          const p = f.step_done[String(s.index)] ?? 0.5;
          return (
            <rect
              key={`${i}-${s.index}`}
              x={x0}
              y={y(s.index) + 1}
              width={w}
              height={ROW - 2}
              rx={2}
              className="hm-cell"
              style={{ opacity: 0.06 + 0.94 * p }}
            />
          );
        });
      })}
      <polyline
        className="hm-path"
        points={[
          `0,${y(0)}`,
          ...result.frames.flatMap((f, i) => [
            `${x(f.time_s - result.every_s)},${y(i === 0 ? 0 : result.path[i - 1])}`,
            `${x(f.time_s - result.every_s)},${y(result.path[i])}`,
            `${x(f.time_s)},${y(result.path[i])}`,
          ]),
        ].join(' ')}
      />
      {[...placed.entries()].map(([k, t]) => (
        <g key={k} transform={`translate(${x(t)} ${y(k)})`} className="hm-truth">
          <line y1={-4} y2={ROW + 4} />
        </g>
      ))}
      {Object.entries(result.finished_at).map(([k, t]) => (
        <rect
          key={k}
          x={x(t) - 4}
          y={y(Number(k)) - 4}
          width={8}
          height={8}
          transform={`rotate(45 ${x(t)} ${y(Number(k))})`}
          className="hm-found"
        />
      ))}
      {time !== null && <line x1={x(time)} x2={x(time)} y1={-6} y2={H + 4} className="hm-playhead" />}
      {[0, 10, 20, 30, 40, 50, 60, 70]
        .filter((t) => t <= duration)
        .map((t) => (
          <text key={t} x={x(t)} y={H + 18} className="hm-tick">
            {t}s
          </text>
        ))}
    </svg>
  );
}
