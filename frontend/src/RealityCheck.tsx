import { useEffect, useRef, useState, type PointerEvent } from 'react';

/**
 * One filmed push of an object across the table, as written by the pipeline's reality-check
 * stage to scenes/<id>/reality_check.json (a list of these).
 */
export interface PushTest {
  object_id: string;
  video?: string | null;
  fit: { v0: number; friction: number; distance_m: number; rms_residual_m: number };
  friction_estimated: number;
  friction_measured: number;
  real_distance_m: number;
  predicted_distance_before_m: number;
  predicted_distance_after_m: number;
  error_before_m: number;
  error_after_m: number;
  track: { t: number[]; s_m: number[] };
}

const G = 9.81;
// Categorical slots 1–3 of the dataviz reference palette (validated on white; aqua needs the
// direct labels and table that this panel always shows).
const COLOR = { real: '#2a78d6', before: '#eb6834', after: '#1baf7a' };

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

function isPushTest(value: unknown): value is PushTest {
  const p = value as PushTest;
  return (
    typeof p?.object_id === 'string' &&
    isNum(p.fit?.v0) &&
    [p.friction_estimated, p.friction_measured, p.real_distance_m, p.predicted_distance_before_m, p.predicted_distance_after_m].every(isNum) &&
    [p.error_before_m, p.error_after_m].every(isNum) &&
    Array.isArray(p.track?.t) &&
    Array.isArray(p.track?.s_m) &&
    p.track.t.length === p.track.s_m.length &&
    p.track.t.every(isNum) &&
    p.track.s_m.every(isNum)
  );
}

/** The scene's push tests, or null when it has none (the file is optional). */
export async function loadPushTests(sceneDir: string): Promise<PushTest[] | null> {
  try {
    const response = await fetch(`${sceneDir}/reality_check.json`);
    if (!response.ok) return null;
    const data = (await response.json()) as unknown;
    const tests = Array.isArray(data) ? data.filter(isPushTest) : [];
    return tests.length > 0 ? tests : null;
  } catch {
    return null;
  }
}

const cm = (meters: number) => `${(meters * 100).toFixed(1)} cm`;
const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / values.length;

/** Distance slid by time t for a push at v0 that stops after `distance` (constant deceleration). */
function slide(v0: number, distance: number, t: number): number {
  const deceleration = (v0 * v0) / (2 * distance);
  const stopsAt = v0 / deceleration;
  return t >= stopsAt ? distance : v0 * t - 0.5 * deceleration * t * t;
}

const PAD = { left: 52, right: 56, top: 12, bottom: 26 };

/** The rendered width of an element, so the chart can be drawn at real pixel size (text stays legible). */
function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(320);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** Distance over time: the tracked real push (dots) against the simulation before and after correction. */
function PushChart({ test }: { test: PushTest }) {
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [wrapper, measured] = useWidth();
  const W = Math.max(280, measured);
  const H = Math.round(Math.min(Math.max(W * 0.45, 180), 260));
  const v0 = test.fit.v0;
  const lastT = Math.max(...test.track.t, 0.1);
  const tMax = Math.max(
    lastT,
    (2 * test.predicted_distance_before_m) / v0,
    (2 * test.predicted_distance_after_m) / v0,
  );
  const sMax = Math.max(test.real_distance_m, test.predicted_distance_before_m, test.predicted_distance_after_m, ...test.track.s_m) * 1.1;
  const x = (t: number) => PAD.left + (t / tMax) * (W - PAD.left - PAD.right);
  const y = (s: number) => H - PAD.bottom - (s / sMax) * (H - PAD.top - PAD.bottom);
  const curve = (distance: number) =>
    Array.from({ length: 61 }, (_, i) => {
      const t = (i / 60) * tMax;
      return `${i === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(slide(v0, distance, t)).toFixed(1)}`;
    }).join(' ');
  const yTicks = [0, sMax / 2, sMax].map((s) => Math.round(s * 100));
  const xTicks = [0, tMax / 2, tMax];
  const last = (i: number, length: number) => i === length - 1;

  const onMove = (event: PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    setHoverT(Math.min(Math.max(fraction, 0), 1) * tMax);
  };
  const nearest =
    hoverT === null
      ? -1
      : test.track.t.reduce((best, t, i) => (Math.abs(t - hoverT) < Math.abs(test.track.t[best] - hoverT) ? i : best), 0);

  return (
    <div className="push-chart" ref={wrapper}>
      <ul className="legend" aria-hidden="true">
        <li>
          <svg width="18" height="10"><circle cx="9" cy="5" r="4" fill={COLOR.real} /></svg>Real push (tracked)
        </li>
        <li>
          <svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" stroke={COLOR.before} strokeWidth="2" strokeDasharray="4 3" /></svg>
          Simulation before
        </li>
        <li>
          <svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" stroke={COLOR.after} strokeWidth="2" /></svg>Simulation after
        </li>
      </ul>
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Distance over time. Real push slid ${cm(test.real_distance_m)}; the simulation said ${cm(
          test.predicted_distance_before_m,
        )} before correction and ${cm(test.predicted_distance_after_m)} after.`}
      >
        {yTicks.map((tick, i) => (
          <g key={tick}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(tick / 100)} y2={y(tick / 100)} className="grid" />
            <text x={PAD.left - 8} y={y(tick / 100) + 4} textAnchor="end" className="tick">
              {last(i, yTicks.length) ? `${tick} cm` : tick}
            </text>
          </g>
        ))}
        {xTicks.map((tick, i) => (
          <text key={tick} x={x(tick)} y={H - PAD.bottom + 18} textAnchor={i === 0 ? 'start' : 'middle'} className="tick">
            {last(i, xTicks.length) ? `${tick.toFixed(1)} s` : tick.toFixed(1)}
          </text>
        ))}

        <path d={curve(test.predicted_distance_before_m)} fill="none" stroke={COLOR.before} strokeWidth="2" strokeDasharray="5 4" />
        <path d={curve(test.predicted_distance_after_m)} fill="none" stroke={COLOR.after} strokeWidth="2" />
        {test.track.t.map((t, i) => (
          <circle key={i} cx={x(t)} cy={y(test.track.s_m[i])} r={i === nearest ? 5.5 : 4} fill={COLOR.real} stroke="#ffffff" strokeWidth="2" />
        ))}
        {/* Direct labels at the curves' ends, in text ink with the series color as a tick mark. */}
        {(
          [
            ['before', test.predicted_distance_before_m],
            ['after', test.predicted_distance_after_m],
          ] as const
        ).map(([name, distance]) => (
          <g key={name}>
            <line x1={W - PAD.right + 4} x2={W - PAD.right + 12} y1={y(distance)} y2={y(distance)} stroke={COLOR[name]} strokeWidth="2" />
            <text x={W - PAD.right + 15} y={y(distance) + 4} className="end-label">
              {name}
            </text>
          </g>
        ))}

        {hoverT !== null && <line x1={x(hoverT)} x2={x(hoverT)} y1={PAD.top} y2={H - PAD.bottom} className="crosshair" />}
        <rect
          x={PAD.left}
          y={PAD.top}
          width={W - PAD.left - PAD.right}
          height={H - PAD.top - PAD.bottom}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => setHoverT(null)}
        />
      </svg>
      {hoverT !== null && nearest >= 0 && (
        <p className="chart-readout">
          At {hoverT.toFixed(2)} s: real {cm(test.track.s_m[nearest])} (at {test.track.t[nearest].toFixed(2)} s) · before{' '}
          {cm(slide(v0, test.predicted_distance_before_m, hoverT))} · after {cm(slide(v0, test.predicted_distance_after_m, hoverT))}
        </p>
      )}
    </div>
  );
}

interface RealityCheckProps {
  tests: PushTest[];
  sceneDir: string;
  labelOf: (objectId: string) => string;
  onClose: () => void;
}

/** Real pushes versus the simulation: how far off it was, and how much closer after learning from them. */
export function RealityCheck({ tests, sceneDir, labelOf, onClose }: RealityCheckProps) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  const before = mean(tests.map((t) => Math.abs(t.error_before_m)));
  const after = mean(tests.map((t) => Math.abs(t.error_after_m)));
  const closer = before > 0 ? Math.round((1 - after / before) * 100) : 0;
  return (
    <section className="reality-panel" aria-labelledby="reality-title">
      <div className="reality-head">
        <h2 id="reality-title">Reality check</h2>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <p className="hero-number">
        {cm(before)} <span aria-hidden="true">→</span> {cm(after)}
      </p>
      <p className="muted">
        Average error in how far a pushed object slides, before and after the simulation learned friction from{' '}
        {tests.length} real push{tests.length > 1 ? 'es' : ''}
        {closer > 0 ? ` (${closer}% closer)` : ''}.
      </p>
      {tests.map((test, i) => (
        <article key={i} className="push-test">
          <h3>
            Push {i + 1}: {labelOf(test.object_id)}
          </h3>
          {test.video && <PushVideo src={`${sceneDir}/${test.video}`} />}
          <PushChart test={test} />
          <table className="numbers">
            <thead>
              <tr>
                <th scope="col"></th>
                <th scope="col">Friction</th>
                <th scope="col">Slide</th>
                <th scope="col">Error</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Real push</th>
                <td>{test.friction_measured.toFixed(2)}</td>
                <td>{cm(test.real_distance_m)}</td>
                <td></td>
              </tr>
              <tr>
                <th scope="row">Simulation before</th>
                <td>{test.friction_estimated.toFixed(2)}</td>
                <td>{cm(test.predicted_distance_before_m)}</td>
                <td>{cm(Math.abs(test.error_before_m))}</td>
              </tr>
              <tr>
                <th scope="row">Simulation after</th>
                <td>{test.friction_measured.toFixed(2)}</td>
                <td>{cm(test.predicted_distance_after_m)}</td>
                <td>{cm(Math.abs(test.error_after_m))}</td>
              </tr>
            </tbody>
          </table>
        </article>
      ))}
    </section>
  );
}

/** The filmed push, if the file is there; quietly hidden otherwise. */
function PushVideo({ src }: { src: string }) {
  const [missing, setMissing] = useState(false);
  if (missing) return null;
  return <video className="push-video" src={src} controls muted playsInline preload="metadata" onError={() => setMissing(true)} />;
}
