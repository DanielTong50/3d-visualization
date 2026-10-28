import { RotateCw } from 'lucide-react';
import { useMemo } from 'react';
import { poseMatrix } from '../engine/choreography';
import type { AssemblyPlan, Orientation, Part } from '../engine/plan';
import { finishedItems, manualArt, stepItems, type Drawing } from '../three/ManualArt';
import { HardwareIcon } from './HardwareIcon';

/**
 * A page of a flat-pack manual, generated from a plan: a cover, the parts, the hardware, then one
 * page per step with the drawing, letter callouts, arrows and the hardware the step uses. Pages
 * are laid out at print size (A4 at 150 dpi) and scaled to fit; the same pages are what the
 * manual reader is tested on.
 */

export const PAGE_W = 1240;
export const PAGE_H = 1754;

export type PageSpec = { kind: 'cover' } | { kind: 'parts' } | { kind: 'hardware' } | { kind: 'step'; index: number };

export function manualPages(plan: AssemblyPlan): PageSpec[] {
  return [{ kind: 'cover' }, { kind: 'parts' }, { kind: 'hardware' }, ...plan.steps.map((s) => ({ kind: 'step' as const, index: s.index }))];
}

const TURN_TEXT: Record<Orientation, string> = {
  upright: 'stand it up',
  on_back: 'lay it on its back',
  on_front: 'lay it face down',
  on_side: 'lay it on its side',
  upside_down: 'turn it upside down',
};

/** Parts grouped by letter, in letter order, with how many of each. */
export function partKinds(plan: AssemblyPlan): { label: string; part: Part; count: number }[] {
  const kinds = new Map<string, { label: string; part: Part; count: number }>();
  for (const part of plan.parts) {
    const kind = kinds.get(part.label);
    if (kind) kind.count++;
    else kinds.set(part.label, { label: part.label, part, count: 1 });
  }
  return [...kinds.values()].sort((a, b) => a.label.localeCompare(b.label));
}

function Picture({ drawing, className }: { drawing: Drawing; className?: string }) {
  return (
    <div className={`mp-picture ${className ?? ''}`} style={{ width: drawing.width, height: drawing.height }}>
      <img src={drawing.url} width={drawing.width} height={drawing.height} alt="" />
      <svg viewBox={`0 0 ${drawing.width} ${drawing.height}`} className="mp-overlay">
        <defs>
          <marker id="mp-head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="#141414" />
          </marker>
        </defs>
        {drawing.arrows.map((a, i) => (
          <line
            key={i}
            x1={a.from[0] * drawing.width}
            y1={a.from[1] * drawing.height}
            x2={a.to[0] * drawing.width}
            y2={a.to[1] * drawing.height}
            stroke="#141414"
            strokeWidth="3.2"
            markerEnd="url(#mp-head)"
          />
        ))}
        {drawing.callouts.map((c) => (
          <g key={c.partId} transform={`translate(${c.x * drawing.width} ${c.y * drawing.height})`}>
            <circle r="25" fill="#fff" stroke="#141414" strokeWidth="3" />
            <text dy="0.36em" textAnchor="middle" className="mp-letter">
              {c.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function Cover({ plan }: { plan: AssemblyPlan }) {
  const drawing = useMemo(() => manualArt().draw(`${plan.id}/cover`, finishedItems(plan), 1000, 1060), [plan]);
  const tools = [...new Set(plan.steps.flatMap((s) => s.tools))];
  const [w, h, d] = (plan.dimensions_mm ?? [0, 0, 0]).map((v) => Math.round(v / 10));
  return (
    <>
      <header className="mp-cover-head">
        <h1>{plan.title.split(' ')[0].toUpperCase()}</h1>
        <p>
          {plan.category} · {w} × {h} × {d} cm
        </p>
      </header>
      <Picture drawing={drawing} className="mp-cover-art" />
      <footer className="mp-cover-foot">
        <div className="mp-tools">
          {tools.map((t) => (
            <HardwareIcon key={t} kind={t} size={96} />
          ))}
        </div>
        <div className="mp-brand">
          <b>darius</b>
          <span>
            {plan.parts.length} parts · {plan.steps.length} steps
          </span>
        </div>
      </footer>
    </>
  );
}

function Parts({ plan }: { plan: AssemblyPlan }) {
  const kinds = partKinds(plan);
  const span = Math.max(...plan.parts.map((p) => Math.max(...p.size_mm))) * 0.001 * 1.08;
  return (
    <div className="mp-grid">
      {kinds.map(({ label, part, count }) => {
        const rotation = part.assembled?.rotation;
        const items = [{ part, matrix: poseMatrix([0, 0, 0], rotation), tone: 'new' as const }];
        const drawing = manualArt().draw(`${plan.id}/part/${label}`, items, 340, 300, span);
        return (
          <div key={label} className="mp-cell">
            <Picture drawing={drawing} />
            <div className="mp-cell-foot">
              <span className="mp-badge">{label}</span>
              <b>{count}×</b>
              <span className="mp-dim">{part.size_mm.map((v) => Math.round(v)).join(' × ')}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HardwarePage({ plan }: { plan: AssemblyPlan }) {
  return (
    <div className="mp-grid mp-hardware">
      {plan.hardware.map((h) => (
        <div key={h.id} className="mp-cell">
          <HardwareIcon kind={h.kind} size={200} />
          <div className="mp-cell-foot">
            <b>{h.count}×</b>
            <span className="mp-article">{h.id}</span>
          </div>
          <span className="mp-dim">{h.name}</span>
        </div>
      ))}
    </div>
  );
}

function StepPage({ plan, index }: { plan: AssemblyPlan; index: number }) {
  const step = plan.steps[index - 1];
  const drawing = useMemo(() => manualArt().draw(`${plan.id}/step/${index}`, stepItems(plan, index), 1120, 1180), [plan, index]);
  const previous = plan.steps[index - 2]?.orientation ?? 'upright';
  const turns = step.orientation !== previous;
  const kinds = new Map(plan.hardware.map((h) => [h.id, h]));
  return (
    <>
      <header className="mp-step-head">
        <div className="mp-step-number">{index}</div>
        <div className="mp-step-side">
          {step.hardware.length > 0 && (
            <div className="mp-box">
              {step.hardware.map((use) => (
                <div key={use.id} className="mp-box-item">
                  <HardwareIcon kind={kinds.get(use.id)?.kind ?? ''} size={72} />
                  <b>{use.count}×</b>
                  <span className="mp-article">{use.id}</span>
                </div>
              ))}
            </div>
          )}
          {step.tools.map((t) => (
            <div key={t} className="mp-tool">
              <HardwareIcon kind={t} size={84} />
            </div>
          ))}
        </div>
      </header>
      <p className="mp-caption">{step.title}</p>
      {turns && (
        <div className="mp-turn">
          <RotateCw size={44} strokeWidth={2.2} />
          <span>{TURN_TEXT[step.orientation]}</span>
        </div>
      )}
      <Picture drawing={drawing} className="mp-step-art" />
    </>
  );
}

/** One page at print size, scaled by `scale`. */
export function ManualPage({ plan, spec, scale = 1 }: { plan: AssemblyPlan; spec: PageSpec; scale?: number }) {
  const pages = manualPages(plan);
  const number = pages.findIndex((p) => JSON.stringify(p) === JSON.stringify(spec)) + 1;
  return (
    <div className="mp-frame" style={{ width: PAGE_W * scale, height: PAGE_H * scale }}>
      <article className={`mp mp-${spec.kind}`} style={{ transform: `scale(${scale})` }} data-page={number}>
        {spec.kind === 'cover' && <Cover plan={plan} />}
        {spec.kind === 'parts' && <Parts plan={plan} />}
        {spec.kind === 'hardware' && <HardwarePage plan={plan} />}
        {spec.kind === 'step' && <StepPage plan={plan} index={spec.index} />}
        <footer className="mp-folio">{number}</footer>
      </article>
    </div>
  );
}
