import { ArrowLeft, Check, RotateCcw } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { loadPlan, type AssemblyPlan } from '../engine/plan';
import { loadManualResult, manualPageUrl, matchRecovered, type ManualResult } from '../engine/results';
import { navigate } from '../navigation';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { FurniturePicker } from '../ui/FurniturePicker';
import { ManualPage, manualPages, PAGE_H } from '../ui/ManualPage';

/**
 * The manual reader, replayed: the generated manual's pages are scanned one by one while what
 * Muse Spark read from them (stored by `darius eval`) fills in: the parts, the hardware, then
 * every step's joints, which build the assembly graph edge by edge. At the end, the recovered
 * graph is scored against the plan the manual was drawn from.
 */

const SCAN_S = 1.5;
const PAGE_S = 2.3;
const ORIENTATION: Record<string, string> = {
  upright: 'upright',
  on_back: 'on its back',
  on_front: 'face down',
  on_side: 'on its side',
  upside_down: 'upside down',
};

function percent(value: number): string {
  return `${Math.round(value * 1000) / 10}`;
}

export function ManualDemo({ furniture }: { furniture: string }) {
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const [result, setResult] = useState<ManualResult | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const pageRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(0.38);

  useEffect(() => {
    Promise.all([loadPlan(furniture), loadManualResult(furniture)])
      .then(([p, r]) => {
        setPlan(p);
        setResult(r);
      })
      .catch((e: Error) => setError(e.message));
  }, [furniture]);

  const pages = useMemo(() => (plan ? manualPages(plan) : []), [plan]);
  const total = pages.length * PAGE_S + 0.6;

  useEffect(() => {
    if (!playing || !pages.length) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // A frame's timestamp can precede the effect's performance.now(): never step backwards.
      const dt = Math.max(0, (now - last) / 1000);
      last = now;
      setTime((t) => {
        const next = t + dt;
        if (next >= total) {
          setPlaying(false);
          return total;
        }
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, pages.length, total]);

  useEffect(() => {
    const fit = () => setScale(Math.min(0.46, Math.max(0.2, (window.innerHeight - 220) / PAGE_H)));
    fit();
    window.addEventListener('resize', fit);
    return () => window.removeEventListener('resize', fit);
  }, []);

  const current = Math.min(pages.length - 1, Math.floor(time / PAGE_S));
  const scanned = Math.min(1, (time - current * PAGE_S) / SCAN_S);
  // Pages whose reading has come back.
  const read = Math.min(pages.length, Math.floor(Math.max(0, time - SCAN_S) / PAGE_S) + (time >= SCAN_S ? 1 : 0));
  const finished = time >= total - 0.01;

  // Elements on the page light up as the scan line passes them.
  useLayoutEffect(() => {
    const page = pageRef.current?.querySelector('.mp');
    if (!page) return;
    const top = page.getBoundingClientRect().top;
    page.querySelectorAll<HTMLElement | SVGElement>('[data-read]').forEach((element) => {
      const at = (element.getBoundingClientRect().top - top) / (PAGE_H * scale);
      element.style.setProperty('--at', String(Math.max(0, Math.min(1, at))));
    });
  }, [current, scale, plan]);

  const matched = useMemo(() => (plan && result ? matchRecovered(plan, result.recovered) : null), [plan, result]);
  const stepsRead = Math.max(0, read - 3);
  const known = useMemo(() => {
    const set = new Set<string>();
    if (matched) for (let k = 1; k <= stepsRead; k++) for (const id of matched.known.get(k) ?? []) set.add(id);
    return set;
  }, [matched, stepsRead]);
  const extra = matched?.extra.filter((e) => e.step <= stepsRead) ?? [];

  if (error) return <p className="page-error">Couldn't load this manual: {error}</p>;
  if (!plan || result === undefined)
    return (
      <div className="screen-loading">
        <span className="loader" />
      </div>
    );

  const seek = (t: number) => {
    setTime(t);
    setPlaying(true);
  };
  const reading = result?.reading;
  const recoveredJoints = result ? result.recovered.joints.length : 0;

  return (
    <div className="demo manual-demo">
      <header className="demo-top">
        <button type="button" className="icon-btn" onClick={() => navigate({ page: 'home' })} aria-label="Back">
          <ArrowLeft size={17} />
        </button>
        <div className="ar-title">
          <span className="eyebrow track">Manual reader · replay of a real run</span>
          <strong>{plan.title}</strong>
        </div>
        <div className="demo-actions">
          {result && <span className="chip mono">{result.model}</span>}
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => seek(0)}>
            <RotateCcw size={14} /> Replay
          </button>
          <FurniturePicker current={furniture} onPick={(id) => navigate({ page: 'manual', furniture: id })} />
        </div>
      </header>

      <main className="manual-grid">
        <section className="manual-viewer" aria-label="Manual pages">
          <div
            className={`manual-sheet${scanned < 1 ? ' is-scanning' : ' is-read'}`}
            ref={pageRef}
            style={{ '--scan': `${SCAN_S}s` } as React.CSSProperties}
          >
            <ManualPage key={current} plan={plan} spec={pages[current]} scale={scale} />
            {scanned < 1 && <div className="scanline" style={{ top: `${scanned * 100}%` }} />}
          </div>
          <ol className="manual-thumbs">
            {pages.map((_, i) => (
              <li key={i}>
                <button type="button" className={i === current ? 'is-now' : i < read ? 'is-read' : ''} onClick={() => seek(i * PAGE_S)}>
                  <img src={manualPageUrl(plan.id, i + 1)} alt={`Page ${i + 1}`} loading="lazy" />
                  {i < read && <Check size={12} className="thumb-check" />}
                </button>
              </li>
            ))}
          </ol>
        </section>

        <section className="manual-reading" aria-label="What the model read">
          {!reading ? (
            <div className="glass panel-pad">
              <span className="eyebrow">No stored run</span>
              <p className="muted">
                Run <code>darius eval --manuals --only {plan.id}</code> to read this manual with the model.
              </p>
            </div>
          ) : (
            <>
              <div className="reading-side">
                <div className="glass panel-pad reading-graph">
                  <div className="panel-head">
                    <span className="eyebrow track">Recovered assembly graph</span>
                    <span className="faint mono">
                      {known.size}/{plan.joints.length} joints
                      {extra.length > 0 && ` +${extra.length} extra`}
                    </span>
                  </div>
                  <AssemblyGraph plan={plan} completed={stepsRead} known={known} extra={extra} active={null} />
                  <p className="panel-note">Nodes are parts at their place in the piece; every edge was read off a step page.</p>
                </div>
                {finished && result && (
                  <div className="glass panel-pad scorecard is-in">
                    <div className="panel-head">
                      <span className="eyebrow done">Scored against the plan</span>
                      <span className="faint mono">
                        {result.pages} pages · {result.seconds.toFixed(0)} s
                        {result.usage.completion_tokens ? ` · ${(result.usage.completion_tokens / 1000).toFixed(1)}k tokens out` : ''}
                      </span>
                    </div>
                    <div className="scores">
                      <div>
                        <b>{percent(result.score.edges_f1)}</b>
                        <span>edges F1</span>
                      </div>
                      <div>
                        <b>{percent(result.score.parts_f1)}</b>
                        <span>parts F1</span>
                      </div>
                      <div>
                        <b>{percent(result.score.steps_exact)}</b>
                        <span>steps exact</span>
                      </div>
                      <div>
                        <b>{percent(result.score.hardware_accuracy)}</b>
                        <span>hardware counts</span>
                      </div>
                    </div>
                    <p className="panel-note">
                      {recoveredJoints} joints recovered, {plan.joints.length} in the plan
                      {matched && matched.missed.length > 0 ? `, ${matched.missed.length} missed` : ''}
                      {extra.length > 0 ? `, ${extra.length} not in the plan (red)` : ''}.
                      {result.notes.length > 0 && ` Notes: ${result.notes.join('; ')}.`}
                    </p>
                  </div>
                )}
              </div>

              <div className="reading-log glass panel-pad">
                <div className="panel-head">
                  <span className="eyebrow">Structured output</span>
                  <span className="faint mono">
                    {read}/{pages.length} pages
                  </span>
                </div>
                {read >= 1 && (
                  <div className="log-row is-in">
                    <span className="log-key">product</span>
                    <span>
                      {reading.title} <span className="faint">· {reading.category}</span>
                    </span>
                  </div>
                )}
                {read >= 2 && (
                  <div className="log-row is-in">
                    <span className="log-key">parts</span>
                    <span className="log-chips">
                      {reading.parts.map((p) => (
                        <span key={p.label} className="chip" title={p.size_mm.join(' × ')}>
                          <span className="letter">{p.label}</span> {p.count > 1 ? `${p.count}× ` : ''}
                          {p.name}
                        </span>
                      ))}
                    </span>
                  </div>
                )}
                {read >= 3 && (
                  <div className="log-row is-in">
                    <span className="log-key">hardware</span>
                    <span className="log-chips">
                      {reading.hardware.map((h) => (
                        <span key={h.id} className="chip mono">
                          <b>{h.count}×</b> {h.id} <span className="faint">{h.kind.replace('_', ' ')}</span>
                        </span>
                      ))}
                    </span>
                  </div>
                )}
                {reading.steps.slice(0, stepsRead).map((step) => (
                  <div key={step.index} className="log-row log-step is-in">
                    <span className="log-key">step {step.index}</span>
                    <span>
                      <span className="log-title">{step.title}</span>
                      <span className="log-meta">
                        {step.adds.length > 0 && <>adds {step.adds.join(', ')} · </>}
                        {ORIENTATION[step.orientation]}
                        {step.hardware.map((h) => ` · ${h.count}× ${h.id}`).join('')}
                      </span>
                      <span className="log-chips">
                        {step.connections.map((c, i) => (
                          <span key={i} className="chip edge">
                            {c.a}–{c.b} <span className="faint">{c.kind.replace('_', ' ')}</span>
                          </span>
                        ))}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
