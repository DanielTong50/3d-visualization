import { ArrowLeft, Pause, Play, RotateCcw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Moment } from '../engine/choreography';
import { hardwareName, loadPlan, type AssemblyPlan } from '../engine/plan';
import { beliefAt, simulateTrack, type SimulatedTrack } from '../engine/simulated';
import { navigate } from '../navigation';
import { Stage, type FrameInfo } from '../three/Stage';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { BeliefBars } from '../ui/BeliefBars';
import { DetectionLayer } from '../ui/DetectionLayer';
import { FurniturePicker } from '../ui/FurniturePicker';
import { HardwareIcon } from '../ui/HardwareIcon';

const PHASES: { phase: Moment['phase']; label: string }[] = [
  { phase: 'reorient', label: 'Turn' },
  { phase: 'locate', label: 'Locate' },
  { phase: 'place', label: 'Place' },
  { phase: 'fasten', label: 'Fasten' },
  { phase: 'verify', label: 'Verify' },
];

const HUD_MS = 60;

/** The close-up of the piece going in: right column, under the belief panel (CSS pixels). */
function closeUpRect(width: number) {
  const w = Math.min(280, width * 0.26);
  return { right: 16, top: 336, width: w, height: Math.round(w * 0.68) };
}

/**
 * The copilot in a simulated session: a choreographed build of a library product, seen through a
 * handheld camera, with the full AR layer and the tracking stack running live on simulated
 * perception. Everything on screen is computed from the plan's geometry and the tracker; nothing
 * is pre-rendered.
 */
export function ArDemo({ furniture }: { furniture: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const timeRef = useRef(0);
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [track, setTrack] = useState<SimulatedTrack | null>(null);
  const playingRef = useRef(playing);
  const speedRef = useRef(speed);
  playingRef.current = playing;
  speedRef.current = speed;

  useEffect(() => {
    let cancelled = false;
    setPlan(null);
    loadPlan(furniture)
      .then((loaded) => !cancelled && setPlan(loaded))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [furniture]);

  useEffect(() => {
    const container = containerRef.current;
    if (!plan || !container) return;
    const stage = new Stage(container, plan, 'ar');
    stage.frame(0, 0.1);
    const place = () => (stage.inset = container.clientWidth > 1000 ? closeUpRect(container.clientWidth) : null);
    place();
    window.addEventListener('resize', place);
    stageRef.current = stage;
    setTrack(simulateTrack(stage.choreography));
    timeRef.current = 0;
    let last = performance.now();
    let hudAt = 0;
    let raf = 0;
    const loop = (now: number) => {
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      if (playingRef.current) {
        timeRef.current += dt * speedRef.current;
        if (timeRef.current >= stage.choreography.duration) timeRef.current = 0;
      }
      const info = stage.update(timeRef.current, now / 1000);
      stage.render();
      if (now - hudAt > HUD_MS) {
        hudAt = now;
        setFrame(info);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', place);
      stage.dispose();
      stageRef.current = null;
    };
  }, [plan]);

  const choreography = stageRef.current?.choreography ?? null;
  const starts = useMemo(() => choreography?.stepStarts() ?? [], [choreography]);
  if (error) {
    return (
      <div className="ar">
        <p className="ar-error">Couldn't load this plan: {error}</p>
      </div>
    );
  }
  const moment = frame?.moment;
  const steps = plan?.steps.length ?? 0;
  const step = plan && moment && moment.step >= 1 && moment.step <= steps ? plan.steps[moment.step - 1] : null;
  const belief = track && moment ? beliefAt(track, moment.time, steps) : null;
  const done = moment?.phase === 'done';
  const tracked = frame?.detections.reduce((sum, d) => sum + d.parts, 0) ?? 0;
  const left = choreography && moment ? choreography.inTray(moment.time) : new Map<string, number>();
  const inset = stageRef.current?.inset ?? null;
  const joining = choreography && moment ? choreography.joining(moment).length : 0;
  const seek = (t: number) => {
    timeRef.current = t;
  };

  return (
    <div className="ar">
      <div ref={containerRef} className="ar-viewport" />
      {frame && <DetectionLayer detections={frame.detections} />}
      <div className="ar-vignette" aria-hidden="true" />

      <header className="ar-top">
        <button type="button" className="icon-btn" onClick={() => navigate({ page: 'home' })} aria-label="Back">
          <ArrowLeft size={17} />
        </button>
        <div className="ar-title">
          <span className="eyebrow">Copilot · simulated session</span>
          <strong>{plan?.title ?? 'Loading…'}</strong>
        </div>
        <div className="ar-status">
          <span className="chip">
            <span className="rec" /> <b>LIVE</b>
          </span>
          <span className="chip">
            <b>{tracked}</b> {tracked === 1 ? 'part' : 'parts'} tracked
          </span>
          <FurniturePicker current={furniture} onPick={(id) => navigate({ page: 'ar', furniture: id })} />
        </div>
      </header>

      {plan && moment && (
        <aside className="ar-graph glass">
          <div className="panel-head">
            <span className="eyebrow track">Assembly graph</span>
            <span className="faint mono">
              {plan.joints.filter((j) => plan.steps.findIndex((s) => s.joints.includes(j.id)) < moment.completed).length}/
              {plan.joints.length} joints
            </span>
          </div>
          <AssemblyGraph plan={plan} completed={moment.completed} active={step ? step.index : null} />
        </aside>
      )}

      {plan && moment && choreography && (
        <aside className="ar-hardware glass">
          <div className="panel-head">
            <span className="eyebrow">Hardware</span>
            <span className="faint mono">{[...left.values()].reduce((a, b) => a + b, 0)} in the tray</span>
          </div>
          <ul className="fittings">
            {plan.hardware.map((item) => {
              const remaining = left.get(item.id) ?? 0;
              const now = frame?.closeUp?.hardware === item.id;
              return (
                <li key={item.id} className={now ? 'is-now' : remaining === 0 ? 'is-empty' : ''}>
                  <HardwareIcon kind={item.kind} size={34} />
                  <span>{item.name}</span>
                  <b>
                    {remaining}/{item.count}
                  </b>
                </li>
              );
            })}
          </ul>
        </aside>
      )}

      {plan && frame?.closeUp && inset && (
        <div className="ar-closeup" style={{ top: inset.top, width: inset.width, height: inset.height }}>
          <span>
            Close-up · {hardwareName(plan, frame.closeUp.hardware)} · {frame.closeUp.done}/{frame.closeUp.total}
          </span>
        </div>
      )}

      {plan && belief && (
        <aside className="ar-belief glass">
          <div className="panel-head">
            <span className="eyebrow done">Progress belief</span>
            <span className="faint mono">HMM · {Math.round(belief.confidence * 100)}%</span>
          </div>
          <BeliefBars plan={plan} belief={belief} />
          <p className="panel-note">
            P(steps done) from per-frame step verification, filtered over the manual's order. Simulated observations, ~1 in 12 misread on
            purpose.
          </p>
        </aside>
      )}

      {plan && moment && (
        <section className={`ar-step glass${done ? ' is-done' : ''}`} aria-live="polite">
          <span className="step-number">{done ? '✓' : step ? step.index : '·'}</span>
          {done ? (
            <>
              <span className="eyebrow done">Verified</span>
              <h2>{plan.title} is assembled.</h2>
              <p className="muted">
                {plan.parts.length} parts, {plan.joints.length} joints, {plan.hardware.reduce((s, h) => s + h.count, 0)} pieces of hardware,{' '}
                {steps} steps.
              </p>
            </>
          ) : step ? (
            <>
              <div className="step-head">
                <span className="eyebrow">
                  Step {step.index} of {steps}
                </span>
                <ol className="phases">
                  {PHASES.filter((p) => p.phase !== 'reorient' || moment.phase === 'reorient').map((p) => {
                    const order = PHASES.findIndex((q) => q.phase === p.phase);
                    const now = PHASES.findIndex((q) => q.phase === moment.phase);
                    return (
                      <li key={p.phase} className={order < now ? 'is-past' : order === now ? 'is-now' : ''}>
                        {p.label}
                      </li>
                    );
                  })}
                </ol>
              </div>
              <h2>{step.title}</h2>
              <p className="muted">
                {moment.phase === 'reorient'
                  ? (step.actions.find((a) => ['flip', 'stand_up', 'rotate'].includes(a.kind))?.text ??
                    `Turn the assembly ${step.orientation.replace('_', ' ')}`)
                  : step.instruction}
              </p>
              <div className="step-chips">
                {step.parts.map((id) => {
                  const part = plan.parts.find((p) => p.id === id)!;
                  return (
                    <span key={id} className="chip guide">
                      <span className="letter">{part.label}</span> {part.name}
                    </span>
                  );
                })}
                {joining > 0 && (
                  <span className="chip guide">
                    <span className="letter">+</span> subassembly · {joining} parts
                  </span>
                )}
                {step.hardware.map((h) => (
                  <span key={h.id} className="chip">
                    <b>{h.count}×</b> {hardwareName(plan, h.id)}
                  </span>
                ))}
              </div>
              {frame?.closeUp && (
                <div className="fasten-progress">
                  <span>
                    {frame.closeUp.done}/{frame.closeUp.total} in
                  </span>
                  <progress value={frame.closeUp.done} max={frame.closeUp.total} />
                  <span>{step.tools[0] ?? 'by hand'}</span>
                </div>
              )}
            </>
          ) : moment.step > steps ? (
            <>
              <span className="eyebrow">Last move</span>
              <h2>Stand it the right way up</h2>
              <p className="muted">Every step is fastened. Turn the {plan.category} upright and it is done.</p>
            </>
          ) : (
            <>
              <span className="eyebrow track">Scanning</span>
              <h2>Finding the parts</h2>
              <p className="muted">{plan.parts.length} parts expected from the manual. Matching each one by shape and size.</p>
            </>
          )}
        </section>
      )}

      {choreography && moment && (
        <footer className="ar-timeline">
          <button type="button" className="icon-btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button type="button" className="icon-btn" onClick={() => seek(0)} aria-label="Restart">
            <RotateCcw size={15} />
          </button>
          <div className="scrub">
            <input
              type="range"
              min={0}
              max={choreography.duration}
              step={0.05}
              value={moment.time}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Session time"
            />
            <div className="scrub-marks" aria-hidden="true">
              {starts.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  style={{ left: `${(s / choreography.duration) * 100}%` }}
                  className={moment.completed > i ? 'is-done' : moment.step === i + 1 ? 'is-now' : ''}
                  onClick={() => seek(s + 0.01)}
                  title={plan?.steps[i].title}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
          <button type="button" className="chip speed" onClick={() => setSpeed((s) => (s === 1 ? 2 : s === 2 ? 0.5 : 1))}>
            <b>{speed}×</b>
          </button>
        </footer>
      )}
    </div>
  );
}
