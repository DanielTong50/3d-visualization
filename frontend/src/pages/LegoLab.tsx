import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Box,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Code2,
  Cpu,
  Download,
  ExternalLink,
  FileText,
  Focus,
  GitBranch,
  Layers3,
  Maximize2,
  Network,
  Package,
  Pause,
  Play,
  RotateCcw,
  Search,
  ShieldCheck,
  SkipForward,
  Sparkles,
  Terminal,
  Workflow,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { LegoStage } from '../lego/LegoStage';
import {
  BUILD_IDS,
  downloadJson,
  instructionUrl,
  isOfficialAssetUrl,
  parseCompiledPlan,
  PHASES,
  STEP_SECONDS,
  timelineDuration,
  timelineAt,
  type LegoPlan,
} from '../lego/types';
import '../lego/lego.css';

type Inspector = 'sequence' | 'graph' | 'inventory' | 'plan';
const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '');
const TITLES: Record<string, { label: string; detail: string; color: string; glyph: string }> = {
  'signal-tower': { label: 'Signal tower', detail: 'Vertical stacking', color: '#e74432', glyph: 'tower' },
  bridge: { label: 'The bridge', detail: 'Spans & dependencies', color: '#dcae1c', glyph: 'bridge' },
  stairway: { label: 'Stairway', detail: 'Layered assembly', color: '#2361d7', glyph: 'stairs' },
};
function BrickGlyph({ type = 'tower', color = '#d94331', small = false }: { type?: string; color?: string; small?: boolean }) {
  return (
    <svg viewBox="0 0 64 64" width={small ? 34 : 52} height={small ? 34 : 52} fill="none" aria-hidden="true">
      {type === 'bridge' ? (
        <>
          <path d="M8 31 32 18l24 13-24 13Z" fill={color} />
          <path d="m8 31 24 13v12L8 43Z" fill={color} opacity=".7" />
          <path d="m32 44 24-13v12L32 56Z" fill={color} opacity=".9" />
          <path d="M12 38v18l8-5V42m24-5v16l8-5V33" stroke={color} strokeWidth="8" />
        </>
      ) : type === 'stairs' ? (
        <>
          <path d="m8 41 16-9 8 4 8-14 16 9v15L32 60 8 46Z" fill={color} opacity=".7" />
          <path d="m8 41 16-9 8 4V20l16-9 8 5v15L40 40l-8-4-8 14Z" fill={color} />
          <path d="m32 20 16-9 8 5-16 9Z" fill="white" opacity=".25" />
        </>
      ) : (
        <>
          <path d="m15 19 17-9 17 9-17 10Z" fill={color} />
          <path d="M15 19v30l17 10V29Z" fill={color} opacity=".75" />
          <path d="M32 29v30l17-10V19Z" fill={color} opacity=".9" />
          <path d="m15 31 17 10 17-10M15 42l17 10 17-10" stroke="white" strokeOpacity=".4" strokeWidth="1.4" />
          <ellipse cx="27" cy="17" rx="4" ry="2.2" fill="white" opacity=".35" />
          <ellipse cx="38" cy="23" rx="4" ry="2.2" fill="white" opacity=".35" />
        </>
      )}
    </svg>
  );
}
function DariusMark() {
  return (
    <svg width="29" height="29" viewBox="0 0 29 29" aria-hidden="true">
      <rect x="1" y="1" width="27" height="27" rx="6" fill="#e63b2e" />
      <rect x="7" y="8" width="15" height="14" rx="2" fill="#fff7eb" />
      <rect x="9" y="5" width="4" height="5" rx="1" fill="#fff7eb" />
      <rect x="17" y="5" width="4" height="5" rx="1" fill="#fff7eb" />
      <rect x="11" y="12" width="6" height="6" rx="1" fill="#e63b2e" />
    </svg>
  );
}
function Graph({ plan, completed, select }: { plan: LegoPlan; completed: number; select: (index: number) => void }) {
  const count = plan.steps.length;
  const height = Math.max(200, Math.ceil(count / 3) * 68 + 75);
  const node = (id: string) => {
    if (id === 'baseplate') return { x: 121, y: 18, i: -1 };
    const i = plan.steps.findIndex((s) => s.brick_id === id);
    return { x: 43 + (i % 3) * 78, y: 78 + Math.floor(i / 3) * 68, i };
  };
  return (
    <div className="lg-graph">
      <div className="lg-inspector-note">Stud contacts form directed support edges. Select a brick to inspect its step.</div>
      <svg viewBox={`0 0 244 ${height}`} role="img" aria-label="Assembly support dependency graph">
        <defs>
          <marker id="lg-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0 0 6 3 0 6" fill="none" stroke="#acaca3" />
          </marker>
        </defs>
        {plan.connections.map((c) => {
          const a = node(c.lower);
          const b = node(c.upper);
          if ((a.i < 0 && c.lower !== 'baseplate') || b.i < 0) return null;
          return (
            <path
              key={c.id}
              d={`M${a.x} ${a.y + 12} C${a.x} ${a.y + 38} ${b.x} ${b.y - 38} ${b.x} ${b.y - 17}`}
              stroke={b.i < completed ? '#4f8b62' : '#b8b8ad'}
              strokeWidth="1.5"
              fill="none"
              markerEnd="url(#lg-arrow)"
            />
          );
        })}
        <g>
          <rect x="89" y="7" width="64" height="22" rx="4" fill="#e9eee0" stroke="#cdd9bb" />
          <text x="121" y="21" textAnchor="middle" fill="#7d9362" fontSize="7" fontFamily="monospace">
            BASEPLATE
          </text>
        </g>
        {plan.steps.map((s, i) => {
          const b = plan.assembly.bricks.find((p) => p.id === s.brick_id)!;
          const p = node(b.id);
          return (
            <g
              key={b.id}
              tabIndex={0}
              role="button"
              aria-label={`Go to step ${i + 1}: ${b.id}`}
              onClick={() => select(i)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') select(i);
              }}
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={p.x - 22}
                y={p.y - 15}
                width="44"
                height="30"
                rx="6"
                fill={i < completed ? '#e6efe6' : '#fff'}
                stroke={i < completed ? '#80a28a' : '#d8d8cd'}
              />
              <rect x={p.x - 14} y={p.y - 5} width="12" height="10" rx="2" fill={b.color_hex} />
              <text x={p.x + 8} y={p.y + 4} textAnchor="middle" fill="#55584e" fontSize="8" fontFamily="monospace">
                {i + 1}
              </text>
              <text x={p.x} y={p.y + 28} textAnchor="middle" fill="#777b70" fontSize="7.5" fontFamily="monospace">
                {b.id}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="lg-graph-key">
        <span>
          <i /> pending
        </span>
        <span>
          <i className="done" /> assembled
        </span>
      </div>
    </div>
  );
}

export function LegoLab() {
  const [selected, setSelected] = useState(() => {
    const id = new URLSearchParams(window.location.search).get('build');
    return BUILD_IDS.includes(id as (typeof BUILD_IDS)[number]) ? id! : 'signal-tower';
  });
  const [plans, setPlans] = useState<Record<string, LegoPlan>>({});
  const [error, setError] = useState('');
  const [time, setTime] = useState(12.5);
  const [playing, setPlaying] = useState(false);
  const [exploded, setExploded] = useState(false);
  const [robot, setRobot] = useState(true);
  const [tab, setTab] = useState<Inspector>('sequence');
  const [setNumber, setSetNumber] = useState('');
  const [lookup, setLookup] = useState<{ url: string; message: string; pdfs?: { url: string; label: string }[] } | null>(null);
  const [lookupBusy, setLookupBusy] = useState(false);
  const [lookupError, setLookupError] = useState('');
  const [exportBusy, setExportBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [help, setHelp] = useState(false);
  const [stageError, setStageError] = useState('');
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<LegoStage | null>(null);
  const importFile = useRef<HTMLInputElement>(null);
  const plan = plans[selected];
  const total = timelineDuration(plan?.steps.length ?? 1);
  const timeline = timelineAt(time, plan?.steps.length ?? 1);
  const current = plan?.steps[timeline.active];
  const currentBrick = plan?.assembly.bricks.find((b) => b.id === current?.brick_id);

  useEffect(() => {
    let active = true;
    Promise.all(
      BUILD_IDS.map(async (id) => {
        const response = await fetch(`${import.meta.env.BASE_URL}assemblies/lego/${id}/plan.json`);
        if (!response.ok)
          throw new Error(`Build ${id} could not be loaded (${response.status}). Generate fixtures with darius lego export.`);
        return [id, await response.json()] as [string, LegoPlan];
      }),
    )
      .then((entries) => {
        if (active) setPlans(Object.fromEntries(entries));
      })
      .catch((e: Error) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!host.current) return;
    try {
      stage.current = new LegoStage(host.current);
    } catch {
      setStageError('WebGL is unavailable. The compiled sequence, graph and exports remain available below.');
    }
    return () => {
      stage.current?.dispose();
      stage.current = null;
    };
  }, []);
  useEffect(() => {
    if (plan) {
      stage.current?.setPlan(plan);
      stage.current?.setState(time, exploded, robot);
    }
  }, [plan]);
  useEffect(() => {
    stage.current?.setState(time, exploded, robot);
  }, [time, exploded, robot]);
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.1);
      previous = now;
      setTime((t) => {
        if (t + dt >= total) {
          setPlaying(false);
          return total;
        }
        return t + dt;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, total]);
  useEffect(() => {
    if (notice) {
      const id = setTimeout(() => setNotice(''), 6500);
      return () => clearTimeout(id);
    }
  }, [notice]);

  async function importPlan(file: File | undefined) {
    if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('Plan files must be smaller than 2 MB.');
      const parsed = parseCompiledPlan(JSON.parse(await file.text()));
      setPlans((existing) => ({ ...existing, imported: parsed }));
      setSelected('imported');
      setTime(0);
      setPlaying(false);
      setExploded(false);
      setTab('plan');
      setNotice('Compiled artifact imported. Browser checks format; use the engine compiler to validate support geometry.');
    } catch (e) {
      setNotice((e as Error).message);
    }
    if (importFile.current) importFile.current.value = '';
  }
  function choose(id: string) {
    setSelected(id);
    setTime(0);
    setPlaying(false);
    setExploded(false);
    const url = new URL(window.location.href);
    url.searchParams.set('build', id);
    window.history.replaceState(null, '', url);
  }
  function seek(index: number) {
    setPlaying(false);
    setTime(Math.min(index * STEP_SECONDS, total));
  }
  function play() {
    if (time >= total) setTime(0);
    setExploded(false);
    setPlaying((p) => !p);
  }
  async function findInstructions(e: React.FormEvent) {
    e.preventDefault();
    setLookupError('');
    setLookup(null);
    let url: string;
    try {
      url = instructionUrl(setNumber.trim());
    } catch (e) {
      setLookupError((e as Error).message);
      return;
    }
    setLookupBusy(true);
    if (!API && setNumber.trim() !== '10698') {
      setLookup({
        url,
        message:
          'Open the official set page to view available instruction booklets. Connect the local API for server-side PDF discovery, or try 10698 for the bundled official metadata.',
      });
      setLookupBusy(false);
      return;
    }
    try {
      const response = await fetch(
        API ? `${API}/lego/instructions/${setNumber.trim()}` : `${import.meta.env.BASE_URL}assemblies/lego/instructions/10698.json`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The lookup service is unavailable.');
      const raw = data.manuals ?? data.booklets ?? data.instructions ?? [];
      const pdfs = Array.isArray(raw)
        ? raw
            .map((p: Record<string, unknown>, i: number) => ({
              url: String(p.pdf_url ?? p.url ?? ''),
              label: String(p.title ?? p.description ?? `Instruction booklet ${i + 1}`),
            }))
            .filter((p: { url: string }) => isOfficialAssetUrl(p.url))
        : [];
      setLookup({
        url,
        message: String(
          data.message ??
            data.note ??
            `${data.title ?? 'Official set'} · ${pdfs.length} instruction booklets discovered. ${!API ? 'Bundled official metadata, retrieved 2026-10-29.' : 'From the official LEGO instructions service.'} Pose verification remains a research stage.`,
        ),
        pdfs,
      });
    } catch {
      setLookup({ url, message: 'The lookup service is unavailable. You can still open the official instructions page for this set.' });
    } finally {
      setLookupBusy(false);
    }
  }
  function exportPlanningDraft(message: string) {
    if (!plan) return;
    downloadJson(`${plan.assembly.id}-training-draft.json`, {
      schema_version: '1.0',
      status: 'planning_draft',
      source: 'deterministic_assembly_compiler',
      real_robot_data: false,
      assembly_id: plan.assembly.id,
      plan,
      observation_contract: ['wrist_rgb', 'overhead_rgb', 'joint_positions', 'gripper_width', 'tool_pose', 'contact_force'],
      action_contract: { type: 'cartesian_delta_and_gripper', frame: 'calibrated_robot_base', units: 'meters_radians' },
      training_status:
        'This planning draft is not a trained robot policy. These waypoints require robot calibration, IK, collision checks, force control, and real demonstrations.',
    });
    setNotice(message);
  }
  async function exportTraining() {
    if (!plan) return;
    if (selected === 'imported' || plan.assembly.source.review_status === 'pending') {
      exportPlanningDraft('Planning draft downloaded. Use the engine CLI to review imported geometry and generate a synthetic dataset.');
      return;
    }
    setExportBusy(true);
    try {
      const url = API
        ? `${API}/lego/builds/${selected}/training?episodes=4&seed=7`
        : `${import.meta.env.BASE_URL}assemblies/lego/${selected}/training.json`;
      const response = await fetch(url);
      if (!response.ok) throw new Error('Training export unavailable.');
      const data = await response.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid training bundle.');
      downloadJson(`${selected}-synthetic-training.json`, data);
      setNotice('Synthetic training bundle downloaded. These are planning trajectories, not robot demonstrations.');
    } catch {
      exportPlanningDraft(
        'The synthetic bundle could not be loaded. A planning draft was downloaded; generate trajectories with the engine CLI.',
      );
    } finally {
      setExportBusy(false);
    }
  }

  return (
    <div className="lego-lab">
      <header className="lg-header">
        <a href={import.meta.env.BASE_URL} className="lg-brand">
          <DariusMark />
          <span>
            darius<span className="lg-brand-slash">/</span>
            <b>LEGO</b>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#assembly" className="active">
            Assembly lab
          </a>
          <a href="#training">
            Training pipeline <span className="lg-soon">DRAFT</span>
          </a>
          <a href="#instructions">Public instructions</a>
        </nav>
        <a
          className="lg-source-link"
          href="https://github.com/DanielTong50/3d-visualization/tree/darius-lego"
          target="_blank"
          rel="noreferrer"
        >
          <GitBranch size={13} /> darius-lego <ArrowUpRight size={13} />
        </a>
      </header>
      <main>
        <section className="lg-intro" id="assembly">
          <div>
            <div className="lg-eyebrow">
              <span /> THE ASSEMBLY INTELLIGENCE LAB
            </div>
            <h1>
              From instructions
              <br />
              to <span>intelligence.</span>
              <span className="lg-title-dot">▦</span>
            </h1>
            <p>
              Every brick is a decision. Turn a build into a dependency graph,
              <br className="lg-wide-break" /> a robot motion plan, and the starting point for a learning system.
            </p>
          </div>
          <div className="lg-intro-aside">
            <div className="lg-mini-flow">
              <FileText />
              <span />
              <Network />
              <span />
              <Workflow />
              <span />
              <Cpu />
            </div>
            <div className="lg-intro-flow-labels">
              <span>MANUAL</span>
              <span>GRAPH</span>
              <span>MOTION</span>
              <span>POLICY</span>
            </div>
            <p>
              Built for the leap from
              <br />
              <b>knowing how → learning to do.</b>
            </p>
            <span className="lg-status">
              <i /> COMPILER LIVE <span>·</span> ROBOT IN PREVIEW
            </span>
          </div>
        </section>
        <section className="lg-workbench" aria-label="Interactive LEGO assembly workbench">
          <div className="lg-workbench-bar">
            <div>
              <span className="lg-window-dot red" />
              <span className="lg-window-dot yellow" />
              <span className="lg-window-dot green" />
              <span className="lg-bar-title">ASSEMBLY WORKSPACE</span>
            </div>
            <div className="lg-bar-right">
              <span className="lg-online-dot" /> LOCAL, DETERMINISTIC{' '}
              <button aria-label="About this simulation" onClick={() => setHelp(!help)}>
                <CircleHelp size={15} />
              </button>
            </div>
          </div>
          <div className="lg-workspace-grid">
            <aside className="lg-library">
              <div className="lg-panel-label">
                <Package size={13} /> BUILD LIBRARY <span>03</span>
              </div>
              <p className="lg-library-note">Small builds. Real constraints.</p>
              <div className="lg-build-list">
                {BUILD_IDS.map((id, i) => (
                  <button
                    key={id}
                    className={`lg-build ${id === selected ? 'selected' : ''}`}
                    onClick={() => choose(id)}
                    aria-pressed={id === selected}
                  >
                    <span className="lg-build-art">
                      <BrickGlyph type={TITLES[id].glyph} color={TITLES[id].color} />
                    </span>
                    <span className="lg-build-info">
                      <span className="lg-build-index">
                        0{i + 1} <span>ORIGINAL BUILD</span>
                      </span>
                      <strong>{TITLES[id].label}</strong>
                      <small>{TITLES[id].detail}</small>
                      <span className="lg-build-meta">
                        {plans[id]?.assembly.bricks.length ?? '—'} bricks <span>·</span> {plans[id]?.steps.length ?? '—'} steps
                      </span>
                    </span>
                    <ChevronRight size={13} />
                  </button>
                ))}
              </div>
              <div className="lg-library-bottom">
                <input
                  ref={importFile}
                  className="visually-hidden"
                  type="file"
                  accept=".json,application/json"
                  aria-label="Import compiled LEGO plan"
                  onChange={(e) => void importPlan(e.target.files?.[0])}
                />
                <button className="lg-import-button" onClick={() => importFile.current?.click()}>
                  <Code2 size={13} /> Import compiled plan <ArrowUpRight size={11} />
                </button>
                <div className="lg-original-note">
                  <ShieldCheck size={15} />
                  <span>
                    Original, reproducible fixtures.
                    <br />
                    No uploads. No API key.
                  </span>
                </div>
                <a href="#instructions">
                  Have a LEGO set number? <ArrowUpRight size={13} />
                </a>
              </div>
            </aside>
            <div className="lg-viewport-panel">
              <div className="lg-view-top">
                <span className="lg-view-name">
                  <span className="lg-tiny-square" /> {TITLES[selected]?.label ?? plan?.assembly.title ?? 'Imported build'}{' '}
                  <ChevronDown size={12} />
                </span>
                <span className="lg-kinematic">KINEMATIC PREVIEW</span>
              </div>
              <div className="lg-stage" ref={host} />
              {(error || stageError) && (
                <div className="lg-stage-error">
                  <Box size={24} />
                  <p>{error || stageError}</p>
                </div>
              )}
              {!plan && !error && (
                <div className="lg-stage-loading">
                  <span className="loader" /> Loading assembly graph…
                </div>
              )}
              <div className="lg-stage-coordinate">
                <span>
                  <i className="x" /> X
                </span>
                <span>
                  <i className="y" /> Y
                </span>
                <span>
                  <i className="z" /> Z
                </span>
                <b>8 mm / stud</b>
              </div>
              <div className="lg-view-tools">
                <button
                  className={exploded ? 'active' : ''}
                  aria-pressed={exploded}
                  onClick={() => {
                    setExploded(!exploded);
                    setPlaying(false);
                  }}
                  title="Toggle exploded view"
                >
                  <Layers3 size={16} />
                  <span>Explode</span>
                </button>
                <button
                  className={robot ? 'active' : ''}
                  aria-pressed={robot}
                  onClick={() => setRobot(!robot)}
                  title="Toggle robot preview"
                >
                  <Workflow size={16} />
                  <span>Robot</span>
                </button>
                <button onClick={() => stage.current?.resetCamera()} aria-label="Reset camera" title="Reset camera">
                  <Focus size={16} />
                </button>
                <button onClick={() => host.current?.parentElement?.requestFullscreen?.()} aria-label="Enter fullscreen" title="Fullscreen">
                  <Maximize2 size={15} />
                </button>
              </div>
              <div className="lg-scene-caption">
                <span className="lg-current-dot" />
                <span>
                  {exploded
                    ? 'Exploded assembly · inspect the layers'
                    : timeline.completed >= (plan?.steps.length ?? 1)
                      ? 'Assembly complete · all bricks placed'
                      : `${PHASES[timeline.phase]} · ${currentBrick?.color ?? 'brick'} ${currentBrick?.studs.join(' × ') ?? ''}`}
                </span>
                <small>DRAG TO ORBIT · SCROLL TO ZOOM</small>
              </div>
              <div className="lg-playback">
                <button
                  className="lg-play-button"
                  onClick={play}
                  disabled={!plan}
                  aria-label={playing ? 'Pause assembly' : 'Play assembly'}
                >
                  {playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}
                </button>
                <button
                  className="lg-step-button"
                  onClick={() => seek(Math.floor(time / STEP_SECONDS) + 1)}
                  disabled={!plan || time >= total}
                  aria-label="Next assembly step"
                >
                  <SkipForward size={17} />
                </button>
                <button
                  className="lg-step-button"
                  onClick={() => {
                    setTime(0);
                    setPlaying(false);
                    setExploded(false);
                  }}
                  aria-label="Restart assembly"
                >
                  <RotateCcw size={15} />
                </button>
                <div className="lg-scrubber">
                  <input
                    type="range"
                    min="0"
                    max={total}
                    value={time}
                    step="0.05"
                    aria-label="Assembly timeline"
                    onChange={(e) => {
                      setTime(Number(e.target.value));
                      setPlaying(false);
                    }}
                  />
                  <div className="lg-timeline-ticks">
                    {Array.from({ length: (plan?.steps.length ?? 1) + 1 }, (_, i) => (
                      <span key={i} />
                    ))}
                  </div>
                </div>
                <span className="lg-time">
                  {Math.floor(time).toString().padStart(2, '0')}
                  <span> / {Number(total.toFixed(1))}s</span>
                </span>
              </div>
            </div>
            <aside className="lg-inspector">
              <div className="lg-inspector-tabs" role="tablist" aria-label="Assembly inspector">
                {(['sequence', 'graph', 'inventory', 'plan'] as Inspector[]).map((name) => (
                  <button
                    key={name}
                    role="tab"
                    aria-selected={tab === name}
                    className={tab === name ? 'active' : ''}
                    onClick={() => setTab(name)}
                  >
                    {name === 'sequence' ? (
                      <Layers3 size={13} />
                    ) : name === 'graph' ? (
                      <Network size={13} />
                    ) : name === 'inventory' ? (
                      <Package size={13} />
                    ) : (
                      <Code2 size={13} />
                    )}
                    <span>{name}</span>
                  </button>
                ))}
              </div>
              <div className="lg-inspector-content" role="tabpanel">
                {plan && tab === 'sequence' && (
                  <>
                    <div className="lg-sequence-heading">
                      <span>BUILD SEQUENCE</span>
                      <b>
                        {timeline.completed.toString().padStart(2, '0')}
                        <span> / {plan.steps.length.toString().padStart(2, '0')}</span>
                      </b>
                    </div>
                    <div className="lg-progress">
                      <span style={{ width: `${(timeline.completed / plan.steps.length) * 100}%` }} />
                    </div>
                    <ol className="lg-sequence-list">
                      {plan.steps.map((s, i) => {
                        const brick = plan.assembly.bricks.find((b) => b.id === s.brick_id)!;
                        const step = plan.assembly.steps.find((p) => p.brick_ids.includes(s.brick_id));
                        return (
                          <li
                            key={s.brick_id}
                            className={
                              i < timeline.completed
                                ? 'done'
                                : i === timeline.active && timeline.completed < plan.steps.length
                                  ? 'current'
                                  : ''
                            }
                          >
                            <button onClick={() => seek(i)}>
                              <span className="lg-step-num">
                                {i < timeline.completed ? <Check size={12} /> : (i + 1).toString().padStart(2, '0')}
                              </span>
                              <span className="lg-step-copy">
                                <strong>{step?.title ?? `Place ${brick.id}`}</strong>
                                <small>
                                  <i style={{ background: brick.color_hex }} />
                                  {brick.studs.join(' × ')} · {brick.color}
                                  <span>{i < timeline.completed ? 'PLACED' : i === timeline.active ? 'NEXT' : ''}</span>
                                </small>
                              </span>
                            </button>
                            {i === timeline.active && timeline.completed < plan.steps.length && (
                              <div className="lg-current-primitives">
                                {PHASES.map((phase, pi) => (
                                  <span key={phase} className={pi <= timeline.phase ? 'active' : ''} title={phase} />
                                ))}
                                <small>
                                  {PHASES[timeline.phase].toLowerCase()} → {PHASES[Math.min(5, timeline.phase + 1)].toLowerCase()}
                                </small>
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ol>
                  </>
                )}
                {plan && tab === 'graph' && <Graph plan={plan} completed={timeline.completed} select={seek} />}
                {plan && tab === 'inventory' && (
                  <>
                    <div className="lg-sequence-heading">
                      <span>BRICK INVENTORY</span>
                      <b>{plan.assembly.bricks.length}</b>
                    </div>
                    <div className="lg-inspector-note">
                      Every physical brick has a unique instance ID. Parts on the tray are still unplaced.
                    </div>
                    <div className="lg-inventory">
                      {plan.steps.map((s, i) => {
                        const b = plan.assembly.bricks.find((p) => p.id === s.brick_id)!;
                        return (
                          <button key={b.id} onClick={() => seek(i)}>
                            <span className="lg-mini-brick" style={{ background: b.color_hex }}>
                              <i />
                              <i />
                            </span>
                            <span>
                              <strong>{b.id}</strong>
                              <small>
                                {b.part_id} · {b.studs.join('×')}
                              </small>
                            </span>
                            <span className={`lg-inventory-state ${i < timeline.completed ? 'placed' : ''}`}>
                              {i < timeline.completed ? <CheckCheck size={13} /> : 'tray'}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
                {plan && tab === 'plan' && (
                  <>
                    <div className="lg-sequence-heading">
                      <span>COMPILED PLAN</span>
                      <button className="lg-small-link" onClick={() => downloadJson(`${selected}-plan.json`, plan)}>
                        <Download size={12} /> JSON
                      </button>
                    </div>
                    <div className="lg-inspector-note">
                      {selected === 'imported' ? (
                        <>
                          Imported file: <code>{plan.assembly.id}</code> compiled plan
                        </>
                      ) : (
                        <>
                          Exact source: <code>assemblies/lego/{selected}/plan.json</code>
                        </>
                      )}
                    </div>
                    <pre className="lg-plan-code">
                      {JSON.stringify(
                        { validation: plan.validation, metrics: plan.metrics, active_step: current, execution: plan.execution },
                        null,
                        2,
                      )}
                    </pre>
                  </>
                )}
              </div>
              <div className="lg-constraint">
                <ShieldCheck size={15} />
                <div>
                  <b>
                    {selected === 'imported'
                      ? 'Imported compiler artifact'
                      : plan?.validation.valid
                        ? 'Graph validated'
                        : 'Validating build graph'}
                  </b>
                  <span>
                    {plan
                      ? `${plan.metrics.stud_connections} stud contacts · ${plan.metrics.connections} support edges`
                      : 'Stud geometry · support · sequence'}
                  </span>
                </div>
                <span className="lg-constraint-check">{plan?.validation.valid ? <Check size={13} /> : '…'}</span>
              </div>
            </aside>
          </div>
          <div className="lg-workbench-footer">
            <div>
              <span className="lg-status-pill">
                <i /> {plan?.validation.valid ? 'COMPILED' : 'LOADING'}
              </span>
              <span>
                STUD GRID <b>8.0 × 3.2 mm</b>
              </span>
              <span>
                FRAME <b>Y-UP</b>
              </span>
              <span>
                PRIMITIVES <b>{plan?.metrics.primitives ?? '—'}</b>
              </span>
            </div>
            <button onClick={() => plan && downloadJson(`${selected}-plan.json`, plan)} disabled={!plan}>
              <Download size={13} /> Export plan <ArrowUpRight size={12} />
            </button>
          </div>
        </section>
        {help && (
          <div className="lg-help" role="status">
            <div>
              <CircleHelp size={20} />
              <p>
                <strong>What you’re seeing</strong>The stud-grid compiler validates these original builds and emits motion primitives. The
                browser animates a kinematic robot; grasping, contact physics, force control and policy inference are not running. The
                timeline uses a 5.1-second display cycle per brick.
              </p>
            </div>
            <button onClick={() => setHelp(false)} aria-label="Close simulation explanation">
              <X size={17} />
            </button>
          </div>
        )}
        <section className="lg-metric-strip" aria-label="Pipeline capabilities">
          <div>
            <span className="lg-metric-icon">
              <Network size={18} />
            </span>
            <p>
              <b>Structure before motion.</b>
              <span>
                A graph with explicit stud contacts
                <br />
                and dependencies for every brick.
              </span>
            </p>
            <ArrowRight size={15} />
          </div>
          <div>
            <span className="lg-metric-icon">
              <Workflow size={18} />
            </span>
            <p>
              <b>Make every action legible.</b>
              <span>
                Pick, transfer, align, press, verify.
                <br />
                Inspectable Cartesian primitives.
              </span>
            </p>
            <ArrowRight size={15} />
          </div>
          <div>
            <span className="lg-metric-icon">
              <Cpu size={18} />
            </span>
            <p>
              <b>Build the data flywheel.</b>
              <span>
                Synthetic trajectories first.
                <br />
                Real demonstrations and policy training next.
              </span>
            </p>
          </div>
        </section>
        <div className="lg-lower-grid">
          <section className="lg-instructions" id="instructions">
            <div className="lg-section-kicker">
              <span>01 / SOURCE</span>
              <span className="lg-neutral-badge">PUBLIC INSTRUCTIONS</span>
            </div>
            <h2>
              The manual is
              <br />
              already out there.
            </h2>
            <p>Start with a set number. Find instruction booklets directly from LEGO’s public library, without uploading a PDF.</p>
            <form onSubmit={findInstructions} className="lg-lookup-form">
              <Search size={16} />
              <input
                inputMode="numeric"
                pattern="[0-9]{4,7}"
                value={setNumber}
                onChange={(e) => setSetNumber(e.target.value)}
                placeholder="Enter a set number, e.g. 10497"
                aria-label="LEGO set number"
                required
              />
              <button disabled={lookupBusy} type="submit">
                {lookupBusy ? 'Finding…' : 'Find manual'}
                <ArrowUpRight size={14} />
              </button>
            </form>
            <div className="lg-quick-sets">
              <span>TRY A SET</span>
              {['10698', '10497', '21058'].map((set) => (
                <button
                  key={set}
                  onClick={() => {
                    setSetNumber(set);
                    setLookupError('');
                  }}
                >
                  {set}
                  <ArrowUpRight size={10} />
                </button>
              ))}
            </div>
            {lookupError && (
              <p className="lg-error-text" role="alert">
                {lookupError}
              </p>
            )}
            {lookup && (
              <div className="lg-lookup-result" role="status">
                <a href={lookup.url} target="_blank" rel="noreferrer">
                  <FileText size={16} /> LEGO set {lookup.url.split('/').pop()} instructions <ExternalLink size={13} />
                </a>
                <p>{lookup.message}</p>
                {lookup.pdfs?.slice(0, 3).map((p) => (
                  <a key={p.url} href={p.url} target="_blank" rel="noreferrer">
                    {p.label}
                    <ArrowDownToLine size={13} />
                  </a>
                ))}
              </div>
            )}
            <div className="lg-source-disclaimer">
              <FileText size={13} />
              <span>
                Official source links; booklet metadata when available. Automatic PDF → verified brick poses is a research stage. Public
                access does not grant training rights.
              </span>
            </div>
          </section>
          <section className="lg-training" id="training">
            <div className="lg-section-kicker">
              <span>02 / LEARN</span>
              <span className="lg-neutral-badge">ROBOT TRAINING BLUEPRINT</span>
            </div>
            <div className="lg-training-heading">
              <h2>
                A path to the
                <br />
                first real build.
              </h2>
              <span className="lg-training-icon">
                <Cpu size={32} />
                <Sparkles size={14} />
              </span>
            </div>
            <div className="lg-training-stages">
              <div className="ready">
                <span>
                  <Check size={12} />
                </span>
                <p>
                  <b>Compile</b>
                  <small>
                    Validated assemblies
                    <br />+ primitive plans
                  </small>
                </p>
                <i>READY</i>
              </div>
              <ChevronRight size={13} />
              <div className="ready">
                <span>
                  <Check size={12} />
                </span>
                <p>
                  <b>Generate</b>
                  <small>
                    Synthetic trajectory
                    <br />
                    seeds + manifests
                  </small>
                </p>
                <i>SYNTHETIC</i>
              </div>
              <ChevronRight size={13} />
              <div>
                <span>3</span>
                <p>
                  <b>Collect & train</b>
                  <small>
                    Teleop → imitation
                    <br />→ hardware evaluation
                  </small>
                </p>
                <i>PLANNED</i>
              </div>
            </div>
            <div className="lg-training-export">
              <div>
                <Terminal size={17} />
                <p>
                  <b>Inspect the training handoff</b>
                  <span>Source plan, provenance, observations and actions.</span>
                </p>
              </div>
              <button onClick={exportTraining} disabled={!plan || exportBusy}>
                <Download size={14} />
                {exportBusy
                  ? 'Preparing…'
                  : selected === 'imported' || plan?.assembly.source.review_status === 'pending'
                    ? 'Export draft'
                    : 'Export bundle'}
              </button>
            </div>
            <p className="lg-training-note">
              {selected === 'imported' || plan?.assembly.source.review_status === 'pending'
                ? 'Imported plan: this export is a planning draft. Use the engine CLI to review geometry and generate a synthetic dataset.'
                : 'Synthetic planning trajectories. No visual robot policy, hardware execution, or native LeRobot dataset is claimed.'}
            </p>
          </section>
        </div>
        <section className="lg-bottom-banner">
          <div className="lg-banner-mark">
            <BrickGlyph color="#f2be32" type="tower" small />
          </div>
          <div>
            <b>From a toy problem to a very real robotics problem.</b>
            <p>Discrete geometry. Contact-rich manipulation. Long-horizon reasoning. One system to connect them.</p>
          </div>
          <a href="https://github.com/DanielTong50/3d-visualization/tree/darius-lego#readme" target="_blank" rel="noreferrer">
            Read the technical blueprint <ArrowUpRight size={16} />
          </a>
        </section>
      </main>
      <footer className="lg-footer">
        <div>
          <DariusMark />
          <span>
            darius <b>/</b> LEGO
          </span>
          <span className="lg-footer-note">ASSEMBLY INTELLIGENCE, ONE BRICK AT A TIME.</span>
        </div>
        <span>Independent research · LEGO® is a trademark of the LEGO Group.</span>
        <a href="?view=legacy">
          Furniture archive <ArrowUpRight size={11} />
        </a>
      </footer>
      {notice && (
        <div className="lg-toast" role="status">
          <CheckCheck size={18} />
          <span>{notice}</span>
          <button onClick={() => setNotice('')} aria-label="Dismiss notification">
            <X size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
