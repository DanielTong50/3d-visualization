import { ArrowRight, ArrowUpRight, Camera, FileText, Play, ScanLine } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Moment } from '../engine/choreography';
import { loadLibrary, loadPlan, type AssemblyPlan, type PlanSummary } from '../engine/plan';
import {
  loadBenchmarks,
  loadSessionResult,
  loadSessionTruth,
  manualPageUrl,
  sessionVideoUrl,
  type Benchmarks,
  type SessionResult,
  type SessionTruth,
} from '../engine/results';
import { Wordmark } from '../Logo';
import { navigate, type Route } from '../navigation';
import { Stage } from '../three/Stage';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { EvidenceHeatmap } from '../ui/EvidenceHeatmap';

/**
 * The front page: the copilot running in a dark studio behind the pitch, the measured results,
 * how the three stages fit together, the four demos, the benchmark table and the engineering.
 * Every number comes from assemblies/benchmarks.json, written by `darius eval`.
 */

const REPO = 'https://github.com/DanielTong50/3d-visualization';
const SOURCE = `${REPO}/blob/main`;
const FEATURED = 'halden-bookcase';
const PHASE_TEXT: Record<string, string> = {
  intro: 'Scanning the parts',
  reorient: 'Turn',
  locate: 'Locate',
  place: 'Place',
  fasten: 'Fasten',
  verify: 'Verify',
  done: 'Verified',
};

function pct(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : `${(value * 100).toFixed(digits).replace(/\.0$/, '')}%`;
}

function Github({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

/** The copilot, building the featured piece on a loop in the studio. Pauses when off screen. */
function HeroStage({ plan, onMoment }: { plan: AssemblyPlan; onMoment: (moment: Moment) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const report = useRef(onMoment);
  report.current = onMoment;

  useEffect(() => {
    const container = ref.current!;
    const stage = new Stage(container, plan, 'hero');
    const frame = () => (container.clientWidth > 900 ? stage.frame(0.27, 0.04) : stage.frame(0, 0.2));
    frame();
    window.addEventListener('resize', frame);
    let visible = true;
    const observer = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
    observer.observe(container);
    let raf = 0;
    let clock = 0;
    let last = performance.now();
    let reported = 0;
    const loop = (now: number) => {
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      if (visible) {
        clock += dt;
        const { moment } = stage.update(clock % stage.choreography.duration, clock);
        stage.render();
        if (now - reported > 150) {
          reported = now;
          report.current(moment);
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('resize', frame);
      stage.dispose();
    };
  }, [plan]);

  return <div ref={ref} className="hero-stage" aria-hidden="true" />;
}

function Go({ route, children, className = 'btn btn-ghost' }: { route: Route; children: React.ReactNode; className?: string }) {
  return (
    <button type="button" className={className} onClick={() => navigate(route)}>
      {children}
    </button>
  );
}

export function Home() {
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const [moment, setMoment] = useState<Moment | null>(null);
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null);
  const [library, setLibrary] = useState<PlanSummary[]>([]);
  const [session, setSession] = useState<SessionResult | null>(null);
  const [truth, setTruth] = useState<SessionTruth | null>(null);

  useEffect(() => {
    loadPlan(FEATURED)
      .then(setPlan)
      .catch(() => setPlan(null));
    loadBenchmarks()
      .then(setBenchmarks)
      .catch(() => setBenchmarks(null));
    loadLibrary()
      .then(setLibrary)
      .catch(() => setLibrary([]));
    loadSessionResult(FEATURED)
      .then(setSession)
      .catch(() => setSession(null));
    loadSessionTruth(FEATURED)
      .then(setTruth)
      .catch(() => setTruth(null));
  }, []);

  const placed = useMemo(
    () => new Map((truth?.segments ?? []).filter((s) => s.phase === 'place').map((s) => [s.step, s.start + s.duration])),
    [truth],
  );
  const sessions = benchmarks ? Object.values(benchmarks.sessions.items) : [];
  const stepsFound = sessions.reduce((n, s) => n + s.steps_found, 0);
  const stepsTotal = sessions.reduce((n, s) => n + s.steps, 0);
  const step = plan && moment && moment.step >= 1 && moment.step <= plan.steps.length ? plan.steps[moment.step - 1] : null;

  return (
    <div className="home">
      <section className="hero">
        {plan && <HeroStage plan={plan} onMoment={setMoment} />}
        <div className="hero-shade" aria-hidden="true" />
        <nav className="home-nav">
          <Wordmark onClick={() => navigate({ page: 'home' })} />
          <div className="home-nav-links">
            <a href="#how">How it works</a>
            <a href="#demos">Demos</a>
            <a href="#results">Results</a>
            <a href={REPO} className="icon-link" aria-label="Source on GitHub">
              <Github size={17} />
            </a>
          </div>
        </nav>
        <div className="hero-copy">
          <span className="eyebrow">Darius 1.0 · AR assembly copilot</span>
          <h1>
            Show it the manual.
            <br />
            It shows you the next move.
          </h1>
          <p className="hero-lede">
            Darius reads a flat-pack manual into an assembly graph of parts, hardware, joints and steps, follows your build through the
            phone camera, and draws every move in place: which part, where it goes, which way to turn the piece, where each screw goes in.
          </p>
          <div className="hero-actions">
            <Go route={{ page: 'ar', furniture: FEATURED }} className="btn btn-guide">
              <Play size={15} /> Watch a session
            </Go>
            <Go route={{ page: 'manual', furniture: FEATURED }}>
              <FileText size={15} /> Read a manual
            </Go>
            <Go route={{ page: 'video', furniture: FEATURED }}>
              <ScanLine size={15} /> Analyze a video
            </Go>
            <Go route={{ page: 'live' }} className="btn btn-ghost">
              <Camera size={15} /> Go live
            </Go>
          </div>
        </div>
        {plan && moment && (
          <aside className="hero-hud glass" aria-hidden="true">
            <span className="eyebrow">{step ? `Step ${step.index} of ${plan.steps.length}` : 'Copilot'}</span>
            <strong>{step ? step.title : moment.phase === 'done' ? `${plan.title}, assembled` : 'Finding the parts'}</strong>
            <span className="hero-hud-phase mono">{PHASE_TEXT[moment.phase]}</span>
          </aside>
        )}
      </section>

      {benchmarks && (
        <section className="stats" aria-label="Measured results">
          <div>
            <b>{pct(benchmarks.manuals.mean.edges_f1)}</b>
            <span>of joints read from manuals (edge F1)</span>
          </div>
          <div>
            <b>{pct(benchmarks.manuals.mean.parts_f1)}</b>
            <span>of parts and {pct(benchmarks.manuals.mean.hardware_accuracy)} of hardware counts read</span>
          </div>
          <div>
            <b>
              {stepsFound}/{stepsTotal}
            </b>
            <span>build steps found in video</span>
          </div>
          <div>
            <b>{benchmarks.sessions.mean.step_error_s?.toFixed(2)} s</b>
            <span>mean error dating each step</span>
          </div>
          <p className="stats-source mono">
            {benchmarks.model} · {Object.keys(benchmarks.manuals.items).length} manuals, {sessions.length} filmed builds ·{' '}
            <code>darius eval</code>, {benchmarks.date}
          </p>
        </section>
      )}

      <section id="how" className="section">
        <header className="section-head">
          <span className="eyebrow track">How it works</span>
          <h2>One graph, three jobs.</h2>
          <p className="muted">
            A manual is a graph in disguise: parts are nodes, joints are edges, and the steps put the edges in order. Darius recovers that
            graph, tracks your progress through it, and turns its geometry into guidance.
          </p>
        </header>
        <div className="how">
          <article className="how-step">
            <span className="how-index mono">01</span>
            <h3>Read the manual</h3>
            <p className="muted">
              All the pages go to Muse Spark with a schema: parts by letter, hardware by article number, and for every step the parts that
              go on, the build orientation and each joint. The reading becomes a validated plan.
            </p>
            <div className="how-visual how-read">
              <img src={manualPageUrl(FEATURED, 6)} alt="A generated manual page: step 3 of the Halden bookcase" loading="lazy" />
              <ArrowRight className="how-arrow" size={20} />
              {plan && (
                <div className="how-graph">
                  <AssemblyGraph plan={plan} completed={plan.steps.length} compact />
                </div>
              )}
            </div>
          </article>
          <article className="how-step">
            <span className="how-index mono">02</span>
            <h3>Track the build</h3>
            <p className="muted">
              Camera frames are checked against the plan a few at a time, with the manual's step pages for reference. A hidden Markov model
              over "steps done" filters the readings along the manual's order, so one misread frame cannot move the build.
            </p>
            <div className="how-visual how-track">
              {plan && session ? <EvidenceHeatmap plan={plan} result={session} placed={placed} /> : <div className="how-placeholder" />}
              <p className="how-caption mono">P(step done) per frame · decoded path · truth</p>
            </div>
          </article>
          <article className="how-step">
            <span className="how-index mono">03</span>
            <h3>Guide the next move</h3>
            <p className="muted">
              The plan knows where every part goes, so the copilot draws it: a ghost at the target, an arc from the part on the floor, a
              ring when the piece must turn, a marker on every hole. Parts it sees are boxed and named.
            </p>
            <div className="how-visual how-guide">
              <img src={`${import.meta.env.BASE_URL}posters/copilot.jpg`} alt="The copilot guiding step 1 of the bookcase" loading="lazy" />
            </div>
          </article>
        </div>
      </section>

      <section id="demos" className="section">
        <header className="section-head">
          <span className="eyebrow">Demos</span>
          <h2>Four ways in.</h2>
          <p className="muted">
            Three replay real runs of the models on the furniture library; the fourth uses your camera. Pick any of the{' '}
            {library.length || 4} products inside each demo.
          </p>
        </header>
        <div className="demos">
          <button type="button" className="demo-card" onClick={() => navigate({ page: 'ar', furniture: FEATURED })}>
            <div className="demo-visual">
              <img src={`${import.meta.env.BASE_URL}posters/copilot.jpg`} alt="" loading="lazy" />
            </div>
            <span className="eyebrow">Copilot session</span>
            <h3>A whole build, guided</h3>
            <p className="muted">Tracked parts, ghosts and arcs, turns, fastener markers, the graph filling in and the tracker's belief.</p>
            <span className="demo-go">
              Watch <ArrowRight size={14} />
            </span>
          </button>
          <button type="button" className="demo-card" onClick={() => navigate({ page: 'manual', furniture: FEATURED })}>
            <div className="demo-visual demo-paper">
              <img src={manualPageUrl(FEATURED, 4)} alt="" loading="lazy" />
              <span className="demo-scan" />
            </div>
            <span className="eyebrow track">Manual reader</span>
            <h3>Pages in, graph out</h3>
            <p className="muted">The model reads a manual page by page; the graph builds edge by edge and is scored against the plan.</p>
            <span className="demo-go">
              Read <ArrowRight size={14} />
            </span>
          </button>
          <button type="button" className="demo-card" onClick={() => navigate({ page: 'video', furniture: FEATURED })}>
            <div className="demo-visual">
              <video src={sessionVideoUrl(FEATURED)} muted loop autoPlay playsInline preload="metadata" />
            </div>
            <span className="eyebrow track">Video analyzer</span>
            <h3>A filmed build, decoded</h3>
            <p className="muted">Every sampled frame read by the model, the build history decoded through it, every joint dated.</p>
            <span className="demo-go">
              Analyze <ArrowRight size={14} />
            </span>
          </button>
          <button type="button" className="demo-card" onClick={() => navigate({ page: 'live' })}>
            <div className="demo-visual demo-viewfinder">
              <Camera size={30} />
              <span className="vf-corner tl" />
              <span className="vf-corner tr" />
              <span className="vf-corner bl" />
              <span className="vf-corner br" />
            </div>
            <span className="eyebrow done">Live</span>
            <h3>Your camera, your build</h3>
            <p className="muted">Photograph the manual, then point the phone at the parts. Runs against the Darius API.</p>
            <span className="demo-go">
              Start <ArrowRight size={14} />
            </span>
          </button>
        </div>
      </section>

      {benchmarks && (
        <section id="results" className="section">
          <header className="section-head">
            <span className="eyebrow done">Results</span>
            <h2>Measured, not claimed.</h2>
            <p className="muted">
              Each manual is generated from a procedural plan and read back by the model; each session is a rendered build with frame-exact
              ground truth. A step counts as done in video once its parts are in place.
            </p>
          </header>
          <div className="table-wrap glass">
            <table className="results">
              <thead>
                <tr>
                  <th rowSpan={2}>Product</th>
                  <th colSpan={4}>Manual → graph</th>
                  <th colSpan={3}>Video → timeline</th>
                </tr>
                <tr>
                  <th>Parts F1</th>
                  <th>Edges F1</th>
                  <th>Steps exact</th>
                  <th>Hardware</th>
                  <th>Frames right</th>
                  <th>Steps found</th>
                  <th>Timing error</th>
                </tr>
              </thead>
              <tbody>
                {library.map((p) => {
                  const m = benchmarks.manuals.items[p.id];
                  const s = benchmarks.sessions.items[p.id];
                  return (
                    <tr key={p.id}>
                      <td>
                        <button type="button" className="link" onClick={() => navigate({ page: 'manual', furniture: p.id })}>
                          {p.title}
                        </button>
                        <span className="faint mono">
                          {p.parts} parts · {p.joints} joints · {p.steps} steps
                        </span>
                      </td>
                      <td>{pct(m?.parts_f1)}</td>
                      <td className={m && m.edges_f1 < 1 ? 'is-miss' : ''}>{pct(m?.edges_f1)}</td>
                      <td>{pct(m?.steps_exact)}</td>
                      <td>{pct(m?.hardware_accuracy)}</td>
                      <td>{pct(s?.frame_accuracy)}</td>
                      <td>{s ? `${s.steps_found}/${s.steps}` : '—'}</td>
                      <td>{s?.step_error_s != null ? `${s.step_error_s.toFixed(2)} s` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="section">
        <header className="section-head">
          <span className="eyebrow">Under the hood</span>
          <h2>The engineering.</h2>
        </header>
        <div className="tiles">
          <a className="tile glass" href={`${SOURCE}/engine/darius/assembly/schema.py`}>
            <h3>Plans are graphs</h3>
            <p className="muted">
              Parts are nodes; joints are edges carrying hardware and fastener positions; steps order the edges and say how the piece lies.
              One Pydantic schema, exported as JSON Schema, mirrored in TypeScript.
            </p>
            <code>assembly/schema.py</code>
          </a>
          <a className="tile glass" href={`${SOURCE}/engine/darius/assembly/tracking.py`}>
            <h3>A tracker that knows the order</h3>
            <p className="muted">
              An HMM over k steps done: stay .86, advance .11, skip .02, back .01. Each frame's evidence is capped at .85 and tempered by
              .7. Forward filtering live; Viterbi and forward-backward for a whole video.
            </p>
            <code>assembly/tracking.py · engine/tracking.ts</code>
          </a>
          <a className="tile glass" href={`${SOURCE}/engine/darius/furniture/builder.py`}>
            <h3>A flat-pack compiler</h3>
            <p className="muted">
              Furniture is declared as boxes; contacts are found from the geometry, hardware is laid out per joint kind (cam locks, dowels,
              nails every 15 cm), and every part gets a spot on the floor.
            </p>
            <code>furniture/builder.py</code>
          </a>
          <a className="tile glass" href={`${SOURCE}/frontend/src/three/ManualArt.ts`}>
            <h3>Manuals from geometry</h3>
            <p className="muted">
              Hidden-line drawings in the flat-pack style. Parts come out along a free way in: shelves between two sides slide in from the
              front. The same pages are the model's test set.
            </p>
            <code>three/ManualArt.ts</code>
          </a>
          <a className="tile glass" href={`${SOURCE}/frontend/src/engine/choreography.ts`}>
            <h3>Choreography to AR</h3>
            <p className="muted">
              Every instant of a build is computed from the plan: turns between build orientations, arcs from the floor, subassemblies built
              on the side and joined later, fasteners from the joints' hole patterns.
            </p>
            <code>engine/choreography.ts</code>
          </a>
          <a className="tile glass" href={`${SOURCE}/engine/darius/eval/benchmark.py`}>
            <h3>Every number is a run</h3>
            <p className="muted">
              <code className="inline">darius eval</code> stores each model output next to its inputs and scores it; the site replays those
              files. <code className="inline">--replay</code> re-scores them without the API.
            </p>
            <code>eval/benchmark.py</code>
          </a>
        </div>
      </section>

      <footer className="home-foot">
        <Wordmark onClick={() => navigate({ page: 'home' })} />
        <span className="faint">Darius 1.0 · MIT license · models: Muse Spark (Meta Model API) · rendering: three.js</span>
        <a href={REPO} className="btn btn-ghost btn-sm">
          <Github size={15} /> Source <ArrowUpRight size={14} />
        </a>
      </footer>
    </div>
  );
}
