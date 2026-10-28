import { ArrowRight, ArrowUpRight, Camera, FileText, Play, ScanLine } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
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
import { BeliefBars } from '../ui/BeliefBars';
import { EvidenceHeatmap } from '../ui/EvidenceHeatmap';
import { HardwareIcon } from '../ui/HardwareIcon';

/**
 * The front page. A showroom where the copilot builds each product of the library in turn, under
 * an IKEA-style product tag; a grid of live demo tiles that keeps rearranging; how the three
 * stages share one graph; the research the methods build on; the stack; and the measured results.
 * Every number comes from assemblies/benchmarks.json, written by `darius eval`.
 */

const REPO = 'https://github.com/DanielTong50/3d-visualization';
const SOURCE = `${REPO}/blob/main`;
const POSTERS = ['copilot', 'copilot-fasten', 'copilot-turn'].map((name) => `${import.meta.env.BASE_URL}posters/${name}.jpg`);
const SHOWROOM_SECONDS = 26;
const SHUFFLE_MS = 4200;
const PHASE_TEXT: Record<string, string> = {
  intro: 'Scanning',
  reorient: 'Turn',
  locate: 'Locate',
  place: 'Place',
  fasten: 'Fasten',
  verify: 'Verify',
  done: 'Done',
};

function pct(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : `${(value * 100).toFixed(digits).replace(/\.0$/, '')}%`;
}

function scholar(title: string): string {
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(`"${title}"`)}`;
}

function Github({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function Go({ route, children, className = 'btn btn-outline' }: { route: Route; children: React.ReactNode; className?: string }) {
  return (
    <button type="button" className={className} onClick={() => navigate(route)}>
      {children}
    </button>
  );
}

/** The copilot building one product in the showroom, from its first step. Pauses off screen. */
function Showroom({ plan, onMoment }: { plan: AssemblyPlan; onMoment: (moment: Moment) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const report = useRef(onMoment);
  report.current = onMoment;

  useEffect(() => {
    const container = ref.current!;
    const stage = new Stage(container, plan, 'hero');
    stage.frame(0, 0.06);
    let visible = true;
    const observer = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
    observer.observe(container);
    let raf = 0;
    let clock = stage.choreography.stepStarts()[0] ?? 0;
    let last = performance.now();
    let reported = 0;
    const loop = (now: number) => {
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      if (visible) {
        clock += dt;
        const { moment } = stage.update(clock % stage.choreography.duration, clock);
        stage.render();
        if (now - reported > 120) {
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
      stage.dispose();
    };
  }, [plan]);

  return <div ref={ref} className="showroom-stage" aria-hidden="true" />;
}

/** The product's tag, the way a showroom labels a piece of furniture. */
function ProductTag({ plan }: { plan: AssemblyPlan }) {
  const [w, h, d] = (plan.dimensions_mm ?? [0, 0, 0]).map((v) => Math.round(v / 10));
  const pieces = plan.hardware.reduce((n, item) => n + item.count, 0);
  return (
    <div className="product-tag" key={plan.id}>
      <b>{plan.title.split(' ')[0].toUpperCase()}</b>
      <span>
        {plan.category[0].toUpperCase() + plan.category.slice(1)}, {w}×{d}×{h} cm
      </span>
      <span className="product-tag-specs mono">
        {plan.parts.length} parts · {plan.joints.length} joints · {pieces} fittings · {plan.steps.length} steps
      </span>
    </div>
  );
}

/** A small loop of time, for tiles that animate on their own. */
function useTicker(periodMs: number): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => setTick((t) => t + 1), periodMs);
    return () => clearInterval(timer);
  }, [periodMs]);
  return tick;
}

type TileId = 'copilot' | 'manual' | 'graph' | 'video' | 'hardware' | 'tracker';

function Tile({ id, slot, children }: { id: TileId; slot: number; children: React.ReactNode }) {
  return (
    <div className={`tile-slot slot-${slot}`} style={{ viewTransitionName: `tile-${id}` } as React.CSSProperties}>
      {children}
    </div>
  );
}

function TileHead({ label, title, route }: { label: string; title: string; route: Route }) {
  return (
    <div className="tile-head">
      <span className="tag">{label}</span>
      <strong>{title}</strong>
      <button type="button" className="tile-open" onClick={() => navigate(route)} aria-label={`Open: ${title}`}>
        <ArrowUpRight size={15} />
      </button>
    </div>
  );
}

/**
 * Six live tiles in a grid that rearranges itself: every few seconds the next tile takes the big
 * slot (a view transition morphs them into place where the browser supports it).
 */
function DemoGrid({ plan, session, truth }: { plan: AssemblyPlan; session: SessionResult | null; truth: SessionTruth | null }) {
  const [order, setOrder] = useState<TileId[]>(['copilot', 'manual', 'graph', 'video', 'hardware', 'tracker']);
  const [paused, setPaused] = useState(false);
  const tick = useTicker(1100);
  const placed = useMemo(
    () => new Map((truth?.segments ?? []).filter((s) => s.phase === 'place').map((s) => [s.step, s.start + s.duration])),
    [truth],
  );

  useEffect(() => {
    if (paused || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => {
      const next = (o: TileId[]) => [...o.slice(1), o[0]];
      const doc = document as Document & { startViewTransition?: (update: () => void) => unknown };
      if (doc.startViewTransition) doc.startViewTransition(() => flushSync(() => setOrder(next)));
      else setOrder(next);
    }, SHUFFLE_MS);
    return () => clearInterval(timer);
  }, [paused]);

  const steps = plan.steps.length;
  const built = tick % (steps + 2);
  const frame = session && session.frames.length ? tick % session.frames.length : 0;
  const pages = 3 + steps;
  const tiles: Record<TileId, React.ReactNode> = {
    copilot: (
      <article className="tile tile-copilot">
        {POSTERS.map((src, i) => (
          <img key={src} src={src} alt="" className={Math.floor(tick / 3) % POSTERS.length === i ? 'is-on' : ''} loading="lazy" />
        ))}
        <TileHead label="Copilot" title="Every move, drawn in place" route={{ page: 'ar', furniture: plan.id }} />
      </article>
    ),
    manual: (
      <article className="tile tile-manual">
        <div className="tile-paper">
          <img src={manualPageUrl(plan.id, 4 + (Math.floor(tick / 2) % (pages - 3)))} alt="" loading="lazy" />
          <span className="tile-scan" />
        </div>
        <TileHead label="Manual reader" title="Pages in, graph out" route={{ page: 'manual', furniture: plan.id }} />
      </article>
    ),
    graph: (
      <article className="tile tile-graph">
        <div className="tile-fill">
          <AssemblyGraph plan={plan} completed={Math.min(built, steps)} active={built < steps ? built + 1 : null} compact />
        </div>
        <TileHead
          label="Assembly graph"
          title={`${plan.parts.length} parts, ${plan.joints.length} joints`}
          route={{ page: 'manual', furniture: plan.id }}
        />
      </article>
    ),
    video: (
      <article className="tile tile-video">
        <video src={sessionVideoUrl(plan.id)} muted loop autoPlay playsInline preload="metadata" />
        {session && (
          <div className="tile-heatmap">
            <EvidenceHeatmap plan={plan} result={session} placed={placed} />
          </div>
        )}
        <TileHead label="Video analyzer" title="Every frame read, every joint dated" route={{ page: 'video', furniture: plan.id }} />
      </article>
    ),
    hardware: (
      <article className="tile tile-hardware">
        <div className="tile-fittings">
          {plan.hardware.map((item, i) => (
            <div key={item.id} className={tick % plan.hardware.length === i ? 'is-on' : ''}>
              <HardwareIcon kind={item.kind} size={46} />
              <b>{item.count}×</b>
              <span className="mono">{item.id}</span>
            </div>
          ))}
        </div>
        <TileHead label="Hardware" title="Dowels, cam locks, screws, bolts" route={{ page: 'ar', furniture: plan.id }} />
      </article>
    ),
    tracker: (
      <article className="tile tile-tracker">
        <div className="tile-fill">
          {session && (
            <BeliefBars
              plan={plan}
              belief={{ probabilities: session.posteriors[frame], completed: session.path[frame], confidence: session.confidence[frame] }}
            />
          )}
        </div>
        <TileHead label="Tracker" title="P(steps done), frame by frame" route={{ page: 'video', furniture: plan.id }} />
      </article>
    ),
  };

  return (
    <div className="demo-grid" onPointerEnter={() => setPaused(true)} onPointerLeave={() => setPaused(false)}>
      {order.map((id, slot) => (
        <Tile key={id} id={id} slot={slot}>
          {tiles[id]}
        </Tile>
      ))}
    </div>
  );
}

const RESEARCH: { venue: string; title: string; authors: string; use: string }[] = [
  {
    venue: 'Proc. IEEE 1989',
    title: 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition',
    authors: 'L. R. Rabiner',
    use: 'The progress tracker is an HMM over "steps done": forward filtering live, the standard recursions in log space.',
  },
  {
    venue: 'IEEE Trans. Inf. Theory 1967',
    title: 'Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm',
    authors: 'A. J. Viterbi',
    use: 'Viterbi decoding gives the most likely build history of a whole video.',
  },
  {
    venue: 'Ann. Math. Stat. 1970',
    title: 'A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains',
    authors: 'L. E. Baum, T. Petrie, G. Soules, N. Weiss',
    use: 'Forward-backward posteriors give the confidence of every decoded moment.',
  },
  {
    venue: 'CVPR 2018',
    title: 'NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning',
    authors: 'A. Richard, H. Kuehne, A. Iqbal, J. Gall',
    use: "Decoding a procedure video under an ordered transcript: here, the manual's steps constrain the path.",
  },
  {
    venue: 'ECCV 2022',
    title: 'Translating a Visual LEGO Manual to a Machine-Executable Plan',
    authors: 'R. Wang, Y. Zhang, J. Mao, C.-Y. Cheng, J. Wu',
    use: 'Manual pages to an executable assembly plan: the problem the manual reader solves for flat-pack furniture.',
  },
  {
    venue: 'NeurIPS 2022 Datasets',
    title: 'IKEA-Manual: Seeing Shape Assembly Step by Step',
    authors: 'R. Wang, Y. Zhang, J. Mao, R. Zhang, C.-Y. Cheng, J. Wu',
    use: 'Furniture manuals aligned with 3D parts and assembly graphs, the representation Darius builds on.',
  },
  {
    venue: 'WACV 2021',
    title: 'The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose',
    authors: 'Y. Ben-Shabat, X. Yu, F. Saleh, D. Campbell, C. Rodriguez-Opazo, H. Li, S. Gould',
    use: 'Furniture assembly as a video-understanding problem: the setting the video analyzer works in.',
  },
  {
    venue: 'CVPR 2022',
    title: 'Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities',
    authors: 'F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, A. Yao',
    use: 'Procedural assembly video, step recognition and progress: the task the tracker addresses.',
  },
  {
    venue: 'ICCV 2023',
    title: 'HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World',
    authors: 'X. Wang, T. Kwon, M. Rad, B. Pan, I. Chakraborty, et al.',
    use: "An assistant that watches a person do a task and guides them: the copilot's interaction model.",
  },
  {
    venue: 'arXiv 2023',
    title: 'Efficient Guided Generation for Large Language Models',
    authors: 'B. T. Willard, R. Louf',
    use: 'Decoding constrained to a JSON schema: every model reading in Darius is schema-constrained.',
  },
  {
    venue: 'SIGGRAPH 2003',
    title: 'Designing Effective Step-by-Step Assembly Instructions',
    authors: 'M. Agrawala, D. Phan, J. Heiser, J. Haymaker, J. Klingner, P. Hanrahan, B. Tversky',
    use: 'Instructions generated from geometry: the generated manuals, with parts introduced as they go on.',
  },
  {
    venue: 'SIGGRAPH 2008',
    title: 'Automated Generation of Interactive 3D Exploded View Diagrams',
    authors: 'W. Li, M. Agrawala, B. Curless, D. Salesin',
    use: 'Explosion directions from blocking constraints: how manual drawings pull each part out along a free way in.',
  },
  {
    venue: 'Artif. Intell. 1994',
    title: 'Geometric Reasoning About Mechanical Assembly',
    authors: 'R. H. Wilson, J.-C. Latombe',
    use: 'Removal directions of a part among the parts in place: the sweep test behind the free way in.',
  },
  {
    venue: 'J. ACM 1975',
    title: 'Efficiency of a Good But Not Linear Set Union Algorithm',
    authors: 'R. E. Tarjan',
    use: 'Union-find gives the subassemblies after every step, in Python and in the browser.',
  },
  {
    venue: 'ICRA 2021',
    title: 'IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks',
    authors: 'Y. Lee, E. S. Hu, J. J. Lim',
    use: 'Furniture as parts with connection points: the plan structure Darius compiles from boxes.',
  },
];

const STACK: { layer: string; items: string[] }[] = [
  {
    layer: 'Perception',
    items: ['Muse Spark 1.3 (Meta Model API)', 'JSON-schema-constrained decoding', 'manual step pages as visual references'],
  },
  {
    layer: 'Engine',
    items: ['Python 3.11', 'Pydantic 2 schema → JSON Schema', 'NumPy HMM in log space', 'OpenCV frame sampling', 'FastAPI'],
  },
  { layer: 'Serving', items: ['Modal (CPU, scale to zero)', 'stateless /frames', 'access-code gating'] },
  { layer: 'Web', items: ['React 19', 'TypeScript 6', 'three.js 0.186 (instancing, custom shaders)', 'Vite 8'] },
  { layer: 'Test data', items: ['Playwright renders manuals and sessions', 'ffmpeg (H.264)', 'frame-exact ground truth'] },
  { layer: 'Quality', items: ['pytest + Vitest', 'Ruff + Prettier', 'CI re-scores stored model results'] },
];

export function Home() {
  const [library, setLibrary] = useState<PlanSummary[]>([]);
  const [plans, setPlans] = useState<AssemblyPlan[]>([]);
  const [index, setIndex] = useState(0);
  const [moment, setMoment] = useState<Moment | null>(null);
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null);
  const [session, setSession] = useState<SessionResult | null>(null);
  const [truth, setTruth] = useState<SessionTruth | null>(null);

  useEffect(() => {
    loadLibrary()
      .then(async (items) => {
        setLibrary(items);
        setPlans(await Promise.all(items.map((p) => loadPlan(p.id))));
      })
      .catch(() => setLibrary([]));
    loadBenchmarks()
      .then(setBenchmarks)
      .catch(() => setBenchmarks(null));
  }, []);

  // The showroom moves on to the next product every so often.
  useEffect(() => {
    if (plans.length < 2) return;
    const timer = setTimeout(() => setIndex((i) => (i + 1) % plans.length), SHOWROOM_SECONDS * 1000);
    return () => clearTimeout(timer);
  }, [index, plans.length]);

  const plan = plans[index] ?? null;
  useEffect(() => {
    if (!plan) return;
    setSession(null);
    loadSessionResult(plan.id)
      .then(setSession)
      .catch(() => setSession(null));
    loadSessionTruth(plan.id)
      .then(setTruth)
      .catch(() => setTruth(null));
  }, [plan]);

  const sessions = benchmarks ? Object.values(benchmarks.sessions.items) : [];
  const stepsFound = sessions.reduce((n, s) => n + s.steps_found, 0);
  const stepsTotal = sessions.reduce((n, s) => n + s.steps, 0);
  const step = plan && moment && moment.step >= 1 && moment.step <= plan.steps.length ? plan.steps[moment.step - 1] : null;
  const featured = plan?.id ?? 'halden-bookcase';

  return (
    <div className="home">
      <nav className="home-nav">
        <Wordmark onClick={() => navigate({ page: 'home' })} />
        <div className="home-nav-links">
          <a href="#demos">Demos</a>
          <a href="#how">How it works</a>
          <a href="#research">Research</a>
          <a href="#results">Results</a>
          <a href={REPO} className="icon-link" aria-label="Source on GitHub">
            <Github size={17} />
          </a>
          <Go route={{ page: 'live' }} className="btn btn-blue btn-sm">
            <Camera size={14} /> Try it live
          </Go>
        </div>
      </nav>

      <header className="hero">
        <div className="hero-copy">
          <span className="tag">Darius 1.1 · assembly copilot</span>
          <h1>
            Show it the manual.
            <br />
            It shows you <mark>the next move</mark>.
          </h1>
          <p className="hero-lede">
            Darius reads a flat-pack manual into an assembly graph, follows the build through a phone camera with a hidden Markov model over
            the manual's steps, and draws every move in place: the part, its target, the turn, each dowel, cam lock and screw.
          </p>
          <div className="hero-actions">
            <Go route={{ page: 'ar', furniture: featured }} className="btn btn-blue">
              <Play size={15} /> Watch a build
            </Go>
            <Go route={{ page: 'manual', furniture: featured }}>
              <FileText size={15} /> Read a manual
            </Go>
            <Go route={{ page: 'video', furniture: featured }}>
              <ScanLine size={15} /> Analyze a video
            </Go>
          </div>
          {benchmarks && (
            <dl className="hero-stats">
              <div>
                <dt>{pct(benchmarks.manuals.mean.edges_f1)}</dt>
                <dd>joints read from manuals</dd>
              </div>
              <div>
                <dt>
                  {stepsFound}/{stepsTotal}
                </dt>
                <dd>steps found in video</dd>
              </div>
              <div>
                <dt>{benchmarks.sessions.mean.step_error_s?.toFixed(2)} s</dt>
                <dd>mean step timing error</dd>
              </div>
            </dl>
          )}
        </div>
        <div className="hero-stage">
          {plan && <Showroom key={plan.id} plan={plan} onMoment={setMoment} />}
          {plan && <ProductTag plan={plan} />}
          {plan && moment && (
            <div className="hero-step">
              <span className="hero-step-number">{step ? step.index : moment.phase === 'done' ? '✓' : '·'}</span>
              <span>
                <b>{step ? step.title : moment.phase === 'done' ? `${plan.title}, assembled` : 'Finding the parts'}</b>
                <span className="mono">{PHASE_TEXT[moment.phase]}</span>
              </span>
            </div>
          )}
          <div className="hero-products" role="tablist" aria-label="Products">
            {library.map((p, i) => (
              <button key={p.id} type="button" role="tab" aria-selected={i === index} onClick={() => setIndex(i)}>
                <img src={manualPageUrl(p.id, 1)} alt="" />
                <span>{p.title.split(' ')[0]}</span>
                {i === index && <i key={index} style={{ animationDuration: `${SHOWROOM_SECONDS}s` }} />}
              </button>
            ))}
          </div>
        </div>
      </header>

      <section id="demos" className="section">
        <header className="section-head">
          <span className="tag">See it work</span>
          <h2>Four demos, one graph.</h2>
          <p className="muted">
            Three replay real runs of the models on the furniture library; live mode uses your camera. The tiles follow the product in the
            showroom.
          </p>
        </header>
        {plan && <DemoGrid plan={plan} session={session} truth={truth} />}
        <div className="demo-links">
          <Go route={{ page: 'ar', furniture: featured }}>
            Copilot session <ArrowRight size={14} />
          </Go>
          <Go route={{ page: 'manual', furniture: featured }}>
            Manual reader <ArrowRight size={14} />
          </Go>
          <Go route={{ page: 'video', furniture: featured }}>
            Video analyzer <ArrowRight size={14} />
          </Go>
          <Go route={{ page: 'live' }}>
            Live <ArrowRight size={14} />
          </Go>
        </div>
      </section>

      <section id="how" className="section">
        <header className="section-head">
          <span className="tag">How it works</span>
          <h2>Read, track, guide.</h2>
          <p className="muted">
            A manual is a graph in disguise: parts are nodes, joints are edges, and the steps order the edges. Each stage works on that one
            graph.
          </p>
        </header>
        <div className="how">
          <article>
            <span className="how-index mono">01 · perception/manual.py</span>
            <h3>Read the manual</h3>
            <p className="muted">
              All pages go to Muse Spark in one request, decoding constrained to a JSON schema. <code>to_plan</code> resolves repeated
              letters to instances, brings in parts drawn without an arrow, and validates the graph.
            </p>
            <pre className="how-code">{`parts    A ×2, B, C, D, E, F, G ×2
joints   16 edges, compared by letter pair
steps    5 · each: adds, joins, orientation`}</pre>
          </article>
          <article>
            <span className="how-index mono">02 · assembly/tracking.py</span>
            <h3>Track the build</h3>
            <p className="muted">
              Frames are checked against the plan with the manual's step pages as references. An HMM over k steps done filters them: stay
              .86, advance .11, skip .02, back .01; evidence capped to [.15, .85] and tempered by .7.
            </p>
            <pre className="how-code">{`log P(obs | k) = 0.7 Σ[i≤k] log pᵢ
               + 0.7 Σ[i>k] log(1 − pᵢ)`}</pre>
          </article>
          <article>
            <span className="how-index mono">03 · engine/choreography.ts</span>
            <h3>Guide the next move</h3>
            <p className="muted">
              Every instant of the build is computed from the plan: turns between build orientations, parts on arcs to their poses, and each
              fitting from the tray to its hole, driven by the step's tool.
            </p>
            <pre className="how-code">{`locate → place → fasten → verify
${plan ? plan.hardware.reduce((n, h) => n + h.count, 0) : 72} fittings, seated from each joint's holes`}</pre>
          </article>
        </div>
      </section>

      <section id="research" className="section">
        <header className="section-head">
          <span className="tag">Research foundations</span>
          <h2>Built on the literature.</h2>
          <p className="muted">
            The methods Darius implements or builds on, and where each one shows up. Darius is independent work, not affiliated with these
            authors.
          </p>
        </header>
        <div className="papers">
          {RESEARCH.map((paper) => (
            <a key={paper.title} className="paper" href={scholar(paper.title)} target="_blank" rel="noreferrer">
              <span className="paper-venue mono">{paper.venue}</span>
              <strong>{paper.title}</strong>
              <span className="paper-authors">{paper.authors}</span>
              <span className="paper-use">{paper.use}</span>
            </a>
          ))}
        </div>
      </section>

      <section id="stack" className="section">
        <header className="section-head">
          <span className="tag">Stack</span>
          <h2>What it runs on.</h2>
        </header>
        <div className="stack">
          {STACK.map((layer) => (
            <div key={layer.layer}>
              <span className="tag">{layer.layer}</span>
              <ul>
                {layer.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {benchmarks && (
        <section id="results" className="section">
          <header className="section-head">
            <span className="tag">Results</span>
            <h2>Measured, not claimed.</h2>
            <p className="muted">
              {benchmarks.model}, {benchmarks.date}. Each manual is generated from a procedural plan and read back by the model; each
              session is a rendered build with frame-exact ground truth. CI re-scores the stored outputs on every commit.
            </p>
          </header>
          <div className="table-wrap">
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
          <p className="results-links">
            <a href={`${SOURCE}/docs/architecture.md`}>
              Architecture <ArrowUpRight size={13} />
            </a>
            <a href={`${SOURCE}/engine/darius/eval/benchmark.py`}>
              Benchmark code <ArrowUpRight size={13} />
            </a>
            <a href={`${SOURCE}/assemblies/benchmarks.json`}>
              Raw results <ArrowUpRight size={13} />
            </a>
          </p>
        </section>
      )}

      <footer className="home-foot">
        <Wordmark onClick={() => navigate({ page: 'home' })} />
        <span className="faint">Darius 1.1 · MIT license · Muse Spark via the Meta Model API · three.js</span>
        <a href={REPO} className="btn btn-outline btn-sm">
          <Github size={15} /> Source
        </a>
      </footer>
    </div>
  );
}
