import { ArrowUpRight } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import type { Moment } from '../engine/choreography';
import { loadLibrary, loadPlan, type AssemblyPlan, type PlanSummary } from '../engine/plan';
import {
  loadBenchmarks,
  loadSessionResult,
  loadSessionTruth,
  loadTrackingResult,
  manualPageUrl,
  sessionVideoUrl,
  type Benchmarks,
  type SessionResult,
  type SessionTruth,
  type TrackingResult,
} from '../engine/results';
import { Wordmark } from '../Logo';
import { navigate, type Route } from '../navigation';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { BeliefBars } from '../ui/BeliefBars';
import { EvidenceHeatmap } from '../ui/EvidenceHeatmap';
import { TrackedVideo } from '../ui/TrackedVideo';
import { ArDemo } from './ArDemo';

/**
 * The front page. The copilot builds each product of the library behind the headline, full bleed
 * and live, HUD and all; when a build is done the next product starts.
 * The page over it is plain: a statement, the numbers, the demos as captioned figures, the models
 * and what each one does, the method, the results, and the references. Every number comes from
 * assemblies/benchmarks.json, written by `darius eval`.
 */

const REPO = 'https://github.com/DanielTong50/3d-visualization';
const SOURCE = `${REPO}/blob/main`;
const POSTERS = ['copilot-fasten', 'copilot', 'copilot-turn'].map((name) => `${import.meta.env.BASE_URL}posters/${name}.jpg`);
const FADE_MS = 700;
const SHUFFLE_MS = 5000;
const PHASE_TEXT: Record<string, string> = {
  intro: 'finding the parts',
  reorient: 'turning the piece',
  locate: 'locating the parts',
  place: 'placing',
  fasten: 'fastening',
  verify: 'verifying',
  done: 'assembled',
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

/** A small loop of time, for figures that animate on their own. */
function useTicker(periodMs: number): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => setTick((t) => t + 1), periodMs);
    return () => clearInterval(timer);
  }, [periodMs]);
  return tick;
}

type FigureId = 'copilot' | 'manual' | 'video' | 'graph';

interface FigureSpec {
  route: Route;
  title: string;
  caption: string;
}

/**
 * The demos as captioned figures in a grid that rearranges itself: every few seconds the next
 * figure takes the large slot (a view transition morphs them into place where supported).
 */
function Figures({
  plan,
  session,
  truth,
  tracks,
}: {
  plan: AssemblyPlan;
  session: SessionResult | null;
  truth: SessionTruth | null;
  tracks: TrackingResult | null;
}) {
  const [order, setOrder] = useState<FigureId[]>(['copilot', 'manual', 'video', 'graph']);
  const [paused, setPaused] = useState(false);
  const tick = useTicker(1200);
  const placed = useMemo(
    () => new Map((truth?.segments ?? []).filter((s) => s.phase === 'place').map((s) => [s.step, s.start + s.duration])),
    [truth],
  );

  useEffect(() => {
    if (paused || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => {
      const next = (o: FigureId[]) => [...o.slice(1), o[0]];
      const doc = document as Document & { startViewTransition?: (update: () => void) => unknown };
      if (doc.startViewTransition) doc.startViewTransition(() => flushSync(() => setOrder(next)));
      else setOrder(next);
    }, SHUFFLE_MS);
    return () => clearInterval(timer);
  }, [paused]);

  const steps = plan.steps.length;
  const built = tick % (steps + 2);
  const frame = session && session.frames.length ? tick % session.frames.length : 0;
  const specs: Record<FigureId, FigureSpec> = {
    copilot: {
      route: { page: 'ar', furniture: plan.id },
      title: 'Copilot session',
      caption: 'Every move drawn in place: parts to their targets, each fitting into its hole, the turn when the piece must flip.',
    },
    manual: {
      route: { page: 'manual', furniture: plan.id },
      title: 'Manual reader',
      caption: 'A real model run replayed: the pages are read into a graph, edge by edge, and scored against the plan.',
    },
    video: {
      route: { page: 'video', furniture: plan.id },
      title: 'Video analyzer',
      caption: 'A filmed build decoded: SAM 3.1 masks on every part, per-frame step verification, the most likely build history.',
    },
    graph: {
      route: { page: 'video', furniture: plan.id },
      title: 'Assembly graph and tracker',
      caption: `${plan.parts.length} parts, ${plan.joints.length} joints; the belief over steps done, frame by frame.`,
    },
  };
  const media: Record<FigureId, React.ReactNode> = {
    copilot: POSTERS.map((src, i) => (
      <img key={src} src={src} alt="" className={Math.floor(tick / 3) % POSTERS.length === i ? 'is-on' : ''} loading="lazy" />
    )),
    manual: (
      <div className="figure-paper">
        <img src={manualPageUrl(plan.id, 4 + (Math.floor(tick / 2) % steps))} alt="" loading="lazy" />
        <span className="figure-scan" />
      </div>
    ),
    video: (
      <>
        <TrackedVideo src={sessionVideoUrl(plan.id)} tracks={tracks} labels={false} />
        {session && (
          <div className="figure-heatmap">
            <EvidenceHeatmap plan={plan} result={session} placed={placed} />
          </div>
        )}
      </>
    ),
    graph: (
      <div className="figure-split">
        <AssemblyGraph plan={plan} completed={Math.min(built, steps)} active={built < steps ? built + 1 : null} compact />
        {session && (
          <BeliefBars
            plan={plan}
            belief={{ probabilities: session.posteriors[frame], completed: session.path[frame], confidence: session.confidence[frame] }}
          />
        )}
      </div>
    ),
  };

  return (
    <div className="figures" onPointerEnter={() => setPaused(true)} onPointerLeave={() => setPaused(false)}>
      {order.map((id, slot) => (
        <figure
          key={id}
          className={`figure slot-${slot}`}
          style={{ viewTransitionName: `figure-${id}` } as React.CSSProperties}
          onClick={() => navigate(specs[id].route)}
        >
          <div className={`figure-media figure-${id}`}>{media[id]}</div>
          <figcaption>
            <b>{specs[id].title}.</b> {specs[id].caption}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

const REFERENCES: { authors: string; title: string; venue: string; use: string }[] = [
  {
    authors: 'L. R. Rabiner',
    title: 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition',
    venue: 'Proceedings of the IEEE, 1989',
    use: 'the progress tracker, an HMM over steps done, filtered forward in log space',
  },
  {
    authors: 'A. J. Viterbi',
    title: 'Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm',
    venue: 'IEEE Transactions on Information Theory, 1967',
    use: 'the most likely build history of a video',
  },
  {
    authors: 'L. E. Baum, T. Petrie, G. Soules, N. Weiss',
    title: 'A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains',
    venue: 'Annals of Mathematical Statistics, 1970',
    use: 'forward–backward posteriors, the confidence of every decoded moment',
  },
  {
    authors: 'A. Richard, H. Kuehne, A. Iqbal, J. Gall',
    title: 'NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning',
    venue: 'CVPR 2018',
    use: "decoding a procedure under an ordered transcript, here the manual's steps",
  },
  {
    authors: 'R. Wang, Y. Zhang, J. Mao, C.-Y. Cheng, J. Wu',
    title: 'Translating a Visual LEGO Manual to a Machine-Executable Plan',
    venue: 'ECCV 2022',
    use: 'manual pages to an executable plan, the problem the manual reader solves',
  },
  {
    authors: 'R. Wang, Y. Zhang, J. Mao, R. Zhang, C.-Y. Cheng, J. Wu',
    title: 'IKEA-Manual: Seeing Shape Assembly Step by Step',
    venue: 'NeurIPS 2022 Datasets and Benchmarks',
    use: 'manuals aligned with parts and assembly graphs',
  },
  {
    authors: 'Y. Ben-Shabat, X. Yu, F. Saleh, D. Campbell, C. Rodriguez-Opazo, H. Li, S. Gould',
    title: 'The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose',
    venue: 'WACV 2021',
    use: 'furniture assembly as video understanding',
  },
  {
    authors: 'F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, A. Yao',
    title: 'Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities',
    venue: 'CVPR 2022',
    use: 'step recognition and progress in procedural assembly video',
  },
  {
    authors: 'X. Wang, T. Kwon, M. Rad, B. Pan, I. Chakraborty, et al.',
    title: 'HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World',
    venue: 'ICCV 2023',
    use: 'an assistant that watches a task and guides it',
  },
  {
    authors:
      'A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick',
    title: 'Segment Anything',
    venue: 'ICCV 2023',
    use: 'segmentation prompted by boxes and points',
  },
  {
    authors: 'N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al.',
    title: 'SAM 2: Segment Anything in Images and Videos',
    venue: 'ICLR 2025',
    use: 'propagating prompted masks through a video, forward and backward',
  },
  {
    authors: 'N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, et al.',
    title: 'SAM 3: Segment Anything with Concepts',
    venue: 'arXiv, 2025',
    use: 'the model family SAM 3.1 belongs to, which tracks every part',
  },
  {
    authors: 'J. Yang, H. Zhang, F. Li, X. Zou, C. Li, J. Gao',
    title: 'Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V',
    venue: 'arXiv, 2023',
    use: 'finding lost parts again from numbered outlines on the first frame',
  },
  {
    authors: 'G. E. Hinton',
    title: 'Training Products of Experts by Minimizing Contrastive Divergence',
    venue: 'Neural Computation, 2002',
    use: 'fusing mask contacts with the frame readings',
  },
  {
    authors: 'B. T. Willard, R. Louf',
    title: 'Efficient Guided Generation for Large Language Models',
    venue: 'arXiv, 2023',
    use: 'schema-constrained decoding, used for every model reading',
  },
  {
    authors: 'M. Agrawala, D. Phan, J. Heiser, J. Haymaker, J. Klingner, P. Hanrahan, B. Tversky',
    title: 'Designing Effective Step-by-Step Assembly Instructions',
    venue: 'SIGGRAPH 2003',
    use: 'instructions generated from geometry, the generated manuals',
  },
  {
    authors: 'W. Li, M. Agrawala, B. Curless, D. Salesin',
    title: 'Automated Generation of Interactive 3D Exploded View Diagrams',
    venue: 'SIGGRAPH 2008',
    use: 'explosion directions from blocking constraints in the manual drawings',
  },
  {
    authors: 'R. H. Wilson, J.-C. Latombe',
    title: 'Geometric Reasoning About Mechanical Assembly',
    venue: 'Artificial Intelligence, 1994',
    use: 'removal directions among the parts in place',
  },
  {
    authors: 'R. E. Tarjan',
    title: 'Efficiency of a Good But Not Linear Set Union Algorithm',
    venue: 'Journal of the ACM, 1975',
    use: 'union-find for the subassemblies after every step',
  },
  {
    authors: 'Y. Lee, E. S. Hu, J. J. Lim',
    title: 'IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks',
    venue: 'ICRA 2021',
    use: 'furniture as parts with connection points',
  },
];

/** What each model does in the pipeline, in order, and the number that measures it. */
function Models({ plan, tracks, benchmarks }: { plan: AssemblyPlan; tracks: TrackingResult | null; benchmarks: Benchmarks | null }) {
  const tracking = benchmarks?.tracking;
  const items = tracking ? Object.values(tracking.items) : [];
  const letters = items.reduce((n, t) => n + t.letters_right, 0);
  const pieces = items.reduce((n, t) => n + t.pieces, 0);
  const refound = items.reduce((n, t) => n + (t.refound ?? 0), 0);
  const spark = benchmarks?.model ?? 'Muse Spark';
  const sam = tracking?.models?.segmentation ?? 'SAM 3.1';
  const stages: { model: string; title: string; text: string; value: string; unit: string }[] = [
    {
      model: spark,
      title: 'Read the manual',
      text: 'Every page in one request, decoding constrained to a JSON schema. A resolver turns the reading into a validated assembly graph.',
      value: pct(benchmarks?.manuals.mean.edges_f1),
      unit: 'edge F1',
    },
    {
      model: spark,
      title: 'Ground the parts',
      text: 'On the first frame, a letter, a box and a point for every piece; a second look at a close crop around them pins down thin parts.',
      value: pieces ? `${letters}/${pieces}` : '—',
      unit: 'letters right, up to look-alikes',
    },
    {
      model: sam,
      title: 'Track every part',
      text: "One video request: each piece's box and point are its prompt, and its mask is propagated forward and backward through the build.",
      value: pct(tracking?.mean.mean_iou),
      unit: 'mask IoU, exact masks',
    },
    {
      model: spark,
      title: 'Find lost parts',
      text: 'SAM drops a part that leaves the frame. Every 4 s the lost ones are looked for again against the first frame, outlined and numbered, and SAM tracks once more.',
      value: items.length ? String(refound) : '—',
      unit: 'prompts added',
    },
    {
      model: spark,
      title: 'Verify each frame',
      text: "Sampled frames are checked against the manual's own step drawings, as P(step done) for every step.",
      value: pct(benchmarks?.sessions.mean.frame_accuracy),
      unit: 'frames right, decoded',
    },
    {
      model: 'tracker',
      title: 'Fuse the evidence',
      text: "Masks that touch for a second are counted against the plan's joints, and join the readings as a product of experts.",
      value: pct(tracking?.mean.frames_right_contacts),
      unit: `frames right, contacts alone (${pct(tracking?.mean.frames_right_exact_contacts)} with exact masks)`,
    },
  ];
  return (
    <div className="models">
      <figure className="models-figure" onClick={() => navigate({ page: 'video', furniture: plan.id })}>
        <TrackedVideo src={sessionVideoUrl(plan.id)} tracks={tracks} />
        <figcaption>
          <b>{plan.title}, tracked.</b> SAM 3.1 masks for {tracks ? tracks.tracks.length : 'all'} parts, prompted by Muse Spark on the first
          frame and wherever a lost part was found again (white corners). A stored run, replayed.
        </figcaption>
      </figure>
      <ol className="pipeline">
        {stages.map((stage, i) => (
          <li key={stage.title}>
            <span className="pipeline-index">{i + 1}</span>
            <div>
              <h3>
                {stage.title} <span>{stage.model}</span>
              </h3>
              <p>{stage.text}</p>
            </div>
            <div className="pipeline-value">
              <b>{stage.value}</b>
              <span>{stage.unit}</span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

const STACK: { layer: string; items: string }[] = [
  {
    layer: 'Perception',
    items:
      'Muse Spark 1.3 (manual reading, part grounding, re-finding, frame verification) and SAM 3.1 (video part tracking), through the Meta Model API',
  },
  { layer: 'Engine', items: 'Python 3.11, Pydantic, NumPy, OpenCV, FastAPI' },
  { layer: 'Serving', items: 'Modal, one CPU function that scales to zero' },
  { layer: 'Web', items: 'React 19, TypeScript, three.js with instancing and custom shaders, Vite' },
  { layer: 'Test data', items: 'Playwright renders the manuals and sessions; ffmpeg encodes them' },
  { layer: 'Quality', items: 'pytest and Vitest, Ruff and Prettier; CI re-scores the stored model results' },
];

export function Home() {
  const [library, setLibrary] = useState<PlanSummary[]>([]);
  const [plans, setPlans] = useState<AssemblyPlan[]>([]);
  const [index, setIndex] = useState(0);
  const [fading, setFading] = useState(false);
  const [moment, setMoment] = useState<Moment | null>(null);
  const [buildSeconds, setBuildSeconds] = useState(60);
  const shown = useRef<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null);
  // A product's stored run, kept with the product it belongs to: during a switch the new plan
  // must never be drawn with the previous product's data.
  const [results, setResults] = useState<{
    id: string;
    session: SessionResult | null;
    truth: SessionTruth | null;
    tracks: TrackingResult | null;
  } | null>(null);

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

  // The background moves on to the next product when a build is done, through a fade.
  const show = (next: number) => {
    setFading(true);
    window.setTimeout(() => {
      setIndex(next);
      setMoment(null);
      shown.current = null;
      setFading(false);
    }, FADE_MS);
  };
  // The copilot reports many times a second; the caption only needs each new step and phase.
  const onMoment = useCallback((next: Moment, duration: number) => {
    const key = `${next.step}:${next.phase}`;
    if (shown.current !== key) {
      shown.current = key;
      setMoment(next);
    }
    setBuildSeconds(duration);
  }, []);

  const plan = plans[index] ?? null;
  useEffect(() => {
    if (!plan) return;
    let cancelled = false;
    Promise.all([
      loadSessionResult(plan.id).catch(() => null),
      loadSessionTruth(plan.id).catch(() => null),
      loadTrackingResult(plan.id).catch(() => null),
    ]).then(([session, truth, tracks]) => {
      if (!cancelled) setResults({ id: plan.id, session, truth, tracks });
    });
    return () => {
      cancelled = true;
    };
  }, [plan]);
  const session = plan && results?.id === plan.id ? results.session : null;
  const truth = plan && results?.id === plan.id ? results.truth : null;
  const tracks = plan && results?.id === plan.id && results.tracks?.plan === plan.id ? results.tracks : null;

  const sessions = benchmarks ? Object.values(benchmarks.sessions.items) : [];
  const stepsFound = sessions.reduce((n, s) => n + s.steps_found, 0);
  const stepsTotal = sessions.reduce((n, s) => n + s.steps, 0);
  const step = plan && moment && moment.step >= 1 && moment.step <= plan.steps.length ? plan.steps[moment.step - 1] : null;
  const featured = plan?.id ?? 'halden-bookcase';

  return (
    <div className="home">
      <header className="hero">
        <div className={`hero-layer${fading ? ' is-fading' : ''}`}>
          {plan && (
            <ArDemo
              key={plan.id}
              furniture={plan.id}
              variant="hero"
              onMoment={onMoment}
              onLoop={() => plans.length > 1 && show((index + 1) % plans.length)}
            />
          )}
        </div>
        <div className="hero-scrim" aria-hidden="true" />

        <nav className="home-nav">
          <Wordmark onClick={() => navigate({ page: 'home' })} />
          <div className="home-nav-links">
            <a href="#demos">Demos</a>
            <a href="#models">Models</a>
            <a href="#method">Method</a>
            <a href="#results">Results</a>
            <a href="#references">References</a>
            <a href={REPO} className="icon-link" aria-label="Source on GitHub">
              <Github size={17} />
            </a>
          </div>
        </nav>

        <div className="hero-text">
          <h1>A copilot for building things.</h1>
          <p>
            Darius reads a flat-pack manual into an assembly graph, follows the build from a phone camera, and shows the next move in place,
            down to each dowel and screw.
          </p>
          <div className="hero-links">
            <button type="button" className="btn btn-dark" onClick={() => navigate({ page: 'ar', furniture: featured })}>
              Watch a build
            </button>
            <button type="button" className="text-link" onClick={() => navigate({ page: 'live' })}>
              Try it with your camera <span aria-hidden="true">→</span>
            </button>
          </div>
        </div>

        {plan && (
          <div className="hero-caption">
            <p>
              <span className="hero-caption-title">
                <b>{plan.title}</b>
                {step ? `, step ${step.index} of ${plan.steps.length}: ${step.title.toLowerCase()}` : ''}
              </span>
              <span>The copilot, live in your browser · {moment ? PHASE_TEXT[moment.phase] : 'loading'}</span>
            </p>
            <div className="hero-products" role="tablist" aria-label="Products">
              {library.map((p, i) => (
                <button key={p.id} type="button" role="tab" aria-selected={i === index} onClick={() => i !== index && show(i)}>
                  {p.title.split(' ')[0]}
                  {i === index && !fading && <i key={index} style={{ animationDuration: `${buildSeconds}s` }} />}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

      <section className="statement">
        <p className="lead">
          A flat-pack manual is a graph in disguise: parts are nodes, joints are edges, and the steps put the edges in order. Darius
          recovers that graph from the pages, follows a build through it from video, and turns its geometry into guidance.
        </p>
        {benchmarks && (
          <>
            <dl className="numbers">
              <div>
                <dt>{pct(benchmarks.manuals.mean.edges_f1)}</dt>
                <dd>of joints recovered from manuals (edge F1)</dd>
              </div>
              <div>
                <dt>{pct(benchmarks.tracking?.mean.mean_iou, 0)}</dt>
                <dd>mask IoU of SAM 3.1's part tracks against exact masks</dd>
              </div>
              <div>
                <dt>
                  {stepsFound}/{stepsTotal}
                </dt>
                <dd>build steps found in video</dd>
              </div>
              <div>
                <dt>{benchmarks.sessions.mean.step_error_s?.toFixed(2)} s</dt>
                <dd>mean error dating each step</dd>
              </div>
            </dl>
            <p className="numbers-note">
              {benchmarks.model} and {benchmarks.tracking?.models?.segmentation ?? 'SAM 3.1'} on a four-product library with generated
              manuals and rendered sessions. Stored runs, re-scored on every commit.
            </p>
          </>
        )}
      </section>

      <section id="demos" className="section">
        <div className="section-head">
          <h2>Demos</h2>
          <p>
            Three replay real runs of the models on the furniture library; live mode uses your camera. They follow the product in the
            background.
          </p>
        </div>
        {plan && <Figures plan={plan} session={session} truth={truth} tracks={tracks} />}
      </section>

      <section id="models" className="section">
        <div className="section-head">
          <h2>Models</h2>
          <p>
            Perception runs on two Meta models through the Meta Model API. Muse Spark reads the manual and the frames and points at the
            parts; SAM 3.1 follows every part through the video from those prompts.
          </p>
        </div>
        {plan && <Models plan={plan} tracks={tracks} benchmarks={benchmarks} />}
      </section>

      <section id="method" className="section">
        <div className="section-head">
          <h2>Method</h2>
          <p>One graph, three jobs. Each stage reads and writes the same assembly plan.</p>
        </div>
        <div className="method">
          <article>
            <span className="method-index">1</span>
            <h3>Read the manual</h3>
            <p>
              All pages go to a vision-language model in one request, its output constrained to a JSON schema. A deterministic resolver
              turns the reading into a validated plan: repeated letters become part instances, parts drawn without an arrow are brought in,
              and each connection resolves to a pair of instances.
            </p>
            <a href={`${SOURCE}/engine/darius/perception/manual.py`}>perception/manual.py</a>
          </article>
          <article>
            <span className="method-index">2</span>
            <h3>Track the build</h3>
            <p>
              Frames are checked against the plan with the manual's own step drawings as references, and the parts' masks add contacts.
              Progress is the hidden state of a Markov model over steps done: stay .86, advance .11, skip .02, back .01, with each frame's
              evidence capped to [.15, .85] and tempered by .7. Viterbi gives the build history, forward–backward its confidence.
            </p>
            <a href={`${SOURCE}/engine/darius/assembly/tracking.py`}>assembly/tracking.py</a>
          </article>
          <article>
            <span className="method-index">3</span>
            <h3>Guide the next move</h3>
            <p>
              Every instant of the build is computed from the plan: the turn into each step's orientation, each part's path to its pose, and
              each fitting's flight from the tray to its seat, which comes from the joint's hole pattern and moves with its part.
            </p>
            <a href={`${SOURCE}/frontend/src/engine/choreography.ts`}>engine/choreography.ts</a>
          </article>
        </div>
      </section>

      {benchmarks && (
        <section id="results" className="section">
          <div className="section-head">
            <h2>Results</h2>
            <p>
              Each manual is generated from a procedural plan and read back by the model; each session is a rendered build with frame-exact
              ground truth, where a step counts as done once its parts are in place, and the renderer's exact mask for every part.
            </p>
          </div>
          <div className="table-wrap">
            <table className="results">
              <thead>
                <tr>
                  <th rowSpan={2}>Product</th>
                  <th colSpan={4}>Manual → graph</th>
                  <th colSpan={3}>Video → timeline</th>
                  <th colSpan={3}>Video → part tracks</th>
                </tr>
                <tr>
                  <th>Parts F1</th>
                  <th>Edges F1</th>
                  <th>Steps exact</th>
                  <th>Hardware</th>
                  <th>Frames right</th>
                  <th>Steps found</th>
                  <th>Timing error</th>
                  <th>Mask IoU</th>
                  <th>Held</th>
                  <th>Joints touched</th>
                </tr>
              </thead>
              <tbody>
                {library.map((p) => {
                  const m = benchmarks.manuals.items[p.id];
                  const s = benchmarks.sessions.items[p.id];
                  const t = benchmarks.tracking?.items[p.id];
                  return (
                    <tr key={p.id}>
                      <td>
                        <button type="button" className="link" onClick={() => navigate({ page: 'manual', furniture: p.id })}>
                          {p.title}
                        </button>
                        <span>
                          {p.parts} parts, {p.joints} joints, {p.steps} steps
                        </span>
                      </td>
                      <td>{pct(m?.parts_f1)}</td>
                      <td className={m && m.edges_f1 < 1 ? 'is-miss' : ''}>{pct(m?.edges_f1)}</td>
                      <td>{pct(m?.steps_exact)}</td>
                      <td>{pct(m?.hardware_accuracy)}</td>
                      <td>{pct(s?.frame_accuracy)}</td>
                      <td>{s ? `${s.steps_found}/${s.steps}` : '—'}</td>
                      <td>{s?.step_error_s != null ? `${s.step_error_s.toFixed(2)} s` : '—'}</td>
                      <td>{pct(t?.mean_iou)}</td>
                      <td>{pct(t?.frames_tracked)}</td>
                      <td>{pct(t?.contacts_recall)}</td>
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

      <section id="references" className="section">
        <div className="section-head">
          <h2>References</h2>
          <p>
            The work Darius implements or builds on, and where each one is used. Darius is independent and not affiliated with these
            authors.
          </p>
        </div>
        <ol className="references">
          {REFERENCES.map((r) => (
            <li key={r.title}>
              <a href={scholar(r.title)} target="_blank" rel="noreferrer">
                {r.authors.endsWith('.') ? r.authors : `${r.authors}.`} <cite>{r.title}.</cite> {r.venue}.
              </a>
              <span>Used for {r.use}.</span>
            </li>
          ))}
        </ol>
      </section>

      <section id="stack" className="section">
        <div className="section-head">
          <h2>Stack</h2>
        </div>
        <dl className="stack">
          {STACK.map((s) => (
            <div key={s.layer}>
              <dt>{s.layer}</dt>
              <dd>{s.items}</dd>
            </div>
          ))}
        </dl>
      </section>

      <footer className="home-foot">
        <Wordmark onClick={() => navigate({ page: 'home' })} />
        <span>Darius 1.3 · MIT license</span>
        <a href={REPO}>
          Source <ArrowUpRight size={13} />
        </a>
      </footer>
    </div>
  );
}
