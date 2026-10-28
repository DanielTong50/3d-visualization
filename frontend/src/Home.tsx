import { ChevronRight } from 'lucide-react';
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent, type ReactNode } from 'react';
import { accessCodeRequired, savedAccessCode } from './api';
import { Wordmark } from './Logo';
import { navigate, SCENES_URL } from './navigation';
import { Pipeline } from './Pipeline';
import { sceneSource } from './replay';

// Loaded after the page itself: the 3D engine is most of the app's weight.
const Showcase = lazy(() => import('./Showcase').then((module) => ({ default: module.Showcase })));
const SceneGraphView = lazy(() => import('./SceneGraphView').then((module) => ({ default: module.SceneGraphView })));

const HERO_SECONDS = 14;

interface DemoScene {
  id: string;
  title: string;
  /** "scanned" or "hand-authored", when the gallery says. */
  kind: string | null;
  objectCount: number;
  description: string | null;
  credit: string | null;
  /** Thumbnail URL, or null to draw simple shapes. */
  thumbnail: string | null;
  /** Appearance quality on held-out frames, for scanned scenes. */
  psnr: number | null;
}

// Used only if /scenes/index.json (written by the pipeline's gallery step) can't be read.
const FALLBACK_DEMO_IDS = ['kitchen-counter', 'objectron-mugs', 'sandbox'];
const MAX_VIDEO_BYTES = 500 * 1024 * 1024;
const SAFE_ID = /^[\w-]+$/;
const SAFE_FILE = /^[\w.-]+$/;
const SOURCE_URL = 'https://github.com/DanielTong50/darius';

const EXPORTS: { name: string; file: string; body: string }[] = [
  {
    name: 'MuJoCo',
    file: 'scene.xml',
    body: 'Free bodies with full inertia tensors, contact parameters from friction and restitution, grasp sites and the filmed cameras.',
  },
  {
    name: 'Isaac Sim',
    file: 'scene.usda',
    body: 'OpenUSD with UsdPhysics: rigid bodies, mass properties, convex-hull colliders and physics materials, +z up.',
  },
  { name: 'URDF', file: 'urdf/*.urdf', body: 'One URDF per object for PyBullet, Drake and Genesis, placed by world.json.' },
  {
    name: 'Grasps',
    file: 'world.json',
    body: 'Antipodal parallel-jaw grasp candidates for every object, scored by friction-cone margin, in each object’s frame.',
  },
];

const SNIPPET = `import darius
from darius.robotics.sim import PushEnv

scene = darius.load_scene("scenes/kitchen-counter")
env = PushEnv(scene, target="obj_03")   # push the bottle to a goal
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0.1, 0.0])`;

// Objectron benchmark, 24 test clips (docs/results.md).
const FIGURES: [string, string][] = [
  ['24/24', 'clips processed end to end'],
  ['100%', 'objects named correctly'],
  ['88%', 'within 25% of true size'],
  ['23.2 dB', 'held-out image quality (PSNR)'],
];

const text = (value: unknown): string | null => (typeof value === 'string' && value.trim() ? value : null);

/** The demo gallery: entries from /scenes/index.json, or the fallback scenes read one by one. */
async function loadDemos(): Promise<DemoScene[]> {
  try {
    const response = await fetch(`${SCENES_URL}/index.json`);
    if (response.ok) {
      const entries = (await response.json()) as unknown;
      if (Array.isArray(entries)) {
        return entries.flatMap((entry: Record<string, unknown>) => {
          const id = text(entry?.id);
          if (!id || !SAFE_ID.test(id)) return [];
          const thumbnail = text(entry.thumbnail);
          return [
            {
              id,
              title: text(entry.title) ?? id,
              kind: text(entry.kind),
              objectCount: typeof entry.object_count === 'number' ? entry.object_count : 0,
              description: text(entry.description),
              credit: text(entry.credit),
              thumbnail: thumbnail && SAFE_FILE.test(thumbnail) ? `${SCENES_URL}/${id}/${thumbnail}` : null,
              psnr: typeof entry.psnr_db === 'number' ? entry.psnr_db : null,
            },
          ];
        });
      }
    }
  } catch {
    // fall through to the fallback list
  }
  const demos = await Promise.all(
    FALLBACK_DEMO_IDS.map(async (id): Promise<DemoScene | null> => {
      try {
        const response = await fetch(`${SCENES_URL}/${id}/scene.json`);
        if (!response.ok) return null;
        const scene = (await response.json()) as { title?: string; objects?: unknown[] };
        return {
          id,
          title: scene.title ?? id,
          kind: null,
          objectCount: scene.objects?.length ?? 0,
          description: null,
          credit: null,
          thumbnail: `${SCENES_URL}/${id}/thumbnail.jpg`,
          psnr: null,
        };
      } catch {
        return null;
      }
    }),
  );
  return demos.filter((demo): demo is DemoScene => demo !== null);
}

/** How long a video file runs, or null if this browser can't read it (the server checks again). */
function videoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    let settled = false;
    const finish = (seconds: number | null) => {
      if (settled) return;
      settled = true;
      URL.revokeObjectURL(url);
      resolve(seconds);
    };
    video.preload = 'metadata';
    video.onloadedmetadata = () => finish(Number.isFinite(video.duration) ? video.duration : null);
    video.onerror = () => finish(null);
    window.setTimeout(() => finish(null), 5000);
    video.src = url;
  });
}

/** A plain-language reason the video can't be used, or null if it looks fine. */
async function problemWithVideo(file: File): Promise<string | null> {
  const looksLikeVideo = file.type.startsWith('video/') || /\.(mp4|mov|m4v)$/i.test(file.name);
  if (!looksLikeVideo) return "That file isn't a video. Choose a video from your camera roll (MP4 or MOV).";
  if (file.size > MAX_VIDEO_BYTES) {
    return `That video is ${Math.round(file.size / 1e6)} MB, over the 500 MB limit. A 30–60 second video is plenty.`;
  }
  const seconds = await videoDuration(file);
  if (seconds !== null && seconds < 8) {
    return `That video is only ${Math.round(seconds)} seconds long. Film 30–60 seconds, walking slowly around the table.`;
  }
  if (seconds !== null && seconds > 180) {
    return `That video is ${Math.round(seconds / 60)} minutes long. Please keep it under 3 minutes; 30–60 seconds is plenty.`;
  }
  return null;
}

const openScene = (sceneId: string, view?: string) => navigate({ page: 'scene', sceneId, ...(view ? { view } : {}) });

interface HomeProps {
  onStart: (file: File, accessCode: string | null) => void;
  /** Set when the server asked for an access code (again, if `wrongCode`). */
  needsCode: { file: File; wrongCode: boolean } | null;
  onCancelCode: () => void;
}

/** Start screen: what Darius does, a live scan, how it works, demo scenes and filming tips. */
export function Home({ onStart, needsCode, onCancelCode }: HomeProps) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [demos, setDemos] = useState<DemoScene[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [askCodeFor, setAskCodeFor] = useState<File | null>(null);
  const [code, setCode] = useState('');

  useEffect(() => {
    let cancelled = false;
    loadDemos().then((loaded) => !cancelled && setDemos(loaded));
    return () => {
      cancelled = true;
    };
  }, []);

  const codeFile = needsCode?.file ?? askCodeFor;
  // The hero turns through the real scans; the hand-built reference scene isn't one of them.
  const scans = useMemo(() => (demos ?? []).filter((demo) => demo.kind !== 'hand-authored'), [demos]);
  const reference = demos?.find((demo) => demo.kind === 'hand-authored') ?? null;
  const [heroIndex, setHeroIndex] = useState(0);
  const [heroShown, setHeroShown] = useState<string | null>(null);
  const [replayId, setReplayId] = useState<string | null>(null);
  const [graphId, setGraphId] = useState<string | null>(null);
  const hero = scans.length ? scans[heroIndex % scans.length] : null;
  const shown = scans.find((demo) => `${SCENES_URL}/${demo.id}` === heroShown) ?? null;
  const replay = scans.find((demo) => demo.id === replayId) ?? scans[0] ?? null;
  const replaySource = useMemo(() => (replay ? sceneSource(`${SCENES_URL}/${replay.id}`) : null), [replay]);
  const graphScene = (demos ?? []).find((demo) => demo.id === graphId) ?? scans[0] ?? null;

  // Next scan every HERO_SECONDS, counted from when the current one appeared.
  useEffect(() => {
    if (scans.length < 2 || !heroShown) return;
    const timer = window.setTimeout(() => {
      if (!document.hidden) setHeroIndex((i) => (i + 1) % scans.length);
    }, HERO_SECONDS * 1000);
    return () => window.clearTimeout(timer);
  }, [heroShown, scans.length]);

  const pickVideo = () => fileInput.current?.click();

  const onPick = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = ''; // so choosing the same file again still triggers a change
    if (!file) return;
    setProblem(null);
    setChecking(true);
    const issue = await problemWithVideo(file);
    if (issue) {
      setChecking(false);
      setProblem(issue);
      return;
    }
    const stored = savedAccessCode();
    const mustAsk = !stored && (await accessCodeRequired());
    setChecking(false);
    if (mustAsk) setAskCodeFor(file);
    else onStart(file, stored);
  };

  const onSubmitCode = (event: FormEvent) => {
    event.preventDefault();
    if (!codeFile || !code.trim()) return;
    setAskCodeFor(null);
    onStart(codeFile, code.trim());
  };

  const cancelCode = () => {
    setAskCodeFor(null);
    onCancelCode();
  };

  const scanButton = (size: 'lg' | 'md', variant: 'primary' | 'outline') => (
    <button type="button" className={`btn btn-${variant}${size === 'lg' ? ' btn-lg' : ''}`} onClick={pickVideo} disabled={checking}>
      {checking && <span className="loader loader-sm" aria-hidden="true" />}
      {checking ? 'Checking video…' : 'Scan a table'}
    </button>
  );

  return (
    <div className="page site">
      <input
        ref={fileInput}
        type="file"
        accept="video/*,.mp4,.mov,.m4v"
        className="visually-hidden"
        onChange={onPick}
        tabIndex={-1}
        aria-hidden="true"
      />

      <header className="nav">
        <Wordmark href={import.meta.env.BASE_URL} />
        <nav className="nav-links" aria-label="Sections">
          <a href="#pipeline">Pipeline</a>
          <a href="#robotics">Robotics</a>
          <a href="#results">Results</a>
          <a href="#scenes">Scenes</a>
          <a href={SOURCE_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
        {scanButton('md', 'primary')}
      </header>

      <main>
        <section className="intro">
          <h1>Film a table. Get a world a robot can learn in.</h1>
          <div className="intro-side">
            <p>
              Darius rebuilds a tabletop in 3D, separates every object, measures it, estimates its mass, friction and bounce,
              and compiles the scene for MuJoCo, Isaac Sim and URDF, with grasp candidates for each object.
            </p>
            <div className="intro-actions">
              {hero && (
                <button type="button" className="btn btn-primary btn-lg" onClick={() => openScene(shown?.id ?? hero.id)}>
                  Open this scene
                </button>
              )}
              {scanButton('lg', 'outline')}
            </div>
          </div>
        </section>

        <section className="scene-stage" aria-label="Live scene">
          <div className="scene-frame">
            {hero && (
              <Suspense fallback={null}>
                <Showcase sceneDir={`${SCENES_URL}/${hero.id}`} onReady={setHeroShown} />
              </Suspense>
            )}
            <span className="mark mark-tl" aria-hidden="true" />
            <span className="mark mark-tr" aria-hidden="true" />
            <span className="mark mark-bl" aria-hidden="true" />
            <span className="mark mark-br" aria-hidden="true" />
          </div>
          {shown && (
            <div className="scene-caption">
              <span>
                {shown.title}
                <span className="faint">
                  {' '}
                  · {shown.objectCount} object{shown.objectCount === 1 ? '' : 's'}
                  {shown.psnr ? ` · ${shown.psnr.toFixed(1)} dB on held-out frames` : ''}
                </span>
              </span>
              <span className="pager" role="group" aria-label="Scanned scenes">
                {scans.map((demo, i) => (
                  <button
                    key={demo.id}
                    type="button"
                    aria-label={demo.title}
                    aria-pressed={demo.id === shown.id}
                    title={demo.title}
                    onClick={() => setHeroIndex(i)}
                  />
                ))}
              </span>
            </div>
          )}
        </section>

        <section id="pipeline" className="band band-dark">
          <div className="band-head band-head-split">
            <div>
              <span className="tag">Pipeline</span>
              <h2>Seven stages, from pixels to physics.</h2>
            </div>
            <p>
              Each scan runs this graph on cloud GPUs in about six minutes. Below is what every stage really produced for one
              scene; pick a stage, or a scene.
            </p>
          </div>
          {replay && replaySource && (
            <>
              <div className="replay-scenes" role="group" aria-label="Scene to replay">
                {scans.map((demo) => (
                  <button key={demo.id} type="button" aria-pressed={demo.id === replay.id} onClick={() => setReplayId(demo.id)}>
                    {demo.title}
                  </button>
                ))}
              </div>
              <Pipeline source={replaySource} autoplay />
            </>
          )}
        </section>

        <section id="robotics" className="band">
          <div className="band-head band-head-split">
            <div>
              <span className="tag">Robotics</span>
              <h2>Every scan compiles to a robot environment.</h2>
            </div>
            <p>
              Mass, inertia and contact parameters come from the estimates; colliders from convex decomposition; grasps from the
              collision geometry. All in one +z-up world frame, checked in MuJoCo and OpenUSD on every commit.
            </p>
          </div>
          {graphScene && (
            <div className="graph-panel">
              <div className="graph-panel-head">
                <div>
                  <h3>3D scene graph</h3>
                  <p>
                    What a planner reads: every object at its center of mass over its collision skeleton, its convex parts
                    and grasp candidates, what supports it, what is near it and which cameras saw it. Drag to orbit; hover
                    or tap a node.
                  </p>
                </div>
                <div className="replay-scenes graph-scenes" role="group" aria-label="Scene graph to show">
                  {[...scans, ...(reference ? [reference] : [])].map((demo) => (
                    <button key={demo.id} type="button" aria-pressed={demo.id === graphScene.id} onClick={() => setGraphId(demo.id)}>
                      {demo.title}
                    </button>
                  ))}
                </div>
              </div>
              <Suspense fallback={<div className="graph-view" />}>
                <SceneGraphView sceneDir={`${SCENES_URL}/${graphScene.id}`} />
              </Suspense>
            </div>
          )}
          <ul className="exports">
            {EXPORTS.map(({ name, file, body }) => (
              <li key={name}>
                <span className="exports-file">{file}</span>
                <h3>{name}</h3>
                <p>{body}</p>
              </li>
            ))}
          </ul>
          <div className="robot-code">
            <pre className="code" aria-label="Python example">
              <code>{SNIPPET}</code>
            </pre>
            <div className="robot-code-side">
              <p>
                The Python package loads any scene, writes the bundle (<span className="mono">darius export</span>), and wraps
                it as a Gymnasium-style pushing environment on MuJoCo.
              </p>
              {hero && (
                <button type="button" className="btn btn-outline" onClick={() => openScene(scans[0].id, 'robot')}>
                  Open the robot view
                </button>
              )}
            </div>
          </div>
        </section>

        <section id="results" className="band band-warm">
          <div className="results">
            <div className="results-lead">
              <span className="tag">Objectron benchmark</span>
              <p className="statement">
                On 24 real phone clips with measured ground truth, the median object comes out at its true size.
              </p>
              <a className="text-link" href={`${SOURCE_URL}/blob/main/docs/results.md`} target="_blank" rel="noreferrer">
                Full results <ChevronRight size={14} aria-hidden="true" />
              </a>
            </div>
            <p className="results-number">
              0.99<span>×</span>
              <small>median estimated ÷ true size</small>
            </p>
          </div>
          <dl className="figures">
            {FIGURES.map(([value, label]) => (
              <div key={label}>
                <dt>{value}</dt>
                <dd>{label}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section id="scenes" className="band">
          <div className="band-head band-head-left">
            <h2>Open a scene.</h2>
            <p>Ten real scans, no filming needed. Each one opens in the full simulator.</p>
          </div>
          {demos === null ? (
            <p className="muted">Loading scenes…</p>
          ) : scans.length === 0 ? (
            <p className="muted">No scenes are available right now.</p>
          ) : (
            <ul className="cards">
              {scans.map((demo) => (
                <li key={demo.id}>
                  <button type="button" className="card" onClick={() => openScene(demo.id)}>
                    <Thumbnail src={demo.thumbnail} />
                    <span className="tag">Real scan</span>
                    <span className="card-title">{demo.title}</span>
                    {demo.description && <span className="card-text">{demo.description}</span>}
                    <span className="card-foot">
                      {demo.objectCount} object{demo.objectCount === 1 ? '' : 's'}
                      {demo.psnr ? ` · ${demo.psnr.toFixed(1)} dB` : ''}
                      {demo.credit ? ` · ${demo.credit}` : ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {reference && (
          <section id="reference" className="band band-warm">
            <div className="reference">
              <div>
                <span className="tag">Reference scene</span>
                <h2>Every scan runs on an engine tested on its own.</h2>
                <p>
                  The simulator was built against a hand-made scene before the first scan existed: one object for every
                  collider type, every value set by hand. Tests hold it to closed-form mechanics (a pushed box slides v0²/2μg,
                  a dropped ball rebounds to e² of its height) in the browser and in MuJoCo. Scans plug into the same engine.
                </p>
                <button type="button" className="btn btn-outline" onClick={() => openScene(reference.id)}>
                  Open the reference scene
                </button>
              </div>
              <dl className="reference-facts">
                <div>
                  <dt>5</dt>
                  <dd>collider types: box, cylinder, sphere, hull, convex parts</dd>
                </div>
                <div>
                  <dt>120 Hz</dt>
                  <dd>fixed physics step, independent of frame rate</dd>
                </div>
                <div>
                  <dt>3</dt>
                  <dd>engines agree: Rapier in the browser, MuJoCo, USD/PhysX</dd>
                </div>
              </dl>
            </div>
          </section>
        )}

        <section id="film" className="band">
          <div className="capture">
            <div>
              <span className="tag">Filming</span>
              <h2>Film your own table.</h2>
              <ol>
                <li>Bright, even light. No direct sun.</li>
                <li>Two slow circles: one at table height, one from higher up.</li>
                <li>Keep the whole table in view. 5 to 8 objects, a few centimeters apart.</li>
                <li>Skip glass, mirrors and shiny metal.</li>
              </ol>
              {scanButton('lg', 'primary')}
              <p className="small muted">30 to 60 seconds, MP4 or MOV. Ready in about six minutes.</p>
            </div>
            <FilmingDiagram />
          </div>
        </section>
      </main>

      <footer className="footer">
        <Wordmark href={import.meta.env.BASE_URL} />
        <span>Real-to-sim scene compiler. Built for the Meta Global AI Developer Hackathon, 2026.</span>
        <span className="footer-links">
          <a href={`${SOURCE_URL}/blob/main/docs/robotics.md`} target="_blank" rel="noreferrer">
            Robotics docs
          </a>
          <a href={SOURCE_URL} target="_blank" rel="noreferrer">
            Source
          </a>
        </span>
      </footer>

      {problem && (
        <Modal title="This video won't work" onClose={() => setProblem(null)}>
          <p className="muted">{problem}</p>
          <div className="modal-actions">
            <button type="button" className="btn btn-ghost" onClick={() => setProblem(null)}>
              Close
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                setProblem(null);
                pickVideo();
              }}
            >
              Choose another video
            </button>
          </div>
        </Modal>
      )}

      {codeFile && (
        <Modal title="Access code" onClose={cancelCode}>
          <form className="modal-form" onSubmit={onSubmitCode}>
            <label htmlFor="access-code" className="muted">
              {needsCode?.wrongCode ? "That code didn't work. Check it and try again." : 'Enter the access code you were given to start processing.'}
            </label>
            <input
              id="access-code"
              className="input"
              value={code}
              onChange={(event) => setCode(event.target.value)}
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              autoFocus
            />
            <div className="modal-actions">
              <button type="button" className="btn btn-ghost" onClick={cancelCode}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={!code.trim()}>
                Start processing
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
}

/** A centered dialog over a dimmed page; Escape or a click outside closes it. */
function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onPointerDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <h2 id="modal-title">{title}</h2>
        {children}
      </section>
    </div>
  );
}

/** The scene's thumbnail, or simple shapes for scenes without one. */
function Thumbnail({ src }: { src: string | null }) {
  const [missing, setMissing] = useState(false);
  if (!src || missing) {
    return (
      <svg className="scene-thumb" viewBox="0 0 160 100" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <rect width="160" height="100" fill="#15181d" />
        <path d="M14 70 L146 70 L132 52 L28 52 Z" fill="#5c4532" />
        <rect x="40" y="46" width="30" height="12" rx="2" fill="#2f5d8a" />
        <rect x="76" y="38" width="10" height="18" rx="2" fill="#c9553a" />
        <circle cx="100" cy="51" r="6" fill="#e0b43c" />
        <rect x="110" y="47" width="18" height="10" rx="2" fill="#5aa59c" />
      </svg>
    );
  }
  return <img className="scene-thumb" src={src} alt="" loading="lazy" onError={() => setMissing(true)} />;
}

/** Top view of a table with the phone's path circling it twice. */
function FilmingDiagram() {
  return (
    <svg
      className="filming-diagram"
      viewBox="0 0 320 240"
      role="img"
      aria-label="Walk in two circles around the table, pointing the phone at its center"
    >
      <ellipse cx="160" cy="124" rx="140" ry="92" fill="none" stroke="#b9b9b9" strokeWidth="1" strokeDasharray="3 5" />
      <ellipse cx="160" cy="124" rx="112" ry="70" fill="none" stroke="#8c8c8c" strokeWidth="1" strokeDasharray="3 5" />
      <rect x="94" y="84" width="132" height="80" rx="4" fill="#ffffff" stroke="#d4d4d4" />
      <circle cx="124" cy="110" r="9" fill="#ff4f00" />
      <rect x="146" y="101" width="28" height="18" rx="2" fill="#7d7d7d" />
      <circle cx="198" cy="140" r="8" fill="#c9c9c9" />
      <rect x="128" y="132" width="20" height="14" rx="2" fill="#000000" opacity="0.55" />
      <g transform="translate(160 16)">
        <rect x="-9" y="0" width="18" height="28" rx="3" fill="#000000" />
      </g>
      <path d="M 160 48 L 160 76" stroke="#000000" strokeWidth="1" strokeDasharray="2 3" />
    </svg>
  );
}
