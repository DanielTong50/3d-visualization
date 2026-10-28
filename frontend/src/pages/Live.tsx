import { ArrowLeft, Camera, Film, FileImage, KeyRound, Loader2, RotateCw, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  checkFrames,
  grab,
  health,
  readManual,
  saveAccessCode,
  savedAccessCode,
  type FrameReading,
  type Health,
  type ManualReadResult,
} from '../engine/api';
import { loadLibrary, loadPlan, hardwareName, type AssemblyPlan, type PlanSummary } from '../engine/plan';
import { manualPageUrl, sessionVideoUrl } from '../engine/results';
import { Tracker, type Belief } from '../engine/tracking';
import { DEFAULT_FURNITURE, navigate } from '../navigation';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { BeliefBars } from '../ui/BeliefBars';

/**
 * Live mode: the copilot on a real camera. Pick a library product or photograph a manual (read
 * by /manuals into a plan), then point the camera at the build. Every few seconds a frame goes
 * to /frames; the answer is filtered by the same tracker as everywhere else, in the browser,
 * and the step card follows. A video file, or one of the library's filmed builds, can stand in
 * for the camera.
 */

const CHECK_EVERY_MS = 3000;
const TURN: Record<string, string> = {
  upright: 'Stand it upright',
  on_back: 'Lay it on its back',
  on_front: 'Lay it face down',
  on_side: 'Lay it on its side',
  upside_down: 'Turn it upside down',
};

type Source = { kind: 'library'; id: string } | { kind: 'manual'; result: ManualReadResult };
type Feed = { kind: 'camera' } | { kind: 'file'; url: string } | { kind: 'sample'; url: string };

interface Check {
  image: string;
  reading: FrameReading;
  seconds: number;
}

function Setup({
  status,
  code,
  setCode,
  source,
  setSource,
  onStart,
}: {
  status: Health | 'down' | null;
  code: string;
  setCode: (code: string) => void;
  source: Source;
  setSource: (source: Source) => void;
  onStart: (feed: Feed) => void;
}) {
  const [library, setLibrary] = useState<PlanSummary[]>([]);
  const [tab, setTab] = useState<'library' | 'photos'>(source.kind === 'manual' ? 'photos' : 'library');
  const [pages, setPages] = useState<File[]>([]);
  const [reading, setReading] = useState<'idle' | 'busy' | { error: string }>('idle');
  const [startedAt, setStartedAt] = useState(0);
  const [, tick] = useState(0);
  const files = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadLibrary()
      .then(setLibrary)
      .catch(() => setLibrary([]));
  }, []);
  useEffect(() => {
    if (reading !== 'busy') return;
    const timer = setInterval(() => tick((n) => n + 1), 500);
    return () => clearInterval(timer);
  }, [reading]);

  const thumbs = useMemo(() => pages.map((p) => URL.createObjectURL(p)), [pages]);
  useEffect(() => () => thumbs.forEach((t) => URL.revokeObjectURL(t)), [thumbs]);

  const read = async () => {
    setReading('busy');
    setStartedAt(performance.now());
    try {
      const result = await readManual(pages, code);
      setSource({ kind: 'manual', result });
      setReading('idle');
    } catch (error) {
      setReading({ error: error instanceof Error ? error.message : String(error) });
    }
  };

  const libraryId = source.kind === 'library' ? source.id : DEFAULT_FURNITURE;
  const down = status === 'down';
  const needsCode = status !== null && status !== 'down' && status.access_code_required && !code;

  return (
    <main className="live-setup">
      <section className="live-card glass">
        <span className="eyebrow track">1 · The manual</span>
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === 'library'} onClick={() => setTab('library')}>
            From the library
          </button>
          <button type="button" role="tab" aria-selected={tab === 'photos'} onClick={() => setTab('photos')}>
            Photograph it
          </button>
        </div>
        {tab === 'library' ? (
          <div className="live-library">
            {library.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`live-product${source.kind === 'library' && source.id === p.id ? ' is-on' : ''}`}
                onClick={() => setSource({ kind: 'library', id: p.id })}
              >
                <img src={manualPageUrl(p.id, 1)} alt="" loading="lazy" />
                <span>
                  <b>{p.title}</b>
                  <span className="faint mono">
                    {p.parts} parts · {p.steps} steps
                  </span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="live-photos">
            <p className="muted">
              Photograph every page in order (cover, parts, hardware, steps). Muse Spark reads them into a plan; it takes about a minute.
            </p>
            <input
              ref={files}
              type="file"
              accept="image/*"
              multiple
              capture="environment"
              hidden
              onChange={(e) => setPages([...pages, ...Array.from(e.target.files ?? [])].slice(0, 50))}
            />
            <div className="live-thumbs">
              {thumbs.map((url, i) => (
                <img key={url} src={url} alt={`Page ${i + 1}`} />
              ))}
              <button type="button" className="live-add" onClick={() => files.current?.click()}>
                <FileImage size={18} /> Add pages
              </button>
            </div>
            <div className="live-row">
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={!pages.length || reading === 'busy' || down || needsCode}
                onClick={read}
              >
                {reading === 'busy' ? (
                  <>
                    <Loader2 size={14} className="spin" /> Reading {pages.length} pages ·{' '}
                    {((performance.now() - startedAt) / 1000).toFixed(0)} s
                  </>
                ) : (
                  <>
                    <Sparkles size={14} /> Read {pages.length || ''} pages
                  </>
                )}
              </button>
              {pages.length > 0 && reading !== 'busy' && (
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setPages([])}>
                  Clear
                </button>
              )}
            </div>
            {typeof reading === 'object' && <p className="live-error">{reading.error}</p>}
            {source.kind === 'manual' && (
              <div className="live-read">
                <span className="eyebrow done">Read</span>
                <b>{source.result.plan.title}</b>
                <span className="faint mono">
                  {source.result.plan.parts.length} parts · {source.result.plan.joints.length} joints · {source.result.plan.steps.length}{' '}
                  steps · {source.result.seconds.toFixed(0)} s
                </span>
                <div className="live-read-graph">
                  <AssemblyGraph plan={source.result.plan} completed={source.result.plan.steps.length} compact spin={false} />
                </div>
                {source.result.notes.length > 0 && <p className="faint">Notes: {source.result.notes.join('; ')}</p>}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="live-card glass">
        <span className="eyebrow">2 · The build</span>
        <p className="muted">
          Point the camera at the parts and the build. A frame is checked every {CHECK_EVERY_MS / 1000} s against{' '}
          {source.kind === 'library' ? 'the plan and its manual pages' : 'the plan read from your photos'}; the tracker keeps the belief in
          your browser.
        </p>
        <div className="live-feeds">
          <button type="button" className="btn btn-guide" disabled={down || needsCode} onClick={() => onStart({ kind: 'camera' })}>
            <Camera size={15} /> Use the camera
          </button>
          <input
            ref={video}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onStart({ kind: 'file', url: URL.createObjectURL(file) });
            }}
          />
          <button type="button" className="btn btn-ghost" disabled={down || needsCode} onClick={() => video.current?.click()}>
            <Film size={15} /> Use a video file
          </button>
          {source.kind === 'library' && (
            <button
              type="button"
              className="btn btn-ghost"
              disabled={down || needsCode}
              onClick={() => onStart({ kind: 'sample', url: sessionVideoUrl(libraryId) })}
            >
              <RotateCw size={15} /> Try a filmed build
            </button>
          )}
        </div>
        <div className="live-server">
          <span className={`dot ${down ? 'is-down' : status ? 'is-up' : ''}`} />
          <span className="mono faint">
            {status === null
              ? 'Checking the Darius API…'
              : down
                ? 'The Darius API is not reachable right now.'
                : `Darius API ${status.version} · ${status.model}`}
          </span>
        </div>
        {status !== null && status !== 'down' && status.access_code_required && (
          <label className="live-code">
            <KeyRound size={14} />
            <input
              type="password"
              placeholder="Access code"
              value={code}
              onChange={(e) => {
                setCode(e.target.value);
                saveAccessCode(e.target.value);
              }}
            />
          </label>
        )}
      </section>
    </main>
  );
}

function Build({ plan, source, feed, code, onStop }: { plan: AssemblyPlan; source: Source; feed: Feed; code: string; onStop: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const tracker = useMemo(() => new Tracker(plan.steps.length), [plan]);
  const [belief, setBelief] = useState<Belief>(() => tracker.current());
  const [last, setLast] = useState<Check | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checks, setChecks] = useState(0);
  const started = useRef(performance.now());

  useEffect(() => {
    const element = video.current!;
    let stream: MediaStream | null = null;
    if (feed.kind === 'camera') {
      navigator.mediaDevices
        ?.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false })
        .then((s) => {
          stream = s;
          element.srcObject = s;
          return element.play();
        })
        .catch(() => setError('The camera is not available (permission, or no camera on this device).'));
    } else {
      element.src = feed.url;
      // A remount (React's dev double effects) interrupts the first play(); the second one wins.
      element.play().catch(() => undefined);
    }
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [feed]);

  useEffect(() => {
    let cancelled = false;
    let inFlight = false;
    let lastSent = 0;
    const timer = setInterval(async () => {
      const element = video.current;
      if (cancelled || inFlight || !element || element.readyState < 2 || performance.now() - lastSent < CHECK_EVERY_MS) return;
      const image = await grab(element);
      if (!image) return;
      inFlight = true;
      lastSent = performance.now();
      setBusy(true);
      const timeS = feed.kind === 'camera' ? (performance.now() - started.current) / 1000 : element.currentTime;
      try {
        const { readings, observations } = await checkFrames(
          [{ image, timeS }],
          source.kind === 'library' ? { id: source.id } : { plan: source.result.plan },
          code,
        );
        if (cancelled) return;
        setBelief(tracker.update(observations[0]));
        setLast({ image: URL.createObjectURL(image), reading: readings[0], seconds: (performance.now() - lastSent) / 1000 });
        setChecks((n) => n + 1);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        inFlight = false;
        if (!cancelled) setBusy(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [feed, source, code, tracker]);

  useEffect(
    () => () => {
      if (last) URL.revokeObjectURL(last.image);
    },
    [last],
  );

  const steps = plan.steps.length;
  const done = belief.completed >= steps;
  const step = done ? null : plan.steps[belief.completed];
  const previous = belief.completed > 0 ? plan.steps[belief.completed - 1].orientation : 'upright';
  const parts = new Map(plan.parts.map((p) => [p.id, p]));

  return (
    <div className="ar live">
      <video ref={video} className="live-video" muted playsInline loop />
      <div className="ar-vignette" aria-hidden="true" />
      <header className="ar-top">
        <button type="button" className="icon-btn" onClick={onStop} aria-label="Back to setup">
          <ArrowLeft size={17} />
        </button>
        <div className="ar-title">
          <span className="eyebrow done">Live · {feed.kind === 'camera' ? 'camera' : 'video'}</span>
          <strong>{plan.title}</strong>
        </div>
        <div className="ar-status">
          <span className="chip">
            <span className="rec" /> <b>{checks}</b> checks
          </span>
          {last && (
            <span className="chip">
              <b>{last.seconds.toFixed(1)}</b> s per check
            </span>
          )}
          {busy && (
            <span className="chip">
              <Loader2 size={12} className="spin" /> checking
            </span>
          )}
        </div>
      </header>

      <aside className="ar-graph glass">
        <div className="panel-head">
          <span className="eyebrow track">Assembly graph</span>
          <span className="faint mono">
            {belief.completed}/{steps} steps
          </span>
        </div>
        <AssemblyGraph plan={plan} completed={belief.completed} active={done ? null : belief.completed + 1} />
      </aside>

      <aside className="ar-belief glass">
        <div className="panel-head">
          <span className="eyebrow done">Progress belief</span>
          <span className="faint mono">HMM · {Math.round(belief.confidence * 100)}%</span>
        </div>
        <BeliefBars plan={plan} belief={belief} />
        {last && (
          <div className="live-last">
            <img src={last.image} alt="The last frame checked" />
            <div className="reading-letters">
              <span className="faint mono">joined</span>
              <span>
                {last.reading.attached.length
                  ? last.reading.attached.map((l, i) => (
                      <span key={i} className="letter is-on">
                        {l}
                      </span>
                    ))
                  : '—'}
              </span>
              <span className="faint mono">loose</span>
              <span>
                {last.reading.loose.length
                  ? last.reading.loose.map((l, i) => (
                      <span key={i} className="letter">
                        {l}
                      </span>
                    ))
                  : '—'}
              </span>
            </div>
          </div>
        )}
      </aside>

      <section className={`ar-step glass${done ? ' is-done' : ''}`} aria-live="polite">
        <span className="step-number">{done ? '✓' : step ? step.index : '·'}</span>
        {error && <p className="live-error">{error}</p>}
        {done ? (
          <>
            <span className="eyebrow done">Verified</span>
            <h2>{plan.title} is assembled.</h2>
          </>
        ) : step ? (
          <>
            <span className="eyebrow">
              Step {step.index} of {steps}
            </span>
            <h2>{step.title}</h2>
            <p className="muted">
              {step.orientation !== previous && <b className="live-turn">{TURN[step.orientation]}. </b>}
              {step.instruction}
            </p>
            <div className="step-chips">
              {step.parts.map((id) => {
                const part = parts.get(id);
                return (
                  <span key={id} className="chip guide">
                    <span className="letter">{part?.label ?? id}</span> {part?.name}
                  </span>
                );
              })}
              {step.hardware.map((h) => (
                <span key={h.id} className="chip">
                  <b>{h.count}×</b> {hardwareName(plan, h.id)}
                </span>
              ))}
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}

export function Live() {
  const [status, setStatus] = useState<Health | 'down' | null>(null);
  const [code, setCode] = useState(savedAccessCode);
  const [source, setSource] = useState<Source>({ kind: 'library', id: DEFAULT_FURNITURE });
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const [feed, setFeed] = useState<Feed | null>(null);

  useEffect(() => {
    health()
      .then(setStatus)
      .catch(() => setStatus('down'));
  }, []);

  useEffect(() => {
    if (source.kind === 'manual') setPlan(source.result.plan);
    else loadPlan(source.id).then(setPlan);
  }, [source]);

  if (feed && plan) {
    return (
      <Build
        key={JSON.stringify(feed)}
        plan={plan}
        source={source}
        feed={feed}
        code={code}
        onStop={() => {
          if (feed.kind === 'file') URL.revokeObjectURL(feed.url);
          setFeed(null);
        }}
      />
    );
  }
  return (
    <div className="demo">
      <header className="demo-top">
        <button type="button" className="icon-btn" onClick={() => navigate({ page: 'home' })} aria-label="Back">
          <ArrowLeft size={17} />
        </button>
        <div className="ar-title">
          <span className="eyebrow done">Live</span>
          <strong>The copilot on your camera</strong>
        </div>
      </header>
      <Setup status={status} code={code} setCode={setCode} source={source} setSource={setSource} onStart={setFeed} />
    </div>
  );
}
