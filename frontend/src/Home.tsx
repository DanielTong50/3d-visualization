import { useEffect, useState, type ChangeEvent, type FormEvent } from 'react';
import { accessCodeRequired, savedAccessCode } from './api';
import { navigate, SCENES_URL } from './navigation';

interface DemoScene {
  id: string;
  title: string;
  objectCount: number;
  description: string | null;
  credit: string | null;
  /** Thumbnail URL, or null to draw simple shapes. */
  thumbnail: string | null;
}

// Used only if /scenes/index.json (written by the pipeline's gallery step) can't be read.
const FALLBACK_DEMO_IDS = ['objectron-mugs', 'sandbox'];
const MAX_VIDEO_BYTES = 500 * 1024 * 1024;
const SAFE_ID = /^[\w-]+$/;
const SAFE_FILE = /^[\w.-]+$/;

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
              objectCount: typeof entry.object_count === 'number' ? entry.object_count : 0,
              description: text(entry.description),
              credit: text(entry.credit),
              thumbnail: thumbnail && SAFE_FILE.test(thumbnail) ? `${SCENES_URL}/${id}/${thumbnail}` : null,
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
          objectCount: scene.objects?.length ?? 0,
          description: null,
          credit: null,
          thumbnail: `${SCENES_URL}/${id}/thumbnail.jpg`,
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

interface HomeProps {
  onStart: (file: File, accessCode: string | null) => void;
  /** Set when the server asked for an access code (again, if `wrongCode`). */
  needsCode: { file: File; wrongCode: boolean } | null;
  onCancelCode: () => void;
}

/** Start screen: scan a tabletop, filming tips, and demo scenes to try. */
export function Home({ onStart, needsCode, onCancelCode }: HomeProps) {
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

  return (
    <main className="page">
      <div className="home">
        <section className="hero">
          <h1>Scan2Sim</h1>
          <p className="lede">
            Film your table with your phone. Get a 3D copy you can play with, where every object knows what it is and how
            heavy it is.
          </p>
          {codeFile ? (
            <form className="code-form" onSubmit={onSubmitCode}>
              <label htmlFor="access-code">
                {needsCode?.wrongCode ? "That code didn't work. Check it and try again." : 'Enter the access code you were given.'}
              </label>
              <div className="code-row">
                <input
                  id="access-code"
                  className="text-input"
                  value={code}
                  onChange={(event) => setCode(event.target.value)}
                  autoComplete="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  autoFocus
                />
                <button type="submit" className="button" disabled={!code.trim()}>
                  Start
                </button>
              </div>
              <button type="button" className="link-button" onClick={cancelCode}>
                Cancel
              </button>
            </form>
          ) : (
            <>
              <label className={`button button-large${checking ? ' is-busy' : ''}`}>
                <input type="file" accept="video/*,.mp4,.mov,.m4v" className="visually-hidden" onChange={onPick} disabled={checking} />
                {checking ? 'Checking video…' : 'Scan a tabletop'}
              </label>
              <p className="muted small">Choose a 30–60 second video of your table.</p>
            </>
          )}
          {problem && (
            <p className="form-error" role="alert">
              {problem}
            </p>
          )}
        </section>

        <section className="tips" aria-labelledby="tips-title">
          <h2 id="tips-title">How to film</h2>
          <FilmingDiagram />
          <ul>
            <li>Bright, even light. No harsh shadows or direct sun.</li>
            <li>Walk slowly in a full circle around the table, then a second, higher circle.</li>
            <li>Keep the whole table in view, with 5–8 objects a few centimeters apart.</li>
            <li>Avoid glass, mirrors and shiny metal. A credit card lying flat helps with size.</li>
          </ul>
        </section>

        <section aria-labelledby="demos-title">
          <h2 id="demos-title">Or try a demo scene</h2>
          {demos === null ? (
            <p className="muted">Loading demo scenes…</p>
          ) : demos.length === 0 ? (
            <p className="muted">No demo scenes are available right now.</p>
          ) : (
            <ul className="scene-grid">
              {demos.map((demo) => (
                <li key={demo.id}>
                  <button type="button" className="scene-card" onClick={() => navigate({ page: 'scene', sceneId: demo.id })}>
                    <Thumbnail src={demo.thumbnail} />
                    <span className="scene-card-title">{demo.title}</span>
                    {demo.description && <span className="small">{demo.description}</span>}
                    <span className="muted small">
                      {demo.objectCount} object{demo.objectCount === 1 ? '' : 's'}
                      {demo.credit ? ` · ${demo.credit}` : ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  );
}

/** The scene's thumbnail, or simple shapes for scenes without one. */
function Thumbnail({ src }: { src: string | null }) {
  const [missing, setMissing] = useState(false);
  if (!src || missing) {
    return (
      <svg className="scene-thumb" viewBox="0 0 160 100" aria-hidden="true">
        <rect width="160" height="100" fill="#d9dee4" />
        <rect x="16" y="58" width="128" height="10" rx="2" fill="#a8774f" />
        <rect x="30" y="40" width="34" height="18" rx="2" fill="#2f5d8a" />
        <rect x="74" y="34" width="12" height="24" rx="2" fill="#e4572e" />
        <circle cx="104" cy="50" r="8" fill="#f3c623" />
        <rect x="118" y="44" width="22" height="14" rx="2" fill="#6cc3b9" />
      </svg>
    );
  }
  return <img className="scene-thumb" src={src} alt="" loading="lazy" onError={() => setMissing(true)} />;
}

/** Top view of a table with the phone's path circling it. */
function FilmingDiagram() {
  return (
    <svg className="filming-diagram" viewBox="0 0 220 140" role="img" aria-label="Walk in a circle around the table, pointing the phone at its center">
      <ellipse cx="110" cy="70" rx="98" ry="58" fill="none" stroke="#5f6b7a" strokeWidth="2" strokeDasharray="6 6" />
      <path d="M 196 52 l 6 10 l -11 1" fill="none" stroke="#5f6b7a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="62" y="40" width="96" height="60" rx="6" fill="#a8774f" />
      <circle cx="84" cy="60" r="8" fill="#e4572e" />
      <rect x="100" y="54" width="22" height="14" rx="2" fill="#2f5d8a" />
      <circle cx="138" cy="80" r="7" fill="#f3c623" />
      <rect x="86" y="78" width="16" height="11" rx="2" fill="#6cc3b9" />
      <g transform="translate(110 4)">
        <rect x="-9" y="0" width="18" height="26" rx="3" fill="#1d2430" />
        <rect x="-6" y="3" width="12" height="17" rx="1" fill="#e7eaee" />
      </g>
      <path d="M 110 32 L 110 38" stroke="#1d2430" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
