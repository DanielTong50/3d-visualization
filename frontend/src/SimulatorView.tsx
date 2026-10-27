import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { accessCodeRequired, ApiError, askWhatIf, saveAccessCode, savedAccessCode } from './api';
import { InfoCard } from './InfoCard';
import { loadPushTests, RealityCheck, type PushTest } from './RealityCheck';
import { parseActions } from './sim/actions';
import type { GravityPreset } from './sim/physics';
import type { SceneObject } from './sim/scene';
import { Simulator } from './sim/Simulator';

type Status = { state: 'loading' } | { state: 'ready'; title: string; warnings: string[] } | { state: 'error'; message: string };

type WhatIf =
  | { state: 'idle' }
  | { state: 'asking' }
  | { state: 'answer'; narration: string; skipped: number }
  | { state: 'error'; message: string }
  | { state: 'code'; wrongCode: boolean };

const GRAVITY_LABELS: [GravityPreset, string][] = [
  ['earth', 'Earth'],
  ['moon', 'Moon'],
  ['mars', 'Mars'],
  ['jupiter', 'Jupiter'],
  ['zero', 'Zero-g'],
];

const IDEAS = ['Someone bumps the table hard', 'The table tilts 30 degrees', "We're on the Moon", 'Everything drops from half a meter'];

function whatIfProblem(error: unknown): string {
  if (error instanceof ApiError && error.status === 0) return "Couldn't reach the server. Check your connection and try again.";
  if (error instanceof ApiError && error.status === 429) return 'Too many requests right now. Try again in a minute.';
  return "Couldn't work that one out. Try saying it differently.";
}

/** The playable scene: the 3D view plus its buttons, hint, info card, physics panel and "what if" box. */
export function SimulatorView({ sceneDir, onHome }: { sceneDir: string; onHome: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const simulatorRef = useRef<Simulator | null>(null);
  const [status, setStatus] = useState<Status>({ state: 'loading' });
  const [selected, setSelected] = useState<SceneObject | null>(null);
  const [edited, setEdited] = useState<ReadonlySet<string>>(new Set());
  const [fps, setFps] = useState<number | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [gravity, setGravity] = useState<GravityPreset>('earth');
  const [slowMotion, setSlowMotion] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [pushTests, setPushTests] = useState<PushTest[] | null>(null);
  const [realityOpen, setRealityOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [whatIf, setWhatIf] = useState<WhatIf>({ state: 'idle' });
  const [code, setCode] = useState('');

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let simulator: Simulator | null = null;
    let cancelled = false;
    setStatus({ state: 'loading' });
    Simulator.create(container, sceneDir, { onSelect: setSelected, onFps: setFps })
      .then((created) => {
        if (cancelled) {
          created.dispose();
          return;
        }
        simulator = created;
        simulatorRef.current = created;
        for (const warning of created.warnings) console.warn(warning);
        setStatus({ state: 'ready', title: created.scene.manifest.title, warnings: created.warnings });
      })
      .catch((error: unknown) => {
        if (!cancelled) setStatus({ state: 'error', message: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      cancelled = true;
      simulator?.dispose();
      simulatorRef.current = null;
    };
  }, [sceneDir]);

  // Offer the download only if the scene package has one (older scans don't), and the reality
  // check only if real pushes were filmed for this scene.
  useEffect(() => {
    let cancelled = false;
    const url = `${sceneDir}/export.zip`;
    fetch(url)
      .then((response) => !cancelled && setExportUrl(response.ok ? url : null))
      .catch(() => !cancelled && setExportUrl(null));
    loadPushTests(sceneDir).then((tests) => !cancelled && setPushTests(tests));
    return () => {
      cancelled = true;
    };
  }, [sceneDir]);

  const labelOf = (objectId: string) =>
    simulatorRef.current?.objectSummaries().find((o) => o.id === objectId)?.label ?? objectId;

  // The physics panel and the info card share the top-right corner: opening one closes the other.
  useEffect(() => {
    if (selected) setPanelOpen(false);
  }, [selected]);
  const togglePanel = () => {
    setPanelOpen((open) => {
      if (!open) simulatorRef.current?.select(null);
      return !open;
    });
  };

  const closeCard = useCallback(() => simulatorRef.current?.select(null), []);
  const markEdited = (ids: string[]) => setEdited((current) => new Set([...current, ...ids]));

  const changeMaterial = (material: string) => {
    if (!selected || !simulatorRef.current) return;
    simulatorRef.current.changeMaterial(selected.id, material);
    markEdited([selected.id]);
  };

  const chooseGravity = (preset: GravityPreset) => {
    setGravity(preset);
    simulatorRef.current?.setGravity(preset);
  };

  const toggleSlowMotion = (on: boolean) => {
    setSlowMotion(on);
    simulatorRef.current?.setSlowMotion(on);
  };

  const runWhatIf = async (scenario: string, accessCode: string | null) => {
    const simulator = simulatorRef.current;
    if (!simulator || !scenario.trim()) return;
    if (!accessCode && (await accessCodeRequired())) {
      setWhatIf({ state: 'code', wrongCode: false });
      return;
    }
    setWhatIf({ state: 'asking' });
    try {
      const objects = simulator.objectSummaries();
      const answer = await askWhatIf(scenario.trim(), objects, accessCode);
      if (accessCode) saveAccessCode(accessCode);
      const { actions, rejected } = parseActions(answer.actions, new Set(objects.map((o) => o.id)));
      for (const reason of rejected) console.warn('Skipped a "what if" action:', reason);
      simulator.runActions(actions);
      markEdited(actions.flatMap((a) => (a.name === 'set_property' || a.name === 'set_material' ? [a.objectId] : [])));
      const newGravity = actions.find((a) => a.name === 'set_gravity');
      if (newGravity) setGravity(newGravity.preset);
      const narration = answer.narration || (actions.length > 0 ? 'Here goes!' : "I couldn't turn that into something to simulate.");
      setWhatIf({ state: 'answer', narration, skipped: rejected.length });
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        saveAccessCode(null);
        setWhatIf({ state: 'code', wrongCode: accessCode !== null });
        return;
      }
      setWhatIf({ state: 'error', message: whatIfProblem(error) });
    }
  };

  const onAsk = (event: FormEvent) => {
    event.preventDefault();
    void runWhatIf(prompt, savedAccessCode());
  };

  const onCode = (event: FormEvent) => {
    event.preventDefault();
    if (code.trim()) void runWhatIf(prompt, code.trim());
  };

  const ready = status.state === 'ready';
  return (
    <div className={`app${selected ? ' has-card' : ''}`}>
      <div ref={containerRef} className="viewport" />

      <header className="top-bar">
        <button type="button" className="brand" onClick={onHome} aria-label="Back to home">
          <strong>
            <span aria-hidden="true">‹ </span>Scan2Sim
          </strong>
          {ready && <span className="scene-title">{status.title}</span>}
        </button>
        <div className="top-actions">
          {fps !== null && <span className="fps">{fps} fps</span>}
          {ready && (
            <>
              <button
                type="button"
                className="button button-secondary"
                aria-expanded={panelOpen}
                onClick={togglePanel}
              >
                Physics
              </button>
              <button type="button" className="button" onClick={() => simulatorRef.current?.reset()}>
                Reset
              </button>
            </>
          )}
        </div>
      </header>

      {ready && panelOpen && (
        <section className="physics-panel" aria-label="Physics settings">
          <fieldset className="segmented">
            <legend>Gravity</legend>
            {GRAVITY_LABELS.map(([preset, label]) => (
              <button key={preset} type="button" aria-pressed={gravity === preset} onClick={() => chooseGravity(preset)}>
                {label}
              </button>
            ))}
          </fieldset>
          <label className="switch">
            <input type="checkbox" checked={slowMotion} onChange={(event) => toggleSlowMotion(event.target.checked)} />
            Slow motion
          </label>
          {pushTests && (
            <button
              type="button"
              className="button reality-button"
              onClick={() => {
                setPanelOpen(false);
                setRealityOpen(true);
              }}
            >
              Reality check: {pushTests.length} real push{pushTests.length > 1 ? 'es' : ''}
            </button>
          )}
          {exportUrl && (
            <p className="download">
              <a href={exportUrl} download>
                Download this scene (.zip)
              </a>
              <span className="muted small"> Physics, MuJoCo model and collision meshes, for robot simulators.</span>
            </p>
          )}
        </section>
      )}

      {status.state === 'loading' && <p className="center-message">Loading scene…</p>}
      {status.state === 'error' && (
        <div className="center-message error">
          <p>Couldn't open this scene: {status.message}</p>
          <button type="button" className="button" onClick={onHome}>
            Back to home
          </button>
        </div>
      )}
      {ready && status.warnings.length > 0 && (
        <p className="warning-note">
          {status.warnings.length} part{status.warnings.length > 1 ? 's' : ''} of this scene couldn't be shown as scanned.
        </p>
      )}

      {ready && (
        <div className="bottom-bar">
          {whatIf.state === 'idle' && !selected && (
            <p className="hint">
              <span className="hint-long">Drag an object to move it, flick to throw · Tap it for details · Drag the background to look around</span>
              <span className="hint-short">Drag to move or throw · Tap for details</span>
            </p>
          )}
          {whatIf.state === 'asking' && <p className="bubble">Working out what would happen…</p>}
          {(whatIf.state === 'answer' || whatIf.state === 'error') && (
            <div className={`bubble${whatIf.state === 'error' ? ' bubble-error' : ''}`} role="status">
              <p>{whatIf.state === 'answer' ? whatIf.narration : whatIf.message}</p>
              {whatIf.state === 'answer' && whatIf.skipped > 0 && (
                <p className="muted small">
                  {whatIf.skipped} step{whatIf.skipped > 1 ? 's' : ''} couldn't be simulated and {whatIf.skipped > 1 ? 'were' : 'was'} skipped.
                </p>
              )}
              <button type="button" className="icon-button" aria-label="Dismiss" onClick={() => setWhatIf({ state: 'idle' })}>
                ×
              </button>
            </div>
          )}
          {whatIf.state === 'code' && (
            <form className="bubble code-form" onSubmit={onCode}>
              <label htmlFor="whatif-code">
                {whatIf.wrongCode ? "That code didn't work. Check it and try again." : '"What if" needs the access code you were given.'}
              </label>
              <div className="code-row">
                <input
                  id="whatif-code"
                  className="text-input"
                  value={code}
                  onChange={(event) => setCode(event.target.value)}
                  autoComplete="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  autoFocus
                />
                <button type="submit" className="button" disabled={!code.trim()}>
                  Go
                </button>
              </div>
            </form>
          )}
          <form className="whatif" onSubmit={onAsk}>
            <label htmlFor="whatif-input" className="visually-hidden">
              What if…
            </label>
            <input
              id="whatif-input"
              className="text-input"
              list="whatif-ideas"
              placeholder="What if… (e.g. someone bumps the table)"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              enterKeyHint="go"
            />
            <datalist id="whatif-ideas">
              {IDEAS.map((idea) => (
                <option key={idea} value={idea} />
              ))}
            </datalist>
            <button type="submit" className="button" disabled={!prompt.trim() || whatIf.state === 'asking'}>
              Try it
            </button>
          </form>
        </div>
      )}

      {selected && <InfoCard object={selected} edited={edited.has(selected.id)} onMaterial={changeMaterial} onClose={closeCard} />}

      {realityOpen && pushTests && (
        <RealityCheck tests={pushTests} sceneDir={sceneDir} labelOf={labelOf} onClose={() => setRealityOpen(false)} />
      )}
    </div>
  );
}
