import {
  ArrowLeft,
  ArrowUp,
  ChartLine,
  CircleAlert,
  Download,
  HandGrab,
  Move,
  Pointer,
  RotateCcw,
  SlidersHorizontal,
  X,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useRef, useState, type FormEvent } from 'react';
import { accessCodeRequired, ApiError, askWhatIf, saveAccessCode, savedAccessCode } from './api';
import { Inspector } from './Inspector';
import { loadPushTests, RealityCheck, type PushTest } from './RealityCheck';
import { parseActions } from './sim/actions';
import { GRAVITY_PRESETS, PHYSICS_HZ, type GravityPreset } from './sim/physics';
import type { SceneObject } from './sim/scene';
import { Simulator, type Telemetry, type Tool, type ViewMode } from './sim/Simulator';

type Status =
  | { state: 'loading' }
  | { state: 'ready'; title: string; objectCount: number; warnings: string[] }
  | { state: 'error'; message: string };

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

const TOOLS: { tool: Tool; label: string; key: string; icon: LucideIcon; hint: string }[] = [
  { tool: 'move', label: 'Move', key: 'M', icon: Move, hint: 'Drag an object to slide or throw it · tap it for details' },
  { tool: 'lift', label: 'Lift', key: 'L', icon: HandGrab, hint: 'Drag to carry an object · Q / E or scroll to turn it · flick to throw' },
  { tool: 'push', label: 'Push', key: 'P', icon: Pointer, hint: 'Tap an object to push it · heavy things barely move' },
];

const VIEW_MODES: { mode: ViewMode; label: string; key: string; hint: string }[] = [
  { mode: 'render', label: 'Scene', key: '1', hint: 'The scan as filmed' },
  { mode: 'debug', label: 'Colliders', key: '2', hint: 'Collision shapes, motion and labels over the scan' },
  { mode: 'physics', label: 'Physics', key: '3', hint: 'Only what the physics engine sees' },
  { mode: 'robot', label: 'Robot', key: '4', hint: 'What a robot stack receives: centers of mass, inertia and grasp candidates, +z up' },
];

const IDEAS = ['Someone bumps the table hard', 'The table tilts 30 degrees', "We're on the Moon", 'Everything drops from half a meter'];

const SHORTCUTS: [string, string][] = [
  ['M  L  P', 'Move, lift, push'],
  ['Q  E', 'Turn what you carry (or scroll)'],
  ['1  2  3  4', 'Scene, colliders, physics, robot view'],
  ['R', 'Reset the scene'],
  ['S', 'Slow motion'],
  ['⌘K', 'Ask "what if…"'],
];


function whatIfProblem(error: unknown): string {
  if (error instanceof ApiError && error.status === 0) return "Couldn't reach the server. Check your connection and try again.";
  if (error instanceof ApiError && error.status === 429) return 'Too many requests right now. Try again in a minute.';
  return "Couldn't work that one out. Try saying it differently.";
}

/** The playable scene: the 3D view with its toolbar, tools, "what if" bar, inspector and settings. */
export function SimulatorView({ sceneDir, onHome, initialView }: { sceneDir: string; onHome: () => void; initialView?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const commandRef = useRef<HTMLInputElement>(null);
  const simulatorRef = useRef<Simulator | null>(null);
  const [status, setStatus] = useState<Status>({ state: 'loading' });
  const [selected, setSelected] = useState<SceneObject | null>(null);
  const [edited, setEdited] = useState<ReadonlySet<string>>(new Set());
  const [fps, setFps] = useState<number | null>(null);
  const [tool, setTool] = useState<Tool>('move');
  const [viewMode, setViewMode] = useState<ViewMode>(() => VIEW_MODES.find((m) => m.mode === initialView)?.mode ?? 'render');
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gravity, setGravity] = useState<GravityPreset>('earth');
  const [slowMotion, setSlowMotion] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [pushTests, setPushTests] = useState<PushTest[] | null>(null);
  const [realityOpen, setRealityOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [commandFocused, setCommandFocused] = useState(false);
  const [whatIf, setWhatIf] = useState<WhatIf>({ state: 'idle' });
  const [code, setCode] = useState('');

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let simulator: Simulator | null = null;
    let cancelled = false;
    setStatus({ state: 'loading' });
    Simulator.create(container, sceneDir, { onSelect: setSelected, onFps: setFps, onTelemetry: setTelemetry }, { viewMode })
      .then((created) => {
        if (cancelled) {
          created.dispose();
          return;
        }
        simulator = created;
        simulatorRef.current = created;
        for (const warning of created.warnings) console.warn(warning);
        const { manifest } = created.scene;
        setStatus({ state: 'ready', title: manifest.title, objectCount: manifest.objects.length, warnings: created.warnings });
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

  // Settings and the inspector share the top-right corner: opening one closes the other.
  useEffect(() => {
    if (selected) setSettingsOpen(false);
  }, [selected]);
  const toggleSettings = () => {
    setSettingsOpen((open) => !open);
    simulatorRef.current?.select(null);
  };

  const closeInspector = useCallback(() => simulatorRef.current?.select(null), []);
  const markEdited = (ids: string[]) => setEdited((current) => new Set([...current, ...ids]));

  const changeMaterial = (material: string) => {
    if (!selected || !simulatorRef.current) return;
    simulatorRef.current.changeMaterial(selected.id, material);
    markEdited([selected.id]);
  };

  const chooseTool = (next: Tool) => {
    setTool(next);
    simulatorRef.current?.setTool(next);
  };

  const chooseViewMode = (mode: ViewMode) => {
    setViewMode(mode);
    simulatorRef.current?.setViewMode(mode);
  };

  const chooseGravity = (preset: GravityPreset) => {
    setGravity(preset);
    simulatorRef.current?.setGravity(preset);
  };

  const chooseSlowMotion = (on: boolean) => {
    setSlowMotion(on);
    simulatorRef.current?.setSlowMotion(on);
  };

  const reset = () => simulatorRef.current?.reset();

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

  const ask = (scenario: string) => {
    setPrompt(scenario);
    commandRef.current?.blur(); // on phones, put the keyboard away so the scene is in view
    void runWhatIf(scenario, savedAccessCode());
  };

  const onAsk = (event: FormEvent) => {
    event.preventDefault();
    ask(prompt);
  };

  const onCode = (event: FormEvent) => {
    event.preventDefault();
    if (code.trim()) void runWhatIf(prompt, code.trim());
  };

  const onKeyDown = useEffectEvent((event: KeyboardEvent) => {
    const typing = event.target instanceof Element && event.target.closest('input, select, textarea') !== null;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      commandRef.current?.focus();
      return;
    }
    if (typing) {
      if (event.key === 'Escape') (event.target as HTMLElement).blur();
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.repeat && !'qQeE'.includes(event.key)) return; // holding Q or E keeps turning
    const toolForKey = TOOLS.find((t) => t.key === event.key.toUpperCase());
    const modeForKey = VIEW_MODES.find((m) => m.key === event.key);
    if (toolForKey) chooseTool(toolForKey.tool);
    else if (modeForKey) chooseViewMode(modeForKey.mode);
    else if (event.key === 'q' || event.key === 'Q') simulatorRef.current?.turnHeld(1);
    else if (event.key === 'e' || event.key === 'E') simulatorRef.current?.turnHeld(-1);
    else if (event.key === 'r' || event.key === 'R') reset();
    else if (event.key === 's' || event.key === 'S') chooseSlowMotion(!slowMotion);
    else if (event.key === '/') {
      event.preventDefault();
      commandRef.current?.focus();
    } else if (event.key === 'Escape') setSettingsOpen(false);
  });

  const ready = status.state === 'ready';
  useEffect(() => {
    if (!ready) return;
    const listener = (event: KeyboardEvent) => onKeyDown(event);
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, [ready]);

  const gravityName = GRAVITY_LABELS.find(([preset]) => preset === gravity)?.[1] ?? gravity;
  const activeTool = TOOLS.find((t) => t.tool === tool)!;
  const showIdeas = commandFocused && !prompt && whatIf.state === 'idle';
  return (
    <div className={`sim${selected || settingsOpen ? ' has-sheet' : ''}`}>
      <div ref={containerRef} className="viewport" />

      <header className="hud-top">
        <div className="hud-title">
          <button type="button" className="icon-btn" onClick={onHome} aria-label="Back to home" title="Back to home">
            <ArrowLeft size={18} aria-hidden="true" />
          </button>
          {ready && (
            <strong className="hud-title-text">{status.title}</strong>
          )}
        </div>
        {ready && (
          <div className="view-modes" role="group" aria-label="View">
            {VIEW_MODES.map(({ mode, label, key, hint }) => (
              <button
                key={mode}
                type="button"
                aria-pressed={viewMode === mode}
                onClick={() => chooseViewMode(mode)}
                title={`${hint} (${key})`}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        {ready && (
          <div className="hud-actions">
            {pushTests && (
              <button
                type="button"
                className="icon-btn labeled"
                onClick={() => {
                  setSettingsOpen(false);
                  setRealityOpen(true);
                }}
                title="Compare with real pushes"
              >
                <ChartLine size={18} aria-hidden="true" />
                <span className="btn-label">Reality check</span>
              </button>
            )}
            <button
              type="button"
              className="icon-btn"
              aria-expanded={settingsOpen}
              aria-label="Physics settings"
              title="Physics settings"
              onClick={toggleSettings}
            >
              <SlidersHorizontal size={18} aria-hidden="true" />
            </button>
            <button type="button" className="icon-btn" onClick={reset} aria-label="Reset the scene" title="Reset (R)">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
          </div>
        )}
      </header>

      {ready && telemetry && (
        <TelemetryReadout
          telemetry={telemetry}
          gravity={gravity}
          slowMotion={slowMotion}
          fps={fps}
          labelOf={labelOf}
        />
      )}

      {ready && settingsOpen && (
        <section className="popover panel" aria-label="Physics settings">
          <div className="popover-head">
            <p className="eyebrow">Physics</p>
            <button type="button" className="icon-btn plain small-btn" onClick={() => setSettingsOpen(false)} aria-label="Close">
              <X size={16} aria-hidden="true" />
            </button>
          </div>
          <div className="popover-section">
            <span className="small muted" id="gravity-label">
              Gravity
            </span>
            <div className="segmented" role="group" aria-labelledby="gravity-label">
              {GRAVITY_LABELS.map(([preset, label]) => (
                <button key={preset} type="button" aria-pressed={gravity === preset} onClick={() => chooseGravity(preset)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          <label className="toggle-row">
            <span>
              Slow motion <span className="kbd">S</span>
            </span>
            <span className="switch">
              <input type="checkbox" checked={slowMotion} onChange={(event) => chooseSlowMotion(event.target.checked)} />
              <span />
            </span>
          </label>
          {exportUrl && (
            <a className="menu-link" href={exportUrl} download>
              <Download size={18} aria-hidden="true" />
              <span>
                Download the robotics bundle
                <small>MuJoCo MJCF, URDF, USD (Isaac Sim), grasps and inertia, +z up (.zip)</small>
              </span>
            </a>
          )}
          <div className="popover-section shortcuts">
            <span className="small muted">Keyboard</span>
            <dl>
              {SHORTCUTS.map(([keys, action]) => (
                <div key={keys}>
                  <dt>
                    {keys.split('  ').map((k) => (
                      <span key={k} className="kbd">
                        {k}
                      </span>
                    ))}
                  </dt>
                  <dd>{action}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>
      )}

      {status.state === 'loading' && (
        <div className="center-message" role="status">
          <span className="loader" aria-hidden="true" />
          <span className="muted">Loading scene…</span>
        </div>
      )}
      {status.state === 'error' && (
        <div className="center-message panel" role="alert">
          <CircleAlert size={22} className="danger" aria-hidden="true" />
          <p>Couldn't open this scene: {status.message}</p>
          <button type="button" className="btn btn-primary" onClick={onHome}>
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
        <div className="dock">
          {whatIf.state === 'idle' &&
            (showIdeas ? (
              <div className="suggestions panel">
                <p className="eyebrow">Try asking</p>
                {IDEAS.map((idea) => (
                  <button
                    key={idea}
                    type="button"
                    className="suggestion"
                    onPointerDown={(event) => event.preventDefault()} // keep the input focused until the click lands
                    onClick={() => ask(idea)}
                  >
                    {idea}
                  </button>
                ))}
              </div>
            ) : (
              <p className="hint">{activeTool.hint}</p>
            ))}
          {whatIf.state === 'asking' && (
            <div className="toast panel" role="status">
              <span className="toast-row">
                <span className="loader loader-sm" aria-hidden="true" />
                Working out what would happen…
              </span>
            </div>
          )}
          {(whatIf.state === 'answer' || whatIf.state === 'error') && (
            <div className={`toast panel${whatIf.state === 'error' ? ' error' : ''}`} role="status">
              <p>{whatIf.state === 'answer' ? whatIf.narration : whatIf.message}</p>
              {whatIf.state === 'answer' && whatIf.skipped > 0 && (
                <p className="faint small">
                  {whatIf.skipped} step{whatIf.skipped > 1 ? 's' : ''} couldn't be simulated and {whatIf.skipped > 1 ? 'were' : 'was'} skipped.
                </p>
              )}
              <button type="button" className="icon-btn plain" aria-label="Dismiss" onClick={() => setWhatIf({ state: 'idle' })}>
                <X size={16} aria-hidden="true" />
              </button>
            </div>
          )}
          {whatIf.state === 'code' && (
            <div className="toast panel">
              <form onSubmit={onCode}>
                <label htmlFor="whatif-code">
                  {whatIf.wrongCode ? "That code didn't work. Check it and try again." : '"What if" needs the access code you were given.'}
                </label>
                <div className="code-row">
                  <input
                    id="whatif-code"
                    className="input"
                    value={code}
                    onChange={(event) => setCode(event.target.value)}
                    autoComplete="off"
                    autoCapitalize="off"
                    spellCheck={false}
                    autoFocus
                  />
                  <button type="submit" className="btn btn-primary" disabled={!code.trim()}>
                    Go
                  </button>
                </div>
              </form>
              <button type="button" className="icon-btn plain" aria-label="Dismiss" onClick={() => setWhatIf({ state: 'idle' })}>
                <X size={16} aria-hidden="true" />
              </button>
            </div>
          )}

          <div className="dock-bar panel">
            <div className="segmented tools" role="group" aria-label="Tool">
              {TOOLS.map(({ tool: t, label, key, icon: Icon }) => (
                <button key={t} type="button" aria-pressed={tool === t} onClick={() => chooseTool(t)} title={`${label} (${key})`}>
                  <Icon size={16} aria-hidden="true" />
                  <span className="tool-label">{label}</span>
                </button>
              ))}
            </div>
            <form className="command" onSubmit={onAsk}>
              <label htmlFor="whatif-input" className="visually-hidden">
                What if…
              </label>
              <input
                ref={commandRef}
                id="whatif-input"
                placeholder="What if… someone bumps the table"
                value={prompt}
                onChange={(event) => {
                  setPrompt(event.target.value);
                  if (whatIf.state === 'answer' || whatIf.state === 'error') setWhatIf({ state: 'idle' });
                }}
                onFocus={() => setCommandFocused(true)}
                onBlur={() => setCommandFocused(false)}
                enterKeyHint="go"
                autoComplete="off"
              />
              {!prompt && <span className="kbd">⌘K</span>}
              <button
                type="submit"
                className="send"
                aria-label="Ask"
                disabled={!prompt.trim() || whatIf.state === 'asking'}
              >
                <ArrowUp size={16} strokeWidth={2.25} aria-hidden="true" />
              </button>
            </form>
          </div>
        </div>
      )}

      {selected && <Inspector object={selected} edited={edited.has(selected.id)} onMaterial={changeMaterial} onClose={closeInspector} />}

      {realityOpen && pushTests && (
        <RealityCheck tests={pushTests} sceneDir={sceneDir} labelOf={labelOf} onClose={() => setRealityOpen(false)} />
      )}
    </div>
  );
}

const fixed = (value: number, digits: number) => value.toFixed(digits);

/** One quiet line of live numbers: the simulation clock, what is moving, and the selected body. */
function TelemetryReadout({
  telemetry,
  gravity,
  slowMotion,
  fps,
  labelOf,
}: {
  telemetry: Telemetry;
  gravity: GravityPreset;
  slowMotion: boolean;
  fps: number | null;
  labelOf: (objectId: string) => string;
}) {
  const { time, steps, energy, moving, selected } = telemetry;
  const parts = [
    `t ${fixed(time, 1)} s`,
    `${steps.toLocaleString('en-US')} steps at ${PHYSICS_HZ} Hz`,
    `g ${fixed(GRAVITY_PRESETS[gravity], 2)}`,
    `${moving} moving`,
    `${energy < 0.0005 ? '0.000' : fixed(energy, 3)} J`,
  ];
  if (slowMotion) parts.push('0.25×');
  if (fps !== null) parts.push(fps > 0 ? `${fps} fps` : 'idle: nothing to redraw');
  return (
    <p className="readout" aria-label="Simulation telemetry">
      {parts.join('\n')}
      {selected && (
        <span>
          {labelOf(selected.id)}: {fixed(selected.speed, 2)} m/s
        </span>
      )}
    </p>
  );
}
