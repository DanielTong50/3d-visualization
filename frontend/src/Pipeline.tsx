import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { StageState } from './api';
import {
  STAGE_GRAPH,
  type ArtifactSource,
  type ImageArtifact,
  type PipelineSummary,
  type PointCloudData,
  type StageName,
} from './replay';

// three.js is only needed once a point cloud or the scene graph is on screen.
const PointCloudView = lazy(() => import('./PointCloudView').then((module) => ({ default: module.PointCloudView })));
const SceneGraphView = lazy(() => import('./SceneGraphView').then((module) => ({ default: module.SceneGraphView })));

const AUTOPLAY_MS = 5200;

/** Where each stage sits in the drawn graph (percent of the graph's box): parallel stages share a column. */
const LAYOUT: Record<StageName, { x: number; y: number }> = {
  ingest: { x: 9, y: 50 },
  reconstruct: { x: 29, y: 24 },
  segment: { x: 29, y: 76 },
  lift: { x: 50, y: 50 },
  physics: { x: 71, y: 24 },
  splat: { x: 71, y: 76 },
  package: { x: 91, y: 50 },
};

const fmt = (value: number | null | undefined, digits = 1, unit = '') =>
  typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${unit}` : '–';
const count = (value: number | undefined) => (typeof value === 'number' ? value.toLocaleString('en-US') : '–');
const grams = (kg: number | undefined) => (typeof kg !== 'number' ? '–' : kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`);

interface PipelineProps {
  source: ArtifactSource;
  /** A running job's stage states; without them every stage is shown as done (a finished scene). */
  states?: Partial<Record<StageName, StageState>>;
  seconds?: Partial<Record<StageName, number | null>>;
  /** Step through the stages on a timer until someone picks one. */
  autoplay?: boolean;
}

/**
 * The compiler as a graph of seven stages, with what each one actually produced for this scene:
 * the keyframes it picked, VGGT's point cloud and camera path, SAM's masks, the separated
 * objects, the physical estimates, the held-out render and the robotics bundle.
 */
export function Pipeline({ source, states, seconds, autoplay = false }: PipelineProps) {
  const [summary, setSummary] = useState<PipelineSummary>({});
  const [cloud, setCloud] = useState<PointCloudData | null>(null);
  const [picked, setPicked] = useState<StageName | null>(null);
  const [auto, setAuto] = useState<StageName>('ingest');
  const rootRef = useRef<HTMLDivElement>(null);

  const stateOf = (name: StageName): StageState => states?.[name] ?? (states ? 'pending' : 'done');
  // A running job follows its newest finished (or running) stage until someone picks one.
  const newest = states ? ([...STAGE_GRAPH].reverse().find((s) => stateOf(s.name) !== 'pending')?.name ?? 'ingest') : null;
  const active: StageName = picked ?? newest ?? auto;

  useEffect(() => {
    let cancelled = false;
    source.summary().then((loaded) => !cancelled && setSummary(loaded));
    return () => {
      cancelled = true;
    };
  }, [source]);

  const wantsCloud = active === 'reconstruct' || active === 'lift';
  useEffect(() => {
    if (!wantsCloud) return;
    let cancelled = false;
    source.points().then((loaded) => !cancelled && loaded && setCloud(loaded));
    return () => {
      cancelled = true;
    };
  }, [source, wantsCloud]);

  const sourceKey = source.version.split(':')[0]; // the scene or job, not its progress
  useEffect(() => {
    setPicked(null);
    setCloud(null);
  }, [sourceKey]);

  useEffect(() => {
    if (!autoplay || picked || states) return;
    let visible = true;
    const observer = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
    if (rootRef.current) observer.observe(rootRef.current);
    const timer = window.setInterval(() => {
      if (!visible || document.hidden) return;
      setAuto((current) => STAGE_GRAPH[(STAGE_GRAPH.findIndex((s) => s.name === current) + 1) % STAGE_GRAPH.length].name);
    }, AUTOPLAY_MS);
    return () => {
      observer.disconnect();
      window.clearInterval(timer);
    };
  }, [autoplay, picked, states]);

  const timings = useMemo(() => {
    const result: Partial<Record<StageName, number | null>> = { ...seconds };
    for (const { stage, seconds: s } of summary.timings ?? []) if (result[stage as StageName] == null) result[stage as StageName] = s;
    return result;
  }, [summary, seconds]);

  const index = STAGE_GRAPH.findIndex((s) => s.name === active);
  const stage = STAGE_GRAPH[index];
  return (
    <div className="pipe" ref={rootRef}>
      <div className="pipe-graph">
        <svg className="pipe-wires" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {STAGE_GRAPH.flatMap(({ name, needs }) =>
            needs.map((need) => {
              const a = LAYOUT[need];
              const b = LAYOUT[name];
              const lit = stateOf(need) === 'done' && stateOf(name) !== 'pending';
              const mid = (a.x + b.x) / 2;
              return (
                <path
                  key={`${need}-${name}`}
                  d={`M ${a.x} ${a.y} C ${mid} ${a.y}, ${mid} ${b.y}, ${b.x} ${b.y}`}
                  className={lit ? 'is-lit' : undefined}
                  vectorEffect="non-scaling-stroke"
                />
              );
            }),
          )}
        </svg>
        <ol>
          {STAGE_GRAPH.map(({ name, title, model }, i) => (
            <li key={name} style={{ left: `${LAYOUT[name].x}%`, top: `${LAYOUT[name].y}%` }}>
              <button
                type="button"
                className={`pipe-node is-${stateOf(name)}${name === active ? ' is-active' : ''}`}
                aria-pressed={name === active}
                onClick={() => setPicked(name)}
              >
                <span className="pipe-step">{String(i + 1).padStart(2, '0')}</span>
                <span className="pipe-title">{title}</span>
                <span className="pipe-model">{model}</span>
                <span className="pipe-time">
                  {stateOf(name) === 'running' ? 'running…' : stateOf(name) === 'failed' ? 'failed' : fmt(timings[name], 0, ' s')}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>
      <div className="pipe-detail" aria-live="polite">
        <div className="pipe-media">
          <StageMedia name={active} source={source} summary={summary} cloud={cloud} ready={stateOf(active) === 'done'} />
        </div>
        <div className="pipe-facts">
          <p className="pipe-kicker">
            {String(index + 1).padStart(2, '0')} · {stage.model}
          </p>
          <h3>{stage.title}</h3>
          <StageFacts name={active} summary={summary} source={source} />
        </div>
      </div>
    </div>
  );
}

function Picture({ source, name, alt }: { source: ArtifactSource; name: ImageArtifact; alt: string }) {
  const [failed, setFailed] = useState(false);
  const url = source.image(name);
  useEffect(() => setFailed(false), [url]);
  if (failed) return <Waiting text="Not produced yet" />;
  return <img src={url} alt={alt} loading="lazy" onError={() => setFailed(true)} />;
}

function Waiting({ text }: { text: string }) {
  return (
    <div className="pipe-waiting">
      <span className="loader loader-sm" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

function StageMedia({
  name,
  source,
  summary,
  cloud,
  ready,
}: {
  name: StageName;
  source: ArtifactSource;
  summary: PipelineSummary;
  cloud: PointCloudData | null;
  ready: boolean;
}) {
  if (!ready) return <Waiting text="Waiting for this stage" />;
  switch (name) {
    case 'ingest':
      return <Picture source={source} name="keyframes" alt="The keyframes picked from the video" />;
    case 'segment':
      return <Picture source={source} name="segmentation" alt="Each object's mask on the keyframe" />;
    case 'splat':
      return <Picture source={source} name="splat_test_view" alt="A held-out frame (left) and the splats rendered from its camera (right)" />;
    case 'reconstruct':
    case 'lift':
      return cloud ? (
        <Suspense fallback={<Waiting text="Loading the point cloud" />}>
          <PointCloudView
            data={cloud}
            colorBy={name === 'lift' ? 'label' : 'rgb'}
            cameras={summary.cameras}
            label={name === 'lift' ? 'The point cloud split into objects' : 'The reconstructed point cloud and camera path'}
          />
        </Suspense>
      ) : (
        <Waiting text="Loading the point cloud" />
      );
    case 'physics':
      return <PhysicsTable summary={summary} />;
    case 'package':
      return source.sceneDir ? (
        <Suspense fallback={<Waiting text="Loading the scene graph" />}>
          <SceneGraphView sceneDir={source.sceneDir} />
        </Suspense>
      ) : (
        <Bundle summary={summary} />
      );
  }
}

function PhysicsTable({ summary }: { summary: PipelineSummary }) {
  const objects = summary.physics?.objects ?? [];
  if (!objects.length) return <Waiting text="Estimating" />;
  return (
    <div className="pipe-table-wrap">
      <table className="pipe-table">
        <thead>
          <tr>
            <th>Object</th>
            <th>Material</th>
            <th>Size (cm)</th>
            <th>Mass</th>
            <th>μ</th>
            <th>e</th>
          </tr>
        </thead>
        <tbody>
          {objects.map((o) => (
            <tr key={o.index}>
              <td>{o.label}</td>
              <td>{o.material ?? '–'}</td>
              <td>{o.measured_dimensions_cm?.map((d) => d.toFixed(1)).join(' × ') ?? '–'}</td>
              <td>{grams(o.mass_kg)}</td>
              <td>{fmt(o.friction, 2)}</td>
              <td>{fmt(o.restitution, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Bundle({ summary }: { summary: PipelineSummary }) {
  const colliders = Object.entries(summary.package?.colliders ?? {});
  const files: [string, string][] = [
    ['scene.json', 'manifest: objects, physics, inertia, grasps'],
    ['*.spz', 'Gaussian splats per object, the table and the room'],
    ['robotics/scene.xml', 'MuJoCo MJCF'],
    ['robotics/scene.usda', 'OpenUSD + UsdPhysics (Isaac Sim)'],
    ['robotics/urdf/*.urdf', 'one URDF per object'],
    ['robotics/world.json', 'poses, mass properties, grasps, cameras, +z up'],
  ];
  return (
    <div className="pipe-bundle">
      <pre>
        {files.map(([file, what]) => (
          <span key={file}>
            <b>{file.padEnd(22)}</b>
            {what}
            {'\n'}
          </span>
        ))}
        {colliders.map(([id, c]) => (
          <span key={id}>
            <b>{`${id} collider`.padEnd(22)}</b>
            {c.used_parts ? `${c.parts} convex parts (hull ${Math.round(c.hull_coverage * 100)}% full)` : 'one convex hull'}
            {'\n'}
          </span>
        ))}
      </pre>
    </div>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

function StageFacts({ name, summary, source }: { name: StageName; summary: PipelineSummary; source: ArtifactSource }) {
  const { frames, reconstruct, segment, lift, physics, splat } = summary;
  switch (name) {
    case 'ingest':
      return (
        <>
          <p>The video is cut into time bins and the sharpest frame of each wins, so the reconstruction never sees motion blur.</p>
          <dl>
            <Fact label="Keyframes">{count(frames?.count)}</Fact>
            <Fact label="Video">{fmt(frames?.duration_s, 1, ' s')}</Fact>
            <Fact label="Resolution">{frames?.size?.join(' × ') ?? '–'}</Fact>
          </dl>
        </>
      );
    case 'reconstruct':
      return (
        <>
          <p>One forward pass of VGGT predicts every camera and a depth map per frame: no feature matching, no bundle adjustment.</p>
          <dl>
            <Fact label="Inference">{fmt(reconstruct?.inference_seconds, 2, ' s')}</Fact>
            <Fact label="GPU">{reconstruct?.gpu?.replace('NVIDIA ', '') ?? '–'}</Fact>
            <Fact label="Peak memory">{fmt(reconstruct?.peak_memory_gb, 1, ' GB')}</Fact>
            <Fact label="Working size">{reconstruct?.resolution?.join(' × ') ?? '–'}</Fact>
          </dl>
        </>
      );
    case 'segment':
      return (
        <>
          <p>Muse Spark names each object and boxes it; SAM 3.1 tracks those boxes through every keyframe.</p>
          <dl>
            {(segment?.objects ?? []).map((o) => (
              <Fact key={o.index} label={o.label}>
                {o.frames_tracked ?? '–'} frames{o.mean_confidence ? ` · ${Math.round(o.mean_confidence * 100)}%` : ''}
              </Fact>
            ))}
          </dl>
        </>
      );
    case 'lift':
      return (
        <>
          <p>
            The table plane defines gravity. Every point carries its pixel's label, and a vote per voxel across all views
            separates the objects.
          </p>
          <dl>
            <Fact label="Points">{count(lift?.points_after_confidence)}</Fact>
            <Fact label="Objects">{lift?.objects?.length ?? '–'}</Fact>
            <Fact label="Table fit">{fmt((lift?.table_inlier_rms ?? NaN) * 1000, 1, ' mm RMS')}</Fact>
            <Fact label="Cameras above">{fmt((lift?.cameras_above_table ?? NaN) * 100, 0, '%')}</Fact>
          </dl>
        </>
      );
    case 'physics':
      return (
        <>
          <p>
            Recognized objects vote on the metric scale. Mass is hull volume × material density × fill fraction, not a guess.
          </p>
          <dl>
            <Fact label="Scale">{physics?.scale_m_per_unit ? `${(physics.scale_m_per_unit * 100).toFixed(1)} cm per unit` : '–'}</Fact>
            <Fact label="Votes">{physics?.scale_votes?.length ?? '–'}</Fact>
          </dl>
        </>
      );
    case 'splat':
      return (
        <>
          <p>3D Gaussian splatting with every Gaussian keeping its object's label, so each object's splats move with its body.</p>
          <dl>
            <Fact label="Held-out PSNR">{fmt(splat?.test_psnr, 1, ' dB')}</Fact>
            <Fact label="Gaussians">{count(splat?.gaussians)}</Fact>
            <Fact label="Training">{fmt(splat?.train_seconds, 0, ' s')}</Fact>
            <Fact label="Method">{splat?.strategy === 'mcmc' ? 'MCMC, anti-aliased' : (splat?.strategy ?? '–')}</Fact>
          </dl>
        </>
      );
    case 'package':
      return (
        <>
          <p>
            One versioned scene package, plus a robotics bundle: MJCF, URDF and USD, mass properties, grasp candidates and the
            3D scene graph shown here.
          </p>
          <Bundle summary={summary} />
          {source.exportUrl && (
            <a className="text-link" href={source.exportUrl} download>
              Download the robotics bundle (.zip)
            </a>
          )}
        </>
      );
  }
}
