import { API_URL } from './api';

/**
 * What the compiler produced at each stage, for drawing the pipeline: from a finished scene's
 * `pipeline/` folder (bundled by the package stage), or live from a running job's artifacts.
 */

export type StageName = 'ingest' | 'reconstruct' | 'segment' | 'lift' | 'physics' | 'splat' | 'package';

/** The stage graph: each stage starts when the stages it reads from are done. */
export const STAGE_GRAPH: { name: StageName; title: string; model: string; needs: StageName[] }[] = [
  { name: 'ingest', title: 'Keyframes', model: 'Laplacian sharpness', needs: [] },
  { name: 'reconstruct', title: 'Reconstruct', model: 'VGGT-1B', needs: ['ingest'] },
  { name: 'segment', title: 'Segment', model: 'Muse Spark → SAM 3.1', needs: ['ingest'] },
  { name: 'lift', title: 'Lift to 3D', model: 'RANSAC · voxel vote', needs: ['reconstruct', 'segment'] },
  { name: 'physics', title: 'Physics', model: 'Muse Spark priors', needs: ['lift'] },
  { name: 'splat', title: 'Appearance', model: 'gsplat · MCMC', needs: ['lift'] },
  { name: 'package', title: 'Package', model: 'MJCF · URDF · USD', needs: ['physics', 'splat'] },
];

export interface PhysicsObject {
  index: number;
  label: string;
  category?: string;
  confidence?: number;
  material?: string;
  mass_kg?: number;
  typical_mass_kg?: number;
  measured_dimensions_cm?: number[];
  typical_dimensions_cm?: number[];
  friction?: number;
  restitution?: number;
  fragile?: boolean;
}

/** The same shape as a scene's pipeline/pipeline.json; every part is optional while a job runs. */
export interface PipelineSummary {
  cameras?: number[][];
  timings?: { stage: string; seconds: number }[];
  frames?: { count?: number; size?: number[]; duration_s?: number; sharpness?: number[] };
  reconstruct?: { frames?: number; resolution?: number[]; weights?: string; gpu?: string; peak_memory_gb?: number; inference_seconds?: number };
  segment?: { keyframe?: number; objects?: { index: number; label: string; frames_tracked?: number; mean_confidence?: number | null }[] };
  lift?: {
    table_inlier_rms?: number;
    cameras_above_table?: number;
    points_after_confidence?: number;
    objects?: { index: number; label: string; points: number }[];
    dropped?: { label: string }[];
  };
  physics?: { scale_m_per_unit?: number; scale_votes?: number[]; scale_source?: string; objects?: PhysicsObject[] };
  splat?: {
    gaussians?: number;
    iterations?: number;
    strategy?: string;
    antialiased?: boolean;
    train_seconds?: number;
    test_psnr?: number | null;
    train_psnr?: number | null;
    gpu?: string;
    train_views?: number;
    test_views?: number;
  };
  package?: { colliders?: Record<string, { parts: number; hull_coverage: number; used_parts: boolean }> };
}

/** points.bin: uint32 count, then float32 xyz, uint8 rgb and uint8 label per point. */
export interface PointCloudData {
  count: number;
  positions: Float32Array;
  colors: Uint8Array;
  labels: Uint8Array;
}

export function parsePoints(buffer: ArrayBuffer): PointCloudData {
  const count = new DataView(buffer).getUint32(0, true);
  if (buffer.byteLength < 4 + 16 * count) throw new Error('points.bin is truncated'); // 12 + 3 + 1 bytes a point
  const positions = new Float32Array(buffer.slice(4, 4 + 12 * count));
  const colors = new Uint8Array(buffer, 4 + 12 * count, 3 * count);
  const labels = new Uint8Array(buffer, 4 + 15 * count, count);
  return { count, positions, colors, labels };
}

export type ImageArtifact = 'keyframes' | 'segmentation' | 'splat_test_view';

export interface ArtifactSource {
  /** Changes whenever new artifacts may exist (a job advanced), so views refetch. */
  version: string;
  image(name: ImageArtifact): string;
  points(): Promise<PointCloudData | null>;
  summary(): Promise<PipelineSummary>;
  /** Where the finished scene's robotics download lives, if there is one. */
  exportUrl: string | null;
  /** The finished scene package's folder (for its scene graph); null while a job runs. */
  sceneDir: string | null;
}

async function json<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    return response.ok ? ((await response.json()) as T) : null;
  } catch {
    return null;
  }
}

async function points(url: string): Promise<PointCloudData | null> {
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    return response.ok ? parsePoints(await response.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

/** A finished scene package: everything is in its pipeline/ folder. */
export function sceneSource(sceneDir: string): ArtifactSource {
  return {
    version: sceneDir,
    image: (name) => `${sceneDir}/pipeline/${name}.jpg`,
    points: () => points(`${sceneDir}/pipeline/points.bin`),
    summary: async () => (await json<PipelineSummary>(`${sceneDir}/pipeline/pipeline.json`)) ?? {},
    exportUrl: `${sceneDir}/export.zip`,
    sceneDir,
  };
}

/** A job in progress: each stage's own output, served by the API as soon as it exists. */
export function jobSource(jobId: string, version: string): ArtifactSource {
  const base = `${API_URL}/jobs/${encodeURIComponent(jobId)}/artifacts`;
  return {
    version: `${jobId}:${version}`,
    image: (name) => `${base}/${name === 'keyframes' ? 'preview/keyframes' : name}.jpg?v=${encodeURIComponent(version)}`,
    points: () => points(`${base}/preview/points.bin`),
    summary: async () => {
      const [frames, reconstruct, segment, lift, physics, splat, pack, cameras] = await Promise.all([
        json<PipelineSummary['frames']>(`${base}/frames.json`),
        json<PipelineSummary['reconstruct']>(`${base}/recon.json`),
        json<PipelineSummary['segment']>(`${base}/detections.json`),
        json<PipelineSummary['lift']>(`${base}/lift.json`),
        json<PipelineSummary['physics']>(`${base}/physics.json`),
        json<PipelineSummary['splat']>(`${base}/splats.json`),
        json<PipelineSummary['package']>(`${base}/package.json`),
        json<{ cameras: number[][] }>(`${base}/preview/cameras.json`),
      ]);
      return {
        ...(frames ? { frames } : {}),
        ...(reconstruct ? { reconstruct } : {}),
        ...(segment ? { segment } : {}),
        ...(lift ? { lift } : {}),
        ...(physics ? { physics } : {}),
        ...(splat ? { splat } : {}),
        ...(pack ? { package: pack } : {}),
        ...(cameras ? { cameras: cameras.cameras } : {}),
      };
    },
    exportUrl: null,
    sceneDir: null,
  };
}
