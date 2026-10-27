import type { BufferGeometry } from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

/**
 * The scene package contract between the pipeline and the simulator: a folder holding
 * scene.json plus asset files. TypeScript mirror of pipeline/scan2sim/schema.py (the source of
 * truth); docs/scene-format.md is the field reference.
 */

export type Vec3 = [number, number, number];
/** Rotation quaternion as [x, y, z, w]. */
export type Quat = [number, number, number, number];

export type ColliderSpec =
  | { type: 'box'; half_extents: Vec3 }
  | { type: 'sphere'; radius: number }
  | { type: 'cylinder'; radius: number; half_height: number }
  | { type: 'convex_hull'; file: string };

/** A colored point cloud: binary little-endian PLY with float xyz and uchar rgb. */
export interface PointsVisual {
  type: 'points';
  file: string;
  count: number;
  point_size_m: number;
}

export type PhysicsSource = 'estimated' | 'measured' | 'hand-authored';

export interface SceneObject {
  id: string;
  label: string;
  confidence: number;
  material: string;
  dimensions_m: Vec3;
  volume_m3: number;
  hollow: boolean;
  fill_fraction: number;
  density_kg_m3: number;
  mass_kg: number;
  /** Friction against the table. Between two objects the larger value applies. */
  friction: number;
  /** Bounciness (0–1) against the table. Between two objects the larger value applies. */
  restitution: number;
  rigidity: 'rigid' | 'soft';
  fragile: boolean;
  /** The object's appearance in object-local coordinates; null means "draw the collider shape". */
  visual?: PointsVisual | null;
  color?: string | null;
  collider: ColliderSpec;
  pose: { position: Vec3; rotation_quat: Quat };
  physics_source: PhysicsSource;
  physics_notes?: string | null;
}

export interface TablePlane {
  point: Vec3;
  normal: Vec3;
  /** Table top size [along x, along z] in meters, centered on `point`. */
  size_m: [number, number];
}

export interface Provenance {
  source: string;
  license?: string | null;
  frames_used: number;
  models: Record<string, string>;
  timings?: { stage: string; seconds: number }[];
  scale_estimates?: number[];
}

export interface SceneManifest {
  version: 1;
  scene_id: string;
  title: string;
  units: 'meters';
  gravity: Vec3;
  scale_source: string;
  background: {
    /** A scanned background in world coordinates; null means "draw a plain table". */
    visual?: PointsVisual | null;
    collider: 'table_plane';
    table_plane: TablePlane;
    table_color?: string | null;
    /** Distance from the table top down to the floor; null means there is no floor. */
    floor_depth_m?: number | null;
  };
  objects: SceneObject[];
  provenance?: Provenance | null;
}

/** Key for the background's point cloud in `LoadedScene.pointClouds`. */
export const BACKGROUND = 'background';

export interface LoadedScene {
  dir: string;
  manifest: SceneManifest;
  /** Convex hull vertices (x, y, z, x, y, z, ...) per object id, read from collider files. */
  hullPoints: Map<string, Float32Array>;
  /** Point clouds per object id, plus BACKGROUND. */
  pointClouds: Map<string, BufferGeometry>;
  /** Problems the loader worked around (a skipped object, a plain shape instead of a scan). */
  warnings: string[];
}

/** How the loader reads a scene's files: over HTTP in the app, from disk in tests. */
export interface SceneFiles {
  text(path: string): Promise<string>;
  binary(path: string): Promise<ArrayBuffer>;
}

async function fetchOk(path: string): Promise<Response> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`could not load ${path} (HTTP ${response.status})`);
  return response;
}

const httpFiles: SceneFiles = {
  text: async (path) => (await fetchOk(path)).text(),
  binary: async (path) => (await fetchOk(path)).arrayBuffer(),
};

/** Load scene.json from a scene folder (a URL or a file path), plus the files it names. */
export async function loadScene(dir: string, files: SceneFiles = httpFiles): Promise<LoadedScene> {
  const { manifest, warnings } = checkManifest(JSON.parse(await files.text(`${dir}/scene.json`)));
  const hullPoints = new Map<string, Float32Array>();
  const pointClouds = new Map<string, BufferGeometry>();

  const loadPoints = async (key: string, visual: PointsVisual, name: string) => {
    try {
      pointClouds.set(key, new PLYLoader().parse(await files.binary(`${dir}/${visual.file}`)));
    } catch (error) {
      warnings.push(`${name}: ${visual.file}: ${(error as Error).message}; showing a plain shape instead`);
    }
  };
  const loadObject = async (object: SceneObject) => {
    const name = `${object.id} (${object.label})`;
    if (object.collider.type === 'convex_hull') {
      try {
        const points = parseObjVertices(await files.text(`${dir}/${object.collider.file}`));
        if (points.length < 12) throw new Error('needs at least 4 vertices');
        hullPoints.set(object.id, points);
      } catch (error) {
        warnings.push(`${name}: ${object.collider.file}: ${(error as Error).message}; using a box instead`);
      }
    }
    if (object.visual) await loadPoints(object.id, object.visual, name);
  };
  const background = manifest.background.visual;
  await Promise.all([
    ...manifest.objects.map(loadObject),
    background ? loadPoints(BACKGROUND, background, 'background') : Promise.resolve(),
  ]);
  return { dir, manifest, hullPoints, pointClouds, warnings };
}

/** Vertex positions from a Wavefront OBJ file ("v x y z" lines); faces are ignored. */
export function parseObjVertices(text: string): Float32Array {
  const values: number[] = [];
  for (const line of text.split('\n')) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] !== 'v') continue;
    const xyz = parts.slice(1, 4).map(Number);
    if (xyz.length !== 3 || !xyz.every(Number.isFinite)) throw new Error(`bad vertex line "${line.trim()}"`);
    values.push(...xyz);
  }
  return new Float32Array(values);
}

// ---------- Checking scene.json ----------

class SceneFormatError extends Error {}

function fail(path: string, expected: string): never {
  throw new SceneFormatError(`scene.json: ${path} should be ${expected}`);
}

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function checkVec3(value: unknown, path: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(isNumber)) fail(path, 'a list of 3 numbers');
  return value as Vec3;
}

function checkNumber(value: unknown, path: string, min: number, max: number): number {
  if (!isNumber(value) || value < min || value > max) fail(path, `a number from ${min} to ${max}`);
  return value;
}

function checkString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value === '') fail(path, 'a non-empty string');
  return value;
}

function checkPointsVisual(value: unknown, path: string): PointsVisual {
  if (!isRecord(value) || value.type !== 'points') fail(path, 'null or {"type": "points", ...}');
  checkString(value.file, `${path}.file`);
  checkNumber(value.point_size_m, `${path}.point_size_m`, 1e-5, 1);
  return value as unknown as PointsVisual;
}

/** A bad appearance doesn't stop anything: report it and fall back to a plain shape. */
function optionalVisual(value: unknown, path: string, warnings: string[]): PointsVisual | null {
  if (value === undefined || value === null) return null;
  try {
    return checkPointsVisual(value, path);
  } catch (error) {
    if (!(error instanceof SceneFormatError)) throw error;
    warnings.push(`${error.message}; showing a plain shape instead`);
    return null;
  }
}

/**
 * Check the parts of scene.json the simulator relies on, filling in schema defaults. A problem
 * with the scene as a whole throws; a problem with one object only skips that object, and a bad
 * appearance falls back to a plain shape (both reported in `warnings`).
 */
export function checkManifest(raw: unknown): { manifest: SceneManifest; warnings: string[] } {
  if (!isRecord(raw)) fail('the file', 'a JSON object');
  if ((raw.version ?? 1) !== 1) fail('version', '1');
  if ((raw.units ?? 'meters') !== 'meters') fail('units', '"meters"');
  const gravity = raw.gravity === undefined ? ([0, -9.81, 0] as Vec3) : checkVec3(raw.gravity, 'gravity');
  const background = raw.background;
  if (!isRecord(background)) fail('background', 'an object');
  const plane = background.table_plane;
  if (!isRecord(plane)) fail('background.table_plane', 'an object');
  checkVec3(plane.point, 'background.table_plane.point');
  const normal = checkVec3(plane.normal, 'background.table_plane.normal');
  if (Math.hypot(...normal) < 1e-6) fail('background.table_plane.normal', 'a non-zero direction');
  const size = plane.size_m;
  if (!Array.isArray(size) || size.length !== 2 || !size.every((v) => isNumber(v) && v > 0)) {
    fail('background.table_plane.size_m', 'a list of 2 positive numbers');
  }
  const floorDepth = background.floor_depth_m;
  if (floorDepth !== undefined && floorDepth !== null) checkNumber(floorDepth, 'background.floor_depth_m', 0.05, 5);
  if (!Array.isArray(raw.objects)) fail('objects', 'a list');

  const warnings: string[] = [];
  const backgroundVisual = optionalVisual(background.visual, 'background.visual', warnings);
  const objects: SceneObject[] = [];
  const ids = new Set<string>();
  raw.objects.forEach((item: unknown, index: number) => {
    try {
      const object = checkObject(item, `objects[${index}]`, warnings);
      if (ids.has(object.id)) fail(`objects[${index}].id`, 'unique');
      ids.add(object.id);
      objects.push(object);
    } catch (error) {
      if (!(error instanceof SceneFormatError)) throw error;
      warnings.push(`${error.message}; the object was left out`);
    }
  });

  const manifest = {
    ...raw,
    version: 1,
    units: 'meters',
    gravity,
    title: typeof raw.title === 'string' && raw.title ? raw.title : String(raw.scene_id ?? 'Untitled scene'),
    background: { ...background, visual: backgroundVisual },
    objects,
  } as unknown as SceneManifest;
  return { manifest, warnings };
}

function checkObject(raw: unknown, path: string, warnings: string[]): SceneObject {
  if (!isRecord(raw)) fail(path, 'an object');
  checkString(raw.id, `${path}.id`);
  checkString(raw.label, `${path}.label`);
  checkVec3(raw.dimensions_m, `${path}.dimensions_m`);
  checkNumber(raw.mass_kg, `${path}.mass_kg`, 0.001, 200);
  checkNumber(raw.friction, `${path}.friction`, 0, 2);
  checkNumber(raw.restitution, `${path}.restitution`, 0, 1);
  checkCollider(raw.collider, `${path}.collider`);
  const pose = raw.pose;
  if (!isRecord(pose)) fail(`${path}.pose`, 'an object');
  const position = checkVec3(pose.position, `${path}.pose.position`);
  const quat = pose.rotation_quat ?? [0, 0, 0, 1];
  if (!Array.isArray(quat) || quat.length !== 4 || !quat.every(isNumber) || Math.hypot(...quat) < 1e-6) {
    fail(`${path}.pose.rotation_quat`, 'a list of 4 numbers [x, y, z, w]');
  }
  return {
    ...(raw as unknown as SceneObject),
    pose: { position, rotation_quat: quat as Quat },
    visual: optionalVisual(raw.visual, `${path}.visual`, warnings),
  };
}

function checkCollider(raw: unknown, path: string): void {
  if (!isRecord(raw)) fail(path, 'an object');
  switch (raw.type) {
    case 'box':
      if (!checkVec3(raw.half_extents, `${path}.half_extents`).every((v) => v > 0)) {
        fail(`${path}.half_extents`, 'positive');
      }
      return;
    case 'sphere':
      checkNumber(raw.radius, `${path}.radius`, 1e-4, 10);
      return;
    case 'cylinder':
      checkNumber(raw.radius, `${path}.radius`, 1e-4, 10);
      checkNumber(raw.half_height, `${path}.half_height`, 1e-4, 10);
      return;
    case 'convex_hull':
      checkString(raw.file, `${path}.file`);
      return;
    default:
      fail(`${path}.type`, '"box", "sphere", "cylinder" or "convex_hull"');
  }
}
