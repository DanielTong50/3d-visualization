import type { BufferGeometry } from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

/**
 * The scene package contract between the pipeline and the simulator: a folder holding
 * scene.json plus asset files. TypeScript mirror of pipeline/darius/scene/schema.py (the source of
 * truth); docs/scene-format.md is the field reference.
 */

export type Vec3 = [number, number, number];
/** Rotation quaternion as [x, y, z, w]. */
export type Quat = [number, number, number, number];

export type ColliderSpec =
  | { type: 'box'; half_extents: Vec3 }
  | { type: 'sphere'; radius: number }
  | { type: 'cylinder'; radius: number; half_height: number }
  | { type: 'convex_hull'; file: string }
  /** Several convex pieces (OBJ files) that together approximate a concave object. */
  | { type: 'convex_parts'; files: string[] };

/** A colored point cloud: binary little-endian PLY with float xyz and uchar rgb. */
export interface PointsVisual {
  type: 'points';
  file: string;
  count: number;
  point_size_m: number;
}

/** Trained Gaussian splats (.spz, .splat or .ply); the file extension says which. */
export interface SplatsVisual {
  type: 'splats';
  file: string;
  count: number;
  sh_degree?: number;
  /** Trained with anti-aliased rasterization (render with Spark's defaults); else classic 3DGS. */
  antialiased?: boolean;
}

export type Visual = PointsVisual | SplatsVisual;

export type PhysicsSource = 'estimated' | 'measured' | 'hand-authored';

/** How an object's mass is distributed, in its body frame (pipeline: darius.robotics.mass). */
export interface Inertial {
  com_m: Vec3;
  /** About the center of mass, body axes: [ixx, iyy, izz, ixy, ixz, iyz]. */
  inertia_kg_m2: [number, number, number, number, number, number];
  model: 'solid' | 'shell';
}

/** A parallel-jaw grasp candidate in the body frame: +z approach, +y closing axis. */
export interface Grasp {
  position: Vec3;
  rotation_quat: Quat;
  width_m: number;
  score: number;
}

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
  visual?: Visual | null;
  /** Shown instead of `visual` if that can't be loaded. */
  fallback_visual?: PointsVisual | null;
  color?: string | null;
  collider: ColliderSpec;
  /** One convex hull of the whole object (OBJ): used for taps and as a collider fallback. */
  pick_hull?: string | null;
  pose: { position: Vec3; rotation_quat: Quat };
  physics_source: PhysicsSource;
  physics_notes?: string | null;
  /** Robotics annotations (darius.robotics): mass distribution and grasp candidates, best first. */
  inertial?: Inertial | null;
  grasps?: Grasp[] | null;
}

export interface TablePlane {
  point: Vec3;
  normal: Vec3;
  /** Table top size [along x, along z] in meters, centered on `point`. */
  size_m: [number, number];
  /** Convex outline of the table top as [x, z] world points; preferred over `size_m` when present. */
  polygon_m?: [number, number][] | null;
}

export interface Provenance {
  source: string;
  license?: string | null;
  frames_used: number;
  models: Record<string, string>;
  timings?: { stage: string; seconds: number }[];
  scale_estimates?: number[];
  appearance?: { test_psnr_db?: number | null; train_psnr_db?: number | null; gaussians: number; train_seconds: number } | null;
}

/** The filmed camera path (after refinement): OpenGL cameras (looking down -z, +y up) in the scene frame. */
export interface Capture {
  cameras: { position: Vec3; rotation_quat: Quat }[];
  fov_deg: number;
  aspect: number;
  image_size?: [number, number] | null;
}

/** A camera the scene was filmed from (scene coordinates): splats look their best from there. */
export interface CaptureView {
  position: Vec3;
  target: Vec3;
  /** Vertical field of view. */
  fov_deg: number;
}

export interface SceneManifest {
  version: 1;
  scene_id: string;
  title: string;
  units: 'meters';
  gravity: Vec3;
  scale_source: string;
  background: {
    /** The scanned table top in world coordinates; null means "draw a plain table". */
    visual?: Visual | null;
    /** Splats beyond the table (walls, floor, the room): reliable only near the filmed viewpoints. */
    backdrop?: SplatsVisual | null;
    fallback_visual?: PointsVisual | null;
    collider: 'table_plane';
    table_plane: TablePlane;
    table_color?: string | null;
    /** Distance from the table top down to the floor; null means there is no floor. */
    floor_depth_m?: number | null;
  };
  objects: SceneObject[];
  /** Where the simulator's camera starts, if the pipeline knows where the video was filmed from. */
  view?: CaptureView | null;
  capture?: Capture | null;
  provenance?: Provenance | null;
}

/** Key for the background in `LoadedScene.pointClouds` and `LoadedScene.splatFiles`. */
export const BACKGROUND = 'background';
/** Key for the backdrop (the surroundings beyond the table) in `LoadedScene.splatFiles`. */
export const BACKDROP = 'backdrop';

export interface LoadedScene {
  dir: string;
  manifest: SceneManifest;
  /** One convex hull per object id (x, y, z, x, y, z, ...): its convex_hull collider or pick_hull. */
  hullPoints: Map<string, Float32Array>;
  /** Convex pieces per object id, for convex_parts colliders. */
  hullParts: Map<string, Float32Array[]>;
  /** Point clouds to draw, per object id plus BACKGROUND. */
  pointClouds: Map<string, BufferGeometry>;
  /** Gaussian splat files to draw, per object id plus BACKGROUND. */
  splatFiles: Map<string, { bytes: ArrayBuffer; name: string }>;
  /** Problems the loader worked around (a skipped object, a fallback appearance or shape). */
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
  const scene: LoadedScene = {
    dir,
    manifest,
    hullPoints: new Map(),
    hullParts: new Map(),
    pointClouds: new Map(),
    splatFiles: new Map(),
    warnings,
  };
  const hull = async (file: string) => {
    const points = parseObjVertices(await files.text(`${dir}/${file}`));
    if (points.length < 12) throw new Error('needs at least 4 vertices');
    return points;
  };

  /** The visual if it loads, else its points fallback, else nothing (plain shape). */
  const loadVisual = async (key: string, name: string, visual?: Visual | null, fallback?: PointsVisual | null) => {
    for (const candidate of [visual, fallback]) {
      if (!candidate) continue;
      try {
        const bytes = await files.binary(`${dir}/${candidate.file}`);
        if (candidate.type === 'splats') scene.splatFiles.set(key, { bytes, name: candidate.file });
        else scene.pointClouds.set(key, new PLYLoader().parse(bytes));
        return;
      } catch (error) {
        warnings.push(`${name}: ${candidate.file}: ${(error as Error).message}`);
      }
    }
  };

  const loadObject = async (object: SceneObject) => {
    const name = `${object.id} (${object.label})`;
    const { collider } = object;
    if (collider.type === 'convex_parts') {
      try {
        scene.hullParts.set(object.id, await Promise.all(collider.files.map(hull)));
      } catch (error) {
        warnings.push(`${name}: convex parts: ${(error as Error).message}; using a single hull instead`);
      }
    }
    const single = collider.type === 'convex_hull' ? collider.file : object.pick_hull;
    if (single) {
      try {
        scene.hullPoints.set(object.id, await hull(single));
      } catch (error) {
        warnings.push(`${name}: ${single}: ${(error as Error).message}`);
      }
    }
    await loadVisual(object.id, name, object.visual, object.fallback_visual);
  };

  const { background } = manifest;
  await Promise.all([
    ...manifest.objects.map(loadObject),
    loadVisual(BACKGROUND, 'background', background.visual, background.fallback_visual),
    background.backdrop ? loadVisual(BACKDROP, 'backdrop', background.backdrop, null) : Promise.resolve(),
  ]);
  return scene;
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

function checkVisual(value: unknown, path: string, allowSplats: boolean): Visual {
  if (!isRecord(value) || (value.type !== 'points' && !(allowSplats && value.type === 'splats'))) {
    fail(path, allowSplats ? 'null, {"type": "points", ...} or {"type": "splats", ...}' : 'null or {"type": "points", ...}');
  }
  checkString(value.file, `${path}.file`);
  if (value.type === 'points') checkNumber(value.point_size_m, `${path}.point_size_m`, 1e-5, 1);
  return value as unknown as Visual;
}

/** A bad appearance doesn't stop anything: report it and fall back to a plain shape. */
function optionalVisual<T extends Visual>(value: unknown, path: string, warnings: string[], allowSplats: boolean): T | null {
  if (value === undefined || value === null) return null;
  try {
    return checkVisual(value, path, allowSplats) as T;
  } catch (error) {
    if (!(error instanceof SceneFormatError)) throw error;
    warnings.push(`${error.message}; showing a plain shape instead`);
    return null;
  }
}

/** The table outline if it's a usable polygon, else null (the rectangle from size_m is used). */
function optionalPolygon(value: unknown, warnings: string[]): [number, number][] | null {
  if (value === undefined || value === null) return null;
  const ok = Array.isArray(value) && value.length >= 3 && value.every((p) => Array.isArray(p) && p.length === 2 && p.every(isNumber));
  if (!ok) {
    warnings.push('scene.json: background.table_plane.polygon_m should be a list of at least 3 [x, z] points; using size_m');
    return null;
  }
  return value as [number, number][];
}

/** The capture view if it's usable, else null (the simulator frames the objects itself). */
function optionalView(value: unknown, warnings: string[]): CaptureView | null {
  if (value === undefined || value === null) return null;
  const vec3 = (v: unknown) => Array.isArray(v) && v.length === 3 && v.every(isNumber);
  const ok =
    isRecord(value) &&
    vec3(value.position) &&
    vec3(value.target) &&
    isNumber(value.fov_deg) &&
    value.fov_deg > 0 &&
    value.fov_deg < 180 &&
    Math.hypot(...(value.position as Vec3).map((p, i) => p - (value.target as Vec3)[i])) > 1e-6;
  if (!ok) {
    warnings.push('scene.json: view should have a position and a different target ([x, y, z]) and fov_deg; ignoring it');
    return null;
  }
  return value as unknown as CaptureView;
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
    background: {
      ...background,
      visual: optionalVisual(background.visual, 'background.visual', warnings, true),
      backdrop: optionalVisual(background.backdrop, 'background.backdrop', warnings, true),
      fallback_visual: optionalVisual(background.fallback_visual, 'background.fallback_visual', warnings, false),
      table_plane: { ...plane, polygon_m: optionalPolygon(plane.polygon_m, warnings) },
    },
    objects,
    view: optionalView(raw.view, warnings),
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
  if (raw.pick_hull !== undefined && raw.pick_hull !== null) checkString(raw.pick_hull, `${path}.pick_hull`);
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
    visual: optionalVisual(raw.visual, `${path}.visual`, warnings, true),
    fallback_visual: optionalVisual(raw.fallback_visual, `${path}.fallback_visual`, warnings, false),
    inertial: optionalInertial(raw.inertial, `${path}.inertial`, warnings),
    grasps: optionalGrasps(raw.grasps, `${path}.grasps`, warnings),
  };
}

const isList = (value: unknown, length: number) => Array.isArray(value) && value.length === length && value.every(isNumber);

/** Robotics annotations are optional extras: a malformed one is dropped, not fatal. */
function optionalInertial(value: unknown, path: string, warnings: string[]): Inertial | null {
  if (value === undefined || value === null) return null;
  if (isRecord(value) && isList(value.com_m, 3) && isList(value.inertia_kg_m2, 6)) return value as unknown as Inertial;
  warnings.push(`scene.json: ${path} should have com_m [x, y, z] and six inertia_kg_m2 values; ignoring it`);
  return null;
}

function optionalGrasps(value: unknown, path: string, warnings: string[]): Grasp[] | null {
  if (value === undefined || value === null) return null;
  const ok =
    Array.isArray(value) &&
    value.every((g) => isRecord(g) && isList(g.position, 3) && isList(g.rotation_quat, 4) && isNumber(g.width_m) && isNumber(g.score));
  if (ok) return value as Grasp[];
  warnings.push(`scene.json: ${path} should be a list of {position, rotation_quat, width_m, score}; ignoring it`);
  return null;
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
    case 'convex_parts':
      if (!Array.isArray(raw.files) || raw.files.length === 0 || !raw.files.every((f) => typeof f === 'string' && f !== '')) {
        fail(`${path}.files`, 'a non-empty list of OBJ file names');
      }
      return;
    default:
      fail(`${path}.type`, '"box", "sphere", "cylinder", "convex_hull" or "convex_parts"');
  }
}
