import * as THREE from 'three';
import type { SceneObject, TablePlane } from './scene';

/**
 * A scanned background is the table top, trained from a phone circling it. Training squeezes the
 * room around the table into big, soft splats hovering over and around it (a blurry backdrop that
 * only looks right from the filmed angles), and leaves stray "needle" splats. Around each object
 * the table is unreliable too: the camera never saw what the object hid, so training painted the
 * object's colors there, and they stay behind as a ghost once it moves. These helpers keep the
 * table's surface, drop the rest, and patch the table where the objects stood.
 */

/** Splats stay fully visible to this far inside the table's edge (negative) ... */
const EDGE_FADE_START_M = -0.06;
/** ... and are gone this far past it. */
const EDGE_FADE_END_M = -0.015;
/** Below the table top, splats fade out over this range of depths. */
const BELOW_FADE_M: [number, number] = [0.015, 0.04];
/** Above it, over this range of heights: the table's surface, not the backdrop over it. */
const ABOVE_FADE_M: [number, number] = [0.02, 0.05];
/** Splats this large (their longest axis) fade out: the table is made of millimeter-sized ones. */
const SIZE_FADE_M: [number, number] = [0.015, 0.04];
/** The table is patched this far around each object's footprint ... */
const PATCH_MARGIN_M = 0.015;
/** ... with colors sampled from the table in a band this wide around the patch ... */
const SAMPLE_BAND_M = 0.03;
/** ... from splats within this distance of the table's surface. */
const SURFACE_BAND_M = 0.003;
/** Patch splats sit on this grid, each this wide (Gaussian sigma) and this thin. */
const PATCH_SPACING_M = 0.003;
const PATCH_SIGMA_M = 0.003;
const PATCH_THICKNESS_M = 0.0002;
/** Objects resting more than this far above the table (stacked on another) leave no patch. */
const RESTING_GAP_M = 0.025;
/** An object's splats fade out from its bounding box's surface to this far outside it. */
const OBJECT_BOX_FADE_M = 0.005;
/** And as their longest axis grows over this range: objects are made of sub-millimeter splats. */
const OBJECT_SIZE_FADE_M: [number, number] = [0.004, 0.01];
/** Long thin "needle" splats (this many times longer than wide, and at least NEEDLE_MIN_M long) go. */
const NEEDLE_RATIO = 10;
const NEEDLE_MIN_M = 0.003;

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = THREE.MathUtils.clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Height of the table's plane above the point (x, z). */
function tableHeight({ point: [px, py, pz], normal: [nx, ny, nz] }: TablePlane): (x: number, z: number) => number {
  return (x, z) => py - (nx * (x - px) + nz * (z - pz)) / ny;
}

/**
 * Signed horizontal distance from the table's outline (its convex polygon, else its rectangle):
 * negative inside, positive outside. Outside a corner it overestimates a little, which only makes
 * corners fade slightly sooner.
 */
export function tableEdgeDistance(plane: TablePlane): (x: number, z: number) => number {
  const [px, , pz] = plane.point;
  const [sx, sz] = plane.size_m;
  const outline: [number, number][] =
    plane.polygon_m && plane.polygon_m.length >= 3
      ? plane.polygon_m
      : [
          [px - sx / 2, pz - sz / 2],
          [px + sx / 2, pz - sz / 2],
          [px + sx / 2, pz + sz / 2],
          [px - sx / 2, pz + sz / 2],
        ];
  let twiceArea = 0;
  outline.forEach(([ax, az], i) => {
    const [bx, bz] = outline[(i + 1) % outline.length];
    twiceArea += ax * bz - bx * az;
  });
  const turn = twiceArea >= 0 ? 1 : -1; // so the edge normals point outward whichever way the outline winds
  const edges = outline.flatMap(([ax, az], i) => {
    const [bx, bz] = outline[(i + 1) % outline.length];
    const length = Math.hypot(bx - ax, bz - az);
    return length > 1e-9 ? [{ ax, az, nx: (turn * (bz - az)) / length, nz: (-turn * (bx - ax)) / length }] : [];
  });
  return (x, z) => edges.reduce((farthest, e) => Math.max(farthest, (x - e.ax) * e.nx + (z - e.az) * e.nz), -Infinity);
}

/**
 * How visible a background splat centered at (x, y, z) with longest axis `size` should be: 1 on
 * the table's surface, fading to 0 past its edge, away from its surface, and as splats get large.
 */
export function backgroundVisibility(plane: TablePlane): (x: number, y: number, z: number, size: number) => number {
  const edgeDistance = tableEdgeDistance(plane);
  const height = tableHeight(plane);
  const fadeOut = ([start, end]: [number, number], value: number) => 1 - smoothstep(start, end, value);
  return (x, y, z, size) => {
    const onTable = 1 - smoothstep(EDGE_FADE_START_M, EDGE_FADE_END_M, edgeDistance(x, z));
    if (onTable === 0) return 0;
    const above = y - height(x, z);
    return onTable * fadeOut(BELOW_FADE_M, -above) * fadeOut(ABOVE_FADE_M, above) * fadeOut(SIZE_FADE_M, size);
  };
}

/** The table under and around one object, replaced by flat splats in the table's own colors. */
export interface TablePatch {
  /** Whether a background splat lies in the patch (at any height near the table): it is dropped. */
  covers(x: number, y: number, z: number): boolean;
  /** Whether a background splat is table surface just around the patch: its color is a sample. */
  samples(x: number, y: number, z: number): boolean;
  /** Where the patch's splats go, on the table's surface. */
  grid: THREE.Vector3[];
}

/** A patch for each object that rests on the table, where it first stood. */
export function tablePatches(objects: SceneObject[], plane: TablePlane): TablePatch[] {
  const height = tableHeight(plane);
  const local = new THREE.Vector3();
  return objects.flatMap(({ pose, dimensions_m }) => {
    const center = new THREE.Vector3(...pose.position);
    const turn = new THREE.Quaternion(...pose.rotation_quat);
    const inverse = turn.clone().invert();
    const [hx, hy, hz] = dimensions_m.map((d) => d / 2);
    const top = center.y + hy;
    if (center.y - hy - height(center.x, center.z) > RESTING_GAP_M) return [];
    // Footprint test in the object's own frame (objects rest upright, so its x and z lie flat).
    const within = (x: number, z: number, grow: number) => {
      local.set(x - center.x, 0, z - center.z).applyQuaternion(inverse);
      return Math.abs(local.x) <= hx + grow && Math.abs(local.z) <= hz + grow;
    };
    const grid: THREE.Vector3[] = [];
    const [gx, gz] = [hx + PATCH_MARGIN_M, hz + PATCH_MARGIN_M];
    for (let x = -gx; x <= gx + 1e-9; x += PATCH_SPACING_M) {
      for (let z = -gz; z <= gz + 1e-9; z += PATCH_SPACING_M) {
        const spot = new THREE.Vector3(x, 0, z).applyQuaternion(turn).add(center);
        spot.y = height(spot.x, spot.z);
        grid.push(spot);
      }
    }
    return [
      {
        covers: (x, y, z) => y < top + PATCH_MARGIN_M && y > height(x, z) - BELOW_FADE_M[1] && within(x, z, PATCH_MARGIN_M),
        samples: (x, y, z) =>
          Math.abs(y - height(x, z)) < SURFACE_BAND_M && within(x, z, PATCH_MARGIN_M + SAMPLE_BAND_M) && !within(x, z, PATCH_MARGIN_M),
        grid,
      },
    ];
  });
}

/**
 * Splats filling a patch: flat discs lying on the table, each colored like a random sample of the
 * table around it (from the middle of the samples' brightness range, so neither shadows nor
 * glints get copied in, and the patch stays smooth). Random but repeatable, so the table looks
 * the same on every visit.
 */
export function patchSplats(
  plane: TablePlane,
  grid: THREE.Vector3[],
  samples: THREE.Color[],
  push: (center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion, opacity: number, color: THREE.Color) => void,
): void {
  if (samples.length === 0) return;
  const brightness = (c: THREE.Color) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  const sorted = [...samples].sort((a, b) => brightness(a) - brightness(b));
  const middle = sorted.slice(Math.floor(sorted.length * 0.3), Math.ceil(sorted.length * 0.7));
  const pool = middle.length > 0 ? middle : sorted;
  const [nx, ny, nz] = plane.normal;
  const flat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(nx, ny, nz).normalize());
  const scales = new THREE.Vector3(PATCH_SIGMA_M, PATCH_THICKNESS_M, PATCH_SIGMA_M);
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    return seed / 2 ** 32;
  };
  for (const spot of grid) push(spot, scales, flat, 1, pool[Math.floor(random() * pool.length)]);
}

/**
 * How visible one of an object's own splats should be, from its object-local center (x, y, z) and
 * its scales: stray splats outside the object's box, oversized ones and needles fade out. These
 * are what training makes of the background seen past the object's edges.
 */
export function objectVisibility(dimensions: [number, number, number]): (center: THREE.Vector3, scales: THREE.Vector3) => number {
  const [hx, hy, hz] = dimensions.map((d) => d / 2);
  return (center, scales) => {
    const outside = Math.max(Math.abs(center.x) - hx, Math.abs(center.y) - hy, Math.abs(center.z) - hz);
    const inBox = 1 - smoothstep(0, OBJECT_BOX_FADE_M, outside);
    if (inBox === 0) return 0;
    const longest = Math.max(scales.x, scales.y, scales.z);
    const shortest = Math.min(scales.x, scales.y, scales.z);
    if (longest > NEEDLE_MIN_M && longest > NEEDLE_RATIO * shortest) return 0;
    return inBox * (1 - smoothstep(OBJECT_SIZE_FADE_M[0], OBJECT_SIZE_FADE_M[1], longest));
  };
}
