import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { ORIENTATIONS, poseMatrix } from '../engine/choreography';
import { subassemblies } from '../engine/graph';
import type { AssemblyPlan, Part } from '../engine/plan';

/**
 * Line drawings in the style of a flat-pack manual: white parts with heavy black outlines, hidden
 * lines removed, a fixed three-quarter view. Step drawings show the assembly in the step's build
 * orientation with the parts that go on now pulled out along the direction they go in, so arrows
 * and letter callouts can be laid over them.
 *
 * One offscreen WebGL renderer draws every picture; each drawing comes back as an image plus the
 * screen positions (fractions of the image) of its callouts and arrows.
 */

const MM = 0.001;
/** The viewing direction: from the front right, above. */
const VIEW = new THREE.Vector3(1.05, 0.95, 1.7).normalize();
const EXPLODE_M = 0.2;

export interface DrawItem {
  part: Part;
  /** World pose (meters). */
  matrix: THREE.Matrix4;
  /** New parts are drawn white, parts already in place a shade darker. */
  tone: 'new' | 'old';
  /** Pulled out by this much (world meters) along the way it goes in; arrows show the motion. */
  explode?: THREE.Vector3;
  callout?: boolean;
  /** Items that move together share one arrow. */
  group?: string;
}

export interface Drawing {
  url: string;
  width: number;
  height: number;
  callouts: { partId: string; label: string; x: number; y: number }[];
  arrows: { from: [number, number]; to: [number, number] }[];
}

/** The parts of a plan as they sit in the finished piece, standing upright. */
export function finishedItems(plan: AssemblyPlan): DrawItem[] {
  return plan.parts.map((part) => ({ part, matrix: assembledMatrix(part), tone: 'new' }));
}

function assembledMatrix(part: Part): THREE.Matrix4 {
  return poseMatrix(part.assembled?.position ?? [0, 0, 0], part.assembled?.rotation);
}

function halfSize(part: Part): THREE.Vector3 {
  return new THREE.Vector3(...part.size_mm).multiplyScalar(MM / 2);
}

/** World-axis-aligned box of a part at its assembled pose (parts are axis-aligned when assembled). */
function assembledBox(part: Part): THREE.Box3 {
  const box = new THREE.Box3().setFromCenterAndSize(new THREE.Vector3(), halfSize(part).multiplyScalar(2));
  return box.applyMatrix4(assembledMatrix(part));
}

/**
 * Which way a part comes off the part it is joined to: along the axis where their boxes touch,
 * pointing from the other part toward it.
 */
function separation(part: Part, other: Part): THREE.Vector3 {
  const a = assembledBox(part);
  const b = assembledBox(other);
  const ca = a.getCenter(new THREE.Vector3());
  const cb = b.getCenter(new THREE.Vector3());
  const ha = a.getSize(new THREE.Vector3()).multiplyScalar(0.5);
  const hb = b.getSize(new THREE.Vector3()).multiplyScalar(0.5);
  const delta = ca.clone().sub(cb);
  let axis = 0;
  let best = -Infinity;
  for (let i = 0; i < 3; i++) {
    const gap = Math.abs(delta.getComponent(i)) - (ha.getComponent(i) + hb.getComponent(i));
    if (gap > best) {
      best = gap;
      axis = i;
    }
  }
  const direction = new THREE.Vector3();
  direction.setComponent(axis, Math.sign(delta.getComponent(axis)) || 1);
  return direction;
}

const DIRECTIONS = [
  [0, 0, 1],
  [0, 1, 0],
  [0, -1, 0],
  [1, 0, 0],
  [-1, 0, 0],
  [0, 0, -1],
].map(([x, y, z]) => new THREE.Vector3(x, y, z));

/** Can these boxes slide away along `direction` without running into any of the obstacles? */
function free(moving: THREE.Box3[], direction: THREE.Vector3, obstacles: THREE.Box3[]): boolean {
  const shrink = new THREE.Vector3(0.0015, 0.0015, 0.0015);
  const travel = direction.clone().multiplyScalar(4);
  return moving.every((box) => {
    const inner = box.clone().expandByVector(shrink.clone().negate());
    const swept = inner.clone().union(inner.clone().translate(travel));
    return obstacles.every((obstacle) => !swept.intersectsBox(obstacle.clone().expandByVector(shrink.clone().negate())));
  });
}

/** The way out: the preferred direction if it is free, else the first free one (front, up, …). */
function removal(moving: THREE.Box3[], preferred: THREE.Vector3, obstacles: THREE.Box3[]): THREE.Vector3 {
  if (free(moving, preferred, obstacles)) return preferred;
  return DIRECTIONS.find((d) => free(moving, d, obstacles))?.clone() ?? preferred;
}

/**
 * The drawing for step k: the (sub)assembly the step works on, turned to the step's orientation,
 * with the parts it adds pulled out. A subassembly built on the side that joins in this step is
 * pulled out as one piece.
 */
export function stepItems(plan: AssemblyPlan, index: number): DrawItem[] {
  const step = plan.steps[index - 1];
  const parts = new Map(plan.parts.map((p) => [p.id, p]));
  const joints = new Map(plan.joints.map((j) => [j.id, j]));
  const after = subassemblies(plan, index);
  const before = subassemblies(plan, index - 1);
  const mainPart = plan.steps[0]?.parts[0] ?? plan.parts[0].id;
  const touched = new Set([...step.parts, ...step.joints.flatMap((id) => joints.get(id)!.parts)]);
  const group = after.find((g) => g.some((id) => touched.has(id))) ?? [];
  const isMain = group.includes(mainPart);

  // What is new: the step's parts, or (a join with no new parts) the smaller side of the join.
  let fresh = new Set(step.parts);
  if (!fresh.size) {
    const sides = before.filter((g) => g.some((id) => touched.has(id)));
    const moving = sides.filter((g) => !g.includes(mainPart)).sort((a, b) => a.length - b.length)[0] ?? [];
    fresh = new Set(moving);
  }
  // The anchor stays put: the first part of the step when everything in it is new.
  const anchored = group.filter((id) => !fresh.has(id));
  if (!anchored.length) {
    const anchor = step.parts.includes(mainPart) ? mainPart : step.parts[0];
    fresh.delete(anchor);
    anchored.push(anchor);
  }
  const joinedSide = !step.parts.length;

  // Breadth-first from the anchored parts over this step's joints: a part's depth is how many new
  // parts lie between it and the anchored ones, so chains explode further out.
  const depth = new Map<string, number>(anchored.map((id) => [id, 0]));
  const direction = new Map<string, THREE.Vector3>();
  const stepJoints = step.joints.map((id) => joints.get(id)!);
  for (let round = 0; round < group.length && depth.size < group.length; round++) {
    for (const joint of stepJoints) {
      const [a, b] = joint.parts;
      for (const [near, far] of [
        [a, b],
        [b, a],
      ]) {
        if (depth.has(near) && !depth.has(far) && fresh.has(far)) {
          depth.set(far, depth.get(near)! + 1);
          direction.set(far, separation(parts.get(far)!, parts.get(near)!));
        }
      }
    }
  }
  const obstacles = group.filter((id) => !fresh.has(id)).map((id) => assembledBox(parts.get(id)!));
  if (joinedSide) {
    // The side assembly comes in as one piece, along a way that is free of the rest.
    const joint = stepJoints.find((j) => fresh.has(j.parts[0]) !== fresh.has(j.parts[1]));
    if (joint) {
      const [inside, outside] = fresh.has(joint.parts[0]) ? joint.parts : [joint.parts[1], joint.parts[0]];
      const boxes = [...fresh].map((id) => assembledBox(parts.get(id)!));
      const way = removal(boxes, separation(parts.get(inside)!, parts.get(outside)!), obstacles);
      for (const id of fresh) {
        depth.set(id, 1);
        direction.set(id, way);
      }
    }
  } else {
    // A part slides in along the way it leaves its partner, unless parts already in place block it
    // (a shelf between two sides comes in from the front).
    for (const [id, way] of direction) direction.set(id, removal([assembledBox(parts.get(id)!)], way, obstacles));
  }

  const box = group.reduce((acc, id) => acc.union(assembledBox(parts.get(id)!)), new THREE.Box3());
  const reach = Math.max(...box.getSize(new THREE.Vector3()).toArray());
  const distance = Math.max(EXPLODE_M, reach * 0.22);

  // Parts pulled out must not land on (or hide) each other: the one further out goes further.
  const pulled = (id: string) => {
    const way = direction.get(id);
    const shift = way ? way.clone().multiplyScalar(distance * (depth.get(id) ?? 1)) : new THREE.Vector3();
    return assembledBox(parts.get(id)!).translate(shift).expandByScalar(0.05);
  };
  const movers = [...fresh].filter((id) => direction.has(id));
  for (let round = 0; round < movers.length && !joinedSide; round++) {
    let moved = false;
    for (const a of movers) {
      for (const b of movers) {
        if (a >= b || !pulled(a).intersectsBox(pulled(b))) continue;
        const way = direction.get(a)!;
        const outer = assembledBox(parts.get(a)!).getCenter(new THREE.Vector3()).dot(way) >
          assembledBox(parts.get(b)!).getCenter(new THREE.Vector3()).dot(way) ? a : b;
        depth.set(outer, (depth.get(outer) ?? 1) + 1);
        moved = true;
      }
    }
    if (!moved) break;
  }

  const center = box.getCenter(new THREE.Vector3());
  const turn = isMain ? ORIENTATIONS[step.orientation] : new THREE.Quaternion();
  const frame = new THREE.Matrix4()
    .makeTranslation(center.x, center.y, center.z)
    .multiply(new THREE.Matrix4().makeRotationFromQuaternion(turn))
    .multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
  return group.map((id) => {
    const part = parts.get(id)!;
    const isFresh = fresh.has(id);
    const way = direction.get(id);
    const explode = isFresh && way ? way.clone().applyQuaternion(turn).multiplyScalar(distance * (depth.get(id) ?? 1)) : undefined;
    return {
      part,
      matrix: frame.clone().multiply(assembledMatrix(part)),
      tone: isFresh || (step.parts.includes(id) && index === 1) ? 'new' : 'old',
      explode,
      callout: step.parts.includes(id) && !joinedSide,
      group: joinedSide && isFresh ? 'side' : undefined,
    } satisfies DrawItem;
  });
}

function groupBy<T>(items: T[], key: (item: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const item of items) groups.set(key(item), [...(groups.get(key(item)) ?? []), item]);
  return groups;
}

export class ManualArt {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly fills: Record<DrawItem['tone'], THREE.MeshLambertMaterial>;
  private readonly line = new LineMaterial({ color: 0x141414, linewidth: 2.4 });
  private readonly thin = new LineMaterial({ color: 0x141414, linewidth: 1.4 });
  private readonly faces = new Map<string, THREE.BoxGeometry>();
  private readonly edges = new Map<string, LineSegmentsGeometry>();
  private readonly cache = new Map<string, Drawing>();

  constructor() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
    this.renderer.setPixelRatio(2);
    const fill = (color: string) =>
      new THREE.MeshLambertMaterial({ color, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    this.fills = { new: fill('#ffffff'), old: fill('#dcdcdc') };
  }

  private geometry(part: Part): { faces: THREE.BoxGeometry; edges: LineSegmentsGeometry } {
    if (!this.faces.has(part.id)) {
      const [w, h, d] = part.size_mm.map((v) => Math.max(v * MM, 0.003));
      const faces = new THREE.BoxGeometry(w, h, d);
      const edges = new LineSegmentsGeometry().setPositions(new THREE.EdgesGeometry(faces).attributes.position.array as Float32Array);
      this.faces.set(part.id, faces);
      this.edges.set(part.id, edges);
    }
    return { faces: this.faces.get(part.id)!, edges: this.edges.get(part.id)! };
  }

  /**
   * Draw the items at a size (CSS pixels); results are cached by key. `span` fixes the view height
   * in meters (to draw several pictures at one scale); by default the drawing fills the image.
   */
  draw(key: string, items: DrawItem[], width: number, height: number, span?: number): Drawing {
    const cached = this.cache.get(key);
    if (cached) return cached;
    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight('#ffffff', 2.1));
    const light = new THREE.DirectionalLight('#ffffff', 1.25);
    light.position.set(-0.6, 1, 0.35);
    scene.add(light);

    const placed = items.map((item) => {
      const matrix = item.explode ? new THREE.Matrix4().makeTranslation(item.explode.x, item.explode.y, item.explode.z).multiply(item.matrix) : item.matrix;
      return { ...item, final: item.matrix, matrix };
    });
    const bounds = new THREE.Box3();
    const corners = (part: Part, matrix: THREE.Matrix4) => {
      const half = halfSize(part);
      return [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => new THREE.Vector3(x * half.x, y * half.y, z * half.z).applyMatrix4(matrix))));
    };
    for (const item of placed) for (const corner of [...corners(item.part, item.matrix), ...corners(item.part, item.final)]) bounds.expandByPoint(corner);

    const center = bounds.getCenter(new THREE.Vector3());
    const camera = new THREE.OrthographicCamera();
    camera.position.copy(center).addScaledVector(VIEW, 20);
    camera.lookAt(center);
    camera.updateMatrixWorld();
    const view = new THREE.Box3();
    for (const item of placed) for (const corner of corners(item.part, item.matrix)) view.expandByPoint(corner.applyMatrix4(camera.matrixWorldInverse));
    const aspect = width / height;
    let halfW = ((view.max.x - view.min.x) / 2) * 1.1;
    let halfH = ((view.max.y - view.min.y) / 2) * 1.1;
    if (span) {
      halfH = span / 2;
      halfW = halfH * aspect;
    } else if (halfW / halfH > aspect) halfH = halfW / aspect;
    else halfW = halfH * aspect;
    const cx = (view.min.x + view.max.x) / 2;
    const cy = (view.min.y + view.max.y) / 2;
    Object.assign(camera, { left: cx - halfW, right: cx + halfW, top: cy + halfH, bottom: cy - halfH, near: 1, far: 60 });
    camera.updateProjectionMatrix();

    for (const item of placed) {
      const { faces, edges } = this.geometry(item.part);
      const mesh = new THREE.Mesh(faces, this.fills[item.tone]);
      const outline = new LineSegments2(edges, item.tone === 'new' ? this.line : this.thin);
      for (const object of [mesh, outline]) {
        object.matrixAutoUpdate = false;
        object.matrix.copy(item.matrix);
        scene.add(object);
      }
    }
    this.renderer.setSize(width, height, false);
    this.renderer.setClearColor(0xffffff, 0);
    this.line.resolution.set(width, height);
    this.thin.resolution.set(width, height);
    this.renderer.render(scene, camera);

    const screen = (point: THREE.Vector3): [number, number] => {
      const p = point.clone().project(camera);
      return [(p.x + 1) / 2, (1 - p.y) / 2];
    };
    const centerOf = (matrix: THREE.Matrix4) => new THREE.Vector3().setFromMatrixPosition(matrix);
    const drawing: Drawing = {
      url: this.renderer.domElement.toDataURL('image/png'),
      width,
      height,
      callouts: placed.filter((i) => i.callout).map((i) => {
        const [x, y] = screen(centerOf(i.matrix));
        return { partId: i.part.id, label: i.part.label, x, y };
      }),
      arrows: [...groupBy(placed.filter((i) => i.explode), (i) => i.group ?? i.part.id).values()].map((items) => {
        const mean = (key: 'matrix' | 'final') =>
          items.reduce((sum, i) => sum.add(centerOf(i[key])), new THREE.Vector3()).divideScalar(items.length);
        const from = mean('matrix');
        const to = mean('final');
        return { from: screen(from.clone().lerp(to, 0.32)), to: screen(from.clone().lerp(to, 0.86)) };
      }),
    };
    this.cache.set(key, drawing);
    return drawing;
  }

  dispose(): void {
    this.faces.forEach((g) => g.dispose());
    this.edges.forEach((g) => g.dispose());
    Object.values(this.fills).forEach((m) => m.dispose());
    this.line.dispose();
    this.thin.dispose();
    this.renderer.dispose();
  }
}

let shared: ManualArt | null = null;

/** One renderer for every manual on the page. */
export function manualArt(): ManualArt {
  shared ??= new ManualArt();
  return shared;
}
