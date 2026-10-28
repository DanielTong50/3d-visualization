import RAPIER from '@dimforge/rapier3d-compat';
import type { Collider, ColliderDesc, ImpulseJoint, RigidBody, World } from '@dimforge/rapier3d-compat';
import type { ColliderSpec, LoadedScene, Quat, SceneObject, Vec3 } from './scene';

/** Physics runs at a fixed 120 steps per second, whatever the screen's frame rate. */
export const PHYSICS_HZ = 120;
export const TABLE_THICKNESS_M = 0.04;

/** Gravity presets (m/s², pulling straight down). */
export const GRAVITY_PRESETS = { earth: 9.81, moon: 1.62, mars: 3.71, jupiter: 24.79, zero: 0 } as const;
export type GravityPreset = keyof typeof GRAVITY_PRESETS;

/** How hard a bumped table shoves what's on it (m/s). */
export const BUMP_SPEEDS = { light: 0.3, medium: 0.7, hard: 1.4 } as const;
export type BumpStrength = keyof typeof BUMP_SPEEDS;

/** A finger poke (N·s): a 60 g ball leaves at ~3 m/s, a 1.2 kg book barely budges. */
export const POKE_IMPULSE_NS = 0.2;

// A little angular damping stands in for rolling resistance, so balls come to rest.
const ANGULAR_DAMPING = 0.25;

// Holding: the grabbed point follows the pointer on a spring that Rapier solves implicitly (stable
// at any stiffness, unlike an explicit spring, which made light objects held by an edge oscillate
// and fly off). The stiffness scales with mass, so every object follows the hand with the same
// natural frequency and sags the same g / omega^2 (1 cm): a 23 kg toaster used to hang half a
// meter below the pointer on a fixed 400 N/m spring, which read as broken gravity. Critically
// damped, so a held object never bounces on the spring.
const GRAB_FREQUENCY_RAD_S = 31; // about 5 Hz
const GRAB_DAMPING_RATIO = 1;
const HELD_ANGULAR_DAMPING = 5; // a hand steadies what it slides: swings settle within a second
const MAX_THROW_SPEED_M_S = 10;
const MAX_PUSH_SPEED_M_S = 8; // fastest a poke or a "what if" push may make an object move
const TILT_SPEED_DEG_PER_S = 60; // the table tilts smoothly rather than jumping

interface XYZ {
  x: number;
  y: number;
  z: number;
}

interface XYZW extends XYZ {
  w: number;
}

export interface SimBody {
  spec: SceneObject;
  body: RigidBody;
  /** How the collider was built: as the scene asked, from the single pick hull, or as a box. */
  shape: 'as-specified' | 'hull-fallback' | 'box-fallback';
}

export interface TableBounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

interface Grab {
  body: RigidBody;
  joint: ImpulseJoint;
  target: XYZ;
  savedAngularDamping: number;
  /** Carried objects stay upright: their rotation is locked until they are let go. */
  upright: boolean;
}

let rapierReady: Promise<void> | null = null;

/** Load Rapier's WebAssembly (only the first call does any work). */
export function initPhysicsEngine(): Promise<void> {
  rapierReady ??= RAPIER.init();
  return rapierReady;
}

/**
 * The Rapier world for one scene: a table that can tilt, a fixed floor, and one rigid body per
 * object, plus the actions the simulator and "what if" commands perform on them.
 */
export class PhysicsWorld {
  readonly world: World;
  readonly bodies: SimBody[] = [];
  /** Objects built differently than the scene asked (for example, a box for a flat convex hull). */
  readonly warnings: string[] = [];
  /** The table top's extent in world x/z. */
  readonly tableBounds: TableBounds;
  private grab: Grab | null = null;
  /** Kinematic and collider-free: the pointer moves it, and a held object hangs from it. */
  private readonly hand: RigidBody;
  /** The table: kinematic, with its origin on the middle of its top surface, so it tilts about that point. */
  private readonly table: RigidBody;
  private readonly tableLevel: XYZW;
  private tilt = { axis: 'x' as 'x' | 'z', degrees: 0, targetDegrees: 0 };

  constructor(private readonly scene: LoadedScene) {
    const [gx, gy, gz] = scene.manifest.gravity;
    this.world = new RAPIER.World({ x: gx, y: gy, z: gz });
    this.world.timestep = 1 / PHYSICS_HZ;
    // Rapier's default 1 m length unit. At 0.1 (tighter tolerances for 10 cm objects), the internal-edge
    // fix of the table mesh let a 3 cm scanned book fall straight through the slab.
    this.world.lengthUnit = 1;
    this.world.numSolverIterations = 8; // steadier resting and stacking than the default 4

    const { table_plane: plane, floor_depth_m } = scene.manifest.background;
    const up = normalize(plane.normal);
    const rotation = rotationFromUp(up);
    const [px, py, pz] = plane.point;
    this.tableLevel = rotation;
    this.table = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(px, py, pz).setRotation(rotation),
    );
    // Table and floor have zero friction and bounciness, so each object's own values
    // decide its contacts with them (objects use the "max" combine rule).
    const { descs, bounds } = tableColliders(plane.polygon_m ?? null, plane.size_m, plane.point, rotation);
    this.tableBounds = bounds;
    for (const desc of descs) this.world.createCollider(desc.setFriction(0).setRestitution(0), this.table);
    if (typeof floor_depth_m === 'number') {
      const [x, y, z] = plane.point.map((v, i) => v - up[i] * (floor_depth_m + 0.05));
      const floor = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(x, y, z).setRotation(rotation));
      this.world.createCollider(RAPIER.ColliderDesc.cuboid(10, 0.05, 10).setFriction(0).setRestitution(0), floor);
    }
    this.hand = this.world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased());
    for (const spec of scene.manifest.objects) this.addObject(spec);
  }

  step(): void {
    if (this.grab) {
      this.hand.setNextKinematicTranslation(this.grab.target);
      limitSpeed(this.grab.body, MAX_THROW_SPEED_M_S);
    }
    this.moveTableTowardTilt();
    this.world.step();
  }

  /** Put every object back where the scene file says, at rest, on a level table. */
  reset(): void {
    this.release();
    this.tilt = { axis: 'x', degrees: 0, targetDegrees: 0 };
    this.table.setRotation(this.tableLevel, true);
    this.table.setNextKinematicRotation(this.tableLevel);
    for (const { spec, body } of this.bodies) {
      const [x, y, z] = spec.pose.position;
      body.setTranslation({ x, y, z }, true);
      body.setRotation(toRotation(spec.pose.rotation_quat), true);
      body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }
  }

  // ---------- Holding objects ----------

  /**
   * Take hold of `body` by the world-space `point` on its surface; it stays put until the target
   * moves. `upright` (carrying) locks its rotation, so it doesn't swing or tip while lifted.
   */
  grabAt(body: RigidBody, point: XYZ, upright = false): void {
    this.release();
    this.hand.setTranslation(point, true);
    this.hand.setNextKinematicTranslation(point);
    const local = rotate(conjugate(body.rotation()), subtract(point, body.translation()));
    const stiffness = body.mass() * GRAB_FREQUENCY_RAD_S ** 2;
    const damping = 2 * GRAB_DAMPING_RATIO * body.mass() * GRAB_FREQUENCY_RAD_S;
    const joint = this.world.createImpulseJoint(
      RAPIER.JointData.spring(0, stiffness, damping, { x: 0, y: 0, z: 0 }, local),
      this.hand,
      body,
      true,
    );
    this.grab = { body, joint, target: { ...point }, savedAngularDamping: body.angularDamping(), upright };
    body.setAngularDamping(HELD_ANGULAR_DAMPING);
    if (upright) {
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
      body.lockRotations(true, true);
    }
    body.wakeUp();
  }

  moveGrabTarget(target: XYZ): void {
    if (this.grab) this.grab.target = { x: target.x, y: target.y, z: target.z };
  }

  /** Turn the held object about the vertical by `radians` (carrying only). */
  turnHeld(radians: number): void {
    if (!this.grab?.upright) return;
    const { body } = this.grab;
    body.setRotation(multiply(axisRotation('y', radians), body.rotation()), true);
  }

  /**
   * Let go. With a `throwVelocity` (the hand's own recent velocity), a flick throws the object
   * as fast as the hand moved; otherwise it keeps whatever velocity the spring gave it.
   */
  release(throwVelocity?: XYZ): void {
    if (!this.grab) return;
    const { body, joint, savedAngularDamping, upright } = this.grab;
    this.grab = null;
    this.world.removeImpulseJoint(joint, true);
    body.setAngularDamping(savedAngularDamping);
    if (upright) body.lockRotations(false, true);
    if (throwVelocity) body.setLinvel(throwVelocity, true);
    limitSpeed(body, MAX_THROW_SPEED_M_S);
  }

  /** A finger poke at `point`, pushing along `direction`; the same push moves heavy objects less. */
  poke(id: string, point: XYZ, direction: XYZ, impulse = POKE_IMPULSE_NS): void {
    const body = this.find(id).body;
    const length = Math.hypot(direction.x, direction.y, direction.z) || 1;
    const scale = impulse / length;
    body.applyImpulseAtPoint({ x: direction.x * scale, y: direction.y * scale, z: direction.z * scale }, point, true);
    limitSpeed(body, MAX_PUSH_SPEED_M_S);
  }

  // ---------- Changing the world ----------

  setGravity(preset: GravityPreset): void {
    this.world.gravity = { x: 0, y: -GRAVITY_PRESETS[preset], z: 0 };
    for (const { body } of this.bodies) body.wakeUp();
  }

  /** Change an object's velocity by `delta` (m/s), as a sudden push would. */
  push(id: string, delta: XYZ): void {
    const body = this.find(id).body;
    const v = body.linvel();
    body.setLinvel({ x: v.x + delta.x, y: v.y + delta.y, z: v.z + delta.z }, true);
    limitSpeed(body, MAX_PUSH_SPEED_M_S);
  }

  /** Jolt the table along `direction`: everything on it lurches that way and hops a little. */
  bumpTable(direction: XYZ, strength: BumpStrength): void {
    const speed = BUMP_SPEEDS[strength];
    const length = Math.hypot(direction.x, direction.z) || 1;
    const top = this.scene.manifest.background.table_plane.point[1];
    const { minX, maxX, minZ, maxZ } = this.tableBounds;
    for (const { spec, body } of this.bodies) {
      const p = body.translation();
      const onTable = p.x >= minX - 0.05 && p.x <= maxX + 0.05 && p.z >= minZ - 0.05 && p.z <= maxZ + 0.05 && p.y >= top - 0.05 && p.y <= top + 0.5;
      if (onTable) {
        this.push(spec.id, { x: (direction.x / length) * speed, y: speed * 0.4, z: (direction.z / length) * speed });
      }
    }
  }

  /** Tilt the table `degrees` about the world x or z axis (0 levels it); it turns at 60°/s. */
  tiltTable(axis: 'x' | 'z', degrees: number): void {
    if (axis !== this.tilt.axis) this.tilt = { axis, degrees: 0, targetDegrees: degrees };
    else this.tilt.targetDegrees = degrees;
    for (const { body } of this.bodies) body.wakeUp();
  }

  /** The table's current tilt, for drawing it. */
  tableTilt(): { axis: 'x' | 'z'; degrees: number } {
    return { axis: this.tilt.axis, degrees: this.tilt.degrees };
  }

  /** Lift an object `height` meters above the table (keeping its spot and turn) and let it fall. */
  dropObject(id: string, height: number): void {
    const { spec, body } = this.find(id);
    const p = body.translation();
    const top = this.scene.manifest.background.table_plane.point[1];
    body.setTranslation({ x: p.x, y: top + height + spec.dimensions_m[1] / 2, z: p.z }, true);
    body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  /** Change an object's physical properties (from material editing or a "what if"); returns its new spec. */
  updateObject(id: string, changes: Partial<SceneObject>): SceneObject {
    const simBody = this.find(id);
    const spec = { ...simBody.spec, ...changes };
    applyPhysics(colliders(simBody.body), spec);
    simBody.body.recomputeMassPropertiesFromColliders(); // otherwise the new mass applies only from the next step
    simBody.spec = spec;
    simBody.body.wakeUp();
    return spec;
  }

  dispose(): void {
    this.world.free();
  }

  private find(id: string): SimBody {
    const found = this.bodies.find((b) => b.spec.id === id);
    if (!found) throw new Error(`no object "${id}" in this scene`);
    return found;
  }

  private moveTableTowardTilt(): void {
    const { axis, degrees, targetDegrees } = this.tilt;
    if (degrees === targetDegrees) return;
    const stepDegrees = TILT_SPEED_DEG_PER_S / PHYSICS_HZ;
    const next = Math.abs(targetDegrees - degrees) <= stepDegrees ? targetDegrees : degrees + Math.sign(targetDegrees - degrees) * stepDegrees;
    this.tilt.degrees = next;
    this.table.setNextKinematicRotation(multiply(axisRotation(axis, (next * Math.PI) / 180), this.tableLevel));
  }

  private addObject(spec: SceneObject): void {
    const [x, y, z] = spec.pose.position;
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(x, y, z)
        .setRotation(toRotation(spec.pose.rotation_quat))
        .setAngularDamping(ANGULAR_DAMPING)
        .setCcdEnabled(true), // fast throws don't tunnel through the table
    );
    const [dx, dy, dz] = spec.dimensions_m.map((d) => Math.max(d, 0.005));
    // Rapier accepts hulls whose points all lie in one plane; treat a nearly flat one as missing.
    const solid = (created: Collider[]) => created.reduce((sum, c) => sum + c.volume(), 0) >= 0.01 * dx * dy * dz;
    const attach = (descs: (ColliderDesc | null)[]): Collider[] | null => {
      if (descs.length === 0 || descs.some((d) => d === null)) return null;
      const created = descs.map((d) => this.world.createCollider(d!, body));
      if (solid(created)) return created;
      for (const c of created) this.world.removeCollider(c, false);
      return null;
    };

    let shape: SimBody['shape'] = 'as-specified';
    let created = attach(colliderShapes(spec.collider, this.scene.hullParts.get(spec.id), this.scene.hullPoints.get(spec.id)));
    if (!created && spec.collider.type !== 'convex_hull') {
      const hull = this.scene.hullPoints.get(spec.id);
      created = hull ? attach([RAPIER.ColliderDesc.convexHull(hull)]) : null;
      if (created) shape = 'hull-fallback';
    }
    if (!created) {
      if (spec.collider.type === 'convex_hull' || spec.collider.type === 'convex_parts') {
        this.warnings.push(`${spec.id} (${spec.label}): its collision shape was missing or flat, so a box stands in`);
      }
      created = [this.world.createCollider(RAPIER.ColliderDesc.cuboid(dx / 2, dy / 2, dz / 2), body)];
      shape = 'box-fallback';
    }
    for (const c of created) {
      c.setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max);
      c.setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max);
    }
    applyPhysics(created, spec);
    body.recomputeMassPropertiesFromColliders();
    this.bodies.push({ spec, body, shape });
  }
}

/**
 * The table top: the scanned outline when the scene has one, else a rectangle. The outline is a
 * closed triangle mesh (a slab) with Rapier's internal-edge fix. A convex slab didn't work: Rapier
 * builds contacts from at most four corners of a face, so a top with the outline's dozens of
 * corners held objects up by only part of it, and they sank in and wandered. Separate wedges
 * didn't either: objects snagged on the seams between them.
 */
function tableColliders(
  polygon: [number, number][] | null,
  size: [number, number],
  point: Vec3,
  rotation: XYZW,
): { descs: ColliderDesc[]; bounds: TableBounds } {
  const [px, , pz] = point;
  if (polygon) {
    const back = conjugate(rotation);
    const desc = slabMesh(polygon.map(([x, z]) => rotate(back, { x: x - px, y: 0, z: z - pz })));
    if (desc) {
      const xs = polygon.map(([x]) => x);
      const zs = polygon.map(([, z]) => z);
      return { descs: [desc], bounds: { minX: Math.min(...xs), maxX: Math.max(...xs), minZ: Math.min(...zs), maxZ: Math.max(...zs) } };
    }
  }
  const [sx, sz] = size;
  return {
    descs: [RAPIER.ColliderDesc.cuboid(sx / 2, TABLE_THICKNESS_M / 2, sz / 2).setTranslation(0, -TABLE_THICKNESS_M / 2, 0)],
    bounds: { minX: px - sx / 2, maxX: px + sx / 2, minZ: pz - sz / 2, maxZ: pz + sz / 2 },
  };
}

/**
 * A closed slab TABLE_THICKNESS_M thick under the outline `corners` (table space, top at y = 0),
 * as a triangle mesh with every face turned outward: a fan around the middle on top and bottom,
 * and a quad down each side.
 */
function slabMesh(corners: { x: number; z: number }[]): ColliderDesc | null {
  const n = corners.length;
  const middle = { x: corners.reduce((sum, c) => sum + c.x, 0) / n, z: corners.reduce((sum, c) => sum + c.z, 0) / n };
  const ring = [middle, ...corners];
  // Vertices 0..n on top (0 is the middle), n+1..2n+1 the same points underneath.
  const vertices = new Float32Array([
    ...ring.flatMap(({ x, z }) => [x, 0, z]),
    ...ring.flatMap(({ x, z }) => [x, -TABLE_THICKNESS_M, z]),
  ]);
  const below = n + 1;
  const indices: number[] = [];
  const at = (i: number) => ({ x: vertices[3 * i], y: vertices[3 * i + 1], z: vertices[3 * i + 2] });
  /** Add triangle a-b-c facing `outward` (flipping it if needed). */
  const triangle = (a: number, b: number, c: number, outward: { x: number; y: number; z: number }) => {
    const [pa, pb, pc] = [at(a), at(b), at(c)];
    const u = { x: pb.x - pa.x, y: pb.y - pa.y, z: pb.z - pa.z };
    const v = { x: pc.x - pa.x, y: pc.y - pa.y, z: pc.z - pa.z };
    const normal = { x: u.y * v.z - u.z * v.y, y: u.z * v.x - u.x * v.z, z: u.x * v.y - u.y * v.x };
    const facing = normal.x * outward.x + normal.y * outward.y + normal.z * outward.z;
    indices.push(...(facing >= 0 ? [a, b, c] : [a, c, b]));
  };
  for (let i = 1; i <= n; i++) {
    const next = (i % n) + 1;
    triangle(0, i, next, { x: 0, y: 1, z: 0 });
    triangle(below, below + i, below + next, { x: 0, y: -1, z: 0 });
    const side = { x: (corners[i - 1].x + corners[next - 1].x) / 2 - middle.x, y: 0, z: (corners[i - 1].z + corners[next - 1].z) / 2 - middle.z };
    triangle(i, next, below + next, side);
    triangle(i, below + next, below + i, side);
  }
  return RAPIER.ColliderDesc.trimesh(
    vertices,
    new Uint32Array(indices),
    RAPIER.TriMeshFlags.FIX_INTERNAL_EDGES | RAPIER.TriMeshFlags.ORIENTED,
  );
}

function colliderShapes(collider: ColliderSpec, parts: Float32Array[] | undefined, hull: Float32Array | undefined): (ColliderDesc | null)[] {
  switch (collider.type) {
    case 'box':
      return [RAPIER.ColliderDesc.cuboid(...collider.half_extents)];
    case 'sphere':
      return [RAPIER.ColliderDesc.ball(collider.radius)];
    case 'cylinder':
      return [RAPIER.ColliderDesc.cylinder(collider.half_height, collider.radius)];
    case 'convex_hull':
      return [hull ? RAPIER.ColliderDesc.convexHull(hull) : null];
    case 'convex_parts':
      return parts ? parts.map((points) => RAPIER.ColliderDesc.convexHull(points)) : [];
  }
}

function colliders(body: RigidBody): Collider[] {
  return Array.from({ length: body.numColliders() }, (_, i) => body.collider(i));
}

/** Give a body's colliders the object's friction and bounciness, and share its mass out by volume. */
function applyPhysics(parts: Collider[], spec: SceneObject): void {
  const volumes = parts.map((c) => c.volume());
  const total = volumes.reduce((sum, v) => sum + v, 0);
  parts.forEach((c, i) => {
    c.setMass(total > 0 ? (spec.mass_kg * volumes[i]) / total : spec.mass_kg / parts.length);
    c.setFriction(spec.friction);
    c.setRestitution(spec.restitution);
  });
}

function limitSpeed(body: RigidBody, maxSpeed: number): void {
  const v = body.linvel();
  const speed = Math.hypot(v.x, v.y, v.z);
  if (speed > maxSpeed) {
    const scale = maxSpeed / speed;
    body.setLinvel({ x: v.x * scale, y: v.y * scale, z: v.z * scale }, true);
  }
}

// ---------- Small vector helpers ----------

export function toRotation([x, y, z, w]: Quat): XYZW {
  const length = Math.hypot(x, y, z, w);
  return { x: x / length, y: y / length, z: z / length, w: w / length };
}

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}

/** The shortest rotation that turns +Y into the unit vector `up`. */
export function rotationFromUp([x, y, z]: Vec3): XYZW {
  if (y < -0.999999) return { x: 1, y: 0, z: 0, w: 0 };
  // Half-way quaternion: axis (0,1,0) × up = (z, 0, -x), with w = 1 + cos(angle).
  const w = 1 + y;
  const length = Math.hypot(z, x, w);
  return { x: z / length, y: 0, z: -x / length, w: w / length };
}

/** Rotation by `radians` about the world x, y or z axis. */
export function axisRotation(axis: 'x' | 'y' | 'z', radians: number): XYZW {
  const s = Math.sin(radians / 2);
  const c = Math.cos(radians / 2);
  return { x: axis === 'x' ? s : 0, y: axis === 'y' ? s : 0, z: axis === 'z' ? s : 0, w: c };
}

function subtract(a: XYZ, b: XYZ): XYZ {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function conjugate(q: XYZW): XYZW {
  return { x: -q.x, y: -q.y, z: -q.z, w: q.w };
}

/** Quaternion product a·b (apply b, then a). */
function multiply(a: XYZW, b: XYZW): XYZW {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

/** Rotate vector `v` by unit quaternion `q`. */
function rotate(q: XYZW, v: XYZ): XYZ {
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}
