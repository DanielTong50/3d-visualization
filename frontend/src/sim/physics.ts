import RAPIER from '@dimforge/rapier3d-compat';
import type { ColliderDesc, RigidBody, World } from '@dimforge/rapier3d-compat';
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

// A little angular damping stands in for rolling resistance, so balls come to rest.
const ANGULAR_DAMPING = 0.4;

// A held object's weight is cancelled, and a spring on its center of mass pulls it toward the
// pointer. The stiffness is the same for every object, so heavy objects follow more slowly and
// feel heavy. (Pulling at the grabbed point instead made light objects held by an edge oscillate
// and fly off: the effective mass there is much smaller than the object's mass.)
const GRAB_STIFFNESS_N_PER_M = 300;
const GRAB_MAX_OMEGA = 50; // rad/s: caps stiffness for light objects so the spring stays stable at 120 Hz
const GRAB_DAMPING_RATIO = 1; // critically damped: follows the pointer without wobbling
const MAX_SPEED_M_S = 6; // fastest a held or thrown object may move
const MAX_PUSH_SPEED_M_S = 8; // fastest a "what if" push may make an object move
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
  /** True when the convex hull was missing or flat, so a box of `dimensions_m` stands in. */
  boxFallback: boolean;
}

interface Grab {
  body: RigidBody;
  /** The body's position relative to the grabbed point, kept fixed while held. */
  offset: XYZ;
  /** Where the grabbed point should go. */
  target: XYZ;
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
  private grab: Grab | null = null;
  private readonly table: RigidBody;
  private readonly tableLevel: { translation: XYZ; rotation: XYZW };
  /** The table tilts about a horizontal axis through this point on its top. */
  private readonly pivot: XYZ;
  private tilt = { axis: 'x' as 'x' | 'z', degrees: 0, targetDegrees: 0 };

  constructor(private readonly scene: LoadedScene) {
    const [gx, gy, gz] = scene.manifest.gravity;
    this.world = new RAPIER.World({ x: gx, y: gy, z: gz });
    this.world.timestep = 1 / PHYSICS_HZ;
    this.world.lengthUnit = 0.1; // objects are about 10 cm: tighter contact tolerances than the 1 m default
    this.world.numSolverIterations = 8; // steadier resting and stacking than the default 4

    const { table_plane: plane, floor_depth_m } = scene.manifest.background;
    const up = normalize(plane.normal);
    const rotation = rotationFromUp(up);
    const [px, py, pz] = plane.point;
    this.pivot = { x: px, y: py, z: pz };
    const [sizeX, sizeZ] = plane.size_m;
    // Table and floor have zero friction and bounciness, so each object's own values
    // decide its contacts with them (objects use the "max" combine rule).
    const slabAt = (depthBelowTop: number, halfThickness: number) => {
      const [x, y, z] = plane.point.map((v, i) => v - up[i] * (depthBelowTop + halfThickness));
      return { x, y, z };
    };
    const tableCenter = slabAt(0, TABLE_THICKNESS_M / 2);
    this.tableLevel = { translation: tableCenter, rotation };
    // Kinematic, so it can be tilted smoothly while objects ride on it; otherwise it never moves.
    this.table = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(tableCenter.x, tableCenter.y, tableCenter.z).setRotation(rotation),
    );
    this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(sizeX / 2, TABLE_THICKNESS_M / 2, sizeZ / 2).setFriction(0).setRestitution(0),
      this.table,
    );
    if (typeof floor_depth_m === 'number') {
      const floorCenter = slabAt(floor_depth_m, 0.05);
      const floor = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed().setTranslation(floorCenter.x, floorCenter.y, floorCenter.z).setRotation(rotation),
      );
      this.world.createCollider(RAPIER.ColliderDesc.cuboid(10, 0.05, 10).setFriction(0).setRestitution(0), floor);
    }
    for (const spec of scene.manifest.objects) this.addObject(spec, scene.hullPoints.get(spec.id));
  }

  step(): void {
    if (this.grab) this.pullTowardPointer(this.grab);
    this.moveTableTowardTilt();
    this.world.step();
  }

  /** Put every object back where the scene file says, at rest, on a level table. */
  reset(): void {
    this.release();
    this.tilt = { axis: 'x', degrees: 0, targetDegrees: 0 };
    this.table.setTranslation(this.tableLevel.translation, true);
    this.table.setRotation(this.tableLevel.rotation, true);
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
   * Pick up `body` by the world-space `point` on its surface; it stays put until the target moves.
   * A held object can't turn (a firm grip), so it never tips while carried past its neighbors.
   */
  grabAt(body: RigidBody, point: XYZ): void {
    this.release();
    this.grab = { body, offset: subtract(body.translation(), point), target: { x: point.x, y: point.y, z: point.z } };
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    body.lockRotations(true, true);
  }

  moveGrabTarget(target: XYZ): void {
    if (this.grab) this.grab.target = { x: target.x, y: target.y, z: target.z };
  }

  /** Let go. The object keeps its current velocity, so a flick throws it. */
  release(): void {
    if (!this.grab) return;
    const { body } = this.grab;
    this.grab = null;
    body.lockRotations(false, true);
    limitSpeed(body, MAX_SPEED_M_S);
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
    const { size_m } = this.scene.manifest.background.table_plane;
    for (const { spec, body } of this.bodies) {
      const p = body.translation();
      const onTable =
        Math.abs(p.x - this.pivot.x) <= size_m[0] / 2 + 0.05 &&
        Math.abs(p.z - this.pivot.z) <= size_m[1] / 2 + 0.05 &&
        p.y >= this.pivot.y - 0.05 &&
        p.y <= this.pivot.y + 0.5;
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
  tableTilt(): { axis: 'x' | 'z'; degrees: number; pivot: XYZ } {
    return { axis: this.tilt.axis, degrees: this.tilt.degrees, pivot: { ...this.pivot } };
  }

  /** Lift an object `height` meters above the table (keeping its spot and turn) and let it fall. */
  dropObject(id: string, height: number): void {
    const { spec, body } = this.find(id);
    const p = body.translation();
    body.setTranslation({ x: p.x, y: this.pivot.y + height + spec.dimensions_m[1] / 2, z: p.z }, true);
    body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  /** Change an object's physical properties (from material editing or a "what if"); returns its new spec. */
  updateObject(id: string, changes: Partial<SceneObject>): SceneObject {
    const simBody = this.find(id);
    const spec = { ...simBody.spec, ...changes };
    const collider = simBody.body.collider(0);
    collider.setMass(spec.mass_kg);
    collider.setFriction(spec.friction);
    collider.setRestitution(spec.restitution);
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
    const turn = axisRotation(axis, (next * Math.PI) / 180);
    const level = this.tableLevel;
    this.table.setNextKinematicRotation(multiply(turn, level.rotation));
    this.table.setNextKinematicTranslation(add(this.pivot, rotate(turn, subtract(level.translation, this.pivot))));
  }

  private pullTowardPointer({ body, offset, target }: Grab): void {
    const mass = body.mass();
    const stiffness = Math.min(GRAB_STIFFNESS_N_PER_M, mass * GRAB_MAX_OMEGA ** 2);
    const damping = 2 * GRAB_DAMPING_RATIO * Math.sqrt(stiffness * mass);
    const goal = add(target, offset);
    const position = body.translation();
    const velocity = body.linvel();
    const gravity = this.world.gravity;
    const dt = this.world.timestep;
    body.applyImpulse(
      {
        x: (stiffness * (goal.x - position.x) - damping * velocity.x - mass * gravity.x) * dt,
        y: (stiffness * (goal.y - position.y) - damping * velocity.y - mass * gravity.y) * dt,
        z: (stiffness * (goal.z - position.z) - damping * velocity.z - mass * gravity.z) * dt,
      },
      true,
    );
    limitSpeed(body, MAX_SPEED_M_S);
  }

  private addObject(spec: SceneObject, hull: Float32Array | undefined): void {
    const [x, y, z] = spec.pose.position;
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(x, y, z)
        .setRotation(toRotation(spec.pose.rotation_quat))
        .setAngularDamping(ANGULAR_DAMPING)
        .setCcdEnabled(true), // fast throws don't tunnel through the table
    );
    const withPhysics = (shape: ColliderDesc) =>
      shape
        .setMass(spec.mass_kg)
        .setFriction(spec.friction)
        .setRestitution(spec.restitution)
        .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max)
        .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max);

    const shape = colliderShape(spec.collider, hull);
    let collider = shape ? this.world.createCollider(withPhysics(shape), body) : null;
    // Rapier accepts a hull of points that all lie in one plane; treat a nearly flat one as missing.
    const [dx, dy, dz] = spec.dimensions_m.map((d) => Math.max(d, 0.005));
    if (collider && spec.collider.type === 'convex_hull' && collider.volume() < 0.01 * dx * dy * dz) {
      this.world.removeCollider(collider, false);
      collider = null;
    }
    const boxFallback = collider === null;
    if (boxFallback) {
      if (hull) this.warnings.push(`${spec.id} (${spec.label}): its convex hull is flat, so a box stands in`);
      this.world.createCollider(withPhysics(RAPIER.ColliderDesc.cuboid(dx / 2, dy / 2, dz / 2)), body);
    }
    this.bodies.push({ spec, body, boxFallback });
  }
}

function colliderShape(collider: ColliderSpec, hull: Float32Array | undefined): ColliderDesc | null {
  switch (collider.type) {
    case 'box':
      return RAPIER.ColliderDesc.cuboid(...collider.half_extents);
    case 'sphere':
      return RAPIER.ColliderDesc.ball(collider.radius);
    case 'cylinder':
      return RAPIER.ColliderDesc.cylinder(collider.half_height, collider.radius);
    case 'convex_hull':
      return hull ? RAPIER.ColliderDesc.convexHull(hull) : null;
  }
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

/** Rotation by `radians` about the world x or z axis. */
export function axisRotation(axis: 'x' | 'z', radians: number): XYZW {
  const s = Math.sin(radians / 2);
  return axis === 'x' ? { x: s, y: 0, z: 0, w: Math.cos(radians / 2) } : { x: 0, y: 0, z: s, w: Math.cos(radians / 2) };
}

function add(a: XYZ, b: XYZ): XYZ {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function subtract(a: XYZ, b: XYZ): XYZ {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
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
