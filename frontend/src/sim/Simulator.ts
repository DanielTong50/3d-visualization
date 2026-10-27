import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import type { WhatIfAction } from './actions';
import { materialChanges } from './materials';
import {
  axisRotation,
  initPhysicsEngine,
  PHYSICS_HZ,
  PhysicsWorld,
  rotationFromUp,
  TABLE_THICKNESS_M,
  type GravityPreset,
  type SimBody,
} from './physics';
import { BACKGROUND, loadScene, type LoadedScene, type SceneObject } from './scene';

export interface SimulatorCallbacks {
  onSelect: (object: SceneObject | null) => void;
  onFps: (fps: number) => void;
}

const MAX_STEPS_PER_FRAME = 8; // if the device can't keep up, slow the simulation down instead of freezing
const TAP_MAX_MOVE_PX = 8;
const TAP_MAX_MS = 350;
const CARRY_CLEARANCE_M = 0.03; // a carried object floats this far above whatever is under it
const CARRY_REACH_M = 0.4; // how far past the table edge an object can be carried
const DOWN = new THREE.Vector3(0, -1, 0);
const DEFAULT_OBJECT_COLOR = '#9aa5b1';
const DEFAULT_TABLE_COLOR = '#a8774f';
const BACKGROUND_COLOR = '#e7eaee';
const SLOW_MOTION = 0.25; // fraction of real time
const ACTION_GAP_MS = 400; // pause between "what if" actions so each one can be seen

interface PointerDown {
  id: number;
  x: number;
  y: number;
  time: number;
  moved: number;
}

interface Holding {
  pointerId: number;
  index: number;
  /**
   * Horizontal plane at the grabbed point's starting height: where the pointer ray crosses it gives
   * the object's spot. It stays at that height even when the object rises over something, because
   * moving it shifted the spot toward the camera and made the object lurch.
   */
  plane: THREE.Plane;
  /** How high the grabbed point sits above the object's lowest point. */
  heightAboveBottom: number;
  /** The object's horizontal extent, relative to the grabbed point. */
  footprint: { minX: number; maxX: number; minZ: number; maxZ: number };
  /** False until the pointer moves, so a tap leaves the object where it is. */
  carrying: boolean;
}

/**
 * The 3D view of a scene: draws it with three.js, runs the physics at a fixed rate,
 * and turns pointer input (mouse or touch) into orbiting, grabbing, throwing and tapping.
 */
export class Simulator {
  static async create(container: HTMLElement, sceneDir: string, callbacks: SimulatorCallbacks): Promise<Simulator> {
    const [scene] = await Promise.all([loadScene(sceneDir), initPhysicsEngine()]);
    return new Simulator(container, scene, callbacks);
  }

  readonly scene: LoadedScene;
  readonly warnings: string[];
  private readonly physics: PhysicsWorld;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly view = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly downRaycaster = new THREE.Raycaster();
  // One entry per object, in the same order as physics.bodies:
  private readonly objectGroups: THREE.Group[] = []; // moved by the physics each frame
  private readonly pickMeshes: THREE.Mesh[] = []; // the collider shape; what taps and drags hit
  private readonly highlighters: ((on: boolean) => void)[] = [];
  /** The table and scanned background; turns about the table's pivot when the table tilts. */
  private readonly tiltRoot = new THREE.Group();
  private timeScale = 1;
  private actionTimers: number[] = [];
  private readonly resizeObserver: ResizeObserver;
  private readonly framing: { center: THREE.Vector3; halfWidth: number; halfDepth: number };
  private cameraMovedByUser = false;
  private selected: number | null = null;
  private down: PointerDown | null = null;
  private holding: Holding | null = null;
  private accumulator = 0;
  private lastFrameTime = -1;
  private fpsFrames = 0;
  private fpsWindowStart = -1;
  private disposed = false;

  private constructor(
    private readonly container: HTMLElement,
    scene: LoadedScene,
    private readonly callbacks: SimulatorCallbacks,
  ) {
    this.scene = scene;
    this.physics = new PhysicsWorld(scene);
    this.warnings = [...scene.warnings, ...this.physics.warnings];

    // Sharp phone screens don't need antialiasing, and skipping it saves the GPU work.
    this.renderer = new THREE.WebGLRenderer({ antialias: window.devicePixelRatio < 2 });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.canvas.style.touchAction = 'none'; // our handlers own every touch gesture on the canvas
    container.appendChild(this.canvas);

    this.view.background = new THREE.Color(BACKGROUND_COLOR);
    this.framing = this.objectsExtent();
    this.addLights(this.framing.center, Math.max(this.framing.halfWidth, this.framing.halfDepth));
    this.addBackground();
    for (const simBody of this.physics.bodies) this.addObject(simBody);

    // Register our pointer handlers before OrbitControls creates its own, so that a press
    // on an object can switch orbiting off before OrbitControls sees it.
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    this.canvas.addEventListener('pointermove', this.onPointerMove);
    this.canvas.addEventListener('pointerup', this.onPointerUp);
    this.canvas.addEventListener('pointercancel', this.onPointerUp);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.copy(this.framing.center);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(84); // never look from under the table
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = 4;
    this.controls.addEventListener('start', () => (this.cameraMovedByUser = true));

    this.resizeObserver = new ResizeObserver(this.resize);
    this.resizeObserver.observe(container);
    this.resize();
    if (new URLSearchParams(window.location.search).has('debug')) this.exposeDebugHooks();
    this.renderer.setAnimationLoop(this.frame);
  }

  /** Put every object back where it started, on a level table. */
  reset(): void {
    this.cancelActions();
    this.holding = null;
    this.controls.enabled = true;
    this.physics.reset();
  }

  setGravity(preset: GravityPreset): void {
    this.physics.setGravity(preset);
  }

  setSlowMotion(on: boolean): void {
    this.timeScale = on ? SLOW_MOTION : 1;
  }

  /** Remake an object in another material (same size); returns its updated description. */
  changeMaterial(id: string, material: string): SceneObject {
    const spec = this.physics.bodies.find((b) => b.spec.id === id)?.spec;
    if (!spec) throw new Error(`no object "${id}"`);
    return this.updateObject(id, materialChanges(spec, material));
  }

  /** What a "what if" request needs to know about each object. */
  objectSummaries(): { id: string; label: string; material: string; mass_kg: number; position: [number, number, number] }[] {
    return this.physics.bodies.map(({ spec, body }) => {
      const p = body.translation();
      const round = (v: number) => Math.round(v * 1000) / 1000;
      return { id: spec.id, label: spec.label, material: spec.material, mass_kg: spec.mass_kg, position: [round(p.x), round(p.y), round(p.z)] };
    });
  }

  /** Play checked "what if" actions one after another, a moment apart. */
  runActions(actions: WhatIfAction[]): void {
    this.cancelActions();
    actions.forEach((action, i) => {
      this.actionTimers.push(window.setTimeout(() => this.runAction(action), i * ACTION_GAP_MS));
    });
  }

  private runAction(action: WhatIfAction): void {
    switch (action.name) {
      case 'apply_impulse': {
        const [x, y, z] = action.direction;
        const length = Math.hypot(x, y, z);
        const s = action.speed / length;
        this.physics.push(action.objectId, { x: x * s, y: y * s, z: z * s });
        return;
      }
      case 'set_property':
        this.updateObject(action.objectId, { [action.property]: action.value });
        return;
      case 'set_material':
        this.changeMaterial(action.objectId, action.material);
        return;
      case 'bump_table': {
        const [x, y, z] = action.direction;
        this.physics.bumpTable({ x, y, z }, action.strength);
        return;
      }
      case 'tilt_table':
        this.physics.tiltTable(action.axis, action.degrees);
        return;
      case 'set_gravity':
        this.physics.setGravity(action.preset);
        return;
      case 'drop_object':
        this.physics.dropObject(action.objectId, action.height);
        return;
    }
  }

  private cancelActions(): void {
    for (const timer of this.actionTimers) window.clearTimeout(timer);
    this.actionTimers = [];
  }

  private updateObject(id: string, changes: Partial<SceneObject>): SceneObject {
    const spec = this.physics.updateObject(id, changes);
    const index = this.physics.bodies.findIndex((b) => b.spec.id === id);
    if (index === this.selected) this.callbacks.onSelect(spec); // refresh the open info card
    return spec;
  }

  /** Highlight an object and report it to the info card; null clears the selection. */
  select(id: string | null): void {
    const index = id === null ? -1 : this.physics.bodies.findIndex((b) => b.spec.id === id);
    this.setSelected(index >= 0 ? index : null);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.cancelActions();
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointercancel', this.onPointerUp);
    this.controls.dispose();
    this.view.traverse((node) => {
      if (!(node instanceof THREE.Mesh || node instanceof THREE.Points)) return;
      node.geometry.dispose();
      const material = node.material as THREE.Material & { map?: THREE.Texture | null };
      material.map?.dispose();
      material.dispose();
    });
    this.renderer.dispose();
    this.canvas.remove();
    this.physics.dispose();
  }

  private get canvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  // ---------- Building the 3D scene ----------

  private addLights(center: THREE.Vector3, radius: number): void {
    this.view.add(new THREE.HemisphereLight('#ffffff', '#8d7b68', 1.6));
    const sun = new THREE.DirectionalLight('#ffffff', 2.2);
    sun.position.copy(center).add(new THREE.Vector3(0.6, 1.5, 0.8));
    sun.target.position.copy(center);
    sun.castShadow = true;
    const touchDevice = window.matchMedia('(pointer: coarse)').matches;
    sun.shadow.mapSize.set(touchDevice ? 1024 : 2048, touchDevice ? 1024 : 2048);
    const extent = radius + 0.4;
    Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.1, far: 5 });
    sun.shadow.bias = -0.0005;
    this.view.add(sun, sun.target);
  }

  /**
   * A scanned background (which already shows the table), or a plain table on legs. Both hang
   * from `tiltRoot`, which sits at the table's pivot, so tilting it tilts the table.
   */
  private addBackground(): void {
    const { table_plane: plane, table_color, floor_depth_m, visual } = this.scene.manifest.background;
    const [px, py, pz] = plane.point;
    this.tiltRoot.position.set(px, py, pz);
    this.view.add(this.tiltRoot);
    const scan = this.scene.pointClouds.get(BACKGROUND);
    if (scan && visual) {
      const cloud = pointCloud(scan, visual.point_size_m); // world coordinates
      cloud.position.set(-px, -py, -pz);
      this.tiltRoot.add(cloud);
    }

    const group = new THREE.Group(); // table space: origin on the table top, +Y along its normal
    const { x, y, z, w } = rotationFromUp(plane.normal);
    group.quaternion.set(x, y, z, w);
    this.tiltRoot.add(group);
    const floorGroup = group.clone(); // the floor stays put when the table tilts
    floorGroup.position.set(px, py, pz);
    this.view.add(floorGroup);
    const [sizeX, sizeZ] = plane.size_m;
    const hasFloor = typeof floor_depth_m === 'number';
    if (!scan) {
      const wood = new THREE.MeshStandardMaterial({ color: table_color ?? DEFAULT_TABLE_COLOR, roughness: 0.8 });
      const top = new THREE.Mesh(new THREE.BoxGeometry(sizeX, TABLE_THICKNESS_M, sizeZ), wood);
      top.position.y = -TABLE_THICKNESS_M / 2;
      top.receiveShadow = true;
      group.add(top);
      if (hasFloor) {
        const legHeight = floor_depth_m - TABLE_THICKNESS_M;
        for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
          const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, legHeight, 0.05), wood);
          leg.position.set(sx * (sizeX / 2 - 0.06), -TABLE_THICKNESS_M - legHeight / 2, sz * (sizeZ / 2 - 0.06));
          leg.castShadow = true;
          group.add(leg);
        }
      }
    }
    if (hasFloor) {
      // A plain floor under a plain table; under a scan, only the shadows of fallen objects show.
      const material = scan
        ? new THREE.ShadowMaterial({ opacity: 0.25 })
        : new THREE.MeshStandardMaterial({ color: '#cfd4da', roughness: 1 });
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), material);
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -floor_depth_m;
      floor.receiveShadow = true;
      floorGroup.add(floor);
    }
  }

  /** An object is drawn as its scanned point cloud if it has one, otherwise as its collider shape. */
  private addObject({ spec, boxFallback }: SimBody): void {
    const group = new THREE.Group();
    const shape = this.colliderGeometry(spec, boxFallback);
    const cloud = this.scene.pointClouds.get(spec.id);
    let pickMesh: THREE.Mesh;
    if (cloud && spec.visual) {
      const points = pointCloud(cloud, spec.visual.point_size_m);
      const material = points.material as THREE.PointsMaterial;
      const baseSize = material.size;
      group.add(points);
      // Point clouds are too dense to hit-test, so an invisible copy of the collider takes the taps.
      pickMesh = new THREE.Mesh(shape, new THREE.MeshBasicMaterial({ visible: false }));
      this.highlighters.push((on) => {
        material.size = on ? baseSize * 1.6 : baseSize;
        material.color.set(on ? '#fff3b0' : '#ffffff');
      });
    } else {
      const color = spec.color ?? DEFAULT_OBJECT_COLOR;
      const isBall = spec.collider.type === 'sphere' && !boxFallback;
      const material = new THREE.MeshStandardMaterial({
        color: isBall ? '#ffffff' : color,
        map: isBall ? stripedTexture(color) : null, // stripes make rolling visible
        roughness: 0.55,
        flatShading: spec.collider.type === 'convex_hull',
      });
      pickMesh = new THREE.Mesh(shape, material);
      pickMesh.castShadow = true;
      pickMesh.receiveShadow = true;
      this.highlighters.push((on) => {
        material.emissive.set(on ? '#ffffff' : '#000000');
        material.emissiveIntensity = on ? 0.22 : 0;
      });
    }
    group.add(pickMesh);
    this.objectGroups.push(group);
    this.pickMeshes.push(pickMesh);
    this.view.add(group);
  }

  private colliderGeometry(spec: SceneObject, boxFallback: boolean): THREE.BufferGeometry {
    const collider = spec.collider;
    if (boxFallback) return new THREE.BoxGeometry(...spec.dimensions_m.map((d) => Math.max(d, 0.005)));
    switch (collider.type) {
      case 'box': {
        const [hx, hy, hz] = collider.half_extents;
        return new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
      }
      case 'sphere':
        return new THREE.SphereGeometry(collider.radius, 32, 16);
      case 'cylinder':
        return new THREE.CylinderGeometry(collider.radius, collider.radius, 2 * collider.half_height, 32);
      case 'convex_hull': {
        const values = this.scene.hullPoints.get(spec.id) ?? new Float32Array();
        const points: THREE.Vector3[] = [];
        for (let i = 0; i < values.length; i += 3) points.push(new THREE.Vector3(values[i], values[i + 1], values[i + 2]));
        return new ConvexGeometry(points);
      }
    }
  }

  /** The area the objects cover on the table (with a margin), to frame them on screen. */
  private objectsExtent(): { center: THREE.Vector3; halfWidth: number; halfDepth: number } {
    const tableTop = this.scene.manifest.background.table_plane.point;
    let [minX, maxX, minZ, maxZ] = [tableTop[0] - 0.2, tableTop[0] + 0.2, tableTop[2] - 0.15, tableTop[2] + 0.15];
    for (const { spec } of this.physics.bodies) {
      const [x, , z] = spec.pose.position;
      const half = Math.max(...spec.dimensions_m) / 2;
      [minX, maxX, minZ, maxZ] = [Math.min(minX, x - half), Math.max(maxX, x + half), Math.min(minZ, z - half), Math.max(maxZ, z + half)];
    }
    const margin = 0.06;
    return {
      center: new THREE.Vector3((minX + maxX) / 2, tableTop[1] + 0.05, (minZ + maxZ) / 2),
      halfWidth: (maxX - minX) / 2 + margin,
      halfDepth: (maxZ - minZ) / 2 + margin,
    };
  }

  /**
   * Frame the objects. Tall phone screens get a wider lens and a more top-down view, and the
   * camera looks along whichever table axis lines up the objects' longer spread with the
   * screen's longer side.
   */
  private placeCamera(): void {
    const portrait = this.camera.aspect < 1;
    this.camera.fov = portrait ? 60 : 45;
    this.camera.updateProjectionMatrix();
    const tanVertical = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanHorizontal = tanVertical * this.camera.aspect;
    const { halfWidth, halfDepth } = this.framing;
    const fromSide = portrait === halfWidth > halfDepth;
    const [across, away] = fromSide ? [halfDepth, halfWidth] : [halfWidth, halfDepth];
    const distance = Math.max(across / tanHorizontal, away / tanVertical);
    const height = portrait ? 1.3 : 0.8;
    const direction = (fromSide ? new THREE.Vector3(1, height, 0) : new THREE.Vector3(0, height, 1)).normalize();
    this.controls.target.copy(this.framing.center);
    this.camera.position.copy(this.framing.center).addScaledVector(direction, distance);
    this.controls.update();
  }

  private resize = (): void => {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (!this.cameraMovedByUser) this.placeCamera(); // e.g. the phone was rotated
  };

  // ---------- Each frame ----------

  private frame = (time: number): void => {
    const dt = this.lastFrameTime < 0 ? 0 : Math.min((time - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = time;
    this.accumulator += dt * this.timeScale;
    const step = 1 / PHYSICS_HZ;
    let steps = 0;
    while (this.accumulator >= step && steps < MAX_STEPS_PER_FRAME) {
      this.physics.step();
      this.accumulator -= step;
      steps++;
    }
    if (steps === MAX_STEPS_PER_FRAME) this.accumulator = 0;

    this.physics.bodies.forEach(({ body }, i) => {
      const p = body.translation();
      const q = body.rotation();
      this.objectGroups[i].position.set(p.x, p.y, p.z);
      this.objectGroups[i].quaternion.set(q.x, q.y, q.z, q.w);
    });
    const tilt = this.physics.tableTilt();
    const turn = axisRotation(tilt.axis, THREE.MathUtils.degToRad(tilt.degrees));
    this.tiltRoot.quaternion.set(turn.x, turn.y, turn.z, turn.w);
    this.controls.update();
    this.renderer.render(this.view, this.camera);
    this.countFrame(time);
  };

  private countFrame(time: number): void {
    if (this.fpsWindowStart < 0) this.fpsWindowStart = time;
    this.fpsFrames++;
    const elapsed = time - this.fpsWindowStart;
    if (elapsed >= 500) {
      this.callbacks.onFps(Math.round((this.fpsFrames * 1000) / elapsed));
      this.fpsFrames = 0;
      this.fpsWindowStart = time;
    }
  }

  // ---------- Pointer input ----------

  private onPointerDown = (event: PointerEvent): void => {
    this.down = { id: event.pointerId, x: event.clientX, y: event.clientY, time: performance.now(), moved: 0 };
    if (this.holding || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const hit = this.pickObject(event);
    if (!hit) return; // empty space: OrbitControls orbits the camera
    this.controls.enabled = false;
    this.canvas.setPointerCapture(event.pointerId);
    const bounds = new THREE.Box3().setFromObject(this.pickMeshes[hit.index]);
    this.holding = {
      pointerId: event.pointerId,
      index: hit.index,
      plane: new THREE.Plane(new THREE.Vector3(0, 1, 0), -hit.point.y),
      heightAboveBottom: hit.point.y - bounds.min.y,
      footprint: {
        minX: bounds.min.x - hit.point.x,
        maxX: bounds.max.x - hit.point.x,
        minZ: bounds.min.z - hit.point.z,
        maxZ: bounds.max.z - hit.point.z,
      },
      carrying: false,
    };
    this.physics.grabAt(this.physics.bodies[hit.index].body, hit.point);
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (this.down?.id === event.pointerId) {
      this.down.moved = Math.max(this.down.moved, Math.hypot(event.clientX - this.down.x, event.clientY - this.down.y));
    }
    const holding = this.holding;
    if (!holding || event.pointerId !== holding.pointerId) return;
    if (!holding.carrying && (this.down?.moved ?? 0) < TAP_MAX_MOVE_PX) return;
    holding.carrying = true;
    // Carry the object across the table under the pointer, floating just above whatever is
    // beneath it, so it can be dragged around, lifted onto other objects, and flicked.
    this.aimRay(event);
    const spot = this.raycaster.ray.intersectPlane(holding.plane, new THREE.Vector3());
    if (!spot) return;
    this.clampToReach(spot);
    spot.y = this.surfaceHeightUnder(spot, holding) + holding.heightAboveBottom + CARRY_CLEARANCE_M;
    this.physics.moveGrabTarget(spot);
  };

  private onPointerUp = (event: PointerEvent): void => {
    const down = this.down?.id === event.pointerId ? this.down : null;
    if (down) this.down = null;
    const tap =
      event.type === 'pointerup' && down !== null && down.moved < TAP_MAX_MOVE_PX && performance.now() - down.time < TAP_MAX_MS;
    if (this.holding && event.pointerId === this.holding.pointerId) {
      const { index } = this.holding;
      this.holding = null;
      this.physics.release();
      this.controls.enabled = true;
      if (tap) this.setSelected(index);
    } else if (tap && !this.holding) {
      this.setSelected(null); // tapped empty space
    }
  };

  private aimRay(event: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
  }

  private pickObject(event: PointerEvent): { index: number; point: THREE.Vector3 } | null {
    this.aimRay(event);
    const [hit] = this.raycaster.intersectObjects(this.pickMeshes, false);
    return hit ? { index: this.pickMeshes.indexOf(hit.object as THREE.Mesh), point: hit.point.clone() } : null;
  }

  /**
   * The highest surface (table or another object's top) under the held object's footprint if it
   * were moved to `spot`, sampled on a 3×3 grid, so it rises before its edge reaches anything
   * taller. The grid covers the middle 70% of the footprint's bounding box: the corners of a box
   * around a round mug are empty air, and sampling them lifted mugs over neighbors they weren't
   * touching. Assumes the table is level.
   */
  private surfaceHeightUnder(spot: THREE.Vector3, holding: Holding): number {
    const tableTop = this.scene.manifest.background.table_plane.point[1];
    const others = this.pickMeshes.filter((_, i) => i !== holding.index);
    const { minX, maxX, minZ, maxZ } = holding.footprint;
    let highest = tableTop;
    for (const fx of [0.15, 0.5, 0.85]) {
      for (const fz of [0.15, 0.5, 0.85]) {
        const x = spot.x + minX + fx * (maxX - minX);
        const z = spot.z + minZ + fz * (maxZ - minZ);
        this.downRaycaster.set(new THREE.Vector3(x, tableTop + 10, z), DOWN);
        const [hit] = this.downRaycaster.intersectObjects(others, false);
        if (hit) highest = Math.max(highest, hit.point.y);
      }
    }
    return highest;
  }

  /** Keep carried objects near the table, even if the pointer ray heads toward the horizon. */
  private clampToReach(spot: THREE.Vector3): void {
    const { point, size_m } = this.scene.manifest.background.table_plane;
    const [halfX, halfZ] = [size_m[0] / 2, size_m[1] / 2];
    spot.x = THREE.MathUtils.clamp(spot.x, point[0] - halfX - CARRY_REACH_M, point[0] + halfX + CARRY_REACH_M);
    spot.z = THREE.MathUtils.clamp(spot.z, point[2] - halfZ - CARRY_REACH_M, point[2] + halfZ + CARRY_REACH_M);
  }

  private setSelected(index: number | null): void {
    if (this.selected !== null) this.highlighters[this.selected](false);
    this.selected = index;
    if (index !== null) this.highlighters[index](true);
    this.callbacks.onSelect(index === null ? null : this.physics.bodies[index].spec);
  }

  /** With ?debug in the URL, browser tests can read object poses and screen locations. */
  private exposeDebugHooks(): void {
    const project = (point: THREE.Vector3) => {
      const ndc = point.clone().project(this.camera);
      const rect = this.canvas.getBoundingClientRect();
      return [rect.left + ((ndc.x + 1) / 2) * rect.width, rect.top + ((1 - ndc.y) / 2) * rect.height];
    };
    const bodyOf = (id: string) => this.physics.bodies.find((b) => b.spec.id === id)?.body;
    (window as unknown as Record<string, unknown>).__scan2sim = {
      positions: () =>
        Object.fromEntries(
          this.physics.bodies.map(({ spec, body }) => {
            const p = body.translation();
            return [spec.id, [p.x, p.y, p.z]];
          }),
        ),
      /** Degrees between the object's up axis and world up. */
      tilt: (id: string) => {
        const q = bodyOf(id)?.rotation();
        if (!q) return null;
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(new THREE.Quaternion(q.x, q.y, q.z, q.w));
        return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(up.y, -1, 1)));
      },
      screenPoint: (id: string) => {
        const p = bodyOf(id)?.translation();
        return p ? project(new THREE.Vector3(p.x, p.y, p.z)) : null;
      },
      project: (x: number, y: number, z: number) => project(new THREE.Vector3(x, y, z)),
      run: (actions: WhatIfAction[]) => this.runActions(actions),
      tableTilt: () => this.physics.tableTilt().degrees,
    };
  }
}

/** A scanned point cloud: each point keeps its captured color and shrinks with distance. */
function pointCloud(geometry: THREE.BufferGeometry, pointSizeM: number): THREE.Points {
  const material = new THREE.PointsMaterial({
    size: pointSizeM,
    sizeAttenuation: true,
    vertexColors: geometry.hasAttribute('color'),
  });
  return new THREE.Points(geometry, material);
}

/** A white sphere texture with colored stripes, so a rolling ball visibly rolls. */
function stripedTexture(color: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context) {
    for (let i = 0; i < 8; i++) {
      context.fillStyle = i % 2 === 0 ? color : '#ffffff';
      context.fillRect(i * 16, 0, 16, 64);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
