import { SparkRenderer, SplatMesh, type PackedSplats } from '@sparkjsdev/spark';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { ViewHelper } from 'three/addons/helpers/ViewHelper.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import type { WhatIfAction } from './actions';
import { materialChanges } from './materials';
import {
  bodyFrame,
  bracketGeometry,
  gridMaterial,
  INK,
  overlayLine,
  studioBackdrop,
  Trail,
  VelocityArrow,
} from './overlays';
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
import { comMarker, gripperGlyph, inertiaRings, worldFrame } from './robotics';
import { BACKDROP, BACKGROUND, loadScene, type LoadedScene, type SceneObject } from './scene';
import { backgroundVisibility, objectVisibility, patchSplats, tablePatches } from './splatCleanup';

export interface SimulatorCallbacks {
  onSelect: (object: SceneObject | null) => void;
  onFps: (fps: number) => void;
  /** A few times a second: the simulation's clock and state, for the telemetry readout. */
  onTelemetry?: (telemetry: Telemetry) => void;
}

/**
 * What the view draws:
 * - render: the scan as filmed (trained splats), with a mark on the selected object;
 * - debug: the scan plus the instruments: collider wireframes, body frames, velocities, trails, labels;
 * - physics: only what the physics engine sees: colliders in flat clay on a metric grid;
 * - robot: what a robot stack receives (the exported model): colliders, each object's center of
 *   mass and inertia ellipsoid, its grasp candidates, and the +z-up robot world frame.
 */
export type ViewMode = 'render' | 'debug' | 'physics' | 'robot';

export interface Telemetry {
  /** Simulated seconds since the scene loaded or was reset (slow motion slows this clock). */
  time: number;
  steps: number;
  /** Total translational kinetic energy, in joules. */
  energy: number;
  /** Objects moving faster than 1 cm/s. */
  moving: number;
  bodies: number;
  /** Camera azimuth and elevation in degrees, distance to the orbit target in meters. */
  camera: { azimuth: number; elevation: number; distance: number };
  /** The selected object's live state, if one is selected. */
  selected: { id: string; position: [number, number, number]; speed: number } | null;
}

/**
 * How pointer drags on objects act:
 * - move: slide the object along the table by the point you hold (it turns and drags like a real one);
 * - lift: pick it up and carry it, hanging from your finger, over anything in the way; let go to drop or stack;
 * - push: tap to poke the object at that point (heavy objects barely budge).
 */
export type Tool = 'move' | 'lift' | 'push';

export interface SimulatorOptions {
  /** A non-interactive, slowly turning view for the home page. */
  showcase?: boolean;
  viewMode?: ViewMode;
}

// Physics keeps real time down to 5 fps (24 steps of 1/120 s per frame); below that the simulation
// slows down rather than freezing. (At 8 steps it ran at half speed on an 8 fps frame, which made
// gravity look weak.)
const MAX_STEPS_PER_FRAME = 24;
const THROW_WINDOW_MS = 80; // a release throws the object as fast as the hand moved over this long
const TURN_STEP_RAD = Math.PI / 12; // Q / E or one wheel notch turns a carried object 15 degrees
// Adaptive resolution: drop the pixel ratio when frames take longer than SLOW_FRAME_MS, raise it
// again after a long run of fast ones.
const SLOW_FRAME_MS = 1000 / 45;
const FAST_FRAME_MS = 1000 / 58;
const PIXEL_RATIO_STEP = 0.25;
const MIN_PIXEL_RATIO = 0.75;
// The backdrop (the room around the table) is only trained from the filmed side: it fades out as
// the camera leaves the filmed path.
const BACKDROP_FADE_M: [number, number] = [0.12, 0.6];
const MAX_GRASP_GLYPHS = 5;
const TAP_MAX_MOVE_PX = 8;
const TAP_MAX_MS = 350;
const LIFT_CLEARANCE_M = 0.04; // a lifted object floats this far above whatever is under it
const CARRY_REACH_M = 0.4; // how far past the table edge an object can be carried
const DOWN = new THREE.Vector3(0, -1, 0);
const UP = new THREE.Vector3(0, 1, 0);
const DEFAULT_OBJECT_COLOR = '#9aa5b1';
const DEFAULT_TABLE_COLOR = '#7b5a40';
const FLOOR_COLOR = '#2a2d33';
const FLOOR_RADIUS_M = 2.5;
const BACKGROUND_COLOR = '#0a0b0d';
const SLOW_MOTION = 0.25; // fraction of real time
const ACTION_GAP_MS = 400; // pause between "what if" actions so each one can be seen
const SCAN_VIEW_HEIGHT = 0.75; // without a filmed view, a scan is viewed from about 37° above the table
const VIEW_MARGIN_M = 0.02; // room left around the objects when backing off from the filmed view
const SWAY_RADIANS = 0.35; // the showcase sways this far either side of its starting view ...
const SWAY_PERIOD_MS = 16000; // ... and back, this often
const SELECT_TINT = new THREE.Color(1.22, 1.17, 1.0);
const HOVER_TINT = new THREE.Color(1.1, 1.08, 1.02);
const NO_TINT = new THREE.Color(1, 1, 1);
const MOVING_MPS = 0.01;
const TELEMETRY_MS = 200;
const GRID_SIZE_M = 40;

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
  tool: 'move' | 'lift';
  /**
   * Horizontal plane at the grabbed point's starting height: where the pointer ray crosses it gives
   * the object's spot. It stays at that height even when the object rises over something, because
   * moving it shifted the spot toward the camera and made the object lurch.
   */
  plane: THREE.Plane;
  /** The grabbed point's starting height. */
  startY: number;
  /** How high the grabbed point sits above the object's lowest point. */
  heightAboveBottom: number;
  /** The object's horizontal extent, relative to the grabbed point. */
  footprint: { minX: number; maxX: number; minZ: number; maxZ: number };
  /** False until the pointer moves, so a tap leaves the object where it is. */
  carrying: boolean;
}

/** What draws one object, and how to tint it when hovered or selected. */
interface ObjectView {
  group: THREE.Group; // follows the rigid body
  pick: THREE.Mesh; // the collider's shape: what taps and drags hit
  tint: (color: THREE.Color) => void;
  /** A soft dark disc on the table under scanned objects (meshes cast real shadows instead). */
  shadow: SplatMesh | null;
  shadowRadius: number;
  /** The trained splats or point cloud, if the object has them. */
  splats: SplatMesh | null;
  /** The collider in flat clay, for the physics view (null when `pick` is already drawn). */
  clay: THREE.Mesh | null;
  wire: THREE.LineSegments;
  brackets: THREE.LineSegments;
  frame: THREE.LineSegments;
  arrow: VelocityArrow;
  trail: Trail;
  label: CSS2DObject;
  speedText: HTMLElement;
  /** How far the label floats above the body's origin. */
  labelHeight: number;
  /** The robot view's marks: center of mass, inertia ellipsoid, grasp candidates (body frame). */
  robot: THREE.Group;
}

/**
 * The 3D view of a scene: draws it with three.js (meshes) and Spark (Gaussian splats), runs the
 * physics at a fixed rate, and turns pointer input (mouse or touch) into orbiting and the tools.
 */
/** Live simulators that installed the ?debug hooks; the newest one owns them. */
const debugSimulators: Simulator[] = [];

export class Simulator {
  static async create(
    container: HTMLElement,
    sceneDir: string,
    callbacks: SimulatorCallbacks,
    options: SimulatorOptions = {},
  ): Promise<Simulator> {
    const [scene] = await Promise.all([loadScene(sceneDir), initPhysicsEngine()]);
    const simulator = new Simulator(container, scene, callbacks, options);
    // Splat files decode in the background; wait for them so the scene appears complete.
    const decoded = await Promise.allSettled(simulator.splatsLoading);
    for (const result of decoded) {
      if (result.status === 'rejected') simulator.warnings.push(`A splat file couldn't be decoded: ${String(result.reason)}`);
    }
    return simulator;
  }

  readonly scene: LoadedScene;
  readonly warnings: string[];
  private readonly physics: PhysicsWorld;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly labels = new CSS2DRenderer();
  private readonly gizmo: ViewHelper | null;
  private readonly view = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly downRaycaster = new THREE.Raycaster();
  private readonly views: ObjectView[] = []; // same order as physics.bodies
  /** The table and scanned background; turns about the table's pivot when the table tilts. */
  private readonly tiltRoot = new THREE.Group();
  private readonly resizeObserver: ResizeObserver;
  private readonly framing: { center: THREE.Vector3; halfWidth: number; halfDepth: number };
  private readonly interactive: boolean;
  private readonly splatsLoading: Promise<unknown>[] = [];
  /** Parts of the scene each view mode shows or hides. */
  private readonly backgroundParts: { splats: SplatMesh | null; clayTable: THREE.Object3D; tableGrid: THREE.Object3D; outline: THREE.Object3D } = {
    splats: null,
    clayTable: new THREE.Group(),
    tableGrid: new THREE.Group(),
    outline: new THREE.Group(),
  };
  private viewMode: ViewMode;
  /** The surroundings beyond the table, faded by distance from the filmed path. */
  private backdrop: SplatMesh | null = null;
  private readonly capturePositions: THREE.Vector3[];
  private readonly robotWorld = worldFrame();
  private readonly dropMarker = dropRing();
  /** Recent hand positions while holding, for throwing on release. */
  private handTrail: { time: number; point: THREE.Vector3 }[] = [];
  /** Render only when something changed: the camera, a moving body, a finished splat sort. */
  private renderRequested = true;
  private renderedLastFrame = false;
  private pixelRatio: number;
  private readonly maxPixelRatio: number;
  private slowFrames = 0;
  private fastFrames = 0;
  private simTime = 0;
  private simSteps = 0;
  private telemetryAt = -1;
  private tool: Tool = 'move';
  private timeScale = 1;
  private actionTimers: number[] = [];
  private cameraMovedByUser = false;
  /** The showcase's starting camera position, which it sways around (null: no swaying). */
  private swayFrom: THREE.Vector3 | null = null;
  private selected: number | null = null;
  private hovered: number | null = null;
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
    options: SimulatorOptions,
  ) {
    this.scene = scene;
    this.interactive = !options.showcase;
    this.physics = new PhysicsWorld(scene);
    this.warnings = [...scene.warnings, ...this.physics.warnings];

    // Splats render straight to the canvas, so Spark blends them in display (sRGB) space, the space
    // gsplat trained them in. Through the old linear, multisampled HalfFloat render target they came
    // out washed out, and that target cost most of the frame (7.9 fps on an M3 at 2x). MSAA only
    // helps the instrument lines, so only low-density screens get it.
    // Splats are fill-rate bound: start at 1.5x at most and let adaptResolution climb if there's room.
    this.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
    this.pixelRatio = Math.min(this.maxPixelRatio, 1.5);
    this.renderer = new THREE.WebGLRenderer({ antialias: window.devicePixelRatio < 1.5, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.canvas.style.touchAction = 'none'; // our handlers own every touch gesture on the canvas
    container.appendChild(this.canvas);
    this.labels.domElement.className = 'sim-labels';
    container.appendChild(this.labels.domElement);

    this.view.background = studioBackdrop();
    this.view.fog = new THREE.Fog(BACKGROUND_COLOR, 2.2, 7);
    const environment = new THREE.PMREMGenerator(this.renderer);
    this.view.environment = environment.fromScene(new RoomEnvironment(), 0.04).texture;
    this.view.environmentIntensity = 0.35;
    environment.dispose();
    // Scans trained with anti-aliasing render with Spark's defaults; classic 3DGS scans carry their
    // 0.3 px^2 dilation inside each splat instead.
    const classic = this.splatVisuals().some((visual) => !visual.antialiased);
    this.view.add(
      new SparkRenderer({
        renderer: this.renderer,
        onDirty: () => this.requestRender(),
        ...(classic ? { preBlurAmount: 0.3, blurAmount: 0 } : {}),
      }),
    );
    this.capturePositions = (scene.manifest.capture?.cameras ?? []).map(({ position }) => new THREE.Vector3(...position));

    this.framing = this.objectsExtent();
    this.addLights(this.framing.center, Math.max(this.framing.halfWidth, this.framing.halfDepth));
    this.addBackground();
    for (const simBody of this.physics.bodies) this.addObject(simBody);
    this.robotWorld.position.set(...this.scene.manifest.background.table_plane.point);
    this.view.add(this.robotWorld, this.dropMarker);
    this.viewMode = options.viewMode ?? 'render';
    this.applyViewMode();

    // Register our pointer handlers before OrbitControls creates its own, so that a press
    // on an object can switch orbiting off before OrbitControls sees it.
    if (this.interactive) {
      this.canvas.addEventListener('pointerdown', this.onPointerDown);
      this.canvas.addEventListener('pointermove', this.onPointerMove);
      this.canvas.addEventListener('pointerup', this.onPointerUp);
      this.canvas.addEventListener('pointercancel', this.onPointerUp);
      this.canvas.addEventListener('pointerleave', this.onPointerLeave);
      this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    }
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.copy(this.framing.center);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(84); // never look from under the table
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = 4;
    this.controls.addEventListener('start', () => (this.cameraMovedByUser = true));
    this.controls.addEventListener('change', () => this.requestRender());
    if (!this.interactive) {
      this.controls.enabled = false;
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) this.swayFrom = new THREE.Vector3();
    }

    // The orientation gizmo: where world x, y and z point from here.
    this.gizmo = this.interactive ? new ViewHelper(this.camera, this.canvas) : null;
    if (this.gizmo) {
      this.gizmo.location.right = 12;
      this.gizmo.location.bottom = 12;
      this.gizmo.setLabelStyle('500 22px "IBM Plex Mono", ui-monospace, monospace', '#0b0c0e', 15);
    }

    this.resizeObserver = new ResizeObserver(this.resize);
    this.resizeObserver.observe(container);
    this.resize();
    if (this.interactive && new URLSearchParams(window.location.search).has('debug')) this.exposeDebugHooks();
    this.renderer.setAnimationLoop(this.frame);
  }

  /** Put every object back where it started, on a level table. */
  reset(): void {
    this.cancelActions();
    this.holding = null;
    this.controls.enabled = true;
    this.physics.reset();
    this.simTime = 0;
    this.simSteps = 0;
    for (const view of this.views) view.trail.clear();
  }

  setViewMode(mode: ViewMode): void {
    this.viewMode = mode;
    this.applyViewMode();
  }

  /** Turn the object being carried by 15 degrees about the vertical (+1 counter-clockwise from above). */
  turnHeld(direction: 1 | -1): void {
    if (this.holding?.tool !== 'lift') return;
    this.physics.turnHeld(direction * TURN_STEP_RAD);
    this.requestRender();
  }

  /** Ask for a frame to be drawn (rendering is on demand). */
  requestRender(): void {
    this.renderRequested = true;
  }

  setTool(tool: Tool): void {
    this.tool = tool;
    this.updateCursor();
  }

  /** Stop or restart drawing and simulating, e.g. while the view is scrolled out of sight. */
  setPaused(paused: boolean): void {
    if (this.disposed) return;
    this.lastFrameTime = -1; // so the first frame back doesn't count the paused time
    this.renderRequested = true;
    this.renderer.setAnimationLoop(paused ? null : this.frame);
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

  /** Highlight an object and report it to the inspector; null clears the selection. */
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
    this.canvas.removeEventListener('pointerleave', this.onPointerLeave);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.controls.dispose();
    this.view.traverse((node) => {
      if (node instanceof SplatMesh) {
        node.dispose();
      } else if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
        node.geometry.dispose();
        const material = node.material as THREE.MeshStandardMaterial;
        material.map?.dispose();
        material.dispose();
      }
    });
    this.view.environment?.dispose();
    (this.view.background as THREE.Texture | null)?.dispose();
    this.gizmo?.dispose();
    this.renderer.dispose();
    this.canvas.remove();
    this.labels.domElement.remove();
    this.physics.dispose();
    // React's strict mode makes and throws away a simulator in development: hand the debug hooks
    // back to one that is still alive instead of leaving them on a freed physics world.
    const debugIndex = debugSimulators.indexOf(this);
    if (debugIndex >= 0) {
      debugSimulators.splice(debugIndex, 1);
      debugSimulators.at(-1)?.exposeDebugHooks();
    }
  }

  private get canvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  private splatVisuals(): { antialiased?: boolean }[] {
    const { background, objects } = this.scene.manifest;
    return [background.visual, background.backdrop, ...objects.map((o) => o.visual)].filter(
      (visual): visual is NonNullable<typeof visual> & { type: 'splats' } => visual?.type === 'splats',
    );
  }

  private runAction(action: WhatIfAction): void {
    switch (action.name) {
      case 'apply_impulse': {
        const [x, y, z] = action.direction;
        const s = action.speed / Math.hypot(x, y, z);
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
    if (index === this.selected) this.callbacks.onSelect(spec); // refresh the open inspector
    return spec;
  }

  // ---------- Building the 3D scene ----------

  private addLights(center: THREE.Vector3, radius: number): void {
    this.view.add(new THREE.HemisphereLight('#dfe6ff', '#1a120c', 0.9));
    const key = new THREE.DirectionalLight('#fff4e6', 2.6);
    key.position.copy(center).add(new THREE.Vector3(0.7, 1.6, 0.9));
    key.target.position.copy(center);
    key.castShadow = true;
    const touchDevice = window.matchMedia('(pointer: coarse)').matches;
    key.shadow.mapSize.set(touchDevice ? 1024 : 2048, touchDevice ? 1024 : 2048);
    const extent = radius + 0.4;
    Object.assign(key.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.1, far: 5 });
    key.shadow.bias = -0.0005;
    key.shadow.radius = 4;
    const rim = new THREE.DirectionalLight('#9db4ff', 0.8);
    rim.position.copy(center).add(new THREE.Vector3(-1.2, 0.8, -1.4));
    this.view.add(key, key.target, rim);
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
    const backgroundSplats = this.splatsFor(BACKGROUND, visual?.type === 'points' ? visual.point_size_m : 0.004);
    if (backgroundSplats) {
      backgroundSplats.position.set(-px, -py, -pz); // its coordinates are world coordinates
      this.tiltRoot.add(backgroundSplats);
    }
    // The room around the table stays put when the table tilts.
    this.backdrop = this.splatsFor(BACKDROP, 0.004);
    if (this.backdrop) this.view.add(this.backdrop);
    const scanned = backgroundSplats !== null;
    this.backgroundParts.splats = backgroundSplats;

    const group = new THREE.Group(); // table space: origin on the table top, +Y along its normal
    const { x, y, z, w } = rotationFromUp(plane.normal);
    group.quaternion.set(x, y, z, w);
    this.tiltRoot.add(group);
    const floorGroup = group.clone(); // the floor stays put when the table tilts
    floorGroup.position.set(px, py, pz);
    this.view.add(floorGroup);
    const [sizeX, sizeZ] = plane.size_m;
    const hasFloor = typeof floor_depth_m === 'number';
    if (!scanned) {
      const wood = new THREE.MeshStandardMaterial({ color: table_color ?? DEFAULT_TABLE_COLOR, roughness: 0.62, metalness: 0 });
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
    this.addTableInstruments(group, scanned);

    // The metric grid: under the table where the floor is, or a little below the table without one.
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(GRID_SIZE_M, GRID_SIZE_M), gridMaterial({ fadeFar: 6, axes: this.interactive }));
    grid.rotation.x = -Math.PI / 2;
    grid.position.y = -(hasFloor ? floor_depth_m : 0.75) + 0.001;
    grid.renderOrder = -1;
    floorGroup.add(grid);

    if (hasFloor) {
      // A pool of dim floor under the table, fading into the dark: things knocked off land
      // somewhere visible, and the scene still floats in the dark.
      const floor = new THREE.Mesh(
        new THREE.CircleGeometry(FLOOR_RADIUS_M, 64),
        new THREE.MeshStandardMaterial({
          color: FLOOR_COLOR,
          roughness: 1,
          metalness: 0,
          alphaMap: radialFade(),
          transparent: true,
          depthWrite: false,
        }),
      );
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -floor_depth_m;
      floor.receiveShadow = true;
      floorGroup.add(floor);
    }
  }

  /**
   * The table as the physics engine sees it: its collider in clay with a 5 cm grid (physics view),
   * and its outline traced on the scan (debug view). `group` is table space.
   */
  private addTableInstruments(group: THREE.Group, scanned: boolean): void {
    const { table_plane: plane } = this.scene.manifest.background;
    const [px, , pz] = plane.point;
    const [sizeX, sizeZ] = plane.size_m;
    // Table-space outline points (x, z), from the scanned polygon (world x, z) or the plane's rectangle.
    const outline: [number, number][] = plane.polygon_m?.length
      ? plane.polygon_m.map(([x, z]) => [x - px, z - pz])
      : [
          [-sizeX / 2, -sizeZ / 2],
          [sizeX / 2, -sizeZ / 2],
          [sizeX / 2, sizeZ / 2],
          [-sizeX / 2, sizeZ / 2],
        ];

    const shape = new THREE.Shape(outline.map(([x, z]) => new THREE.Vector2(x, -z)));
    const slab = new THREE.ExtrudeGeometry(shape, { depth: TABLE_THICKNESS_M, bevelEnabled: false });
    slab.rotateX(-Math.PI / 2);
    slab.translate(0, -TABLE_THICKNESS_M, 0);
    const clay = new THREE.Mesh(slab, new THREE.MeshStandardMaterial({ color: INK.clayTable, roughness: 0.9, metalness: 0 }));
    clay.receiveShadow = true;
    this.backgroundParts.clayTable = clay;
    group.add(clay);

    // The grid shader reads the mesh's local xy, so the mesh turns rather than its vertices.
    const tableGrid = new THREE.Mesh(new THREE.ShapeGeometry(shape), gridMaterial({ minor: 0.05, major: 0.25, fadeFar: 3, axes: false, opacity: 0.8 }));
    tableGrid.rotation.x = -Math.PI / 2;
    tableGrid.position.y = 0.0008;
    this.backgroundParts.tableGrid = tableGrid;
    group.add(tableGrid);

    const edge = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(outline.map(([x, z]) => new THREE.Vector3(x, 0.001, z))),
      overlayLine({ color: INK.wire, opacity: scanned ? 0.7 : 0.5, depthTest: true }),
    );
    this.backgroundParts.outline = edge;
    group.add(edge);
  }

  /**
   * An object is drawn as its trained splats, or its point cloud drawn as soft splats, or (with
   * neither) as its collider shape. An invisible copy of the collider takes taps and drags either way.
   */
  private addObject({ spec, shape }: SimBody): void {
    const group = new THREE.Group();
    const geometry = this.colliderGeometry(spec, shape === 'box-fallback');
    const pointSize = spec.visual?.type === 'points' ? spec.visual.point_size_m : spec.fallback_visual?.point_size_m ?? 0.003;
    const splats = this.splatsFor(spec.id, pointSize);
    let pick: THREE.Mesh;
    let tint: (color: THREE.Color) => void;
    let shadow: SplatMesh | null = null;
    let clay: THREE.Mesh | null = null;
    const [dx, , dz] = spec.dimensions_m;
    const shadowRadius = 0.5 * Math.max(dx, dz);
    if (splats) {
      group.add(splats);
      pick = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ visible: false }));
      tint = (color) => (splats.recolor = color);
      shadow = blobShadow();
      this.view.add(shadow);
      clay = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: INK.clay, roughness: 0.75, metalness: 0, flatShading: true }));
      clay.castShadow = true;
      clay.receiveShadow = true;
      group.add(clay);
    } else {
      const color = spec.color ?? DEFAULT_OBJECT_COLOR;
      const isBall = spec.collider.type === 'sphere' && shape === 'as-specified';
      const material = new THREE.MeshStandardMaterial({
        color: isBall ? '#ffffff' : color,
        map: isBall ? stripedTexture(color) : null, // stripes make rolling visible
        roughness: 0.5,
        metalness: 0,
        flatShading: spec.collider.type === 'convex_hull' || spec.collider.type === 'convex_parts',
      });
      pick = new THREE.Mesh(geometry, material);
      pick.castShadow = true;
      pick.receiveShadow = true;
      tint = (color) => {
        material.emissive.set(color.equals(NO_TINT) ? '#000000' : '#ffffff');
        material.emissiveIntensity = color.equals(SELECT_TINT) ? 0.16 : color.equals(HOVER_TINT) ? 0.07 : 0;
      };
    }
    group.add(pick);

    // Instruments: the collider's edges, corner brackets, the body frame, velocity and trail.
    const wire = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 25), overlayLine({ color: INK.wire, opacity: 0.55 }));
    wire.renderOrder = 8;
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!.clone().expandByScalar(0.008);
    const brackets = new THREE.LineSegments(bracketGeometry(box), overlayLine({ color: INK.wireSelected }));
    brackets.renderOrder = 12;
    const frame = bodyFrame(Math.max(0.03, Math.min(...spec.dimensions_m) * 0.6));
    group.add(wire, brackets, frame);
    const arrow = new VelocityArrow();
    const trail = new Trail();
    this.view.add(arrow.object, trail.line);

    // The robot view: center of mass, inertia ellipsoid and grasp candidates, all in the body frame.
    const robot = new THREE.Group();
    if (spec.inertial) robot.add(comMarker(spec.inertial), inertiaRings(spec.mass_kg, spec.inertial));
    (spec.grasps ?? []).slice(0, MAX_GRASP_GLYPHS).forEach((grasp, k) => robot.add(gripperGlyph(grasp, k === 0)));
    robot.visible = false;
    group.add(robot);

    const tag = document.createElement('div');
    tag.className = 'sim-tag';
    const name = document.createElement('span');
    name.className = 'sim-tag-name';
    name.textContent = spec.label;
    const speedText = document.createElement('span');
    speedText.className = 'sim-tag-value';
    speedText.textContent = `${formatMass(spec.mass_kg)}`;
    tag.append(name, speedText);
    const label = new CSS2DObject(tag);
    label.center.set(0.5, 1);
    this.view.add(label);

    this.views.push({
      group,
      pick,
      tint,
      shadow,
      shadowRadius,
      splats,
      clay,
      wire,
      brackets,
      frame,
      arrow,
      trail,
      label,
      speedText,
      labelHeight: box.max.length() + 0.02,
      robot,
    });
    this.view.add(group);
  }

  /** Show what the view mode calls for (see ViewMode), and the selection marks. */
  private applyViewMode(): void {
    const mode = this.viewMode;
    const clay = mode === 'physics' || mode === 'robot'; // the colliders instead of the scan
    const instruments = mode === 'debug' || mode === 'physics'; // wireframes, frames, trails, velocities
    const robot = mode === 'robot';
    const scanned = this.backgroundParts.splats !== null;
    if (this.backgroundParts.splats) this.backgroundParts.splats.visible = !clay;
    if (this.backdrop) this.backdrop.visible = mode === 'render' || mode === 'debug';
    this.backgroundParts.clayTable.visible = clay && scanned;
    this.backgroundParts.tableGrid.visible = clay;
    this.backgroundParts.outline.visible = instruments || robot;
    this.robotWorld.visible = robot;
    this.views.forEach((view, i) => {
      const marked = i === this.selected || i === this.hovered;
      if (view.splats) view.splats.visible = !clay;
      if (view.shadow) view.shadow.visible = !clay;
      if (view.clay) view.clay.visible = clay;
      view.wire.visible = instruments;
      (view.wire.material as THREE.LineBasicMaterial).color.set(i === this.selected ? INK.wireSelected : INK.wire);
      (view.wire.material as THREE.LineBasicMaterial).opacity = i === this.selected ? 0.9 : clay ? 0.35 : 0.55;
      view.brackets.visible = marked;
      (view.brackets.material as THREE.LineBasicMaterial).opacity = i === this.selected ? 1 : 0.5;
      view.frame.visible = instruments;
      view.trail.line.visible = instruments;
      view.robot.visible = robot;
      view.label.visible = instruments || robot || marked;
      view.label.element.classList.toggle('is-selected', i === this.selected);
    });
    this.requestRender();
  }

  /** Splats for an object (or BACKGROUND): from its splat file, or built from its point cloud. */
  private splatsFor(key: string, pointSizeM: number): SplatMesh | null {
    const file = this.scene.splatFiles.get(key);
    if (file) {
      const object = this.scene.manifest.objects.find((o) => o.id === key);
      // Scenes packaged with a capture path were cleaned by the pipeline itself (ghosts dropped, the
      // table filled in under each object, the backdrop split off). Older ones are tidied here.
      const legacy = !this.scene.manifest.capture;
      const mesh = new SplatMesh({
        fileBytes: file.bytes,
        fileName: file.name,
        onLoad: (loaded) => {
          if (legacy && object) tidySplats(loaded, objectVisibility(object.dimensions_m));
          else if (legacy && key === BACKGROUND) this.tidyBackground(loaded);
          this.requestRender();
        },
      });
      this.splatsLoading.push(mesh.initialized);
      return mesh;
    }
    const cloud = this.scene.pointClouds.get(key);
    return cloud ? pointsAsSplats(cloud, pointSizeM) : null;
  }

  /**
   * Keep the table's surface, dropping the blurry backdrop trained around it, and patch the table
   * where the objects stood (see splatCleanup).
   */
  private tidyBackground(mesh: SplatMesh): void {
    const plane = this.scene.manifest.background.table_plane;
    const visibility = backgroundVisibility(plane);
    const patches = tablePatches(this.scene.manifest.objects, plane);
    const samples: THREE.Color[][] = patches.map(() => []);
    const freed = tidySplats(mesh, ({ x, y, z }, { x: sx, y: sy, z: sz }, color) => {
      if (patches.some((patch) => patch.covers(x, y, z))) return 0;
      const keep = visibility(x, y, z, Math.max(sx, sy, sz));
      if (keep > 0.9) patches.forEach((patch, i) => patch.samples(x, y, z) && samples[i].push(color.clone()));
      return keep;
    });
    // Patch splats take over the slots of dropped ones: adding splats past the end would need the
    // view-dependent color textures resized too.
    const splats = mesh.packedSplats;
    if (!splats) return;
    let next = 0;
    patches.forEach((patch, i) =>
      patchSplats(plane, patch.grid, samples[i], (center, scales, quaternion, opacity, color) => {
        if (next >= freed.length) return;
        const index = freed[next++];
        splats.setSplat(index, center, scales, quaternion, opacity, color);
        clearViewDependentColor(splats, index);
      }),
    );
  }

  private colliderGeometry(spec: SceneObject, boxFallback: boolean): THREE.BufferGeometry {
    const collider = spec.collider;
    const hull = this.scene.hullPoints.get(spec.id);
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
      case 'convex_hull':
      case 'convex_parts':
        // Taps use the single hull (the pipeline ships one with convex parts as `pick_hull`).
        return hull ? convexGeometry(hull) : new THREE.BoxGeometry(...spec.dimensions_m);
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
   * Frame the objects. A scan opens where its video was filmed from (the scene's `view`), backed
   * off along that line if this screen is too narrow to show every object from there: trained
   * splats only look right near the filmed angles. Without a view, a scan is seen from its +z side
   * (where the pipeline puts the camera), a little above the table. A hand-made scene is viewed
   * along whichever table axis lines up the objects' longer spread with the screen's longer side,
   * from higher up on tall phone screens. Phones get a wider lens.
   */
  private placeCamera(): void {
    const portrait = this.camera.aspect < 1;
    const view = this.scene.manifest.view;
    this.camera.fov = view?.fov_deg ?? (portrait ? 60 : 45);
    this.camera.updateProjectionMatrix();
    const tanVertical = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanHorizontal = tanVertical * this.camera.aspect;
    if (view) {
      const target = new THREE.Vector3(...view.target);
      const offset = new THREE.Vector3(...view.position).sub(target);
      const filmedFrom = offset.length();
      const fits = this.objectsRadius(target) / Math.sin(Math.atan(Math.min(tanVertical, tanHorizontal)));
      this.controls.target.copy(target);
      this.camera.position.copy(target).addScaledVector(offset.normalize(), Math.max(filmedFrom, fits));
    } else {
      const { halfWidth, halfDepth } = this.framing;
      const scanned = Boolean(this.scene.manifest.background.visual);
      const fromSide = !scanned && portrait === halfWidth > halfDepth;
      const [across, away] = fromSide ? [halfDepth, halfWidth] : [halfWidth, halfDepth];
      // A hand-made scene's table is plain, so step back and show all of it.
      const distance = Math.max(across / tanHorizontal, away / tanVertical) * (scanned ? 1 : 1.35);
      const height = scanned ? SCAN_VIEW_HEIGHT : portrait ? 1.3 : 0.8;
      const direction = (fromSide ? new THREE.Vector3(1, height, 0) : new THREE.Vector3(0, height, 1)).normalize();
      this.controls.target.copy(this.framing.center);
      this.camera.position.copy(this.framing.center).addScaledVector(direction, distance);
    }
    this.swayFrom?.copy(this.camera.position);
    this.controls.update();
  }

  /** How far from `center` the objects reach (as they started, with a little margin). */
  private objectsRadius(center: THREE.Vector3): number {
    const reach = this.scene.manifest.objects.map(({ pose, dimensions_m }) => {
      const halfDiagonal = 0.5 * Math.hypot(...dimensions_m);
      return center.distanceTo(new THREE.Vector3(...pose.position)) + halfDiagonal;
    });
    return Math.max(0.1, ...reach) + VIEW_MARGIN_M;
  }

  private resize = (): void => {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.renderer.setSize(width, height);
    this.labels.setSize(width, height);
    this.requestRender();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (!this.cameraMovedByUser) this.placeCamera(); // e.g. the phone was rotated
  };

  // ---------- Each frame ----------

  private frame = (time: number): void => {
    const dt = this.lastFrameTime < 0 ? 0 : Math.min((time - this.lastFrameTime) / 1000, 0.25);
    const frameMs = this.lastFrameTime < 0 ? 0 : time - this.lastFrameTime;
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
    this.simSteps += steps;
    this.simTime += steps * step;

    const tableTop = this.scene.manifest.background.table_plane.point[1];
    const instruments = this.viewMode === 'debug' || this.viewMode === 'physics';
    const velocity = new THREE.Vector3();
    let anyAwake = this.holding !== null;
    this.physics.bodies.forEach(({ body }, i) => {
      const view = this.views[i];
      const v = body.linvel();
      velocity.set(v.x, v.y, v.z);
      const awake = !body.isSleeping();
      const moving = velocity.length() > MOVING_MPS && awake;
      if (!awake && view.trail.line.geometry.drawRange.count === 0) return; // nothing to move or fade
      anyAwake = true;
      const p = body.translation();
      const q = body.rotation();
      const { group, shadow, shadowRadius, arrow, trail, label, labelHeight } = view;
      group.position.set(p.x, p.y, p.z);
      group.quaternion.set(q.x, q.y, q.z, q.w);
      arrow.update(group.position, velocity, instruments);
      trail.update(group.position, moving);
      label.position.set(p.x, p.y + labelHeight, p.z);
      if (shadow) {
        // The disc stays on the table under the object, fading and spreading as it rises.
        const height = Math.max(0, p.y - tableTop);
        shadow.position.set(p.x, tableTop + 0.002, p.z);
        shadow.scale.setScalar(shadowRadius * (1 + height * 2));
        shadow.opacity = THREE.MathUtils.clamp(0.55 - height * 1.8, 0, 0.55);
      }
    });
    const tilt = this.physics.tableTilt();
    if (tilt.degrees !== 0 || !this.tiltRoot.quaternion.equals(LEVEL)) {
      const turn = axisRotation(tilt.axis, THREE.MathUtils.degToRad(tilt.degrees));
      this.tiltRoot.quaternion.set(turn.x, turn.y, turn.z, turn.w);
      anyAwake = true;
    }
    if (this.swayFrom) {
      const target = this.controls.target;
      const angle = SWAY_RADIANS * Math.sin((2 * Math.PI * time) / SWAY_PERIOD_MS);
      this.camera.position.copy(this.swayFrom).sub(target).applyAxisAngle(UP, angle).add(target);
      anyAwake = true;
    }
    this.controls.update(); // fires 'change' (a render) while the camera moves or glides to a stop
    this.updateDropMarker();

    if (anyAwake || this.renderRequested) {
      this.renderRequested = false;
      this.fadeBackdrop();
      this.renderer.render(this.view, this.camera);
      this.labels.render(this.view, this.camera);
      if (this.gizmo && this.container.clientWidth >= 720) {
        this.renderer.autoClear = false;
        this.gizmo.render(this.renderer);
        this.renderer.autoClear = true;
      }
      if (this.renderedLastFrame) this.adaptResolution(frameMs);
      this.renderedLastFrame = true;
      this.countFrame(time);
    } else {
      // Idle: nothing changed, so nothing is drawn (and there is no frame rate to report).
      if (this.renderedLastFrame) this.callbacks.onFps(0);
      this.renderedLastFrame = false;
      this.fpsWindowStart = -1;
    }
    if (time - this.telemetryAt >= TELEMETRY_MS) {
      this.telemetryAt = time;
      this.reportTelemetry();
    }
  };

  /**
   * Hold the frame rate by trading resolution: a run of slow frames lowers the pixel ratio a step,
   * a long run of fast ones raises it again (never above the screen's own, capped at 2).
   */
  private adaptResolution(frameMs: number): void {
    if (frameMs <= 0 || frameMs > 1000) return; // a stall (tab switch, GC) says nothing about load
    this.slowFrames = frameMs > SLOW_FRAME_MS ? this.slowFrames + 1 : 0;
    this.fastFrames = frameMs < FAST_FRAME_MS ? this.fastFrames + 1 : 0;
    let next = this.pixelRatio;
    if (this.slowFrames >= 8) next = Math.max(MIN_PIXEL_RATIO, this.pixelRatio - PIXEL_RATIO_STEP);
    else if (this.fastFrames >= 240) next = Math.min(this.maxPixelRatio, this.pixelRatio + PIXEL_RATIO_STEP);
    if (next === this.pixelRatio) return;
    this.pixelRatio = next;
    this.slowFrames = 0;
    this.fastFrames = 0;
    this.renderer.setPixelRatio(next);
    this.resize();
  }

  /** The backdrop is only right near the filmed path: fade it with the camera's distance from it. */
  private fadeBackdrop(): void {
    if (!this.backdrop || !this.backdrop.visible || this.capturePositions.length === 0) return;
    let nearest = Infinity;
    for (const position of this.capturePositions) nearest = Math.min(nearest, position.distanceTo(this.camera.position));
    const [near, far] = BACKDROP_FADE_M;
    this.backdrop.opacity = 1 - THREE.MathUtils.smoothstep(nearest, near, far);
  }

  /** While carrying, a ring on the surface below shows where the object will land. */
  private updateDropMarker(): void {
    const holding = this.holding;
    const show = holding !== null && holding.tool === 'lift' && holding.carrying;
    if (this.dropMarker.visible !== show) this.requestRender();
    this.dropMarker.visible = show;
    if (!holding || !show) return;
    const p = this.physics.bodies[holding.index].body.translation();
    const below = this.surfaceHeightUnder(new THREE.Vector3(p.x, 0, p.z), { ...holding, footprint: CENTER_ONLY });
    this.dropMarker.position.set(p.x, below + 0.003, p.z);
    this.dropMarker.scale.setScalar(Math.max(0.03, this.views[holding.index].shadowRadius));
  }

  /** Numbers for the telemetry readout and the object tags; a few times a second is plenty to read. */
  private reportTelemetry(): void {
    let energy = 0;
    let moving = 0;
    this.physics.bodies.forEach(({ body, spec }, i) => {
      const v = body.linvel();
      const speed = Math.hypot(v.x, v.y, v.z);
      energy += 0.5 * body.mass() * speed * speed;
      if (speed > MOVING_MPS) moving++;
      this.views[i].speedText.textContent = speed > MOVING_MPS ? `${speed.toFixed(2)} m/s` : formatMass(spec.mass_kg);
      this.views[i].speedText.classList.toggle('is-moving', speed > MOVING_MPS);
    });
    if (!this.callbacks.onTelemetry) return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const distance = offset.length();
    const selectedBody = this.selected === null ? null : this.physics.bodies[this.selected];
    const p = selectedBody?.body.translation();
    const v = selectedBody?.body.linvel();
    this.callbacks.onTelemetry({
      time: this.simTime,
      steps: this.simSteps,
      energy,
      moving,
      bodies: this.physics.bodies.length,
      camera: {
        azimuth: THREE.MathUtils.radToDeg(Math.atan2(offset.x, offset.z)),
        elevation: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(offset.y / distance, -1, 1))),
        distance,
      },
      selected:
        selectedBody && p && v
          ? { id: selectedBody.spec.id, position: [p.x, p.y, p.z], speed: Math.hypot(v.x, v.y, v.z) }
          : null,
    });
  }

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
    if (this.tool === 'push') {
      // Poke along the view direction, flattened toward the table so it pushes rather than presses.
      const direction = this.raycaster.ray.direction.clone();
      direction.y *= 0.25;
      this.physics.poke(this.physics.bodies[hit.index].spec.id, hit.point, direction);
      return;
    }
    const bounds = new THREE.Box3().setFromObject(this.views[hit.index].pick);
    this.holding = {
      pointerId: event.pointerId,
      index: hit.index,
      tool: this.tool,
      plane: new THREE.Plane(new THREE.Vector3(0, 1, 0), -hit.point.y),
      startY: hit.point.y,
      heightAboveBottom: hit.point.y - bounds.min.y,
      footprint: {
        minX: bounds.min.x - hit.point.x,
        maxX: bounds.max.x - hit.point.x,
        minZ: bounds.min.z - hit.point.z,
        maxZ: bounds.max.z - hit.point.z,
      },
      carrying: false,
    };
    this.physics.grabAt(this.physics.bodies[hit.index].body, hit.point, this.tool === 'lift');
    this.handTrail = [{ time: performance.now(), point: hit.point.clone() }];
    this.updateCursor();
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (this.down?.id === event.pointerId) {
      this.down.moved = Math.max(this.down.moved, Math.hypot(event.clientX - this.down.x, event.clientY - this.down.y));
    }
    const holding = this.holding;
    if (!holding) {
      if (event.pointerType === 'mouse' && !this.down) this.setHovered(this.pickObject(event)?.index ?? null);
      return;
    }
    if (event.pointerId !== holding.pointerId) return;
    if (!holding.carrying && (this.down?.moved ?? 0) < TAP_MAX_MOVE_PX) return;
    holding.carrying = true;
    this.aimRay(event);
    const spot = this.raycaster.ray.intersectPlane(holding.plane, new THREE.Vector3());
    if (!spot) return;
    this.clampToReach(spot);
    // Move slides the object along the table at the height you grabbed it; Lift carries it just
    // above whatever is underneath, so it can be set down on top of another object.
    spot.y =
      holding.tool === 'lift'
        ? this.surfaceHeightUnder(spot, holding) + holding.heightAboveBottom + LIFT_CLEARANCE_M
        : holding.startY;
    this.physics.moveGrabTarget(spot);
    const now = performance.now();
    this.handTrail.push({ time: now, point: spot.clone() });
    while (this.handTrail.length > 2 && now - this.handTrail[0].time > THROW_WINDOW_MS) this.handTrail.shift();
  };

  private onPointerUp = (event: PointerEvent): void => {
    const down = this.down?.id === event.pointerId ? this.down : null;
    if (down) this.down = null;
    const tap =
      event.type === 'pointerup' && down !== null && down.moved < TAP_MAX_MOVE_PX && performance.now() - down.time < TAP_MAX_MS;
    if (this.holding && event.pointerId === this.holding.pointerId) {
      const { index, carrying, tool } = this.holding;
      this.holding = null;
      this.physics.release(carrying ? this.throwVelocity(tool) : undefined);
      this.controls.enabled = true;
      if (tap) this.setSelected(index);
    } else if (this.tool === 'push' && !this.controls.enabled) {
      this.controls.enabled = true; // the poke is done
    } else if (tap && !this.holding) {
      this.setSelected(null); // tapped empty space
    }
    this.updateCursor();
  };

  private onPointerLeave = (): void => {
    if (!this.holding) this.setHovered(null);
  };

  /** While carrying, the wheel turns the object instead of zooming. */
  private onWheel = (event: WheelEvent): void => {
    if (this.holding?.tool !== 'lift') return;
    event.preventDefault();
    this.turnHeld(event.deltaY > 0 ? -1 : 1);
  };

  /**
   * How fast the hand was moving just before letting go (over THROW_WINDOW_MS), so a flick
   * throws the object at the flick's speed. A hand that had stopped sets the object down still;
   * sliding (move) throws only along the table.
   */
  private throwVelocity(tool: 'move' | 'lift'): { x: number; y: number; z: number } {
    const now = performance.now();
    const recent = this.handTrail.filter((sample) => now - sample.time <= THROW_WINDOW_MS);
    this.handTrail = [];
    if (recent.length < 2) return { x: 0, y: 0, z: 0 };
    const first = recent[0];
    const last = recent[recent.length - 1];
    const seconds = Math.max((last.time - first.time) / 1000, 1 / 120);
    const v = last.point.clone().sub(first.point).divideScalar(seconds);
    if (tool === 'move') v.y = 0;
    return v.length() < 0.05 ? { x: 0, y: 0, z: 0 } : { x: v.x, y: v.y, z: v.z };
  }

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
    const picks = this.views.map((v) => v.pick);
    const [hit] = this.raycaster.intersectObjects(picks, false);
    return hit ? { index: picks.indexOf(hit.object as THREE.Mesh), point: hit.point.clone() } : null;
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
    const others = this.views.filter((_, i) => i !== holding.index).map((v) => v.pick);
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
    const { minX, maxX, minZ, maxZ } = this.physics.tableBounds;
    spot.x = THREE.MathUtils.clamp(spot.x, minX - CARRY_REACH_M, maxX + CARRY_REACH_M);
    spot.z = THREE.MathUtils.clamp(spot.z, minZ - CARRY_REACH_M, maxZ + CARRY_REACH_M);
  }

  private setSelected(index: number | null): void {
    this.selected = index;
    this.refreshTints();
    this.callbacks.onSelect(index === null ? null : this.physics.bodies[index].spec);
  }

  private setHovered(index: number | null): void {
    if (index === this.hovered) return;
    this.hovered = index;
    this.refreshTints();
    this.updateCursor();
  }

  private refreshTints(): void {
    this.views.forEach((v, i) => v.tint(i === this.selected ? SELECT_TINT : i === this.hovered ? HOVER_TINT : NO_TINT));
    this.applyViewMode();
  }

  private updateCursor(): void {
    if (!this.interactive) return;
    const overObject = this.hovered !== null;
    this.canvas.style.cursor = this.holding ? 'grabbing' : overObject ? (this.tool === 'push' ? 'pointer' : 'grab') : '';
  }

  /** With ?debug in the URL, browser tests can read object poses and screen locations. */
  private exposeDebugHooks(): void {
    const project = (point: THREE.Vector3) => {
      const ndc = point.clone().project(this.camera);
      const rect = this.canvas.getBoundingClientRect();
      return [rect.left + ((ndc.x + 1) / 2) * rect.width, rect.top + ((1 - ndc.y) / 2) * rect.height];
    };
    const bodyOf = (id: string) => this.physics.bodies.find((b) => b.spec.id === id)?.body;
    if (!debugSimulators.includes(this)) debugSimulators.push(this);
    (window as unknown as Record<string, unknown>).__darius = {
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
      /** Sway the camera like the home page does, so every frame renders (for frame-rate tests). */
      sway: (on: boolean) => {
        this.swayFrom = on ? this.camera.position.clone() : null;
      },
      pixelRatio: () => this.pixelRatio,
    };
  }
}

const LEVEL = new THREE.Quaternion();
/** Sample only the middle of the footprint (the drop marker looks straight down). */
const CENTER_ONLY = { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };

/** The landing mark under a carried object: a thin ring, scaled to the object's footprint. */
function dropRing(): THREE.Mesh {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.92, 1, 48),
    new THREE.MeshBasicMaterial({ color: '#ff4f00', transparent: true, opacity: 0.9, depthWrite: false }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.renderOrder = 15;
  ring.visible = false;
  return ring;
}

/**
 * A scanned point cloud drawn as soft, round Gaussian splats (one per point), which reads as a
 * continuous surface instead of a speckle of squares. Trained splats replace this when the
 * pipeline provides them.
 */
function pointsAsSplats(geometry: THREE.BufferGeometry, pointSizeM: number): SplatMesh {
  const positions = geometry.getAttribute('position');
  const colors = geometry.hasAttribute('color') ? geometry.getAttribute('color') : null;
  const radius = pointSizeM * 0.6;
  const scales = new THREE.Vector3(radius, radius, radius);
  const turn = new THREE.Quaternion();
  const center = new THREE.Vector3();
  const color = new THREE.Color();
  return new SplatMesh({
    maxSplats: positions.count,
    constructSplats: (splats) => {
      for (let i = 0; i < positions.count; i++) {
        center.fromBufferAttribute(positions, i);
        if (colors) {
          // PLYLoader stores colors in linear space; splats take display (sRGB) colors.
          color.setRGB(colors.getX(i), colors.getY(i), colors.getZ(i)).convertLinearToSRGB();
        } else {
          color.set('#b8bec6');
        }
        splats.pushSplat(center, scales, turn, 0.9, color);
      }
    },
  });
}

/**
 * Scale each splat's opacity by `keep` (0 to 1) of its center, scales and color. Only the opacity
 * byte (the top 8 bits of each splat's first word) changes, so nothing else is decoded and
 * re-encoded. Returns the indices of the splats it hid completely, free for reuse.
 */
function tidySplats(
  mesh: SplatMesh,
  keep: (center: THREE.Vector3, scales: THREE.Vector3, color: THREE.Color) => number,
): number[] {
  const splats = mesh.packedSplats;
  const words = splats?.packedArray;
  if (!splats || !words) return [];
  const hidden: number[] = [];
  splats.forEachSplat((index, center, scales, _quaternion, _opacity, color) => {
    const amount = keep(center, scales, color);
    if (amount >= 1) return;
    const first = index * 4;
    words[first] = (words[first] & 0x00ffffff) | (Math.round((words[first] >>> 24) * amount) << 24);
    if (amount <= 0) hidden.push(index);
  });
  splats.needsUpdate = true; // re-upload the splat data ...
  mesh.needsUpdate = true; // ... and redraw it
  return hidden;
}

/** Zero a splat's spherical-harmonic (view-dependent) color terms, so it looks the same from every side. */
function clearViewDependentColor(splats: PackedSplats, index: number): void {
  const { sh1, sh2, sh3 } = splats.extra;
  sh1?.fill(0, index * 2, index * 2 + 2);
  sh2?.fill(0, index * 4, index * 4 + 4);
  sh3?.fill(0, index * 4, index * 4 + 4);
  for (const texture of [splats.extra.sh1Texture, splats.extra.sh2Texture, splats.extra.sh3Texture]) {
    if (texture) texture.value.needsUpdate = true;
  }
}

/** One wide, flat, dark splat: a soft contact shadow that composites correctly with the scan. */
function blobShadow(): SplatMesh {
  return new SplatMesh({
    maxSplats: 1,
    constructSplats: (splats) => {
      splats.pushSplat(new THREE.Vector3(), new THREE.Vector3(0.55, 0.002, 0.55), new THREE.Quaternion(), 1, new THREE.Color(0, 0, 0));
    },
  });
}

function formatMass(kg: number): string {
  return kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`;
}

function convexGeometry(values: Float32Array): THREE.BufferGeometry {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < values.length; i += 3) points.push(new THREE.Vector3(values[i], values[i + 1], values[i + 2]));
  return new ConvexGeometry(points);
}

/** White in the middle fading to black at the edge: an alpha map that blends a disc into the dark. */
function radialFade(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createRadialGradient(128, 128, 0, 128, 128, 128);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.35, '#8c8c8c');
    gradient.addColorStop(1, '#000000');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 256, 256);
  }
  return new THREE.CanvasTexture(canvas);
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
