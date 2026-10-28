import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { Choreography, ORIENTATIONS, type Moment, type PartState } from '../engine/choreography';
import { stepParts } from '../engine/graph';
import type { AssemblyPlan, Part } from '../engine/plan';
import { floorMaterial, partMaterial, wallMaterial } from './materials';

/**
 * The 3D world of an assembly session: a room, the furniture parts, a handheld-phone camera, and
 * the AR layer drawn into the scene (ghosts at target poses, guidance arcs, a rotation arc when the
 * assembly must be turned, fastener markers). It renders one moment of a Choreography at a time and
 * reports where every part appears on screen, so the HUD can draw detection boxes that come from
 * the real geometry.
 *
 * Modes: 'ar' (the copilot view), 'hero' (a cinematic orbit for the home page), 'clean' (no AR
 * layer, a deterministic camera: frames a vision model can be tested on).
 */

export type StageMode = 'ar' | 'hero' | 'clean';

export const AR = {
  guide: new THREE.Color('#ff6a1a'),
  done: new THREE.Color('#34d399'),
  track: new THREE.Color('#5ccfff'),
} as const;

export interface Detection {
  id: string;
  label: string;
  name: string;
  kind: 'part' | 'assembly';
  role: 'loose' | 'active' | 'assembly';
  /** How many parts the box covers (more than one for an assembly). */
  parts: number;
  box: { x: number; y: number; w: number; h: number };
  confidence: number;
}

export interface FrameInfo {
  moment: Moment;
  detections: Detection[];
  /** Screen position (0..1) of the current step's target, for the HUD's callout line. */
  focus: { x: number; y: number } | null;
}

const MM = 0.001;

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
}

/** A soft ring around a dot, drawn once and tinted per use: a fastener marker, sized in screen pixels. */
function markerTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const g = canvas.getContext('2d')!;
  const c = size / 2;
  const glow = g.createRadialGradient(c, c, 0, c, c, c);
  glow.addColorStop(0, 'rgba(255,255,255,0.6)');
  glow.addColorStop(0.4, 'rgba(255,255,255,0.16)');
  glow.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = glow;
  g.fillRect(0, 0, size, size);
  g.strokeStyle = '#fff';
  g.lineWidth = 8;
  g.beginPath();
  g.arc(c, c, 36, 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = '#fff';
  g.beginPath();
  g.arc(c, c, 14, 0, Math.PI * 2);
  g.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** A tube along a curve whose dashes run toward the target (the guidance arc). */
function flowMaterial(color: THREE.Color): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { uTime: { value: 0 }, uColor: { value: color }, uOpacity: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `
      uniform float uTime; uniform vec3 uColor; uniform float uOpacity; varying vec2 vUv;
      void main() {
        float dash = step(0.45, fract(vUv.x * 14.0 - uTime * 1.6));
        float fade = smoothstep(0.0, 0.08, vUv.x) * (0.55 + 0.45 * vUv.x);
        gl_FragColor = vec4(uColor * 1.15, (0.25 + 0.75 * dash) * fade * uOpacity);
      }`,
  });
}

interface PartView {
  part: Part;
  mesh: THREE.Mesh;
  edges: THREE.LineSegments;
  ghost: THREE.Group;
  corners: THREE.Vector3[];
}

export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.PerspectiveCamera(52, 1, 0.02, 60);
  readonly choreography: Choreography;
  private readonly scene = new THREE.Scene();
  private readonly views = new Map<string, PartView>();
  private readonly ar = new THREE.Group();
  private readonly arcs: THREE.Mesh[] = [];
  private readonly flow: THREE.ShaderMaterial;
  private readonly rotation: THREE.Group;
  private readonly fastenerPool: THREE.Group[] = [];
  private readonly markers: Record<'guide' | 'done', THREE.SpriteMaterial>;
  /** Where the look-at point sits on screen, as a fraction of the viewport from its center (+x right, +y up). */
  private framing = { x: 0, y: 0 };
  private readonly resizeObserver: ResizeObserver;
  private readonly lookAt = new THREE.Vector3();
  private readonly focusPoint = new THREE.Vector3();
  private disposed = false;

  constructor(
    private readonly container: HTMLElement,
    readonly plan: AssemblyPlan,
    readonly mode: StageMode,
  ) {
    this.choreography = new Choreography(plan);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: mode === 'clean' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.appendChild(this.renderer.domElement);

    const environment = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = environment.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.55;
    environment.dispose();
    this.scene.background = new THREE.Color('#d9d4cc');
    this.scene.fog = new THREE.Fog('#d9d4cc', 7, 16);
    this.buildRoom();
    for (const part of plan.parts) this.addPart(part);

    this.flow = flowMaterial(AR.guide);
    const marker = markerTexture();
    this.markers = {
      guide: new THREE.SpriteMaterial({ map: marker, color: AR.guide, depthTest: false, transparent: true, sizeAttenuation: false }),
      done: new THREE.SpriteMaterial({ map: marker, color: AR.done, depthTest: false, transparent: true, sizeAttenuation: false }),
    };
    this.rotation = this.rotationArc();
    this.ar.add(this.rotation);
    this.scene.add(this.ar);
    this.ar.visible = mode !== 'clean';

    const c = this.choreography;
    this.lookAt.copy(c.workCenter).setY(c.size().y * 0.35);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  private buildRoom(): void {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), floorMaterial());
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const wall = wallMaterial();
    const back = new THREE.Mesh(new THREE.PlaneGeometry(14, 3.2), wall);
    back.position.set(0, 1.6, -2.4);
    back.receiveShadow = true;
    const side = new THREE.Mesh(new THREE.PlaneGeometry(14, 3.2), wall);
    side.position.set(-3.6, 1.6, 0);
    side.rotation.y = Math.PI / 2;
    side.receiveShadow = true;
    const skirting = new THREE.Mesh(new THREE.BoxGeometry(14, 0.08, 0.015), new THREE.MeshStandardMaterial({ color: '#f4f2ee', roughness: 0.5 }));
    skirting.position.set(0, 0.04, -2.39);
    this.scene.add(back, side, skirting);

    this.scene.add(new THREE.HemisphereLight('#fff6ea', '#8a7460', 0.55));
    const sun = new THREE.DirectionalLight('#fff1dc', 2.2);
    sun.position.set(2.6, 4.2, 2.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 12 });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);
  }

  private addPart(part: Part): void {
    const [w, h, d] = part.size_mm.map((v) => Math.max(v * MM, 0.002));
    const radius = Math.min(0.0025, Math.min(w, h, d) / 3);
    const geometry = new RoundedBoxGeometry(w, h, d, 2, radius);
    const mesh = new THREE.Mesh(geometry, partMaterial(part.material, part.color));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)),
      new THREE.LineBasicMaterial({ color: '#2a2622', transparent: true, opacity: 0.22 }),
    );
    edges.matrixAutoUpdate = false;
    const ghost = new THREE.Group();
    ghost.add(
      new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshBasicMaterial({ color: AR.guide, transparent: true, opacity: 0.14, depthWrite: false }),
      ),
      new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)),
        new THREE.LineBasicMaterial({ color: AR.guide, transparent: true, opacity: 0.95, depthTest: false }),
      ),
    );
    ghost.matrixAutoUpdate = false;
    ghost.visible = false;
    ghost.renderOrder = 5;
    this.ar.add(ghost);
    this.scene.add(mesh, edges);
    const corners = [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => new THREE.Vector3((x * w) / 2, (y * h) / 2, (z * d) / 2))));
    this.views.set(part.id, { part, mesh, edges, ghost, corners });
  }

  /** A three-quarter ring with an arrowhead: "turn the assembly this way". */
  private rotationArc(): THREE.Group {
    const group = new THREE.Group();
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1, 0.012, 8, 96, Math.PI * 1.45),
      new THREE.MeshBasicMaterial({ color: AR.guide, transparent: true, opacity: 0.95, depthTest: false }),
    );
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(0.045, 0.11, 18),
      new THREE.MeshBasicMaterial({ color: AR.guide, transparent: true, opacity: 0.95, depthTest: false }),
    );
    const angle = Math.PI * 1.45;
    head.position.set(Math.cos(angle), Math.sin(angle), 0);
    head.rotation.z = angle;
    ring.renderOrder = head.renderOrder = 6;
    group.add(ring, head);
    group.visible = false;
    return group;
  }

  /** A marker at a fastener: a screen-sized ring on the hole, and a pin along the way the fastener goes in. */
  private fastenerMarker(index: number): THREE.Group {
    while (this.fastenerPool.length <= index) {
      const marker = new THREE.Group();
      const ring = new THREE.Sprite(this.markers.guide);
      const pin = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0035, 0.0035, 0.07, 8),
        new THREE.MeshBasicMaterial({ color: AR.guide, depthTest: false, transparent: true, opacity: 0.85 }),
      );
      pin.position.y = 0.035;
      ring.renderOrder = pin.renderOrder = 7;
      marker.add(ring, pin);
      this.ar.add(marker);
      this.fastenerPool.push(marker);
    }
    return this.fastenerPool[index];
  }

  private arc(index: number, from: THREE.Vector3, to: THREE.Vector3): void {
    while (this.arcs.length <= index) {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.flow);
      mesh.renderOrder = 6;
      this.ar.add(mesh);
      this.arcs.push(mesh);
    }
    const lift = 0.25 + from.distanceTo(to) * 0.25;
    const middle = from.clone().lerp(to, 0.5).setY(Math.max(from.y, to.y) + lift);
    const curve = new THREE.QuadraticBezierCurve3(from, middle, to);
    const mesh = this.arcs[index];
    mesh.geometry.dispose();
    mesh.geometry = new THREE.TubeGeometry(curve, 48, 0.006, 6, false);
    mesh.visible = true;
  }

  /** Lay out the scene for time t; returns what the HUD needs about this moment. */
  update(time: number, clock = time): FrameInfo {
    const c = this.choreography;
    const moment = c.moment(time);
    const states = c.partStates(moment);
    const active = stepParts(this.plan, moment.step);
    const ar = this.mode !== 'clean';
    for (const [id, view] of this.views) {
      const state = states.get(id)!;
      view.mesh.matrix.copy(state.matrix);
      view.edges.matrix.copy(state.matrix);
      const showGhost = ar && state.target !== null && active.has(id) && ['locate', 'place'].includes(moment.phase) && state.status !== 'attached';
      view.ghost.visible = showGhost;
      if (showGhost) {
        view.ghost.matrix.copy(state.target!);
        const pulse = 0.5 + 0.5 * Math.sin(clock * 5);
        ((view.ghost.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = 0.08 + 0.1 * pulse;
      }
    }
    this.layoutAr(moment, states, clock);
    this.moveCamera(moment, states, clock);
    return { moment, detections: ar ? this.detect(moment, states) : [], focus: this.screenOf(this.focusPoint) };
  }

  private layoutAr(moment: Moment, states: Map<string, PartState>, clock: number): void {
    this.flow.uniforms.uTime.value = clock;
    this.arcs.forEach((a) => (a.visible = false));
    this.fastenerPool.forEach((m) => (m.visible = false));
    this.rotation.visible = false;
    if (this.mode === 'clean') return;
    const step = this.plan.steps[moment.step - 1];
    const c = this.choreography;
    const assemblyCenter = c.workCenter.clone().setY(c.size().y * 0.3);
    this.focusPoint.copy(assemblyCenter);
    if (step && ['locate', 'place'].includes(moment.phase)) {
      let k = 0;
      for (const id of step.parts) {
        const state = states.get(id)!;
        if (!state.target || state.status === 'attached') continue;
        const from = new THREE.Vector3().setFromMatrixPosition(state.matrix);
        const to = new THREE.Vector3().setFromMatrixPosition(state.target);
        this.arc(k++, from, to);
        if (k === 1) this.focusPoint.copy(to);
      }
      // A subassembly built on the side travels as one: a single arc from its center to where it goes.
      const joining = c.joining(moment).map((id) => states.get(id)!);
      if (joining.length && joining.every((s) => s.target)) {
        const center = (key: 'matrix' | 'target') =>
          joining.reduce((sum, s) => sum.add(new THREE.Vector3().setFromMatrixPosition(s[key]!)), new THREE.Vector3()).divideScalar(joining.length);
        const to = center('target');
        this.arc(k++, center('matrix'), to);
        if (k === 1) this.focusPoint.copy(to);
      }
      this.flow.uniforms.uOpacity.value = moment.phase === 'locate' ? Math.min(1, moment.u * 2.5) : 1 - Math.max(0, moment.u - 0.75) * 4;
    }
    if (moment.phase === 'reorient') {
      // A ring around the axis the assembly turns about, centered on it.
      const from = c.orientationOf(moment.step - 1);
      const to = c.orientationOf(moment.step);
      const turn = ORIENTATIONS[to].clone().multiply(ORIENTATIONS[from].clone().invert());
      const axis = new THREE.Vector3(turn.x, turn.y, turn.z);
      if (axis.lengthSq() < 1e-9) axis.set(1, 0, 0);
      axis.normalize();
      const radius = Math.max(c.size().x, c.size().y, c.size().z) * 0.62;
      this.rotation.visible = true;
      this.rotation.position.copy(c.workCenter).setY(radius * 0.75);
      this.rotation.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      this.rotation.scale.setScalar(radius);
      this.rotation.rotateZ(clock * 0.6);
      this.focusPoint.copy(this.rotation.position);
    }
    if (step && ['fasten', 'verify'].includes(moment.phase)) {
      const fasteners = c.fasteners(moment);
      fasteners.forEach((f, i) => {
        const appear = moment.phase === 'verify' ? 1 : Math.min(1, Math.max(0, moment.u * (fasteners.length + 2) - i));
        if (appear <= 0) return;
        const marker = this.fastenerMarker(i);
        marker.visible = true;
        marker.position.copy(f.position);
        marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), f.axis.clone().negate());
        const done = moment.phase === 'verify';
        const [ring, pin] = marker.children as [THREE.Sprite, THREE.Mesh];
        ring.material = done ? this.markers.done : this.markers.guide;
        (pin.material as THREE.MeshBasicMaterial).color.copy(done ? AR.done : AR.guide);
        const pulse = done ? 1 : 1 + 0.22 * Math.sin(clock * 6 + i);
        ring.scale.setScalar(0.03 * pulse * appear);
        pin.visible = !done;
      });
      if (fasteners.length) this.focusPoint.copy(fasteners[Math.floor(fasteners.length / 2)].position);
    }
  }

  /** A person holding a phone: a slow walk around the work, a little hand shake, attention on the action. */
  private moveCamera(moment: Moment, states: Map<string, PartState>, clock: number): void {
    const c = this.choreography;
    const size = c.size();
    const reach = Math.max(size.x, size.y, size.z);
    const radius = 1.45 + reach * 1.45;
    let azimuth: number;
    let height: number;
    if (this.mode === 'hero') {
      azimuth = 0.55 + clock * 0.09;
      height = 1.3 + reach * 0.6;
    } else {
      azimuth = 0.62 + 0.38 * Math.sin(clock * 0.11) + 0.1 * Math.sin(clock * 0.37);
      height = 1.5 + reach * 0.62 + 0.05 * Math.sin(clock * 0.53);
    }
    const shake = this.mode === 'ar' ? 0.006 : this.mode === 'clean' ? 0.003 : 0;
    const center = c.workCenter.clone().lerp(c.layoutCenter, 0.3);
    const position = new THREE.Vector3(
      center.x + Math.sin(azimuth) * radius + shake * Math.sin(clock * 7.3),
      height + shake * Math.sin(clock * 5.1 + 1),
      center.z + Math.cos(azimuth) * radius + shake * Math.sin(clock * 6.7 + 2),
    );
    let attention = c.workCenter.clone().lerp(c.layoutCenter, 0.3).setY(size.y * 0.3);
    const step = this.plan.steps[moment.step - 1];
    if (step && moment.phase === 'locate') {
      const loose = step.parts.map((id) => new THREE.Vector3().setFromMatrixPosition(states.get(id)!.matrix));
      if (loose.length) attention = attention.lerp(loose.reduce((a, b) => a.add(b), new THREE.Vector3()).divideScalar(loose.length), 0.55);
    }
    this.lookAt.lerp(attention, this.mode === 'clean' ? 1 : 0.04);
    this.camera.position.copy(position);
    this.camera.lookAt(this.lookAt);
  }

  private screenOf(point: THREE.Vector3): { x: number; y: number } | null {
    const p = point.clone().project(this.camera);
    if (p.z > 1 || Math.abs(p.x) > 1.2 || Math.abs(p.y) > 1.2) return null;
    return { x: (p.x + 1) / 2, y: (1 - p.y) / 2 };
  }

  /**
   * Screen boxes (0..1) as a part tracker would report them: every loose part on its own, the parts
   * being placed now, and each (sub)assembly as one object.
   */
  private detect(moment: Moment, states: Map<string, PartState>): Detection[] {
    const c = this.choreography;
    const step = this.plan.steps[moment.step - 1];
    const placing = new Set(moment.phase === 'intro' ? [] : (step?.parts ?? []));
    const settled = c.settledSteps(moment);
    const project = (view: PartView, matrix: THREE.Matrix4) => {
      const box = new THREE.Box2();
      for (const corner of view.corners) {
        const p = corner.clone().applyMatrix4(matrix).project(this.camera);
        if (p.z > 1) return null;
        box.expandByPoint(new THREE.Vector2((p.x + 1) / 2, (1 - p.y) / 2));
      }
      return box;
    };
    const onScreen = (box: THREE.Box2) => !(box.max.x < 0 || box.min.x > 1 || box.max.y < 0 || box.min.y > 1);
    const toRect = (box: THREE.Box2) => ({ x: box.min.x, y: box.min.y, w: box.max.x - box.min.x, h: box.max.y - box.min.y });
    const singles: Detection[] = [];
    const groups = new Map<string, { box: THREE.Box2; ids: string[]; moving: boolean }>();
    for (const [id, view] of this.views) {
      const state = states.get(id)!;
      const box = project(view, state.matrix);
      if (!box) continue;
      const alone = state.status === 'loose' || (state.status === 'moving' && placing.has(id));
      if (!alone) {
        const key = c.componentOf(id, settled);
        const group = groups.get(key) ?? { box: new THREE.Box2(), ids: [], moving: false };
        group.box.union(box);
        group.ids.push(id);
        group.moving ||= state.status === 'moving';
        groups.set(key, group);
        continue;
      }
      if (!onScreen(box)) continue;
      singles.push({
        id,
        label: view.part.label,
        name: view.part.name,
        kind: 'part',
        role: placing.has(id) ? 'active' : 'loose',
        parts: 1,
        box: toRect(box),
        confidence: 0.9 + 0.09 * hash(id + Math.floor(moment.time * 2)),
      });
    }
    const assemblies: Detection[] = [];
    for (const [key, group] of groups) {
      if (!onScreen(group.box)) continue;
      if (group.ids.length === 1) {
        const view = this.views.get(group.ids[0])!;
        singles.push({
          id: view.part.id,
          label: view.part.label,
          name: view.part.name,
          kind: 'part',
          role: placing.has(view.part.id) || group.moving ? 'active' : 'loose',
          parts: 1,
          box: toRect(group.box),
          confidence: 0.9 + 0.09 * hash(view.part.id + Math.floor(moment.time * 2)),
        });
        continue;
      }
      const main = key === c.mainPart;
      assemblies.push({
        id: `assembly-${key}`,
        label: '',
        name: `${main ? 'assembly' : 'subassembly'} · ${group.ids.length} parts`,
        kind: 'assembly',
        role: group.moving ? 'active' : 'assembly',
        parts: group.ids.length,
        box: toRect(group.box),
        confidence: main ? 0.97 : 0.94,
      });
    }
    return [...assemblies, ...singles];
  }

  render(): void {
    if (!this.disposed) this.renderer.render(this.scene, this.camera);
  }

  /** Move the look-at point on screen (fractions of the viewport from its center), e.g. to clear the HUD. */
  frame(x: number, y: number): void {
    this.framing = { x, y };
    this.resize();
  }

  resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.fov = width < height ? 62 : 50;
    this.camera.setViewOffset(width, height, -this.framing.x * width, this.framing.y * height, width, height);
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.disposed = true;
    this.resizeObserver.disconnect();
    this.scene.traverse((node) => {
      if (node instanceof THREE.Mesh || node instanceof THREE.LineSegments) {
        node.geometry.dispose();
        const material = node.material as THREE.Material | THREE.Material[];
        (Array.isArray(material) ? material : [material]).forEach((m) => m.dispose());
      }
    });
    this.markers.guide.map?.dispose();
    Object.values(this.markers).forEach((m) => m.dispose());
    this.scene.environment?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
