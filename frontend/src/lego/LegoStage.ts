import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { brickFootprint, renderPlacement, timelineAt, type LegoBrick, type LegoPlan } from './types';

const UP = new THREE.Vector3(0, 1, 0);
const COLORS = { red: '#d82e28', dark: '#252d39', white: '#f7f5ed' };
const material = (color: string, props: THREE.MeshStandardMaterialParameters = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.29, metalness: 0.02, ...props });

/** A deliberately kinematic robot preview. No physical contact or controller is simulated. */
export class LegoStage {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(37, 1, 0.1, 220);
  private controls: OrbitControls;
  private observer: ResizeObserver;
  private frame = 0;
  private plan: LegoPlan | null = null;
  private models: { mesh: THREE.Group; target: THREE.Vector3; tray: THREE.Vector3; trayYaw: number }[] = [];
  private root = new THREE.Group();
  private robot = new THREE.Group();
  private arm1: THREE.Mesh;
  private arm2: THREE.Mesh;
  private joint: THREE.Mesh;
  private wrist = new THREE.Group();
  private fingers: THREE.Mesh[] = [];
  private ghost = new THREE.Group();
  private toolpath: THREE.Line;
  private time = 0;
  private exploded = false;
  private showRobot = true;
  private disposed = false;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor('#f4f3ee', 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.domElement.setAttribute(
      'aria-label',
      'Interactive 3D LEGO assembly and kinematic robot preview. Drag to orbit; scroll to zoom.',
    );
    this.renderer.domElement.setAttribute('role', 'img');
    host.appendChild(this.renderer.domElement);
    this.camera.position.set(27, 24, 30);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 2.2, 0);
    this.controls.enableDamping = true;
    this.controls.minDistance = 16;
    this.controls.maxDistance = 65;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#b9aaa0', 3));
    const key = new THREE.DirectionalLight('#fffdf3', 4.2);
    key.position.set(6, 22, 12);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = key.shadow.camera.bottom = -25;
    key.shadow.camera.right = key.shadow.camera.top = 25;
    key.shadow.normalBias = 0.035;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight('#dce8ff', 1.7);
    fill.position.set(-15, 8, -10);
    this.scene.add(fill);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(180, 180), material('#efeee8', { roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.01;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(80, 80, '#d8d8cf', '#e1e1d9');
    grid.position.y = -0.005;
    this.scene.add(grid);
    this.scene.add(this.root, this.robot, this.ghost);
    const dark = material(COLORS.dark);
    const white = material(COLORS.white);
    const red = material(COLORS.red);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(2, 2.3, 0.8, 40), dark);
    base.position.set(-12, 0.15, -4);
    base.castShadow = true;
    this.robot.add(base);
    const shoulder = new THREE.Mesh(new THREE.CylinderGeometry(0.85, 1.2, 3.4, 32), white);
    shoulder.position.set(-12, 2.1, -4);
    shoulder.castShadow = true;
    this.robot.add(shoulder);
    const shoulderRing = new THREE.Mesh(new THREE.CylinderGeometry(1.22, 1.22, 0.3, 32), red);
    shoulderRing.position.set(-12, 0.8, -4);
    this.robot.add(shoulderRing);
    this.arm1 = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.85, 1, 24), white);
    this.arm2 = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.62, 1, 24), white);
    this.joint = new THREE.Mesh(new THREE.SphereGeometry(0.85, 24, 16), red);
    this.arm1.castShadow = this.arm2.castShadow = this.joint.castShadow = true;
    this.robot.add(this.arm1, this.arm2, this.joint, this.wrist);
    const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.65, 0.65, 0.7, 24), dark);
    this.wrist.add(cuff);
    for (const sign of [-1, 1]) {
      const finger = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1, 0.48), dark);
      finger.position.set(sign * 0.65, -0.6, 0);
      finger.castShadow = true;
      this.wrist.add(finger);
      this.fingers.push(finger);
    }
    const path = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.toolpath = new THREE.Line(
      path,
      new THREE.LineDashedMaterial({ color: '#c45f40', dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.5 }),
    );
    this.scene.add(this.toolpath);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.tick();
  }

  private brick(brick: LegoBrick, ghost = false): THREE.Group {
    const mesh = new THREE.Group();
    const [width, depth] = brickFootprint(brick);
    const height = brick.height_plates * 0.4;
    const mat = material(brick.color_hex, ghost ? { transparent: true, opacity: 0.2, depthWrite: false } : {});
    const box = new THREE.Mesh(new THREE.BoxGeometry(width - 0.035, height - 0.035, depth - 0.035), mat);
    box.position.y = height / 2;
    box.castShadow = !ghost;
    box.receiveShadow = !ghost;
    mesh.add(box);
    const stud = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 20);
    for (let x = 0; x < width; x++)
      for (let z = 0; z < depth; z++) {
        const top = new THREE.Mesh(stud, mat);
        top.position.set(x - width / 2 + 0.5, height + 0.09, z - depth / 2 + 0.5);
        top.castShadow = !ghost;
        mesh.add(top);
      }
    if (ghost) {
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(box.geometry),
        new THREE.LineBasicMaterial({ color: brick.color_hex, transparent: true, opacity: 0.6 }),
      );
      edges.position.copy(box.position);
      mesh.add(edges);
    }
    return mesh;
  }

  setPlan(plan: LegoPlan): void {
    this.clearGroup(this.root);
    this.clearGroup(this.ghost);
    this.plan = plan;
    const [bw, bd] = plan.assembly.baseplate.studs;
    const plateHeight = (plan.assembly.baseplate.height_mm ?? 3.2) / 8;
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(bw, plateHeight, bd),
      material(plan.assembly.baseplate.color_hex, { roughness: 0.7 }),
    );
    plate.position.y = plateHeight / 2;
    plate.receiveShadow = true;
    this.root.add(plate);
    const studGeo = new THREE.CylinderGeometry(0.27, 0.27, 0.15, 12);
    const studMat = material(plan.assembly.baseplate.color_hex, { roughness: 0.7 });
    for (let x = 0; x < bw; x++)
      for (let z = 0; z < bd; z++) {
        const stud = new THREE.Mesh(studGeo, studMat);
        stud.position.set(x - bw / 2 + 0.5, plateHeight + 0.075, z - bd / 2 + 0.5);
        stud.receiveShadow = true;
        this.root.add(stud);
      }
    this.models = plan.steps.map((step, i) => {
      const brick = plan.assembly.bricks.find((b) => b.id === step.brick_id)!;
      const pose = renderPlacement(plan, step, brick, i);
      const target = new THREE.Vector3(...pose.target);
      const tray = new THREE.Vector3(...pose.source);
      const mesh = this.brick(brick);
      this.root.add(mesh);
      return { mesh, target, tray, trayYaw: pose.trayYaw };
    });
    const trayMinX = Math.min(...this.models.map((m) => m.tray.x)) - 3;
    const trayMaxX = Math.max(...this.models.map((m) => m.tray.x)) + 3;
    const trayMinZ = Math.min(...this.models.map((m) => m.tray.z)) - 3;
    const trayMaxZ = Math.max(...this.models.map((m) => m.tray.z)) + 3;
    const tray = new THREE.Mesh(
      new THREE.BoxGeometry(trayMaxX - trayMinX, 0.16, trayMaxZ - trayMinZ),
      material('#dfded6', { roughness: 0.8 }),
    );
    tray.position.set((trayMinX + trayMaxX) / 2, -0.08, (trayMinZ + trayMaxZ) / 2);
    tray.receiveShadow = true;
    this.root.add(tray);
    const activeBrick = plan.assembly.bricks.find((b) => b.id === plan.steps[0].brick_id)!;
    this.ghost.add(this.brick(activeBrick, true));
    this.update();
  }

  setState(time: number, exploded: boolean, showRobot: boolean): void {
    this.time = time;
    this.exploded = exploded;
    this.showRobot = showRobot;
    this.update();
  }
  resetCamera(): void {
    this.camera.position.set(27, 24, 30);
    this.controls.target.set(0, 2.2, 0);
    this.controls.update();
  }
  private link(mesh: THREE.Mesh, start: THREE.Vector3, end: THREE.Vector3): void {
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.scale.y = start.distanceTo(end);
    mesh.quaternion.setFromUnitVectors(UP, end.clone().sub(start).normalize());
  }
  private update(): void {
    if (!this.plan) return;
    const t = timelineAt(this.time, this.models.length);
    this.robot.visible = this.showRobot && !this.exploded;
    this.toolpath.visible = this.showRobot && !this.exploded && t.completed < this.models.length;
    this.ghost.visible = !this.exploded && t.completed < this.models.length;
    const active = this.models[t.active];
    this.models.forEach((model, i) => {
      model.mesh.position.copy(i < t.completed || this.exploded ? model.target : model.tray);
      model.mesh.rotation.y = i < t.completed || this.exploded ? 0 : model.trayYaw;
      if (this.exploded) model.mesh.position.y += i * 1.5 + 0.5;
    });
    if (!active) return;
    const p = t.fraction;
    const target = active.target.clone();
    const from = active.tray.clone();
    const tip = from.clone();
    if (p < 0.18) tip.y += 5 * (1 - p / 0.18) + 1.7;
    else if (p < 0.32) tip.y += 1.7;
    else if (p < 0.64) {
      const q = (p - 0.32) / 0.32;
      tip.lerp(target, q);
      tip.y += Math.sin(q * Math.PI) * 6 + 1.7;
    } else if (p < 0.8) tip.copy(target).add(new THREE.Vector3(0, 2.7 - (p - 0.64) / 0.16, 0));
    else if (p < 0.94) tip.copy(target).add(new THREE.Vector3(0, 1.7, 0));
    else tip.copy(target).add(new THREE.Vector3(0, 1.7 + ((p - 0.94) / 0.06) * 3, 0));
    if (t.completed < this.models.length && !this.exploded) {
      if (p > 0.32 && p < 0.94) active.mesh.position.copy(tip).add(new THREE.Vector3(0, -1.7, 0));
      active.mesh.rotation.y = active.trayYaw * (1 - THREE.MathUtils.clamp((p - 0.32) / 0.32, 0, 1));
      if (p >= 0.94) active.mesh.position.copy(target);
      const currentBrick = this.plan.assembly.bricks.find((b) => b.id === this.plan!.steps[t.active].brick_id)!;
      const ghostId = this.ghost.userData.brick;
      if (ghostId !== currentBrick.id) {
        this.clearGroup(this.ghost);
        this.ghost.add(this.brick(currentBrick, true));
        this.ghost.userData.brick = currentBrick.id;
      }
      this.ghost.position.copy(target);
    }
    const shoulder = new THREE.Vector3(-12, 4, -4);
    const elbow = shoulder.clone().lerp(tip, 0.48);
    elbow.y = Math.max(9, tip.y + 4);
    elbow.z -= 2;
    this.link(this.arm1, shoulder, elbow);
    this.link(this.arm2, elbow, tip);
    this.joint.position.copy(elbow);
    this.wrist.position.copy(tip);
    this.fingers.forEach((finger, i) => {
      finger.position.x = (i ? 1 : -1) * (p > 0.2 && p < 0.94 ? 0.48 : 0.8);
    });
    const curve = new THREE.QuadraticBezierCurve3(
      from.clone().add(new THREE.Vector3(0, 2, 0)),
      from
        .clone()
        .lerp(target, 0.5)
        .add(new THREE.Vector3(0, 12, 0)),
      target.clone().add(new THREE.Vector3(0, 2, 0)),
    );
    this.toolpath.geometry.dispose();
    this.toolpath.geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(48));
    this.toolpath.computeLineDistances();
  }
  private resize(): void {
    const { width, height } = this.host.getBoundingClientRect();
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
  private tick = (): void => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.tick);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
  private clearGroup(group: THREE.Object3D): void {
    group.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      mesh.geometry?.dispose();
      if (mesh.material) (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((m) => m.dispose());
    });
    group.clear();
  }
  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.controls.dispose();
    this.clearGroup(this.scene);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
