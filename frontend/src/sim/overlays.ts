import * as THREE from 'three';
import type { Pass } from 'three/addons/postprocessing/Pass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/**
 * The simulator's instrument layer: the metric grid, collider wireframes, body frames, velocity
 * vectors, motion trails and the final color grade. Everything here is drawn on top of (or under)
 * the scan, and nothing here touches the physics.
 */

/** Overlay ink, in display colors. Kept apart from the scan's own colors: cool and desaturated. */
export const INK = {
  grid: '#2c3036',
  gridMajor: '#454b54',
  axisX: '#e5484d',
  axisY: '#46a758',
  axisZ: '#3e8ef7',
  wire: '#7cc4ff',
  wireSelected: '#ffffff',
  velocity: '#ffb224',
  trail: '#7cc4ff',
  clay: '#b4b9c1',
  clayTable: '#5d6168',
} as const;

/**
 * An infinite-looking metric grid: 10 cm cells, 1 m major lines, the two horizontal axes in red
 * and blue, fading out with distance so it never ends in a hard edge. Drawn on a plane mesh; the
 * plane's local XY are the grid's coordinates in meters.
 */
export function gridMaterial({ opacity = 1, minor = 0.1, major = 1, fadeFar = 4, axes = true } = {}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uColor: { value: new THREE.Color(INK.grid) },
      uMajorColor: { value: new THREE.Color(INK.gridMajor) },
      uAxisX: { value: new THREE.Color(INK.axisX) },
      uAxisZ: { value: new THREE.Color(INK.axisZ) },
      uMinor: { value: minor },
      uMajor: { value: major },
      uFadeFar: { value: fadeFar },
      uOpacity: { value: opacity },
      uAxes: { value: axes ? 1 : 0 },
    },
    vertexShader: /* glsl */ `
      varying vec2 vLocal;
      varying vec3 vWorld;
      void main() {
        vLocal = position.xy;
        vec4 world = modelMatrix * vec4(position, 1.0);
        vWorld = world.xyz;
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor, uMajorColor, uAxisX, uAxisZ;
      uniform float uMinor, uMajor, uFadeFar, uOpacity, uAxes;
      varying vec2 vLocal;
      varying vec3 vWorld;
      // Anti-aliased lines of constant screen width, every 'spacing' meters.
      float lines(vec2 p, float spacing, float width) {
        vec2 c = p / spacing;
        vec2 d = fwidth(c);
        vec2 g = abs(fract(c - 0.5) - 0.5) / max(d, 1e-5);
        return 1.0 - min(min(g.x, g.y) / width, 1.0);
      }
      float line(float v, float width) {
        return 1.0 - min(abs(v) / max(fwidth(v), 1e-5) / width, 1.0);
      }
      void main() {
        float dist = distance(vWorld, cameraPosition);
        float fade = 1.0 - smoothstep(uFadeFar * 0.35, uFadeFar, dist);
        // Minor lines thin out before they alias into a gray mush.
        float minorFade = 1.0 - smoothstep(uFadeFar * 0.2, uFadeFar * 0.6, dist);
        float minor = lines(vLocal, uMinor, 1.0) * minorFade;
        float major = lines(vLocal, uMajor, 1.4);
        vec3 color = mix(uColor, uMajorColor, major);
        float alpha = max(minor * 0.5, major * 0.7);
        if (uAxes > 0.5) {
          float ax = line(vLocal.y, 1.6); // the x axis runs where local y = 0
          float az = line(vLocal.x, 1.6);
          color = mix(color, uAxisX, ax);
          color = mix(color, uAxisZ, az * (1.0 - ax));
          alpha = max(alpha, max(ax, az) * 0.6);
        }
        gl_FragColor = vec4(color, alpha * fade * uOpacity);
        if (gl_FragColor.a < 0.004) discard;
      }
    `,
  });
}

/** Corner brackets around a box: the selection mark, which reads as "tracked" rather than "glowing". */
export function bracketGeometry(box: THREE.Box3, fraction = 0.28): THREE.BufferGeometry {
  const { min, max } = box;
  const size = box.getSize(new THREE.Vector3());
  const arm = Math.min(size.x, size.y, size.z) * fraction + 0.004;
  const points: number[] = [];
  for (const x of [min.x, max.x]) {
    for (const y of [min.y, max.y]) {
      for (const z of [min.z, max.z]) {
        const sx = x === min.x ? 1 : -1;
        const sy = y === min.y ? 1 : -1;
        const sz = z === min.z ? 1 : -1;
        points.push(x, y, z, x + sx * arm, y, z);
        points.push(x, y, z, x, y + sy * arm, z);
        points.push(x, y, z, x, y, z + sz * arm);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return geometry;
}

/** A body frame: short red/green/blue lines along the object's own x/y/z axes. */
export function bodyFrame(length: number): THREE.LineSegments {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, length, 0, 0, 0, 0, 0, 0, length, 0, 0, 0, 0, 0, 0, length], 3));
  const colors = [INK.axisX, INK.axisX, INK.axisY, INK.axisY, INK.axisZ, INK.axisZ].flatMap((hex) => new THREE.Color(hex).toArray());
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const frame = new THREE.LineSegments(geometry, overlayLine({ vertexColors: true }));
  frame.renderOrder = 10;
  return frame;
}

/** Lines that stay visible through the scan (overlays are instruments, not objects). */
export function overlayLine({ color = '#ffffff', opacity = 1, vertexColors = false, depthTest = false } = {}): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({ color, opacity, transparent: true, depthTest, depthWrite: false, vertexColors });
}

/** A velocity vector: a shaft and a flat arrowhead, rescaled each frame. */
export class VelocityArrow {
  readonly object = new THREE.Group();
  private readonly shaft: THREE.Line;
  private readonly head: THREE.Mesh;
  private static readonly HEAD = 0.014;

  constructor() {
    const material = overlayLine({ color: INK.velocity });
    this.shaft = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 1, 0)]), material);
    const headGeometry = new THREE.ConeGeometry(VelocityArrow.HEAD * 0.42, VelocityArrow.HEAD, 12);
    headGeometry.translate(0, -VelocityArrow.HEAD / 2, 0);
    this.head = new THREE.Mesh(
      headGeometry,
      new THREE.MeshBasicMaterial({ color: INK.velocity, transparent: true, depthTest: false, depthWrite: false }),
    );
    this.object.add(this.shaft, this.head);
    this.object.renderOrder = 11;
    this.shaft.renderOrder = 11;
    this.head.renderOrder = 11;
    this.object.visible = false;
  }

  /** Point from `origin` along `velocity`; 1 m/s draws 25 cm, capped so a throw doesn't cross the screen. */
  update(origin: THREE.Vector3, velocity: THREE.Vector3, show: boolean): void {
    const speed = velocity.length();
    this.object.visible = show && speed > 0.03;
    if (!this.object.visible) return;
    const length = Math.min(0.03 + speed * 0.25, 0.4);
    this.object.position.copy(origin);
    this.object.quaternion.setFromUnitVectors(UP, velocity.clone().divideScalar(speed));
    this.shaft.scale.set(1, length, 1);
    this.head.position.set(0, length, 0);
  }
}

const UP = new THREE.Vector3(0, 1, 0);
const TRAIL_POINTS = 90; // about 1.5 s at 60 fps

/** Where an object has been lately: a line that fades toward its tail and retracts once it rests. */
export class Trail {
  readonly line: THREE.Line;
  private readonly positions = new Float32Array(TRAIL_POINTS * 3);
  private readonly colors = new Float32Array(TRAIL_POINTS * 4);
  private count = 0;
  private readonly tint = new THREE.Color(INK.trail);

  constructor() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 4).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.line = new THREE.Line(geometry, overlayLine({ vertexColors: true }));
    this.line.frustumCulled = false;
    this.line.renderOrder = 9;
  }

  update(point: THREE.Vector3, moving: boolean): void {
    if (moving) {
      // Newest point first; skip points that barely moved so a slow drift doesn't flood the buffer.
      const last = this.count > 0 ? new THREE.Vector3().fromArray(this.positions, 0) : null;
      if (!last || last.distanceToSquared(point) > 1e-6) {
        this.positions.copyWithin(3, 0, (TRAIL_POINTS - 1) * 3);
        point.toArray(this.positions, 0);
        this.count = Math.min(this.count + 1, TRAIL_POINTS);
      }
    } else if (this.count > 0) {
      this.count = Math.max(0, this.count - 2);
    }
    for (let i = 0; i < this.count; i++) {
      const alpha = (1 - i / TRAIL_POINTS) ** 1.6 * 0.85;
      this.colors.set([this.tint.r, this.tint.g, this.tint.b, alpha], i * 4);
    }
    const geometry = this.line.geometry;
    geometry.setDrawRange(0, this.count);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }

  clear(): void {
    this.count = 0;
    this.line.geometry.setDrawRange(0, 0);
  }
}

/**
 * The final grade, in linear light before the output transform: a soft highlight shoulder (bright
 * scans otherwise clip to flat white), a touch of contrast and saturation, a lens vignette and a
 * faint moving grain that keeps flat areas from banding.
 */
export function gradePass(): Pass & { uniforms: Record<string, THREE.IUniform> } {
  return new ShaderPass({
    name: 'Grade',
    uniforms: {
      tDiffuse: { value: null },
      uExposure: { value: 0.82 },
      uKnee: { value: 0.5 },
      uSaturation: { value: 1.08 },
      uVignette: { value: 0.32 },
      uGrain: { value: 0.018 },
      uTime: { value: 0 },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse;
      uniform float uExposure, uKnee, uSaturation, uVignette, uGrain, uTime;
      varying vec2 vUv;
      vec3 shoulder(vec3 x) {
        vec3 over = max(x - uKnee, 0.0);
        float room = 1.0 - uKnee;
        return min(x, vec3(uKnee)) + room * (1.0 - exp(-over / room));
      }
      float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
      void main() {
        vec4 texel = texture2D(tDiffuse, vUv);
        vec3 c = shoulder(texel.rgb * uExposure);
        float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = max(mix(vec3(luma), c, uSaturation), 0.0);
        vec2 q = vUv - 0.5;
        c *= 1.0 - uVignette * smoothstep(0.25, 0.85, length(q * vec2(1.15, 1.0)));
        c += (hash(vUv * 1024.0 + uTime) - 0.5) * uGrain * (0.35 + luma);
        gl_FragColor = vec4(c, texel.a);
      }
    `,
  }) as Pass & { uniforms: Record<string, THREE.IUniform> };
}

/** A vertical gradient for the void around the scene: a lit studio rather than a black hole. */
export function studioBackdrop(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 512;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, 512);
    gradient.addColorStop(0, '#1b1e23');
    gradient.addColorStop(0.55, '#111316');
    gradient.addColorStop(1, '#0a0b0d');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 2, 512);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
