import * as THREE from 'three';
import type { Grasp, Inertial } from './scene';

/**
 * What the robot view draws for each object, from the scene's robotics annotations (computed by
 * the pipeline, darius.robotics): its center of mass, its inertia as an equivalent ellipsoid, and
 * its parallel-jaw grasp candidates. Plus the robot world frame, which is +z up.
 */

/** Robot-view ink. */
export const ROBOT_INK = {
  com: '#ff4f00',
  inertia: '#7cc4ff',
  grasp: '#ffffff',
  bestGrasp: '#ff4f00',
} as const;

/** How deep the drawn gripper fingers reach past the contacts, and the stem behind the palm. */
const FINGER_M = 0.035;
const STEM_M = 0.03;

export type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]];

export function inertiaTensor({ inertia_kg_m2: [ixx, iyy, izz, ixy, ixz, iyz] }: Inertial): Matrix3 {
  return [
    [ixx, ixy, ixz],
    [ixy, iyy, iyz],
    [ixz, iyz, izz],
  ];
}

/**
 * Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations. Returns eigenvalues
 * and the matching eigenvectors as the columns of a proper rotation (determinant +1).
 */
export function symmetricEigen(matrix: Matrix3): { values: [number, number, number]; vectors: Matrix3 } {
  const a = matrix.map((row) => [...row]) as Matrix3;
  const v: Matrix3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-15 * (Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]) + 1e-300)) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const det =
    v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) -
    v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) +
    v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
  if (det < 0) for (let k = 0; k < 3; k++) v[k][2] = -v[k][2];
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: v };
}

/**
 * Semi-axes of the uniform solid ellipsoid with the same mass and inertia (the "equivalent
 * ellipsoid"): a^2 = 5 / (2m) (I2 + I3 - I1), and so on. Long axes are directions mass is spread
 * along, which is what makes an object hard to spin about the others.
 */
export function equivalentEllipsoid(mass: number, inertial: Inertial): { semiAxes: THREE.Vector3; rotation: THREE.Quaternion } {
  const { values: [i1, i2, i3], vectors } = symmetricEigen(inertiaTensor(inertial));
  const axis = (a: number, b: number, c: number) => Math.sqrt(Math.max((5 / (2 * mass)) * (b + c - a), 0));
  const basis = new THREE.Matrix4().set(
    vectors[0][0], vectors[0][1], vectors[0][2], 0,
    vectors[1][0], vectors[1][1], vectors[1][2], 0,
    vectors[2][0], vectors[2][1], vectors[2][2], 0,
    0, 0, 0, 1,
  ); // prettier-ignore
  return {
    semiAxes: new THREE.Vector3(axis(i1, i2, i3), axis(i2, i1, i3), axis(i3, i1, i2)),
    rotation: new THREE.Quaternion().setFromRotationMatrix(basis),
  };
}

function lineMaterial(color: string, opacity = 1): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({ color, opacity, transparent: true, depthTest: false, depthWrite: false });
}

/** The equivalent ellipsoid as three rings in its principal planes, placed at the center of mass. */
export function inertiaRings(mass: number, inertial: Inertial): THREE.LineSegments {
  const { semiAxes, rotation } = equivalentEllipsoid(mass, inertial);
  const points: number[] = [];
  const segments = 48;
  const ring = (u: number, v: number) => {
    for (let i = 0; i < segments; i++) {
      for (const k of [i, i + 1]) {
        const angle = (2 * Math.PI * k) / segments;
        const p = [0, 0, 0];
        p[u] = Math.cos(angle) * semiAxes.getComponent(u);
        p[v] = Math.sin(angle) * semiAxes.getComponent(v);
        points.push(...p);
      }
    }
  };
  ring(0, 1);
  ring(0, 2);
  ring(1, 2);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const rings = new THREE.LineSegments(geometry, lineMaterial(ROBOT_INK.inertia, 0.45));
  rings.quaternion.copy(rotation);
  rings.position.set(...inertial.com_m);
  rings.renderOrder = 12;
  return rings;
}

/** A center-of-mass mark: a small cross that stays visible through the object. */
export function comMarker(inertial: Inertial, size = 0.012): THREE.LineSegments {
  const s = size;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-s, 0, 0, s, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, -s, 0, 0, s], 3));
  const marker = new THREE.LineSegments(geometry, lineMaterial(ROBOT_INK.com));
  marker.position.set(...inertial.com_m);
  marker.renderOrder = 13;
  return marker;
}

/**
 * A parallel-jaw gripper at a grasp, in the grasp frame (+z approach, +y closing axis): two fingers
 * reaching to the contacts, the palm behind them and the approach stem.
 */
export function gripperGlyph(grasp: Grasp, best: boolean): THREE.LineSegments {
  const half = grasp.width_m / 2;
  const back = -FINGER_M;
  const points = [
    0, -half, 0, 0, -half, back, // finger
    0, half, 0, 0, half, back, // finger
    0, -half, back, 0, half, back, // palm
    0, 0, back, 0, 0, back - STEM_M, // approach stem
  ]; // prettier-ignore
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const glyph = new THREE.LineSegments(geometry, lineMaterial(best ? ROBOT_INK.bestGrasp : ROBOT_INK.grasp, best ? 1 : 0.25 + 0.6 * grasp.score));
  glyph.position.set(...grasp.position);
  glyph.quaternion.set(...grasp.rotation_quat);
  glyph.renderOrder = best ? 14 : 13;
  return glyph;
}

/**
 * The robot world frame at the table origin: x (red) along the scene's x, y (green) away from the
 * filming side, z (blue) up. Every robotics export uses this frame.
 */
export function worldFrame(length = 0.12): THREE.LineSegments {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, length, 0, 0, 0, 0, 0, 0, 0, -length, 0, 0, 0, 0, length, 0], 3));
  const colors = ['#e5484d', '#e5484d', '#46a758', '#46a758', '#3e8ef7', '#3e8ef7'].flatMap((hex) => new THREE.Color(hex).toArray());
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const frame = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthTest: false, depthWrite: false }),
  );
  frame.renderOrder = 12;
  return frame;
}
