import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import type { SceneObject, TablePlane, Vec3 } from './scene';
import { backgroundVisibility, objectVisibility, patchSplats, tableEdgeDistance, tablePatches } from './splatCleanup';

const TABLE: TablePlane = { point: [0, 0, 0], normal: [0, 1, 0], size_m: [1, 0.8] };

/** A 10 cm cube with the given center. */
function cube(position: Vec3): SceneObject {
  return {
    id: 'obj_01',
    label: 'box',
    confidence: 1,
    material: 'cardboard',
    dimensions_m: [0.1, 0.1, 0.1],
    volume_m3: 0.001,
    hollow: false,
    fill_fraction: 1,
    density_kg_m3: 500,
    mass_kg: 0.5,
    friction: 0.5,
    restitution: 0.1,
    rigidity: 'rigid',
    fragile: false,
    collider: { type: 'box', half_extents: [0.05, 0.05, 0.05] },
    pose: { position, rotation_quat: [0, 0, 0, 1] },
    physics_source: 'hand-authored',
  };
}

describe('tableEdgeDistance', () => {
  it('is negative inside the table rectangle and positive outside it', () => {
    const distance = tableEdgeDistance(TABLE);
    expect(distance(0, 0)).toBeCloseTo(-0.4);
    expect(distance(0.6, 0)).toBeCloseTo(0.1);
  });

  it('uses the outline polygon, whichever way it winds', () => {
    const square: [number, number][] = [
      [-0.5, -0.5],
      [0.5, -0.5],
      [0.5, 0.5],
      [-0.5, 0.5],
    ];
    for (const outline of [square, [...square].reverse()]) {
      const distance = tableEdgeDistance({ ...TABLE, polygon_m: outline });
      expect(distance(0, 0)).toBeCloseTo(-0.5);
      expect(distance(0, 0.7)).toBeCloseTo(0.2);
    }
  });
});

describe('backgroundVisibility', () => {
  const visibility = backgroundVisibility(TABLE);

  it('keeps small splats on the table surface', () => {
    expect(visibility(0, 0.001, 0, 0.002)).toBe(1);
  });

  it('drops the backdrop over the table, splats past its edge or under it, and big blurry ones', () => {
    expect(visibility(0, 0.2, 0, 0.002)).toBe(0);
    expect(visibility(0.6, 0, 0, 0.002)).toBe(0);
    expect(visibility(0, -0.1, 0, 0.002)).toBe(0);
    expect(visibility(0, 0, 0, 0.1)).toBe(0);
  });
});

describe('tablePatches', () => {
  it('covers what was left of an object where it stood, above and below the table', () => {
    const [patch] = tablePatches([cube([0, 0.05, 0])], TABLE);
    expect(patch.covers(0, 0.08, 0)).toBe(true); // floating where the box was
    expect(patch.covers(0.06, 0.005, 0)).toBe(true); // just past its side, within the margin
    expect(patch.covers(0, -0.01, 0)).toBe(true); // sunk below the table under it
    expect(patch.covers(0.2, 0.005, 0)).toBe(false);
  });

  it('samples table colors from a band around the patch, not from inside it', () => {
    const [patch] = tablePatches([cube([0, 0.05, 0])], TABLE);
    expect(patch.samples(0.08, 0.001, 0)).toBe(true);
    expect(patch.samples(0.08, 0.02, 0)).toBe(false); // off the table surface
    expect(patch.samples(0, 0.001, 0)).toBe(false); // inside the patch
    expect(patch.samples(0.3, 0.001, 0)).toBe(false); // too far away
  });

  it('lays its splats on the table over the footprint and its margin', () => {
    const [patch] = tablePatches([cube([0, 0.05, 0])], TABLE);
    const xs = patch.grid.map((p) => p.x);
    expect(Math.min(...xs)).toBeCloseTo(-0.065, 3);
    expect(Math.max(...xs)).toBeCloseTo(0.065, 2);
    expect(patch.grid.every((p) => p.y === 0)).toBe(true);
  });

  it('leaves no patch under an object resting on another', () => {
    expect(tablePatches([cube([0, 0.2, 0])], TABLE)).toEqual([]);
  });
});

describe('patchSplats', () => {
  it('colors the patch from the middle of the samples, skipping shadows and glints', () => {
    const samples = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1].map((v) => new THREE.Color(v, v, v));
    const colors: number[] = [];
    const grid = Array.from({ length: 50 }, (_, i) => new THREE.Vector3(i * 0.003, 0, 0));
    patchSplats(TABLE, grid, samples, (_center, _scales, _quaternion, opacity, color) => {
      expect(opacity).toBe(1);
      colors.push(color.r);
    });
    expect(colors).toHaveLength(50);
    expect(colors.every((r) => r === 0.5)).toBe(true);
  });
});

describe('objectVisibility', () => {
  const visibility = objectVisibility([0.1, 0.1, 0.1]);
  const small = new THREE.Vector3(0.001, 0.001, 0.001);

  it('keeps small splats inside the object', () => {
    expect(visibility(new THREE.Vector3(0.02, 0, 0), small)).toBe(1);
  });

  it('drops splats outside its box, needles and oversized splats', () => {
    expect(visibility(new THREE.Vector3(0.07, 0, 0), small)).toBe(0);
    expect(visibility(new THREE.Vector3(), new THREE.Vector3(0.005, 0.0002, 0.0002))).toBe(0);
    expect(visibility(new THREE.Vector3(), new THREE.Vector3(0.02, 0.02, 0.02))).toBe(0);
  });
});
