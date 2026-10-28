import { describe, expect, it } from 'vitest';
import { equivalentEllipsoid, symmetricEigen, type Matrix3 } from './robotics';

function multiply(a: Matrix3, b: Matrix3): Matrix3 {
  return a.map((row) => [0, 1, 2].map((j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0))) as Matrix3;
}

function transpose(a: Matrix3): Matrix3 {
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[j][i])) as Matrix3;
}

describe('symmetricEigen', () => {
  it('diagonalizes a symmetric matrix with a proper rotation', () => {
    const m: Matrix3 = [
      [4, 1, -2],
      [1, 3, 0.5],
      [-2, 0.5, 5],
    ];
    const { values, vectors } = symmetricEigen(m);
    const back = multiply(multiply(vectors, [[values[0], 0, 0], [0, values[1], 0], [0, 0, values[2]]]), transpose(vectors));
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(back[i][j]).toBeCloseTo(m[i][j], 10);
    const identity = multiply(transpose(vectors), vectors);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(identity[i][j]).toBeCloseTo(i === j ? 1 : 0, 10);
    const det =
      vectors[0][0] * (vectors[1][1] * vectors[2][2] - vectors[1][2] * vectors[2][1]) -
      vectors[0][1] * (vectors[1][0] * vectors[2][2] - vectors[1][2] * vectors[2][0]) +
      vectors[0][2] * (vectors[1][0] * vectors[2][1] - vectors[1][1] * vectors[2][0]);
    expect(det).toBeCloseTo(1, 10);
  });

  it('leaves a diagonal matrix alone', () => {
    const { values } = symmetricEigen([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
    ]);
    expect(values).toEqual([1, 2, 3]);
  });
});

describe('equivalentEllipsoid', () => {
  it("recovers a solid box's proportions", () => {
    // A uniform 0.2 x 0.1 x 0.05 m box of 1 kg: I = m/12 (b^2 + c^2), ...
    const [a, b, c, m] = [0.2, 0.1, 0.05, 1];
    const inertial = {
      com_m: [0, 0, 0] as [number, number, number],
      inertia_kg_m2: [(m / 12) * (b * b + c * c), (m / 12) * (a * a + c * c), (m / 12) * (a * a + b * b), 0, 0, 0] as [
        number,
        number,
        number,
        number,
        number,
        number,
      ],
      model: 'solid' as const,
    };
    const { semiAxes } = equivalentEllipsoid(m, inertial);
    // An ellipsoid with the box's inertia has semi-axes sqrt(5/12) times the box's full sides.
    const k = Math.sqrt(5 / 12);
    expect(semiAxes.x).toBeCloseTo(k * a, 6);
    expect(semiAxes.y).toBeCloseTo(k * b, 6);
    expect(semiAxes.z).toBeCloseTo(k * c, 6);
  });
});
