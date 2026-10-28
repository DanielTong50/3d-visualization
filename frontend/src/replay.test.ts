import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { parsePoints, STAGE_GRAPH } from './replay';

describe('parsePoints', () => {
  it('reads points.bin as the pipeline writes it', () => {
    const count = 3;
    const buffer = new ArrayBuffer(4 + 16 * count);
    const view = new DataView(buffer);
    view.setUint32(0, count, true);
    [0.1, 0.2, 0.3, 1, 2, 3, -1, -2, -3].forEach((v, i) => view.setFloat32(4 + 4 * i, v, true));
    new Uint8Array(buffer, 4 + 12 * count, 3 * count).set([255, 0, 0, 0, 255, 0, 0, 0, 255]);
    new Uint8Array(buffer, 4 + 15 * count, count).set([0, 1, 2]);
    const cloud = parsePoints(buffer);
    expect(cloud.count).toBe(3);
    expect(Array.from(cloud.positions)).toEqual([0.1, 0.2, 0.3, 1, 2, 3, -1, -2, -3].map((v) => Math.fround(v)));
    expect(Array.from(cloud.colors.slice(3, 6))).toEqual([0, 255, 0]);
    expect(Array.from(cloud.labels)).toEqual([0, 1, 2]);
  });

  it("reads a shipped scene's replay cloud", () => {
    const path = fileURLToPath(new URL('../../scenes/kitchen-counter/pipeline/points.bin', import.meta.url));
    const bytes = readFileSync(path);
    const cloud = parsePoints(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    expect(cloud.count).toBeGreaterThan(1000);
    expect(new Set(cloud.labels).size).toBeGreaterThan(1); // background plus objects
  });

  it('rejects a truncated file', () => {
    const buffer = new ArrayBuffer(4 + 10);
    new DataView(buffer).setUint32(0, 5, true);
    expect(() => parsePoints(buffer)).toThrow(/truncated/);
  });
});

describe('STAGE_GRAPH', () => {
  it('lists every stage after the stages it needs', () => {
    const seen = new Set<string>();
    for (const stage of STAGE_GRAPH) {
      for (const need of stage.needs) expect(seen.has(need)).toBe(true);
      seen.add(stage.name);
    }
    expect(seen.size).toBe(7);
  });
});
