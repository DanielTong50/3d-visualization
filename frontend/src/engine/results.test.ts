import { describe, expect, it } from 'vitest';
import { manualResult, plan, sessionResult } from '../test/library';
import { matchRecovered } from './results';

describe('laying a recovered plan over the truth', () => {
  it('matches every joint of a perfect reading', () => {
    const truth = plan('halden-bookcase');
    const matched = matchRecovered(truth, manualResult('halden-bookcase').recovered);
    expect([...matched.known.values()].flat()).toHaveLength(truth.joints.length);
    expect(matched.extra).toEqual([]);
    expect(matched.missed).toEqual([]);
  });

  it('reports a joint the plan does not have', () => {
    const truth = plan('varde-nightstand');
    const result = manualResult('varde-nightstand');
    const matched = matchRecovered(truth, result.recovered);
    expect(matched.missed).toEqual([]);
    expect(matched.extra.length).toBe(result.recovered.joints.length - truth.joints.length);
    expect(matched.extra.length).toBeGreaterThan(0);
  });

  it('agrees with the stored edge precision', () => {
    for (const id of ['halden-bookcase', 'varde-nightstand', 'lund-side-table', 'ostra-desk']) {
      const result = manualResult(id);
      const matched = matchRecovered(plan(id), result.recovered);
      const hits = [...matched.known.values()].flat().length;
      expect(hits / result.recovered.joints.length).toBeCloseTo(result.score.edges_precision, 3);
    }
  });
});

describe('a stored session', () => {
  it('decodes a monotone build history that ends with the piece built', () => {
    for (const id of ['halden-bookcase', 'ostra-desk']) {
      const result = sessionResult(id);
      expect(result.path.every((k, i) => i === 0 || k >= result.path[i - 1])).toBe(true);
      expect(result.path.at(-1)).toBe(plan(id).steps.length);
    }
  });
});
