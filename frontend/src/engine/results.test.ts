import { describe, expect, it } from 'vitest';
import { manualResult, plan, sessionResult, trackingResult } from '../test/library';
import { contactGraph, matchRecovered } from './results';

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

describe("SAM 3.1's stored part tracks", () => {
  const ids = ['halden-bookcase', 'varde-nightstand', 'lund-side-table', 'ostra-desk'];

  it('show the joints the stored score says were touched, by the end', () => {
    for (const id of ids) {
      const tracks = trackingResult(id);
      const { seen } = contactGraph(plan(id), tracks, Infinity);
      expect(seen.size / plan(id).joints.length).toBeCloseTo(tracks.score.contacts_recall, 3);
    }
  });

  it('build the contact graph up over time, never down', () => {
    const tracks = trackingResult('ostra-desk');
    let before = 0;
    for (let t = 0; t <= tracks.frames / tracks.fps; t += 2) {
      const now = contactGraph(plan('ostra-desk'), tracks, t);
      expect(now.seen.size + now.extra.length).toBeGreaterThanOrEqual(before);
      before = now.seen.size + now.extra.length;
    }
  });

  it('keep every outline inside the clip, with prompts on the anchor frame for every track', () => {
    for (const id of ids) {
      const tracks = trackingResult(id);
      const [width, height] = tracks.size;
      for (const track of tracks.tracks) {
        expect(tracks.prompts.some((p) => p.object === track.object && p.frame === tracks.anchor_frame)).toBe(true);
        for (const frame of track.frames) {
          expect(frame.polygon.length % 2).toBe(0);
          expect(frame.polygon.every((v, i) => v >= 0 && v <= (i % 2 ? height : width))).toBe(true);
        }
      }
    }
  });
});
