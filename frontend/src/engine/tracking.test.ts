import { describe, expect, it } from 'vitest';
import { ADVANCE, BACK, decode, SKIP, STAY, Tracker, transitionMatrix, type Observation } from './tracking';

const reading = (timeS: number, steps: number, done: number): Observation => ({
  timeS,
  stepDone: Object.fromEntries(Array.from({ length: steps }, (_, i) => [i + 1, i + 1 <= done ? 0.9 : 0.1])),
});

describe('the transition model', () => {
  it('mostly stays, sometimes advances, rarely skips or undoes, and every row sums to 1', () => {
    const t = transitionMatrix(4);
    const total = STAY + ADVANCE + SKIP + BACK;
    expect(t[2][2]).toBeCloseTo(STAY / total);
    expect(t[2][3]).toBeCloseTo(ADVANCE / total);
    expect(t[2][4]).toBeCloseTo(SKIP / total);
    expect(t[2][1]).toBeCloseTo(BACK / total);
    for (const row of t) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });
});

describe('the live tracker', () => {
  it('follows consistent evidence to the end of the build', () => {
    const tracker = new Tracker(3);
    let belief = tracker.current();
    for (let k = 0; k <= 3; k++) for (let i = 0; i < 6; i++) belief = tracker.update(reading(k * 6 + i, 3, k));
    expect(belief.completed).toBe(3);
    expect(belief.confidence).toBeGreaterThan(0.9);
  });

  it('does not jump ahead on one misread frame', () => {
    const tracker = new Tracker(5);
    for (let i = 0; i < 8; i++) tracker.update(reading(i, 5, 1));
    const belief = tracker.update(reading(8, 5, 5)); // one frame says everything is done
    expect(belief.completed).toBeLessThanOrEqual(2);
  });
});

describe('decoding a whole video', () => {
  it('dates each step where the evidence changes, through a glitch', () => {
    const observations = Array.from({ length: 40 }, (_, t) => reading(t, 3, Math.min(3, Math.floor(t / 10))));
    observations[15] = reading(15, 3, 3);
    const timeline = decode(observations, 3);
    expect(timeline.path.at(-1)).toBe(3);
    expect(timeline.finishedAt).toEqual({ 1: 10, 2: 20, 3: 30 });
  });
});
