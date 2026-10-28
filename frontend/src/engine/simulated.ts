import type { Choreography } from './choreography';
import { Tracker, type Belief, type Observation } from './tracking';

/**
 * Simulated perception for the demo session: what a step-verification model would report about
 * each frame of the choreographed build, with realistic flaws. Steps done read ~0.9, steps not
 * started ~0.1, the step in progress rises while it is fastened; and about one frame in twelve is
 * occluded or misread, flipping one step's reading. The same tracker as the live system turns this
 * into a belief, so the demo shows the real filtering at work, on simulated evidence.
 */

export const OBSERVATION_EVERY_S = 0.5;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SimulatedTrack {
  observations: Observation[];
  beliefs: Belief[];
  /** Indices of frames that were misread on purpose. */
  glitches: Set<number>;
}

export function simulateTrack(choreography: Choreography, seed = 7): SimulatedTrack {
  const steps = choreography.plan.steps.length;
  const random = mulberry32(seed);
  const noise = () => (random() + random() + random() - 1.5) * 0.12;
  const tracker = new Tracker(steps);
  const observations: Observation[] = [];
  const beliefs: Belief[] = [];
  const glitches = new Set<number>();
  for (let t = OBSERVATION_EVERY_S, i = 0; t < choreography.duration; t += OBSERVATION_EVERY_S, i++) {
    const moment = choreography.moment(t);
    const stepDone: Record<number, number> = {};
    for (let k = 1; k <= steps; k++) {
      let p: number;
      if (k <= moment.completed) p = 0.88;
      else if (k === moment.completed + 1 && moment.phase === 'fasten') p = 0.3 + 0.35 * moment.u;
      else if (k === moment.completed + 1 && moment.phase === 'verify') p = 0.78;
      else p = 0.1;
      stepDone[k] = Math.min(0.99, Math.max(0.01, p + noise()));
    }
    if (random() < 0.085 && steps > 1) {
      const k = 1 + Math.floor(random() * steps);
      stepDone[k] = 1 - stepDone[k];
      glitches.add(i);
    }
    const observation = { timeS: t, stepDone };
    observations.push(observation);
    beliefs.push(tracker.update(observation));
  }
  return { observations, beliefs, glitches };
}

export function beliefAt(track: SimulatedTrack, time: number, steps: number): Belief {
  const index = Math.floor(time / OBSERVATION_EVERY_S) - 1;
  if (index < 0) {
    const probabilities = new Array<number>(steps + 1).fill(0);
    probabilities[0] = 1;
    return { probabilities, completed: 0, confidence: 1 };
  }
  return track.beliefs[Math.min(index, track.beliefs.length - 1)];
}
