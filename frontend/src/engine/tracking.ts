/**
 * Plan-conditioned progress tracking (mirror of engine/darius/assembly/tracking.py): a hidden
 * Markov model over k, the number of completed steps. Transitions mostly stay, sometimes advance
 * one step, rarely skip or undo. Each observation gives P(step i done) for the steps a frame can
 * see, capped to [0.15, 0.85] and tempered, so a single frame never moves the build on its own.
 */

export const STAY = 0.86;
export const ADVANCE = 0.11;
export const SKIP = 0.02;
export const BACK = 0.01;
export const TEMPER = 0.7;
export const EVIDENCE_CAP = 0.85;

export interface Observation {
  timeS: number;
  /** step index (1-based) -> P(done) */
  stepDone: Record<number, number>;
}

export interface Belief {
  probabilities: number[];
  completed: number;
  confidence: number;
}

export function transitionMatrix(steps: number): number[][] {
  const n = steps + 1;
  return Array.from({ length: n }, (_, a) => {
    const row = new Array<number>(n).fill(0);
    for (const [b, p] of [
      [a, STAY],
      [a + 1, ADVANCE],
      [a + 2, SKIP],
      [a - 1, BACK],
    ] as const) {
      if (b >= 0 && b < n) row[b] += p;
    }
    const total = row.reduce((s, v) => s + v, 0);
    return row.map((v) => v / total);
  });
}

export function logEmission(observation: Observation, steps: number): number[] {
  const out = new Array<number>(steps + 1).fill(0);
  for (const [key, value] of Object.entries(observation.stepDone)) {
    const index = Number(key);
    if (!(index >= 1 && index <= steps)) continue;
    const p = Math.min(Math.max(value, 1 - EVIDENCE_CAP), EVIDENCE_CAP);
    for (let k = 0; k <= steps; k++) out[k] += TEMPER * (k >= index ? Math.log(p) : Math.log(1 - p));
  }
  return out;
}

function normalizeLog(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const total = exp.reduce((s, v) => s + v, 0);
  return exp.map((v) => v / total);
}

function toBelief(p: number[]): Belief {
  let completed = 0;
  for (let k = 1; k < p.length; k++) if (p[k] > p[completed]) completed = k;
  return { probabilities: p, completed, confidence: p[completed] };
}

/** Online belief, updated one observation at a time (the live session). */
export class Tracker {
  private readonly transitions: number[][];
  private belief: number[];

  constructor(readonly steps: number) {
    this.transitions = transitionMatrix(steps);
    this.belief = new Array<number>(steps + 1).fill(0);
    this.belief[0] = 1;
  }

  update(observation: Observation): Belief {
    const n = this.steps + 1;
    const predicted = new Array<number>(n).fill(0);
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) predicted[b] += this.belief[a] * this.transitions[a][b];
    const emission = logEmission(observation, this.steps);
    this.belief = normalizeLog(predicted.map((p, k) => Math.log(Math.max(p, 1e-300)) + emission[k]));
    return toBelief(this.belief);
  }

  current(): Belief {
    return toBelief(this.belief);
  }

  reset(completed = 0): void {
    this.belief = new Array<number>(this.steps + 1).fill(0);
    this.belief[completed] = 1;
  }
}

export interface Timeline {
  times: number[];
  path: number[];
  confidence: number[];
  posteriors: number[][];
  finishedAt: Record<number, number>;
}

function logSumExp(values: number[]): number {
  const max = Math.max(...values);
  if (max === -Infinity) return -Infinity;
  return max + Math.log(values.reduce((s, v) => s + Math.exp(v - max), 0));
}

/** Viterbi path and forward-backward posteriors over a whole recording. */
export function decode(observations: Observation[], steps: number): Timeline {
  const T = observations.length;
  if (T === 0) return { times: [], path: [], confidence: [], posteriors: [], finishedAt: {} };
  const n = steps + 1;
  const logT = transitionMatrix(steps).map((row) => row.map((v) => Math.log(Math.max(v, 1e-300))));
  const emissions = observations.map((o) => logEmission(o, steps));
  const start = Array.from({ length: n }, (_, k) => (k === 0 ? 0 : -Infinity));

  let score = start.map((s, k) => s + emissions[0][k]);
  const back: number[][] = [new Array<number>(n).fill(0)];
  for (let t = 1; t < T; t++) {
    const next = new Array<number>(n);
    const pointers = new Array<number>(n);
    for (let b = 0; b < n; b++) {
      let best = -Infinity;
      let arg = 0;
      for (let a = 0; a < n; a++) {
        const v = score[a] + logT[a][b];
        if (v > best) {
          best = v;
          arg = a;
        }
      }
      next[b] = best + emissions[t][b];
      pointers[b] = arg;
    }
    score = next;
    back.push(pointers);
  }
  const path = [score.indexOf(Math.max(...score))];
  for (let t = T - 1; t > 0; t--) path.push(back[t][path[path.length - 1]]);
  path.reverse();

  const alpha: number[][] = [start.map((s, k) => s + emissions[0][k])];
  for (let t = 1; t < T; t++) {
    alpha.push(Array.from({ length: n }, (_, b) => logSumExp(alpha[t - 1].map((v, a) => v + logT[a][b])) + emissions[t][b]));
  }
  const beta: number[][] = new Array(T);
  beta[T - 1] = new Array<number>(n).fill(0);
  for (let t = T - 2; t >= 0; t--) {
    beta[t] = Array.from({ length: n }, (_, a) => logSumExp(logT[a].map((v, b) => v + emissions[t + 1][b] + beta[t + 1][b])));
  }
  const posteriors = alpha.map((row, t) => normalizeLog(row.map((v, k) => v + beta[t][k])));
  const times = observations.map((o) => o.timeS);
  const finishedAt: Record<number, number> = {};
  for (let t = 1; t < T; t++) for (let k = path[t - 1] + 1; k <= path[t]; k++) finishedAt[k] ??= times[t];
  return { times, path, confidence: posteriors.map((p, t) => p[path[t]]), posteriors, finishedAt };
}
