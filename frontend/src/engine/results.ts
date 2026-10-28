import { madeAt } from './graph';
import { ASSEMBLIES_URL, type AssemblyPlan, type Orientation } from './plan';

/**
 * The stored outputs of `darius eval` (engine/darius/eval/benchmark.py): what the models read
 * from each generated manual and saw in each rendered session, with their scores. The demos
 * replay these files; nothing here is made up in the browser.
 */

export interface ManualScore {
  parts_f1: number;
  edges_precision: number;
  edges_recall: number;
  edges_f1: number;
  steps_exact: number;
  hardware_accuracy: number;
}

export interface ManualStepReading {
  index: number;
  title: string;
  adds: string[];
  hardware: { id: string; count: number }[];
  orientation: Orientation;
  tools: string[];
  connections: { a: string; b: string; kind: string }[];
}

export interface ManualResult {
  plan: string;
  model: string;
  pages: number;
  seconds: number;
  usage: {
    prompt_tokens?: number;
    completion_tokens?: number;
    completion_tokens_details?: { reasoning_tokens?: number | null };
  };
  score: ManualScore;
  notes: string[];
  reading: {
    title: string;
    category: string;
    parts: { label: string; name: string; count: number; size_mm: number[] }[];
    hardware: { id: string; kind: string; name: string; count: number }[];
    steps: ManualStepReading[];
  };
  recovered: AssemblyPlan;
}

export interface SessionScore {
  frames: number;
  frame_accuracy: number;
  raw_frame_accuracy: number;
  steps_found: number;
  steps: number;
  step_error_s: number | null;
  step_offset_s: number | null;
  edges_recall: number;
}

export interface SessionFrame {
  time_s: number;
  step_done: Record<string, number>;
  attached: string[];
  loose: string[];
  orientation: string;
}

export interface SessionResult {
  score: SessionScore;
  plan: string;
  model: string;
  every_s: number;
  duration_s: number;
  frames: SessionFrame[];
  path: number[];
  confidence: number[];
  posteriors: number[][];
  finished_at: Record<string, number>;
  joints: {
    joint: string;
    parts: string[];
    labels: string[];
    kind: string;
    step: number;
    made_at_s: number | null;
  }[];
  seconds: number;
}

export interface SessionTruth {
  fps: number;
  duration: number;
  frames: number;
  size: [number, number];
  steps: { index: number; start: number; done_at: number }[];
  segments: { step: number; phase: string; start: number; duration: number }[];
  completed: number[];
}

export interface TrackingScore {
  pieces: number;
  grounded: number;
  /** Letters right up to look-alikes (parts of the same size, told apart only by where they go). */
  letters_right: number;
  letters_exact: number;
  /** Mean over tracks of each track's mean IoU with its part's exact mask, wherever the part is visible. */
  mean_iou: number;
  /** Share of visible part-frames where the part's track overlaps it by IoU 0.5 or more. */
  frames_tracked: number;
  contacts_precision: number;
  contacts_recall: number;
  frames_right_spark: number;
  frames_right_contacts: number;
  frames_right_fused: number;
  frames_right_exact_contacts: number;
  step_error_fused_s: number | null;
}

export interface TrackFrame {
  frame: number;
  box: [number, number, number, number];
  area: number;
  confidence: number | null;
  /** Outline in clip pixels: x0, y0, x1, y1, ... */
  polygon: number[];
}

export interface PartPrompt {
  object: number;
  label: string;
  frame: number;
  box: [number, number, number, number];
  point: [number, number] | null;
}

/** Muse Spark's prompts and SAM 3.1's part tracks through a session (session/tracks.json). */
export interface TrackingResult {
  score: TrackingScore;
  /** Each object's matched part (by overlap on the anchor frame) and its mean IoU with it. */
  parts: Record<string, { part: string; iou: number | null }>;
  /** Pairs of objects whose masks touched for a second without a break, and when that was. */
  contacts: { objects: [number, number]; letters: [string, string]; at: number }[];
  plan: string;
  model: string;
  grounding_model: string;
  fps: number;
  size: [number, number];
  anchor_frame: number;
  frames: number;
  prompts: PartPrompt[];
  tracks: { object: number; label: string; frames: TrackFrame[] }[];
  seconds: number;
}

export interface Benchmarks {
  model: string | null;
  date: string;
  manuals: {
    items: Record<string, ManualScore & { seconds: number; pages: number }>;
    mean: Partial<ManualScore>;
  };
  sessions: {
    items: Record<string, SessionScore & { seconds: number; duration_s: number }>;
    mean: Partial<SessionScore>;
  };
  tracking?: {
    models: { segmentation: string; grounding: string } | null;
    items: Record<string, TrackingScore & { seconds: number; frames: number; prompts?: number; refound?: number }>;
    mean: Partial<TrackingScore>;
  };
}

async function json<T>(url: string): Promise<T | null> {
  const response = await fetch(url);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`could not load ${url} (HTTP ${response.status})`);
  return (await response.json()) as T;
}

export const manualPageUrl = (id: string, page: number) => `${ASSEMBLIES_URL}/${id}/manual/page-${String(page).padStart(2, '0')}.png`;
export const sessionVideoUrl = (id: string) => `${ASSEMBLIES_URL}/${id}/session/session.mp4`;
export const loadManualResult = (id: string) => json<ManualResult>(`${ASSEMBLIES_URL}/${id}/manual/reading.json`);
export const loadSessionResult = (id: string) => json<SessionResult>(`${ASSEMBLIES_URL}/${id}/session/analysis.json`);
export const loadSessionTruth = (id: string) => json<SessionTruth>(`${ASSEMBLIES_URL}/${id}/session/truth.json`);
export const loadTrackingResult = (id: string) => json<TrackingResult>(`${ASSEMBLIES_URL}/${id}/session/tracks.json`);
export const loadBenchmarks = () => json<Benchmarks>(`${ASSEMBLIES_URL}/benchmarks.json`);

/** The plan's joints SAM 3.1's contacts have shown by a time, and contacts between parts no joint joins. */
export function contactGraph(plan: AssemblyPlan, tracks: TrackingResult, time: number) {
  const byPair = new Map(plan.joints.map((j) => [[...j.parts].sort().join('|'), j.id]));
  const seen = new Set<string>();
  const extra: { id: string; parts: [string, string] }[] = [];
  for (const contact of tracks.contacts) {
    if (contact.at > time) continue;
    const [a, b] = contact.objects.map((o) => tracks.parts[String(o)]?.part);
    if (!a || !b || a === b) continue;
    const joint = byPair.get([a, b].sort().join('|'));
    if (joint) seen.add(joint);
    else extra.push({ id: `c-${contact.objects.join('-')}`, parts: [a, b] });
  }
  return { seen, extra };
}

export interface Matched {
  /** Truth joints matched by each recovered step. */
  known: Map<number, string[]>;
  /** Recovered joints with no counterpart in the truth (drawn between parts with those letters). */
  extra: { id: string; step: number; parts: [string, string]; kind: string }[];
  /** Truth joints nothing recovered matches. */
  missed: string[];
}

/**
 * Lay a recovered plan over the truth: each recovered joint takes a truth joint that joins
 * parts with the same letters (from the same step if there is one), the way the scores count.
 */
export function matchRecovered(truth: AssemblyPlan, recovered: AssemblyPlan): Matched {
  const truthLabel = new Map(truth.parts.map((p) => [p.id, p.label]));
  const recoveredLabel = new Map(recovered.parts.map((p) => [p.id, p.label]));
  const firstOf = new Map<string, string>();
  for (const part of truth.parts) if (!firstOf.has(part.label)) firstOf.set(part.label, part.id);
  const key = (labels: string[]) => [...labels].sort().join('–');
  const truthStep = madeAt(truth);
  const open = new Map<string, string[]>();
  for (const joint of truth.joints) {
    const k = key(joint.parts.map((p) => truthLabel.get(p)!));
    open.set(k, [...(open.get(k) ?? []), joint.id]);
  }
  const joints = new Map(recovered.joints.map((j) => [j.id, j]));
  const known = new Map<number, string[]>();
  const extra: Matched['extra'] = [];
  for (const step of recovered.steps) {
    for (const id of step.joints) {
      const joint = joints.get(id)!;
      const labels = joint.parts.map((p) => recoveredLabel.get(p) ?? p);
      const pool = open.get(key(labels)) ?? [];
      if (pool.length) {
        const same = pool.findIndex((t) => truthStep.get(t) === step.index);
        const [match] = pool.splice(same >= 0 ? same : 0, 1);
        known.set(step.index, [...(known.get(step.index) ?? []), match]);
      } else {
        const [a, b] = labels.map((l) => firstOf.get(l));
        if (a && b && a !== b)
          extra.push({
            id: `x-${id}`,
            step: step.index,
            parts: [a, b],
            kind: joint.kind,
          });
      }
    }
  }
  return { known, extra, missed: [...open.values()].flat() };
}
