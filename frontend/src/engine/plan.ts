/**
 * The assembly plan: TypeScript mirror of engine/darius/assembly/schema.py (the source of truth;
 * docs/plan.schema.json is generated from it). Lengths are millimeters in the furniture frame:
 * +y up, the floor is y = 0, the front faces +z. Quaternions are [x, y, z, w].
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

export type Orientation = 'upright' | 'on_back' | 'on_front' | 'on_side' | 'upside_down';

export interface Pose {
  position: Vec3;
  rotation?: Quat;
}

export interface Part {
  id: string;
  label: string;
  name: string;
  kind: string;
  size_mm: Vec3;
  material: string;
  color: string;
  assembled?: Pose | null;
  layout?: Pose | null;
}

export interface Hardware {
  id: string;
  name: string;
  kind: string;
  count: number;
  size_mm: number[];
}

export interface HardwareUse {
  id: string;
  count: number;
}

export interface Fastener {
  hardware: string;
  position: Vec3;
  axis: Vec3;
}

export interface Joint {
  id: string;
  parts: [string, string];
  kind: string;
  hardware: HardwareUse[];
  fasteners: Fastener[];
}

export interface Action {
  kind: string;
  target: string;
  axis?: 'x' | 'y' | 'z' | null;
  degrees?: number | null;
  text: string;
}

export interface Step {
  index: number;
  title: string;
  instruction: string;
  parts: string[];
  joints: string[];
  hardware: HardwareUse[];
  actions: Action[];
  orientation: Orientation;
  tools: string[];
}

export interface AssemblyPlan {
  version: 1;
  id: string;
  title: string;
  category: string;
  source: 'procedural' | 'manual' | 'hand-authored';
  units: 'mm';
  dimensions_mm?: Vec3 | null;
  parts: Part[];
  hardware: Hardware[];
  joints: Joint[];
  steps: Step[];
  notes: string[];
}

export interface PlanSummary {
  id: string;
  title: string;
  category: string;
  dimensions_mm: Vec3;
  parts: number;
  joints: number;
  steps: number;
  hardware: number;
}

/** Plans are served next to the app, so this works under any base path (e.g. GitHub Pages). */
export const ASSEMBLIES_URL = `${import.meta.env.BASE_URL}assemblies`;

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (HTTP ${response.status})`);
  return (await response.json()) as T;
}

export function loadLibrary(): Promise<PlanSummary[]> {
  return json<PlanSummary[]>(`${ASSEMBLIES_URL}/index.json`);
}

const cache = new Map<string, Promise<AssemblyPlan>>();

export function loadPlan(id: string): Promise<AssemblyPlan> {
  if (!cache.has(id)) cache.set(id, json<AssemblyPlan>(`${ASSEMBLIES_URL}/${id}/plan.json`).then(checkPlan));
  return cache.get(id)!;
}

/** The checks the views rely on: every joint's parts exist and every step's joints exist. */
export function checkPlan(plan: AssemblyPlan): AssemblyPlan {
  const parts = new Set(plan.parts.map((p) => p.id));
  const joints = new Set(plan.joints.map((j) => j.id));
  for (const joint of plan.joints) {
    for (const part of joint.parts) if (!parts.has(part)) throw new Error(`joint ${joint.id} uses unknown part ${part}`);
  }
  for (const step of plan.steps) {
    for (const joint of step.joints) if (!joints.has(joint)) throw new Error(`step ${step.index} makes unknown joint ${joint}`);
  }
  return plan;
}

export function partById(plan: AssemblyPlan): Map<string, Part> {
  return new Map(plan.parts.map((p) => [p.id, p]));
}

export function jointById(plan: AssemblyPlan): Map<string, Joint> {
  return new Map(plan.joints.map((j) => [j.id, j]));
}

export function hardwareName(plan: AssemblyPlan, id: string): string {
  return plan.hardware.find((h) => h.id === id)?.name ?? id;
}
