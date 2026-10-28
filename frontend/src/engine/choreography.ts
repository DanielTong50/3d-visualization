import * as THREE from 'three';
import { introducedAt, subassemblies } from './graph';
import type { AssemblyPlan, Orientation, Part, Step } from './plan';

/**
 * The build as it plays out in space and time: where every part is at time t.
 *
 * Each step is a sequence of phases: turn the assembly to the step's build orientation (a bookcase
 * is built on its side, nailed face down, stood up), locate the step's parts on the floor, carry
 * them along arcs to their targets, fasten, verify. The assembly is placed so it rests on the
 * floor at the work area; a subassembly built separately (a drawer box, the second frame of a
 * table) waits upright beside it until the step that joins it on.
 *
 * Everything is in meters (plans are in millimeters), three.js axes: +y up.
 */

export type Phase = 'intro' | 'reorient' | 'locate' | 'place' | 'fasten' | 'verify' | 'done';

export const PHASE_SECONDS: Record<Exclude<Phase, 'reorient' | 'intro' | 'done'>, number> = {
  locate: 2.2,
  place: 3.4,
  fasten: 2.0,
  verify: 1.6,
};
export const REORIENT_SECONDS = 2.4;
export const INTRO_SECONDS = 1.6;
export const DONE_SECONDS = 4.0;
const STAGGER_S = 0.35;
const SIDE_GAP_M = 0.35;

const MM = 0.001;
const HALF = Math.SQRT1_2;
export const ORIENTATIONS: Record<Orientation, THREE.Quaternion> = {
  upright: new THREE.Quaternion(),
  on_back: new THREE.Quaternion(-HALF, 0, 0, HALF),
  on_front: new THREE.Quaternion(HALF, 0, 0, HALF),
  on_side: new THREE.Quaternion(0, 0, HALF, HALF),
  upside_down: new THREE.Quaternion(0, 0, 1, 0),
};

export interface Segment {
  step: number; // 1-based; 0 for the intro, steps + 1 for the finale
  phase: Phase;
  start: number;
  duration: number;
}

export interface Moment {
  time: number;
  /** The step in progress (1-based), or 0 before the first, or steps + 1 when done. */
  step: number;
  phase: Phase;
  /** 0..1 through the phase. */
  u: number;
  /** Steps fully done. */
  completed: number;
}

export interface PartState {
  matrix: THREE.Matrix4;
  /** loose: on the floor; moving: being carried; attached: part of a (sub)assembly. */
  status: 'loose' | 'moving' | 'attached';
  /** Where it is headed while it belongs to the current step (null otherwise). */
  target: THREE.Matrix4 | null;
}

function box(part: Part): THREE.Box3 {
  const center = new THREE.Vector3(...(part.assembled?.position ?? [0, 0, 0])).multiplyScalar(MM);
  const half = new THREE.Vector3(...part.size_mm).multiplyScalar(MM / 2);
  return new THREE.Box3(center.clone().sub(half), center.clone().add(half));
}

export function poseMatrix(position: number[], rotation?: number[]): THREE.Matrix4 {
  const q = rotation ? new THREE.Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]) : new THREE.Quaternion();
  return new THREE.Matrix4().compose(new THREE.Vector3(...position).multiplyScalar(MM), q, new THREE.Vector3(1, 1, 1));
}

function smooth(u: number): number {
  const x = Math.min(Math.max(u, 0), 1);
  return x * x * (3 - 2 * x);
}

export class Choreography {
  readonly segments: Segment[] = [];
  readonly duration: number;
  /** The part whose component is "the assembly" (the first part of step 1). */
  readonly mainPart: string;
  private readonly parts: Map<string, Part>;
  private readonly assembled: Map<string, THREE.Matrix4>;
  private readonly layout: Map<string, THREE.Matrix4>;
  private readonly introduced: Map<string, number>;
  /** Components after k steps, as part -> component representative, k = 0..steps. */
  private readonly components: Map<string, string>[] = [];
  private readonly furniture: THREE.Box3;
  readonly workCenter = new THREE.Vector3();
  readonly layoutCenter = new THREE.Vector3();

  constructor(readonly plan: AssemblyPlan) {
    this.parts = new Map(plan.parts.map((p) => [p.id, p]));
    this.assembled = new Map(plan.parts.map((p) => [p.id, poseMatrix(p.assembled?.position ?? [0, 0, 0], p.assembled?.rotation)]));
    this.layout = new Map(
      plan.parts.map((p) => [p.id, poseMatrix(p.layout?.position ?? p.assembled?.position ?? [0, 0, 0], p.layout?.rotation)]),
    );
    this.introduced = introducedAt(plan);
    this.mainPart = plan.steps[0]?.parts[0] ?? plan.parts[0].id;
    for (let k = 0; k <= plan.steps.length; k++) {
      const map = new Map<string, string>();
      for (const group of subassemblies(plan, k))
        for (const id of group) map.set(id, group.includes(this.mainPart) ? this.mainPart : group[0]);
      this.components.push(map);
    }
    this.furniture = plan.parts.reduce((acc, p) => acc.union(box(p)), new THREE.Box3());
    const layoutPoints = plan.parts.map((p) => new THREE.Vector3(...(p.layout?.position ?? [0, 0, 0])).multiplyScalar(MM));
    this.layoutCenter.copy(layoutPoints.reduce((a, b) => a.add(b), new THREE.Vector3()).divideScalar(Math.max(1, layoutPoints.length)));
    this.layoutCenter.y = 0;

    let t = 0;
    this.segments.push({ step: 0, phase: 'intro', start: t, duration: INTRO_SECONDS });
    t += INTRO_SECONDS;
    plan.steps.forEach((step, i) => {
      const previous = i > 0 ? plan.steps[i - 1].orientation : step.orientation;
      if (step.orientation !== previous) {
        this.segments.push({ step: step.index, phase: 'reorient', start: t, duration: REORIENT_SECONDS });
        t += REORIENT_SECONDS;
      }
      for (const phase of ['locate', 'place', 'fasten', 'verify'] as const) {
        const duration = phase === 'place' ? PHASE_SECONDS.place + STAGGER_S * Math.max(0, step.parts.length - 1) : PHASE_SECONDS[phase];
        this.segments.push({ step: step.index, phase, start: t, duration });
        t += duration;
      }
    });
    const last = plan.steps[plan.steps.length - 1];
    if (last && last.orientation !== 'upright') {
      this.segments.push({ step: plan.steps.length + 1, phase: 'reorient', start: t, duration: REORIENT_SECONDS });
      t += REORIENT_SECONDS;
    }
    this.segments.push({ step: plan.steps.length + 1, phase: 'done', start: t, duration: DONE_SECONDS });
    this.duration = t + DONE_SECONDS;
  }

  /** When each step starts (its first segment), for the scrubber. */
  stepStarts(): number[] {
    return this.plan.steps.map((s) => this.segments.find((seg) => seg.step === s.index)!.start);
  }

  moment(time: number): Moment {
    const t = Math.min(Math.max(time, 0), this.duration - 1e-6);
    const segment = this.segments.find((s) => t >= s.start && t < s.start + s.duration) ?? this.segments[this.segments.length - 1];
    const u = Math.min(1, (t - segment.start) / segment.duration);
    const steps = this.plan.steps.length;
    let completed = Math.min(steps, Math.max(0, segment.step - 1));
    if (segment.phase === 'done' || segment.step > steps) completed = steps;
    return { time: t, step: segment.step, phase: segment.phase, u, completed };
  }

  /** How the assembly lies during a step (step steps + 1: standing, finished). */
  orientationOf(step: number): Orientation {
    if (step <= 0) return this.plan.steps[0]?.orientation ?? 'upright';
    if (step > this.plan.steps.length) return 'upright';
    return this.plan.steps[step - 1].orientation;
  }

  /** The transform that turns the assembled furniture to `orientation` and rests it on the floor at the work area. */
  mainTransform(orientation: Orientation): THREE.Matrix4 {
    const rotation = new THREE.Matrix4().makeRotationFromQuaternion(ORIENTATIONS[orientation]);
    const turned = this.furniture.clone().applyMatrix4(rotation);
    const center = turned.getCenter(new THREE.Vector3());
    const offset = new THREE.Vector3(-center.x, -turned.min.y, -center.z).add(this.workCenter);
    return new THREE.Matrix4().makeTranslation(offset.x, offset.y, offset.z).multiply(rotation);
  }

  private mainAt(moment: Moment): THREE.Matrix4 {
    if (moment.phase === 'reorient') {
      const from = this.mainTransform(this.orientationOf(moment.step - 1));
      const to = this.mainTransform(this.orientationOf(moment.step));
      return blend(from, to, smooth(moment.u), 0.22 * this.furniture.getSize(new THREE.Vector3()).length());
    }
    return this.mainTransform(this.orientationOf(moment.step === 0 ? 1 : moment.step));
  }

  /** A component built apart from the assembly waits upright beside it, resting on the floor. */
  private sideTransform(representative: string, components: Map<string, string>): THREE.Matrix4 {
    const members = this.plan.parts.filter((p) => components.get(p.id) === representative);
    const group = members.reduce((acc, p) => acc.union(box(p)), new THREE.Box3());
    const furnitureSize = this.furniture.getSize(new THREE.Vector3());
    const groupSize = group.getSize(new THREE.Vector3());
    const center = group.getCenter(new THREE.Vector3());
    const x = this.workCenter.x + Math.max(furnitureSize.x, furnitureSize.y) / 2 + SIDE_GAP_M + groupSize.x / 2;
    return new THREE.Matrix4().makeTranslation(x - center.x, -group.min.y, this.workCenter.z - center.z);
  }

  /** Where a part sits when attached to its component after `steps` completed steps, during `moment`. */
  private attachedMatrix(partId: string, steps: number, moment: Moment): THREE.Matrix4 {
    const components = this.components[steps];
    const representative = components.get(partId)!;
    const frame = representative === this.mainPart ? this.mainAt(moment) : this.sideTransform(representative, components);
    return frame.clone().multiply(this.assembled.get(partId)!);
  }

  /** The representative of the part's component after k steps (mainPart for the assembly itself). */
  componentOf(partId: string, k: number): string {
    return this.components[Math.max(0, Math.min(k, this.components.length - 1))].get(partId) ?? partId;
  }

  /** How many steps the parts' poses reflect at this moment: the current step counts once its parts go on. */
  settledSteps(moment: Moment): number {
    const steps = this.plan.steps.length;
    const step = this.plan.steps[moment.step - 1];
    const on = ['place', 'fasten', 'verify', 'done'].includes(moment.phase);
    return Math.min(step && on ? step.index : moment.completed, steps);
  }

  partStates(moment: Moment): Map<string, PartState> {
    const states = new Map<string, PartState>();
    const steps = this.plan.steps.length;
    const step: Step | undefined = this.plan.steps[moment.step - 1];
    const before = Math.min(moment.completed, steps);
    const after = step ? step.index : before; // components once the current step is done
    const placing = moment.phase === 'place';
    const placed = ['fasten', 'verify'].includes(moment.phase);
    const newParts = new Set(step?.parts ?? []);
    for (const part of this.plan.parts) {
      const at = this.introduced.get(part.id)!;
      const isNew = newParts.has(part.id);
      if (at > after || (isNew && !placing && !placed && moment.phase !== 'done')) {
        // Not used yet (or about to be): lying on the floor.
        const target = isNew ? this.attachedMatrix(part.id, after, { ...moment, phase: 'locate' }) : null;
        states.set(part.id, { matrix: this.layout.get(part.id)!.clone(), status: 'loose', target });
        continue;
      }
      if (isNew && placing) {
        const order = step!.parts.indexOf(part.id);
        const local = (moment.u * (PHASE_SECONDS.place + STAGGER_S * (step!.parts.length - 1)) - STAGGER_S * order) / PHASE_SECONDS.place;
        const target = this.attachedMatrix(part.id, after, moment);
        const u = smooth(local);
        states.set(part.id, {
          matrix: blend(this.layout.get(part.id)!, target, u, 0.45),
          status: u <= 0 ? 'loose' : u >= 1 ? 'attached' : 'moving',
          target,
        });
        continue;
      }
      // Already attached. If its component joins the assembly in this step, it travels there while placing.
      const wasMain = this.components[before].get(part.id) === this.mainPart;
      const joinsMain = this.components[after].get(part.id) === this.mainPart;
      if (!wasMain && joinsMain && placing) {
        const from = this.attachedMatrix(part.id, before, moment);
        const to = this.attachedMatrix(part.id, after, moment);
        states.set(part.id, { matrix: blend(from, to, smooth(moment.u), 0.3), status: 'moving', target: to });
        continue;
      }
      if (!wasMain && joinsMain && moment.phase === 'locate') {
        // A subassembly about to go on: show where.
        states.set(part.id, {
          matrix: this.attachedMatrix(part.id, before, moment),
          status: 'attached',
          target: this.attachedMatrix(part.id, after, moment),
        });
        continue;
      }
      const settled = placing || placed || moment.phase === 'done' ? after : before;
      states.set(part.id, {
        matrix: this.attachedMatrix(part.id, Math.min(settled, steps), moment),
        status: 'attached',
        target: isNew ? this.attachedMatrix(part.id, after, moment) : null,
      });
    }
    return states;
  }

  /** Parts of subassemblies built on the side that join the assembly in the current step. */
  joining(moment: Moment): string[] {
    const step = this.plan.steps[moment.step - 1];
    if (!step) return [];
    const before = this.components[step.index - 1];
    const after = this.components[step.index];
    return this.plan.parts
      .map((p) => p.id)
      .filter((id) => before.get(id) !== this.mainPart && after.get(id) === this.mainPart && this.introduced.get(id)! < step.index);
  }

  /** The current step's fasteners in world space (meters) with their insertion axes. */
  fasteners(moment: Moment): { position: THREE.Vector3; axis: THREE.Vector3; hardware: string; joint: string }[] {
    const step = this.plan.steps[moment.step - 1];
    if (!step) return [];
    const frame = this.mainAt(moment);
    const joints = new Map(this.plan.joints.map((j) => [j.id, j]));
    const components = this.components[step.index];
    return step.joints.flatMap((id) => {
      const joint = joints.get(id)!;
      const representative = components.get(joint.parts[0])!;
      const f = representative === this.mainPart ? frame : this.sideTransform(representative, components);
      const rotation = new THREE.Matrix3().setFromMatrix4(f);
      return joint.fasteners.map((fastener) => ({
        position: new THREE.Vector3(...fastener.position).multiplyScalar(MM).applyMatrix4(f),
        axis: new THREE.Vector3(...fastener.axis).applyMatrix3(rotation).normalize(),
        hardware: fastener.hardware,
        joint: id,
      }));
    });
  }

  /** Overall size of the finished piece (meters). */
  size(): THREE.Vector3 {
    return this.furniture.getSize(new THREE.Vector3());
  }
}

/**
 * Interpolate two rigid transforms: positions along an arc that rises by `lift` meters in the
 * middle, rotations by slerp.
 */
export function blend(from: THREE.Matrix4, to: THREE.Matrix4, u: number, lift: number): THREE.Matrix4 {
  const [p0, q0, p1, q1] = [new THREE.Vector3(), new THREE.Quaternion(), new THREE.Vector3(), new THREE.Quaternion()];
  const scale = new THREE.Vector3();
  from.decompose(p0, q0, scale);
  to.decompose(p1, q1, scale);
  const t = Math.min(Math.max(u, 0), 1);
  const position = p0.clone().lerp(p1, t);
  position.y += Math.sin(Math.PI * t) * lift;
  return new THREE.Matrix4().compose(position, q0.clone().slerp(q1, t), new THREE.Vector3(1, 1, 1));
}
