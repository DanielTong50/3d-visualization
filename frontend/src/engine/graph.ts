import type { AssemblyPlan } from './plan';

/** Graph operations on a plan (mirror of engine/darius/assembly/graph.py, the parts the views need). */

export function subassemblies(plan: AssemblyPlan, completed: number): string[][] {
  const parent = new Map(plan.parts.map((p) => [p.id, p.id]));
  const find = (id: string): string => {
    let root = id;
    while (parent.get(root) !== root) root = parent.get(root)!;
    parent.set(id, root);
    return root;
  };
  const joints = new Map(plan.joints.map((j) => [j.id, j]));
  for (const step of plan.steps.slice(0, completed)) {
    for (const id of step.joints) {
      const [a, b] = joints.get(id)!.parts;
      parent.set(find(a), find(b));
    }
  }
  const groups = new Map<string, string[]>();
  for (const part of plan.parts) {
    const root = find(part.id);
    groups.set(root, [...(groups.get(root) ?? []), part.id]);
  }
  return [...groups.values()].sort((a, b) => b.length - a.length);
}

/** The step (1-based) that brings each part in; parts never brought in get Infinity. */
export function introducedAt(plan: AssemblyPlan): Map<string, number> {
  const at = new Map<string, number>(plan.parts.map((p) => [p.id, Infinity]));
  for (const step of plan.steps) for (const part of step.parts) at.set(part, Math.min(at.get(part)!, step.index));
  return at;
}

/** The step (1-based) that makes each joint. */
export function madeAt(plan: AssemblyPlan): Map<string, number> {
  const at = new Map<string, number>();
  for (const step of plan.steps) for (const joint of step.joints) at.set(joint, step.index);
  return at;
}

/** Which parts take part in a step: brought in, or joined by it. */
export function stepParts(plan: AssemblyPlan, index: number): Set<string> {
  const step = plan.steps[index - 1];
  if (!step) return new Set();
  const joints = new Map(plan.joints.map((j) => [j.id, j]));
  return new Set([...step.parts, ...step.joints.flatMap((id) => joints.get(id)!.parts)]);
}
