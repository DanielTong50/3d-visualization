"""Graph operations on assembly plans: validation, the state after k steps, and comparison.

The assembly graph has one node per part and one edge per joint. A plan is valid
when the graph and its step order are consistent: every joint is made by exactly
one step, a joint's parts are in hand by then, hardware is not over-used, and the
finished graph is connected (the furniture is one piece). Comparison measures how
well one plan (read from a manual) recovers another (the ground truth), on the
level that matters to the person building: which kinds of parts are joined to
which, with what, in what order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schema import AssemblyPlan, Step


class DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        result: dict[str, list[str]] = {}
        for item in self.parent:
            result.setdefault(self.find(item), []).append(item)
        return sorted((sorted(group) for group in result.values()), key=lambda g: (-len(g), g))


def validate(plan: AssemblyPlan) -> list[str]:
    """Problems that make the plan inconsistent (empty when it is valid)."""
    problems: list[str] = []
    part_ids = [p.id for p in plan.parts]
    joint_ids = [j.id for j in plan.joints]
    for kind, ids in (("part", part_ids), ("joint", joint_ids), ("hardware", [h.id for h in plan.hardware])):
        for item, n in Counter(ids).items():
            if n > 1:
                problems.append(f"{kind} id {item!r} is used {n} times")
    parts, hardware = set(part_ids), {h.id: h.count for h in plan.hardware}
    for joint in plan.joints:
        for part in joint.parts:
            if part not in parts:
                problems.append(f"joint {joint.id} uses unknown part {part!r}")
        if joint.parts[0] == joint.parts[1]:
            problems.append(f"joint {joint.id} joins {joint.parts[0]} to itself")
        for use in joint.hardware:
            if use.id not in hardware:
                problems.append(f"joint {joint.id} uses unknown hardware {use.id!r}")
    if [s.index for s in plan.steps] != list(range(1, len(plan.steps) + 1)):
        problems.append("steps must be numbered 1, 2, 3, ... in order")
    made: Counter[str] = Counter(j for s in plan.steps for j in s.joints)
    for joint in joint_ids:
        if made[joint] != 1:
            problems.append(f"joint {joint} is made by {made[joint]} steps (expected exactly one)")
    for joint in made:
        if joint not in joint_ids:
            problems.append(f"a step makes unknown joint {joint!r}")
    in_hand: set[str] = set()
    used: Counter[str] = Counter()
    for step in plan.steps:
        in_hand |= set(step.parts)
        for part in step.parts:
            if part not in parts:
                problems.append(f"step {step.index} brings in unknown part {part!r}")
        for joint_id in step.joints:
            if joint_id not in joint_ids:
                continue
            for part in plan.joint(joint_id).parts:
                if part not in in_hand:
                    problems.append(f"step {step.index} makes {joint_id} before part {part} is brought in")
        for use in step.hardware:
            used[use.id] += use.count
    for item, count in used.items():
        if item in hardware and count > hardware[item]:
            problems.append(f"steps use {count} of hardware {item}, the inventory has {hardware[item]}")
    if plan.parts and not problems and len(subassemblies(plan, len(plan.steps))) > 1:
        problems.append("the finished assembly is not one connected piece")
    return problems


def subassemblies(plan: AssemblyPlan, completed: int) -> list[list[str]]:
    """Groups of parts that are joined together after the first `completed` steps (largest first)."""
    sets = DisjointSet(p.id for p in plan.parts)
    for step in plan.steps[:completed]:
        for joint_id in step.joints:
            a, b = plan.joint(joint_id).parts
            sets.union(a, b)
    return sets.groups()


def joints_done(plan: AssemblyPlan, completed: int) -> list[str]:
    return [j for s in plan.steps[:completed] for j in s.joints]


def step_of_joint(plan: AssemblyPlan) -> dict[str, int]:
    return {j: s.index for s in plan.steps for j in s.joints}


def degree(plan: AssemblyPlan) -> dict[str, int]:
    counts = Counter(p for j in plan.joints for p in j.parts)
    return {p.id: counts[p.id] for p in plan.parts}


def hardware_for(step: Step) -> int:
    return sum(use.count for use in step.hardware)


# ---------- Comparing a recovered plan with the truth ----------


@dataclass(frozen=True)
class GraphScore:
    """How well a recovered plan matches the truth, by part label (instances are interchangeable:
    which of two identical side panels is 'A1' is arbitrary)."""

    parts_precision: float
    parts_recall: float
    edges_precision: float
    edges_recall: float
    steps_exact: float
    hardware_accuracy: float

    @property
    def parts_f1(self) -> float:
        return _f1(self.parts_precision, self.parts_recall)

    @property
    def edges_f1(self) -> float:
        return _f1(self.edges_precision, self.edges_recall)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _overlap(predicted: Counter, truth: Counter) -> tuple[float, float]:
    hits = sum((predicted & truth).values())
    precision = hits / max(1, sum(predicted.values()))
    recall = hits / max(1, sum(truth.values()))
    return precision, recall


def edge_labels(plan: AssemblyPlan) -> Counter[tuple[str, str]]:
    """Joints as unordered pairs of part labels, with multiplicity."""
    label = {p.id: p.label for p in plan.parts}
    return Counter(tuple(sorted((label[a], label[b]))) for a, b in (j.parts for j in plan.joints))


def step_edges(plan: AssemblyPlan) -> list[Counter[tuple[str, str]]]:
    label = {p.id: p.label for p in plan.parts}
    return [Counter(tuple(sorted(label[x] for x in plan.joint(j).parts)) for j in s.joints) for s in plan.steps]


def compare(predicted: AssemblyPlan, truth: AssemblyPlan) -> GraphScore:
    parts_p, parts_r = _overlap(Counter(p.label for p in predicted.parts), Counter(p.label for p in truth.parts))
    edges_p, edges_r = _overlap(edge_labels(predicted), edge_labels(truth))
    truth_steps, predicted_steps = step_edges(truth), step_edges(predicted)
    exact = sum(1 for k, edges in enumerate(truth_steps) if k < len(predicted_steps) and predicted_steps[k] == edges)
    truth_hw = {h.id: h.count for h in truth.hardware}
    predicted_hw = {h.id: h.count for h in predicted.hardware}
    hw = sum(1 for item, count in truth_hw.items() if predicted_hw.get(item) == count) / max(1, len(truth_hw))
    return GraphScore(parts_p, parts_r, edges_p, edges_r, exact / max(1, len(truth_steps)), hw)
