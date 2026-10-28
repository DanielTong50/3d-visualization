"""Building assembly plans from boxes: the geometry kernel of the furniture library.

Flat-pack furniture is boxes (panels, legs, rails) joined face to face. A designer
places each part's box in the finished piece; the builder works out the rest:

- joints: where two boxes touch, the contact face, its normal, and fastener positions
  spread along it (dowels near the ends, cam bolts inside, nails every ~15 cm);
- hardware totals, per joint and per step;
- a floor layout: every part lying on its largest face, packed in rows in front of
  the work area, which is where the AR view first finds them.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..assembly.schema import (
    Action,
    AssemblyPlan,
    Fastener,
    Hardware,
    HardwareUse,
    Joint,
    Orientation,
    Part,
    Pose,
    Step,
)

TOUCH_MM = 0.6
LAYOUT_GAP_MM = 70
LAYOUT_ROW_MM = 2400


@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray

    @property
    def lo(self) -> np.ndarray:
        return self.center - self.size / 2

    @property
    def hi(self) -> np.ndarray:
        return self.center + self.size / 2


@dataclass(frozen=True)
class Contact:
    axis: int  # the contact normal is along this axis
    sign: int  # +1: the second box lies on the + side of the first
    plane: float  # coordinate of the shared face along `axis`
    lo: np.ndarray  # the overlap rectangle, as a 3D box flattened on `axis`
    hi: np.ndarray


def contact(a: Box, b: Box) -> Contact | None:
    """The face two touching boxes share, or None if they don't touch face to face."""
    for axis in range(3):
        for sign, (face_a, face_b) in ((1, (a.hi[axis], b.lo[axis])), (-1, (a.lo[axis], b.hi[axis]))):
            if abs(face_a - face_b) > TOUCH_MM:
                continue
            others = [i for i in range(3) if i != axis]
            lo = np.maximum(a.lo, b.lo)
            hi = np.minimum(a.hi, b.hi)
            if all(hi[i] - lo[i] > 1.0 for i in others):
                lo[axis] = hi[axis] = (face_a + face_b) / 2
                return Contact(axis, sign, float(lo[axis]), lo, hi)
    return None


def spread(lo: np.ndarray, hi: np.ndarray, count: int, inset: float) -> list[np.ndarray]:
    """`count` points along the longest side of the contact rectangle, centered on the other."""
    extent = hi - lo
    long_axis = int(np.argmax(extent))
    center = (lo + hi) / 2
    if count == 1:
        return [center.copy()]
    inset = min(inset, extent[long_axis] / (2 * count))
    start, end = lo[long_axis] + inset, hi[long_axis] - inset
    points = []
    for k in range(count):
        p = center.copy()
        p[long_axis] = start + (end - start) * k / (count - 1)
        points.append(p)
    return points


def lying_rotation(size: np.ndarray) -> list[float]:
    """Quaternion that lays a box on its largest face (its thinnest axis vertical)."""
    thin = int(np.argmin(size))
    h = math.sqrt(0.5)
    if thin == 0:  # x up: rotate +90 degrees about z
        return [0.0, 0.0, h, h]
    if thin == 2:  # z up: rotate -90 degrees about x
        return [-h, 0.0, 0.0, h]
    return [0.0, 0.0, 0.0, 1.0]


def lying_footprint(size: np.ndarray) -> tuple[float, float]:
    """(extent along x, extent along z) of a box laid on its largest face by `lying_rotation`."""
    thin = int(np.argmin(size))
    if thin == 0:
        return float(size[1]), float(size[2])
    if thin == 2:
        return float(size[0]), float(size[1])
    return float(size[0]), float(size[2])


# Fastener patterns per joint kind: hardware ids (in order along the joint) and how far from
# the ends the first and last sit.
PATTERNS: dict[str, tuple[list[str], float]] = {
    "cam_lock": (["dowel", "cam_bolt", "cam_bolt", "dowel"], 37.0),
    "dowel": (["dowel", "dowel"], 40.0),
    "screw": (["screw", "screw"], 40.0),
    "bolt": (["bolt"], 0.0),
    "pin": (["shelf_pin", "shelf_pin"], 40.0),
    "slot": ([], 0.0),
}


@dataclass
class FurnitureBuilder:
    id: str
    title: str
    category: str
    color: str = "#efece6"
    material: str = "particleboard, white foil"
    parts: list[Part] = field(default_factory=list)
    boxes: dict[str, Box] = field(default_factory=dict)
    hardware: dict[str, Hardware] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)  # hardware kind -> id
    joints: list[Joint] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def item(self, hardware_id: str, name: str, kind: str, size_mm: list[float]) -> None:
        self.hardware[hardware_id] = Hardware(id=hardware_id, name=name, kind=kind, count=1, size_mm=size_mm)
        self.kinds[kind] = hardware_id

    def part(
        self, part_id: str, label: str, name: str, kind: str, size, center, color: str | None = None, material=None
    ) -> str:
        size, center = np.asarray(size, float), np.asarray(center, float)
        self.boxes[part_id] = Box(center, size)
        self.parts.append(
            Part(
                id=part_id,
                label=label,
                name=name,
                kind=kind,
                size_mm=[round(float(v), 1) for v in size],
                material=material or self.material,
                color=color or self.color,
                assembled=Pose(position=[round(float(v), 1) for v in center]),
            )
        )
        return part_id

    def join(self, a: str, b: str, kind: str, fasteners: int | None = None) -> str:
        """Join two touching parts; fasteners follow the kind's pattern along the shared face."""
        touch = contact(self.boxes[a], self.boxes[b])
        if touch is None:
            raise ValueError(f"{a} and {b} do not touch, so they cannot be joined")
        joint_id = f"J{len(self.joints) + 1}"
        if kind in ("nail",):
            length = float(np.max(touch.hi - touch.lo))
            pattern, inset = ["nail"] * max(2, min(8, round(length / 150))), 30.0
        else:
            pattern, inset = PATTERNS[kind]
            if fasteners is not None:
                pattern = (pattern * fasteners)[:fasteners]
        axis = np.zeros(3)
        axis[touch.axis] = touch.sign
        points = spread(touch.lo, touch.hi, len(pattern), inset) if pattern else []
        placed = [
            Fastener(hardware=self.kinds[name], position=[round(float(v), 1) for v in point], axis=axis.tolist())
            for name, point in zip(pattern, points, strict=True)
        ]
        counts = Counter(f.hardware for f in placed)
        if kind == "cam_lock":  # every cam bolt is turned into a cam lock seated in the other panel
            counts[self.kinds["cam_lock"]] += counts[self.kinds["cam_bolt"]]
        self.joints.append(
            Joint(
                id=joint_id,
                parts=[a, b],
                kind=kind,
                hardware=[HardwareUse(id=h, count=n) for h, n in counts.items()],
                fasteners=placed,
            )
        )
        return joint_id

    def step(
        self,
        title: str,
        instruction: str,
        parts: list[str],
        joints: list[str],
        orientation: Orientation = "upright",
        actions: list[Action] | None = None,
        tools: list[str] | None = None,
    ) -> None:
        uses: Counter[str] = Counter()
        for joint_id in joints:
            for use in next(j for j in self.joints if j.id == joint_id).hardware:
                uses[use.id] += use.count
        self.steps.append(
            Step(
                index=len(self.steps) + 1,
                title=title,
                instruction=instruction,
                parts=parts,
                joints=joints,
                hardware=[HardwareUse(id=h, count=n) for h, n in sorted(uses.items())],
                actions=actions or [],
                orientation=orientation,
                tools=tools or [],
            )
        )

    def layout(self) -> None:
        """Lay every part flat on the floor in rows in front of the work area, in the order the
        steps use them (so the first parts needed are nearest)."""
        order = [p for s in self.steps for p in s.parts]
        order += [p.id for p in self.parts if p.id not in order]
        front = max(b.hi[2] for b in self.boxes.values()) + 450
        x, z, row_depth = 0.0, front, 0.0
        rows: list[list[tuple[str, float, float, float]]] = [[]]
        for part_id in order:
            width, depth = lying_footprint(self.boxes[part_id].size)
            if x + width > LAYOUT_ROW_MM and rows[-1]:
                rows.append([])
                z += row_depth + LAYOUT_GAP_MM
                x, row_depth = 0.0, 0.0
            rows[-1].append((part_id, x + width / 2, z + depth / 2, width))
            x += width + LAYOUT_GAP_MM
            row_depth = max(row_depth, depth)
        by_id = {p.id: p for p in self.parts}
        for row in rows:
            row_width = row[-1][1] + row[-1][3] / 2
            for part_id, cx, cz, _ in row:
                size = self.boxes[part_id].size
                height = float(np.min(size))
                by_id[part_id].layout = Pose(
                    position=[round(cx - row_width / 2, 1), round(height / 2, 1), round(cz, 1)],
                    rotation=lying_rotation(size),
                )

    def build(self) -> AssemblyPlan:
        totals: Counter[str] = Counter()
        for joint in self.joints:
            for use in joint.hardware:
                totals[use.id] += use.count
        hardware = [h.model_copy(update={"count": totals[h.id]}) for h in self.hardware.values() if totals[h.id]]
        self.layout()
        lo = np.min([b.lo for b in self.boxes.values()], axis=0)
        hi = np.max([b.hi for b in self.boxes.values()], axis=0)
        return AssemblyPlan(
            id=self.id,
            title=self.title,
            category=self.category,
            source="procedural",
            dimensions_mm=[round(float(v), 1) for v in hi - lo],
            parts=self.parts,
            hardware=hardware,
            joints=self.joints,
            steps=self.steps,
            notes=self.notes,
        )
