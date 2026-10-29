"""Deterministic stud-contact graph and guarded Cartesian task compiler.

This is a task-space reference, not an IK solution or a robot driver. A geometric
contact is an intended snap connection; the compiler does not predict clutch
force or certify stability under robot loads. Blockers remain in every plan.
"""

from __future__ import annotations

from .schema import (
    Brick,
    CompiledPlan,
    CompiledStep,
    Connection,
    Execution,
    LegoAssembly,
    Metrics,
    Primitive,
    Validation,
)

STUD_MM = 8.0
PLATE_MM = 3.2
# Supported original curriculum parts: nominal rectangular bricks and plates.
PARTS: dict[str, tuple[int, int, int]] = {
    "3001": (2, 4, 3),
    "3002": (2, 3, 3),
    "3003": (2, 2, 3),
    "3004": (1, 2, 3),
    "3005": (1, 1, 3),
    "3010": (1, 4, 3),
    "3020": (2, 4, 1),
    "3021": (2, 3, 1),
    "3022": (2, 2, 1),
    "3023": (1, 2, 1),
    "3024": (1, 1, 1),
    "3034": (2, 8, 1),
    "3035": (4, 8, 1),
    "3031": (4, 4, 1),
}
BLOCKERS = [
    "No robot connected: all primitives are nominal task-space references.",
    "Calibrate camera, baseplate, tray and TCP transforms before hardware execution.",
    "Solve joint-space IK and collision checking for the selected arm and gripper.",
    "Measure grip and insertion forces; validate guarded press and emergency stop on hardware.",
]


def brick_center_mm(brick: Brick, baseplate_height_mm: float = 3.2) -> list[float]:
    width, depth = brick.footprint
    x, y, z = brick.position
    return [
        (x + width / 2) * STUD_MM,
        baseplate_height_mm + (y + brick.height_plates / 2) * PLATE_MM,
        (z + depth / 2) * STUD_MM,
    ]


def _tray_centers(assembly: LegoAssembly, ordered_bricks: list[Brick]) -> list[list[float]]:
    """Reserve disjoint feed slots with 16mm clearance around the largest footprint."""
    # Feed bricks are indexed at source_yaw_deg=0, independently of final orientation.
    width = max(brick.studs[0] for brick in ordered_bricks) * STUD_MM
    depth = max(brick.studs[1] for brick in ordered_bricks) * STUD_MM
    spacing = max(width, depth) + 16.0
    return [
        [
            assembly.baseplate.studs[0] * STUD_MM + 24 + width / 2 + (slot % 4) * spacing,
            brick.height_plates * PLATE_MM / 2,
            24 + depth / 2 + (slot // 4) * spacing,
        ]
        for slot, brick in enumerate(ordered_bricks)
    ]


def _primitives(brick: Brick, source_center: list[float], baseplate_height_mm: float) -> list[Primitive]:
    # TCP at the top of the brick, +6mm jaw allowance. A real adapter must calibrate this.
    center = brick_center_mm(brick, baseplate_height_mm)
    target = [center[0], center[1] + brick.height_plates * PLATE_MM / 2 + 6, center[2]]
    pickup = [source_center[0], source_center[1] + brick.height_plates * PLATE_MM / 2 + 6, source_center[2]]
    clearance = max(100.0, target[1] + 50)

    def p(
        kind: str,
        pos: list[float],
        duration: float,
        grip: str,
        text: str,
        force: float | None = None,
        *,
        source: bool = False,
    ):
        return Primitive(
            kind=kind,
            position_mm=pos,
            yaw_deg=0.0 if source else float(brick.rotation),
            duration_s=duration,
            gripper=grip,
            force_limit_n=force,
            description=text,
        )

    return [
        p("move", [pickup[0], clearance, pickup[2]], 0.7, "open", "Approach indexed tray slot", source=True),
        p("move", pickup, 0.4, "open", "Descend to nominal grasp TCP", source=True),
        p("grasp", pickup, 0.35, "closed", "Close jaws; hardware adapter must verify possession", source=True),
        p("move", [pickup[0], clearance, pickup[2]], 0.5, "closed", "Lift above fixture", source=True),
        p("move", [target[0], clearance, target[2]], 0.8, "closed", "Transfer over target"),
        p("move", [target[0], target[1] + 6, target[2]], 0.5, "closed", "Align over target studs"),
        p("press", target, 0.7, "closed", "Guarded nominal seating; 12N is an uncalibrated placeholder", 12.0),
        p("release", target, 0.25, "open", "Release seated brick"),
        p("move", [target[0], clearance, target[2]], 0.5, "open", "Retract to observation clearance"),
        p(
            "verify",
            [target[0], clearance, target[2]],
            0.4,
            "open",
            "Require visual and force verification before advance",
        ),
    ]


def compile_assembly(assembly: LegoAssembly) -> CompiledPlan:
    """Compile an assembly and report all geometric errors without emitting unsafe hardware commands.

    A valid plan has no occupied-volume intersection, fits its fixture, and every
    placed brick has at least one intended stud contact with an earlier brick or
    the fixture. Partial support is reported, not treated as a force guarantee.
    """
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {b.id: b for b in assembly.bricks}
    order = [(s, by_id[bid]) for s in assembly.steps for bid in s.brick_ids]
    cell_cache = {brick.id: brick.cells for brick in assembly.bricks}
    tray_centers = _tray_centers(assembly, [brick for _, brick in order])
    for brick in assembly.bricks:
        expected = PARTS.get(brick.part_id)
        if expected is None:
            errors.append(
                f"{brick.id}: unsupported part {brick.part_id}; only rectangular curriculum parts are compiled"
            )
        elif tuple(brick.studs + [brick.height_plates]) != expected:
            errors.append(f"{brick.id}: part {brick.part_id} dimensions do not match the supported inventory")
        x, _, z = brick.position
        width, depth = brick.footprint
        if x + width > assembly.baseplate.studs[0] or z + depth > assembly.baseplate.studs[1]:
            errors.append(f"{brick.id}: footprint is outside the baseplate")
    for i, first in enumerate(assembly.bricks):
        for other in assembly.bricks[i + 1 :]:
            vertical = max(first.position[1], other.position[1]) < min(
                first.position[1] + first.height_plates, other.position[1] + other.height_plates
            )
            if vertical and cell_cache[first.id] & cell_cache[other.id]:
                errors.append(f"{first.id}/{other.id}: intersecting occupied volumes")
    connections: list[Connection] = []
    compiled: list[CompiledStep] = []
    placed: list[Brick] = []
    for slot, (step, brick) in enumerate(order):
        deps: list[str] = []
        contact_ids: list[str] = []
        supported: set[tuple[int, int]] = set()
        candidates = (
            [("baseplate", cell_cache[brick.id])]
            if brick.position[1] == 0
            else [
                (lower.id, cell_cache[lower.id] & cell_cache[brick.id])
                for lower in placed
                if lower.position[1] + lower.height_plates == brick.position[1]
            ]
        )
        for lower_id, cells in candidates:
            if not cells:
                continue
            cid = f"c{len(connections) + 1:03}"
            connections.append(
                Connection(
                    id=cid, lower=lower_id, upper=brick.id, studs=[list(v) for v in sorted(cells)], step=slot + 1
                )
            )
            contact_ids.append(cid)
            supported |= cells
            if lower_id != "baseplate":
                deps.append(lower_id)
        if not supported:
            errors.append(f"{brick.id}: no stud support from fixture or earlier bricks (floating or wrong step order)")
        ratio = len(supported) / len(cell_cache[brick.id])
        if 0 < ratio < 1:
            warnings.append(
                f"{brick.id}: {ratio:.0%} stud support; physical stability and clutch force require validation"
            )
        # Straight-down insertion is obstructed by previously placed solids in the same footprint.
        for obstruction in placed:
            if (
                cell_cache[obstruction.id] & cell_cache[brick.id]
                and obstruction.position[1] >= brick.position[1] + brick.height_plates
            ):
                errors.append(f"{brick.id}: vertical insertion blocked by earlier brick {obstruction.id}")
        compiled.append(
            CompiledStep(
                index=slot + 1,
                source_step=step.index,
                title=step.title,
                brick_id=brick.id,
                dependencies=sorted(deps),
                connection_ids=contact_ids,
                support_ratio=ratio,
                source_center_mm=tray_centers[slot],
                target_center_mm=brick_center_mm(brick, assembly.baseplate.height_mm),
                source_yaw_deg=0.0,
                target_yaw_deg=float(brick.rotation),
                primitives=_primitives(brick, tray_centers[slot], assembly.baseplate.height_mm),
            )
        )
        placed.append(brick)
    duration = sum(p.duration_s for step in compiled for p in step.primitives)
    return CompiledPlan(
        assembly=assembly,
        connections=connections,
        steps=compiled,
        validation=Validation(valid=not errors, errors=errors, warnings=warnings),
        metrics=Metrics(
            bricks=len(assembly.bricks),
            connections=len(connections),
            stud_connections=sum(len(c.studs) for c in connections),
            steps=len(compiled),
            primitives=sum(len(s.primitives) for s in compiled),
            nominal_duration_s=round(duration, 3),
        ),
        execution=Execution(blockers=BLOCKERS.copy()),
    )
