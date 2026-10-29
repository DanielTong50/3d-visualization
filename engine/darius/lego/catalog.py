"""Original, rights-clear curriculum assemblies, deliberately small enough to inspect."""

from __future__ import annotations

import json
from pathlib import Path

from .compiler import PARTS, compile_assembly
from .schema import Baseplate, Brick, BuildStep, LegoAssembly

COLORS = {
    "coral": "#f4553b",
    "yellow": "#ffd33d",
    "teal": "#31aaa2",
    "blue": "#3769cf",
    "cream": "#f4efe4",
    "charcoal": "#30343b",
}


def _build(build_id: str, title: str, description: str, specs: list[tuple]) -> LegoAssembly:
    bricks, steps = [], []
    for i, (part, color, position, rotation, caption) in enumerate(specs, 1):
        dims = PARTS[part]
        brick_id = f"b{i:02}"
        bricks.append(
            Brick(
                id=brick_id,
                part_id=part,
                color=color,
                color_hex=COLORS[color],
                studs=list(dims[:2]),
                height_plates=dims[2],
                position=list(position),
                rotation=rotation,
            )
        )
        steps.append(BuildStep(index=i, title=caption, brick_ids=[brick_id]))
    return LegoAssembly(
        id=build_id,
        title=title,
        description=description,
        baseplate=Baseplate(studs=[16, 16]),
        bricks=bricks,
        steps=steps,
        notes=[
            "Nominal grid geometry; no measured clutch-force or hardware success claims.",
            "Original curriculum build, independent of official LEGO instruction sets.",
        ],
    )


def signal_tower() -> LegoAssembly:
    return _build(
        "signal-tower",
        "Signal tower",
        "A compact beacon: cross-bonded courses teach alignment, support and guarded insertion.",
        [
            ("3001", "coral", (6, 0, 4), 0, "Anchor the first foundation brick"),
            ("3001", "coral", (8, 0, 4), 0, "Complete the foundation course"),
            ("3001", "yellow", (6, 3, 4), 90, "Cross-bond the front course"),
            ("3001", "yellow", (6, 3, 6), 90, "Cross-bond the rear course"),
            ("3003", "teal", (7, 6, 5), 0, "Raise the central beacon"),
            ("3003", "cream", (7, 9, 5), 0, "Cap the signal tower"),
        ],
    )


def bridge() -> LegoAssembly:
    return _build(
        "bridge",
        "Stud bridge",
        "Two piers and a spanning plate introduce multiple support parents and partial-contact reasoning.",
        [
            ("3003", "teal", (3, 0, 6), 0, "Seat the left pier foot"),
            ("3003", "teal", (9, 0, 6), 0, "Seat the right pier foot"),
            ("3003", "cream", (3, 3, 6), 0, "Raise the left pier"),
            ("3003", "cream", (9, 3, 6), 0, "Raise the right pier"),
            ("3034", "yellow", (3, 6, 6), 90, "Span both piers with one plate"),
            ("3003", "coral", (6, 7, 6), 0, "Add the centered bridge marker"),
        ],
    )


def stairway() -> LegoAssembly:
    return _build(
        "stairway",
        "Gradient stairway",
        "A six-brick staircase trains height-conditioned placement with an inspectable dependency graph.",
        [
            ("3001", "coral", (4, 0, 5), 0, "Place the first tread base"),
            ("3001", "yellow", (6, 0, 5), 0, "Place the middle tread base"),
            ("3001", "teal", (8, 0, 5), 0, "Place the upper tread base"),
            ("3001", "yellow", (6, 3, 5), 0, "Raise the middle tread"),
            ("3001", "teal", (8, 3, 5), 0, "Raise the upper tread"),
            ("3001", "cream", (8, 6, 5), 0, "Finish the top tread"),
        ],
    )


CATALOG = {"signal-tower": signal_tower, "bridge": bridge, "stairway": stairway}


def get_assembly(build_id: str) -> LegoAssembly:
    if build_id not in CATALOG:
        raise KeyError(f"unknown LEGO curriculum build {build_id!r}")
    return CATALOG[build_id]()


def library_index() -> list[dict]:
    return [
        {
            "id": plan.assembly.id,
            "title": plan.assembly.title,
            "description": plan.assembly.description,
            "source": "original",
            **plan.metrics.model_dump(),
            "valid": plan.validation.valid,
        }
        for plan in [compile_assembly(factory()) for factory in CATALOG.values()]
    ]


def export_library(output: Path) -> list[dict]:
    from ..robotics.dataset import training_bundle

    output.mkdir(parents=True, exist_ok=True)
    for build_id in CATALOG:
        assembly = get_assembly(build_id)
        compiled = compile_assembly(assembly)
        if not compiled.validation.valid:
            raise ValueError(f"invalid curriculum {build_id}: {compiled.validation.errors}")
        folder = output / build_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "assembly.json").write_text(assembly.model_dump_json(indent=2) + "\n")
        (folder / "plan.json").write_text(compiled.model_dump_json(indent=2) + "\n")
        # A small reproducible bundle keeps the static browser demo useful without a backend.
        bundle = training_bundle(compiled, episodes=2, seed=7, fps=5)
        (folder / "training.json").write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    index = library_index()
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
