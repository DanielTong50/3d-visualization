"""The `darius` command line.

    darius library                         list the furniture library
    darius export [--out ../assemblies]    write every library plan (and an index) for the web app
    darius show PLAN                       a plan's parts, joints and steps, and its graph
    darius validate PLAN...                check plans for consistency
    darius schema                          print the AssemblyPlan JSON Schema
    darius manual PAGE...  [--truth PLAN]  read manual pages into a plan (Meta Model API)
    darius track VIDEO --plan PLAN         decode a build video into a timeline (Meta Model API)

PLAN is a library id (e.g. halden-bookcase) or a path to a plan.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

REPO = Path(__file__).resolve().parents[2]


def load_plan(ref: str):
    from .assembly.schema import AssemblyPlan
    from .furniture import catalog

    if ref in catalog.CATALOG:
        return catalog.build(ref)
    return AssemblyPlan.model_validate_json(Path(ref).read_text())


def _library(args: argparse.Namespace) -> int:
    from .furniture import catalog

    for plan_id in catalog.CATALOG:
        plan = catalog.build(plan_id)
        w, h, d = (round(v / 10) for v in plan.dimensions_mm)
        print(f"{plan_id:<20} {plan.title:<20} {w} x {d} x {h} cm  {len(plan.parts):>2} parts  "
              f"{len(plan.joints):>2} joints  {len(plan.steps)} steps")  # fmt: skip
    return 0


def _export(args: argparse.Namespace) -> int:
    from .furniture import catalog

    out = Path(args.out)
    index = []
    for plan_id in catalog.CATALOG:
        plan = catalog.build(plan_id)
        folder = out / plan_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(plan.model_dump_json(indent=1, exclude_none=True) + "\n")
        index.append(
            {
                "id": plan.id,
                "title": plan.title,
                "category": plan.category,
                "dimensions_mm": plan.dimensions_mm,
                "parts": len(plan.parts),
                "joints": len(plan.joints),
                "steps": len(plan.steps),
                "hardware": sum(h.count for h in plan.hardware),
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(index)} plans -> {out}")
    return 0


def _show(args: argparse.Namespace) -> int:
    from .assembly import graph

    plan = load_plan(args.plan)
    print(f"{plan.title} ({plan.id}): {len(plan.parts)} parts, {len(plan.joints)} joints, {len(plan.steps)} steps")
    for part in plan.parts:
        print(f"  {part.id:<4} {part.label}  {part.name:<16} {' x '.join(f'{v:g}' for v in part.size_mm)} mm")
    print()
    for step in plan.steps:
        hardware = ", ".join(f"{u.count}x{u.id}" for u in step.hardware) or "no hardware"
        print(f"  {step.index}. {step.title}  [{step.orientation}; {hardware}]")
        for joint_id in step.joints:
            joint = plan.joint(joint_id)
            print(f"       {joint.parts[0]} -- {joint.kind} -- {joint.parts[1]}")
    pieces = [len(g) for g in graph.subassemblies(plan, len(plan.steps))]
    print(f"\n  finished: {len(pieces)} piece(s); graph degree {graph.degree(plan)}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .assembly import graph

    failures = 0
    for ref in args.plans:
        problems = graph.validate(load_plan(ref))
        print(f"{'ok  ' if not problems else 'FAIL'} {ref}")
        for problem in problems:
            print(f"     - {problem}")
        failures += bool(problems)
    return 1 if failures else 0


def _schema(args: argparse.Namespace) -> int:
    from .assembly.schema import AssemblyPlan

    json.dump(AssemblyPlan.model_json_schema(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _manual(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from .assembly import graph
    from .perception import manual

    load_dotenv(REPO / ".env")
    plan = manual.read_manual([Path(p) for p in args.pages], plan_id=args.id, title=args.title)
    text = plan.model_dump_json(indent=1, exclude_none=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    if args.truth:
        score = graph.compare(plan, load_plan(args.truth))
        print(f"edges F1 {score.edges_f1:.3f}  parts F1 {score.parts_f1:.3f}  exact steps {score.steps_exact:.2f}  "
              f"hardware {score.hardware_accuracy:.2f}", file=sys.stderr)  # fmt: skip
    return 0


def _track(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from .perception import video

    load_dotenv(REPO / ".env")
    result = video.analyze(Path(args.video), load_plan(args.plan), every_s=args.every)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darius", description="Assembly copilot: plans, perception and tracking.")
    parser.add_argument("--version", action="version", version=f"darius {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("library", help="list the furniture library").set_defaults(run=_library)
    command = commands.add_parser("export", help="write every library plan for the web app")
    command.add_argument("--out", default=str(REPO / "assemblies"))
    command.set_defaults(run=_export)
    command = commands.add_parser("show", help="print a plan and its graph")
    command.add_argument("plan")
    command.set_defaults(run=_show)
    command = commands.add_parser("validate", help="check plans for consistency")
    command.add_argument("plans", nargs="+")
    command.set_defaults(run=_validate)
    commands.add_parser("schema", help="print the AssemblyPlan JSON Schema").set_defaults(run=_schema)
    command = commands.add_parser("manual", help="read manual pages into a plan (Meta Model API)")
    command.add_argument("pages", nargs="+")
    command.add_argument("--id", default="manual")
    command.add_argument("--title", default="")
    command.add_argument("--truth", help="a plan to score the result against")
    command.add_argument("--out")
    command.set_defaults(run=_manual)
    command = commands.add_parser("track", help="decode a build video into a timeline (Meta Model API)")
    command.add_argument("video")
    command.add_argument("--plan", required=True)
    command.add_argument("--every", type=float, default=2.0, help="seconds between analyzed frames")
    command.add_argument("--out")
    command.set_defaults(run=_track)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
