"""The `darius` command line: LEGO assembly intelligence and retained furniture tools.

    darius lego library                    list original LEGO curriculum builds
    darius lego compile BUILD              compile an id or source JSON into a nominal robot plan
    darius lego instructions SET           discover official instruction booklet links
    darius lego draft-set SET --out DIR    acquire selected public pages and infer a review-required draft
    darius lego dataset BUILD --out DIR    export hashed synthetic reference trajectories
    darius lego sim BUILD --out FILE       export a nominal MuJoCo scene scaffold

    darius library                         list the furniture library
    darius export [--out ../assemblies]    write every library plan (and an index) for the web app
    darius show PLAN                       a plan's parts, joints and steps, and its graph
    darius validate PLAN...                check plans for consistency
    darius schema                          print the AssemblyPlan JSON Schema
    darius manual PAGE...  [--truth PLAN]  read manual pages into a plan (Meta Model API)
    darius track VIDEO --plan PLAN         decode a build video into a timeline (Meta Model API)
    darius parts VIDEO --plan PLAN         ground and track every part through a video (Muse Spark + SAM 3.1)
    darius eval [--tracking] [--replay]    benchmark the models on the library sessions

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


def _pages(paths: list[str]) -> list[Path]:
    """Page images from files and folders (a folder contributes its images in name order)."""
    pages: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            pages += sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        else:
            pages.append(path)
    return pages


def _manual(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    from PIL import Image

    from .assembly import graph
    from .perception import manual

    load_dotenv(REPO / ".env")
    pages = [Image.open(p).convert("RGB") for p in _pages(args.pages)]
    result = manual.read_manual(pages, plan_id=args.id)
    text = result.plan.model_dump_json(indent=1, exclude_none=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    if args.truth:
        score = graph.compare(result.plan, load_plan(args.truth))
        print(f"edges F1 {score.edges_f1:.3f}  parts F1 {score.parts_f1:.3f}  exact steps {score.steps_exact:.2f}  "
              f"hardware {score.hardware_accuracy:.2f}", file=sys.stderr)  # fmt: skip
    return 0


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} batches", end="\n" if done == total else "", file=sys.stderr, flush=True)


def _track(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from .perception import video

    load_dotenv(REPO / ".env")
    result = video.analyze(Path(args.video), load_plan(args.plan), every_s=args.every, on_batch=_progress)
    text = result.model_dump_json(indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    finished = ", ".join(f"step {k} at {t:.1f} s" for k, t in sorted(result.finished_at.items()))
    print(f"{len(result.frames)} frames, finished: {finished or 'nothing'}", file=sys.stderr)
    return 0


def _parts(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    from PIL import Image

    from .perception import parts

    load_dotenv(REPO / ".env")
    page = Image.open(args.parts_page).convert("RGB") if args.parts_page else None
    result = parts.track_parts(Path(args.video), load_plan(args.plan), parts_page=page, anchor_s=args.anchor)
    text = result.model_dump_json(indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    labels = " ".join(t.label for t in result.tracks)
    print(f"{len(result.tracks)} parts ({labels}), {result.frames} frames, {result.seconds} s", file=sys.stderr)
    return 0


def _eval(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from .eval import benchmark
    from .furniture import catalog

    load_dotenv(REPO / ".env")
    ids = args.only or list(catalog.CATALOG)
    if args.check:
        problems = benchmark.check(ids)
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(ids)} plans: stored results {'match' if not problems else 'DO NOT match'} their re-scored values")
        return 1 if problems else 0
    every = not (args.manuals or args.sessions or args.tracking)
    for plan_id in ids:
        if args.manuals or every:
            score = benchmark.eval_manual(plan_id, replay=args.replay)["score"]
            print(f"{plan_id:18} manual   edges F1 {score['edges_f1']:.3f}  parts F1 {score['parts_f1']:.3f}  "
                  f"exact steps {score['steps_exact']:.2f}  hardware {score['hardware_accuracy']:.2f}")  # fmt: skip
        if args.sessions or every:
            print(f"{plan_id:18} session  reading frames…", file=sys.stderr)
            score = benchmark.eval_session(plan_id, every_s=args.every, replay=args.replay)["score"]
            print(f"{plan_id:18} session  frame accuracy {score['frame_accuracy']:.3f} (one frame alone "
                  f"{score['raw_frame_accuracy']:.3f})  steps found {score['steps_found']}/{score['steps']}  "
                  f"step error {score['step_error_s']} s")  # fmt: skip
        if args.tracking or every:
            print(f"{plan_id:18} tracking grounding and tracking the parts…", file=sys.stderr)
            score = benchmark.eval_tracking(plan_id, replay=args.replay)["score"]
            print(f"{plan_id:18} tracking mask IoU {score['mean_iou']:.3f}  tracked {score['frames_tracked']:.2f}  "
                  f"letters {score['letters_right']}/{score['pieces']}  "
                  f"frames right: Muse Spark {score['frames_right_spark']:.3f}  "
                  f"contacts {score['frames_right_contacts']:.3f}  "
                  f"fused {score['frames_right_fused']:.3f}")  # fmt: skip
    benchmark.summarize()
    return 0


def _lego(args: argparse.Namespace) -> int:
    from .lego import compile_assembly, get_assembly, library_index
    from .lego.catalog import export_library
    from .lego.instructions import InstructionLookupError, lookup_instructions
    from .lego.schema import LegoAssembly

    def compiled():
        assembly = (
            get_assembly(args.build)
            if not Path(args.build).is_file()
            else LegoAssembly.model_validate_json(Path(args.build).read_text())
        )
        return compile_assembly(assembly)

    def emit(value):
        text = value.model_dump_json(indent=2) if hasattr(value, "model_dump_json") else json.dumps(value, indent=2)
        if getattr(args, "out", None):
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            print(f"wrote {path}")
        else:
            print(text)

    try:
        if args.lego_command == "library":
            emit(library_index())
        elif args.lego_command == "schema":
            emit(LegoAssembly.model_json_schema())
        elif args.lego_command == "compile":
            result = compiled()
            emit(result)
            return 0 if result.validation.valid else 1
        elif args.lego_command == "instructions":
            emit(lookup_instructions(args.set_number, refresh=args.refresh))
        elif args.lego_command == "export":
            export_library(Path(args.out))
            print(f"exported LEGO curriculum to {args.out}")
        elif args.lego_command == "manual":
            from dotenv import load_dotenv
            from PIL import Image

            from .lego.manual import read_manual_draft

            load_dotenv(REPO / ".env")
            paths = _pages(args.pages)
            if not 1 <= len(paths) <= 30:
                raise ValueError("provide 1 to 30 ordered page images")
            emit(
                read_manual_draft(
                    [Image.open(path).convert("RGB") for path in paths], args.id, source_url=args.source_url
                )
            )
        elif args.lego_command in {"fetch", "draft-set"}:
            from .lego.documents import fetch_booklet

            selected = [int(page.strip()) for page in args.pages.split(",")] if args.pages else None
            result = fetch_booklet(
                args.set_number, Path(args.out), booklet_index=args.booklet, pages=selected, refresh=args.refresh
            )
            if args.lego_command == "fetch":
                print(json.dumps(result, indent=2))
            else:
                from dotenv import load_dotenv
                from PIL import Image

                from .lego.manual import read_manual_draft

                load_dotenv(REPO / ".env")
                if not result["page_paths"]:
                    raise ValueError("draft-set requires selected rendered pages via --pages")
                draft = read_manual_draft(
                    [Image.open(page).convert("RGB") for page in result["page_paths"]],
                    f"set-{args.set_number}-draft",
                    source_url=f"https://www.lego.com/en-us/service/building-instructions/{args.set_number}",
                )
                draft.source_page_map = result["manifest"]["selected_pages"]
                draft.acquisition = result["manifest"]
                path = Path(args.out) / "draft.json"
                path.write_text(draft.model_dump_json(indent=2) + "\n")
                print(f"wrote unapproved candidate draft {path}; review required")
        elif args.lego_command == "dataset":
            from .robotics.dataset import export_dataset

            result = export_dataset(compiled(), Path(args.out), episodes=args.episodes, seed=args.seed, fps=args.fps)
            print(json.dumps(result, indent=2))
        elif args.lego_command == "sim":
            from .robotics.mujoco import export_scene

            print(export_scene(compiled(), Path(args.out)))
    except (KeyError, ValueError, InstructionLookupError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="darius",
        description="LEGO assembly intelligence: compile builds, discover manuals, generate learning data.",
    )
    parser.add_argument("--version", action="version", version=f"darius {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    lego = commands.add_parser("lego", help="LEGO task compiler, public instructions and robot-training drafts")
    lego_commands = lego.add_subparsers(dest="lego_command", required=True)
    lego_commands.add_parser("library", help="list original LEGO curriculum builds").set_defaults(run=_lego)
    command = lego_commands.add_parser("schema", help="export the LEGO assembly JSON Schema")
    command.add_argument("--out")
    command.set_defaults(run=_lego)
    command = lego_commands.add_parser("compile", help="compile a curriculum id or assembly JSON into a task plan")
    command.add_argument("build")
    command.add_argument("--out")
    command.set_defaults(run=_lego)
    command = lego_commands.add_parser("instructions", help="discover official public instructions by set number")
    command.add_argument("set_number")
    command.add_argument("--refresh", action="store_true", help="refresh metadata directly from LEGO.com")
    command.add_argument("--out")
    command.set_defaults(run=_lego)
    command = lego_commands.add_parser("export", help="export curriculum assembly and compiled plan fixtures")
    command.add_argument("--out", default=str(REPO / "assemblies" / "lego"))
    command.set_defaults(run=_lego)
    command = lego_commands.add_parser("manual", help="read ordered page images into an uncertain LEGO plan draft")
    command.add_argument("pages", nargs="+")
    command.add_argument("--id", default="manual-draft")
    command.add_argument("--source-url")
    command.add_argument("--out")
    command.set_defaults(run=_lego)
    for name in ("fetch", "draft-set"):
        command = lego_commands.add_parser(
            name,
            help="fetch official booklet"
            + (" and infer an unapproved plan draft" if name == "draft-set" else " metadata/PDF/pages"),
        )
        command.add_argument("set_number")
        command.add_argument("--booklet", type=int, default=1)
        command.add_argument(
            "--pages",
            default="",
            required=name == "draft-set",
            help="comma-separated 1-based PDF pages to render (max16)",
        )
        command.add_argument("--refresh", action="store_true")
        command.add_argument("--out", required=True)
        command.set_defaults(run=_lego)
    command = lego_commands.add_parser("dataset", help="export synthetic Cartesian reference episodes")
    command.add_argument("build")
    command.add_argument("--out", required=True)
    command.add_argument("--episodes", type=int, default=8)
    command.add_argument("--seed", type=int, default=7)
    command.add_argument("--fps", type=int, default=10)
    command.set_defaults(run=_lego)
    command = lego_commands.add_parser("sim", help="export a nominal MuJoCo scene (no robot or clutch model)")
    command.add_argument("build")
    command.add_argument("--out", required=True)
    command.set_defaults(run=_lego)

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
    command.add_argument("pages", nargs="+", help="page images, or folders of them, in order")
    command.add_argument("--id", default="manual")
    command.add_argument("--truth", help="a plan to score the result against")
    command.add_argument("--out")
    command.set_defaults(run=_manual)
    command = commands.add_parser("track", help="decode a build video into a timeline (Meta Model API)")
    command.add_argument("video")
    command.add_argument("--plan", required=True)
    command.add_argument("--every", type=float, default=1.0, help="seconds between analyzed frames")
    command.add_argument("--out")
    command.set_defaults(run=_track)
    command = commands.add_parser("parts", help="ground and track every part through a video (Muse Spark + SAM 3.1)")
    command.add_argument("video")
    command.add_argument("--plan", required=True)
    command.add_argument("--parts-page", help="the manual's parts page, to name the parts by")
    command.add_argument("--anchor", type=float, default=1.0, help="seconds in, where every part lies apart")
    command.add_argument("--out")
    command.set_defaults(run=_parts)
    command = commands.add_parser("eval", help="benchmark the perception models on the library (Meta Model API)")
    command.add_argument("--manuals", action="store_true", help="only read the manuals")
    command.add_argument("--sessions", action="store_true", help="only watch the session videos")
    command.add_argument("--tracking", action="store_true", help="only track the sessions' parts (SAM 3.1)")
    command.add_argument("--only", nargs="+", help="plan ids")
    command.add_argument("--every", type=float, default=1.0, help="seconds between analyzed frames")
    command.add_argument("--replay", action="store_true", help="re-score the stored model outputs, no API calls")
    command.add_argument("--check", action="store_true", help="fail if any stored score differs from a re-score")
    command.set_defaults(run=_eval)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
