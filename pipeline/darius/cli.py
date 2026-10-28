"""The `darius` command line.

    darius info SCENE                  what is in a scene package
    darius validate SCENE...           check packages against the contract and physical sanity
    darius export SCENE [--out DIR]    write the robotics bundle (MJCF, URDF, USD, world.json)
    darius graph SCENE [--json]        the 3D scene graph: what supports what, what is near what
    darius sim SCENE [--seconds 4]     simulate the scene in MuJoCo and check it comes to rest
    darius gallery [--check]           rebuild (or check) every shipped scene's derived files
    darius schema                      print the scene.json JSON Schema

The video-to-scene compiler itself runs in the cloud: see `modal_app.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _info(args: argparse.Namespace) -> int:
    from .scene.io import load_scene

    package = load_scene(args.scene)
    scene = package.manifest
    print(f"{scene.title}  ({scene.scene_id}, scene format v{scene.version})")
    if scene.provenance:
        print(f"source: {scene.provenance.source}  ·  scale: {scene.scale_source}")
    plane = scene.background.table_plane
    print(f"table: {plane.size_m[0]:.2f} x {plane.size_m[1]:.2f} m  ·  gravity {scene.gravity}")
    print()
    header = f"{'id':<8}{'label':<22}{'material':<11}{'mass':>9}{'mu':>6}{'e':>6}  {'collider':<16}{'grasps':>7}"
    print(header)
    print("-" * len(header))
    for obj in scene.objects:
        collider = obj.collider.type + (f" x{len(obj.collider.files)}" if obj.collider.type == "convex_parts" else "")
        grasps = "-" if obj.grasps is None else str(len(obj.grasps))
        mass = f"{obj.mass_kg * 1000:.0f} g" if obj.mass_kg < 1 else f"{obj.mass_kg:.2f} kg"
        print(
            f"{obj.id:<8}{obj.label[:21]:<22}{obj.material:<11}{mass:>9}{obj.friction:>6.2f}"
            f"{obj.restitution:>6.2f}  {collider:<16}{grasps:>7}"
        )
    robotics = package.root / "robotics"
    print()
    print(f"robotics bundle: {'present' if robotics.is_dir() else 'missing (darius export)'}")
    return 0


def _graph(args: argparse.Namespace) -> int:
    from .robotics import graph
    from .robotics.model import build
    from .scene.io import load_scene

    scene = build(load_scene(args.scene))
    data = graph.build(scene).as_dict(scene)
    if args.json:
        print(json.dumps(data, indent=1))
        return 0
    labels = {n["id"]: n.get("label", n["id"]) for n in data["nodes"]}
    print(f"{data['title']}: {data['counts']['nodes']}, {data['counts']['edges']} edges")
    for edge in data["edges"]:
        if edge["relation"] in ("supports", "near"):
            extra = f"  ({edge['distance_m'] * 100:.1f} cm apart)" if "distance_m" in edge else ""
            print(f"  {labels[edge['source']]:>22}  {edge['relation']:<9} {labels[edge['target']]}{extra}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .tools.validate import validate

    failures = 0
    for scene in args.scenes:
        problems = validate(Path(scene))
        print(f"{'ok  ' if not problems else 'FAIL'} {scene}")
        for problem in problems:
            print(f"     - {problem}")
        failures += bool(problems)
    return 1 if failures else 0


def _export(args: argparse.Namespace) -> int:
    from .robotics.export import bundle

    folder = bundle.export(Path(args.scene), Path(args.out) if args.out else None)
    print(folder)
    return 0


def _sim(args: argparse.Namespace) -> int:
    from .robotics.sim import PushEnv

    env = PushEnv(args.scene)
    report = env.settle(args.seconds)
    failures = 0
    for name, values in report.items():
        ok = values["drift_m"] < 0.03 and values["motion_m"] < 0.002 and values["turn_deg"] < 2
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {name}: drift {values['drift_m'] * 100:.1f} cm, "
            f"still moving {values['motion_m'] * 1000:.2f} mm / {values['turn_deg']:.2f} deg in the last 0.5 s, "
            f"tilt {values['tilt_deg']:.1f} deg"
        )
    return 1 if failures else 0


def _gallery(args: argparse.Namespace) -> int:
    from .tools import gallery

    return gallery.main(check=args.check)


def _schema(args: argparse.Namespace) -> int:
    from .scene.schema import Scene

    json.dump(Scene.model_json_schema(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darius", description="Real-to-sim scene compiler: tools for scene packages.")
    parser.add_argument("--version", action="version", version=f"darius {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("info", help="summarize a scene package")
    command.add_argument("scene")
    command.set_defaults(run=_info)

    command = commands.add_parser("validate", help="check scene packages against the contract")
    command.add_argument("scenes", nargs="+")
    command.set_defaults(run=_validate)

    command = commands.add_parser("export", help="write the robotics bundle (MJCF, URDF, USD, world.json)")
    command.add_argument("scene")
    command.add_argument("--out", help="folder to write (default: SCENE/robotics, plus SCENE/export.zip)")
    command.set_defaults(run=_export)

    command = commands.add_parser("graph", help="print the 3D scene graph (supports, near, grasps, parts)")
    command.add_argument("scene")
    command.add_argument("--json", action="store_true", help="print the whole graph as JSON")
    command.set_defaults(run=_graph)

    command = commands.add_parser("sim", help="simulate the scene in MuJoCo and check that it comes to rest")
    command.add_argument("scene")
    command.add_argument("--seconds", type=float, default=4.0)
    command.set_defaults(run=_sim)

    command = commands.add_parser("gallery", help="rebuild every shipped scene's annotations, bundles and index")
    command.add_argument("--check", action="store_true", help="fail if anything committed is out of date")
    command.set_defaults(run=_gallery)

    command = commands.add_parser("schema", help="print the scene.json JSON Schema")
    command.set_defaults(run=_schema)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
