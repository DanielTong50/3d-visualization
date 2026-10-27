"""Rebuild scenes/index.json (the demo gallery) and every scene's export bundle.

    uv run python -m scan2sim.gallery

CI runs this and fails if the committed files differ, so the gallery can't drift
from the scene folders.
"""

from __future__ import annotations

import json
from pathlib import Path

from .package import write_export_bundle
from .schema import Scene

SCENES = Path(__file__).resolve().parents[2] / "scenes"

# Hand-written blurbs; scenes not listed here get a generic description.
DESCRIPTIONS = {
    "objectron-mugs": "Real scan of two mugs on a garden table, from a 7-second phone clip.",
    "sandbox": "Hand-authored reference scene: a heavy book, a light tissue box, a ceramic mug, a rubber ball "
    "and a convex-hull doorstop.",
}
ORDER = ["objectron-mugs", "sandbox"]


def listing(scene_dir: Path) -> dict:
    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    scanned = scene.provenance is not None
    entry = {
        "id": scene_dir.name,
        "title": scene.title,
        "description": DESCRIPTIONS.get(
            scene_dir.name, f"{len(scene.objects)} objects" + (" scanned from video." if scanned else ".")
        ),
        "kind": "scanned" if scanned else "hand-authored",
        "object_count": len(scene.objects),
        "objects": [o.label for o in scene.objects],
        "thumbnail": "thumbnail.jpg" if (scene_dir / "thumbnail.jpg").exists() else None,
        "export": "export.zip",
    }
    if scanned:
        entry["credit"] = scene.provenance.source + (
            f" ({scene.provenance.license})" if scene.provenance.license else ""
        )
    return entry


def main() -> None:
    def rank(scene_dir: Path) -> tuple[int, str]:
        return (ORDER.index(scene_dir.name) if scene_dir.name in ORDER else len(ORDER), scene_dir.name)

    dirs = sorted((p.parent for p in SCENES.glob("*/scene.json")), key=rank)
    for scene_dir in dirs:
        write_export_bundle(scene_dir)
    index = [listing(d) for d in dirs]
    (SCENES / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(index)} scenes -> {SCENES / 'index.json'}")


if __name__ == "__main__":
    main()
