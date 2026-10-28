"""Render the README's pipeline figure for one processed job.

    uv run python scripts/make_figures.py d683afbd459c --name objectron-mugs

Panels: keyframe -> SAM 3.1 masks -> reconstructed, gravity-aligned scene ->
separated objects with their collision hulls and estimated physics.
Writes docs/media/<name>-pipeline.png.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import modal  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from darius.core import geometry  # noqa: E402
from darius.core.materials import PALETTE  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def scatter(ax, points, colors, size=0.6):
    # Scene is y-up; matplotlib's 3D axes are z-up.
    ax.scatter(points[:, 0], -points[:, 2], points[:, 1], c=colors, s=size, linewidths=0, depthshade=False)


def style(ax, points, elev=28, azim=-60):
    lo, hi = np.percentile(points, [1, 99], axis=0)
    center, radius = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(-center[2] - radius, -center[2] + radius)
    ax.set_zlim(-0.2 * radius, 1.2 * radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.7))
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    volume = modal.Volume.from_name("darius-jobs")
    job = f"/{args.job}"
    detections = json.loads(read(volume, f"{job}/detections.json"))
    physics = json.loads(read(volume, f"{job}/physics.json"))
    keyframe = Image.open(io.BytesIO(read(volume, f"{job}/frames/{detections['keyframe']:03d}.jpg")))
    overlay = Image.open(io.BytesIO(read(volume, f"{job}/segmentation.jpg")))
    lift = np.load(io.BytesIO(read(volume, f"{job}/lift.npz")))
    scale = physics["scale_m_per_unit"]
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(16, 4.6), dpi=150)
    titles = [
        "1 · Keyframe",
        "2 · SAM 3.1 masks (box prompts from Muse Spark)",
        "3 · VGGT reconstruction, gravity-aligned",
        "4 · Separated objects + collision hulls",
    ]
    for i, image in enumerate([keyframe, overlay]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=10)

    objects = {o["index"]: o for o in physics["objects"]}
    object_points = np.concatenate([lift[f"obj{i}_points"] for i in objects]) * scale
    center = object_points.mean(axis=0)
    radius = np.percentile(np.linalg.norm(object_points - center, axis=1), 95)
    background = lift["background_points"] * scale
    near = np.flatnonzero(np.linalg.norm((background - center)[:, [0, 2]], axis=1) < 3.5 * radius)
    pick = rng.choice(near, min(len(near), 60_000), replace=False)
    cloud = [background[pick]]
    colors = [lift["background_colors"][pick] / 255]
    for index in objects:
        pts = lift[f"obj{index}_points"] * scale
        sel = rng.choice(len(pts), min(len(pts), 15_000), replace=False)
        cloud.append(pts[sel])
        colors.append(lift[f"obj{index}_colors"][sel] / 255)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    scatter(ax, np.concatenate(cloud), np.concatenate(colors), size=0.4)
    style(ax, np.concatenate(cloud[1:]) if len(cloud) > 1 else cloud[0], azim=-70)
    ax.set_title(titles[2], fontsize=10)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    table = background[near][np.abs(background[near][:, 1]) < 0.004]
    table = table[rng.choice(len(table), min(len(table), 6000), replace=False)]
    scatter(ax, table, [(0.75, 0.75, 0.75)], size=0.3)
    all_points = []
    for index, props in objects.items():
        pts = lift[f"obj{index}_points"] * scale
        all_points.append(pts)
        color = np.array(PALETTE[(index - 1) % len(PALETTE)]) / 255
        sel = rng.choice(len(pts), min(len(pts), 6000), replace=False)
        scatter(ax, pts[sel], [color], size=0.5)
        hull = geometry.convex_hull(geometry.complete_bottom(pts, geometry.SUPPORT_BAND * np.ptp(pts[:, 1])))
        for face in hull.faces:
            tri = hull.vertices[np.append(face, face[0])]
            ax.plot(tri[:, 0], -tri[:, 2], tri[:, 1], color=color, lw=0.35, alpha=0.6)
        top = pts[np.argmax(pts[:, 1])]
        ax.text(
            top[0],
            -top[2],
            top[1] + 0.03,
            f"{props['label']}\n{props['mass_kg'] * 1000:.0f} g · μ {props['friction']:.2f} "
            f"· e {props['restitution']:.2f}",
            fontsize=7,
            ha="center",
            color=color * 0.7,
        )
    style(ax, np.concatenate(all_points), azim=-70)
    ax.set_title(titles[3], fontsize=10)

    fig.tight_layout()
    out = REPO / "docs" / "media" / f"{args.name}-pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
