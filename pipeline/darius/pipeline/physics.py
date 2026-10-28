"""Stage 5, physics: identify each object, recover metric scale, and estimate its physics.

VGGT reconstructs up to an unknown scale. Muse Spark knows what a mug is and
roughly how big mugs are, so each confidently recognized object votes for a
scale: expected size / measured size. The scene scale is the confidence-weighted
median of those votes, which tolerates one badly recognized object.

Mass is computed, not guessed: hull volume x material density x fill fraction,
where fill fraction accounts for hollow objects (a mug's hull includes its
cavity). Muse Spark's direct mass guess is only a sanity check: if the computed
mass disagrees by more than MASS_TOLERANCE, the guess wins and the reason is
recorded in `physics_notes`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..core import geometry, materials
from ..models import meta_api
from .jobs import load_arrays

MASS_TOLERANCE = 2.5
# Optionally tell the model the scan's measured proportions. On Objectron (docs/results.md)
# the hint tightens the median size error slightly but produces more outliers, so it is off.
PROPORTION_HINT = "\nIts 3D scan has these relative proportions (length : height : width, height is vertical): {}."
MIN_SCALE_CONFIDENCE = 0.5

PROPERTIES_PROMPT = """These {n} photos show the same object, highlighted, from different angles on a tabletop.{hint}

Identify it and estimate its physical properties for a rigid-body simulation:
- label: a short common noun phrase, e.g. "coffee mug"
- material: its main material, one of: {materials}
- typical_dimensions_cm: [length, height, width] of a typical object like this, as it sits on the table
- typical_mass_kg: what a typical object like this weighs
- hollow: true if its convex outline encloses mostly empty space (mugs, bowls, boxes, shoes)
- fill_fraction: the fraction of its convex outline that is solid material (1.0 for a solid block,
  about 0.25 for a ceramic mug, 0.05 for an empty cardboard box)
- friction: Coulomb friction coefficient against a wooden table (0.1 to 1.2)
- restitution: bounciness on a hard surface (0 = no bounce, 0.9 = superball)
- fragile: true if it would likely break when dropped on a hard floor
- rigidity: "rigid" or "soft"
- confidence: 0 to 1, how sure you are about the label"""


class ObjectProperties(BaseModel):
    label: str
    material: str
    typical_dimensions_cm: list[float] = Field(min_length=3, max_length=3)
    typical_mass_kg: float = Field(gt=0)
    hollow: bool
    fill_fraction: float = Field(gt=0, le=1)
    friction: float = Field(ge=0, le=2)
    restitution: float = Field(ge=0, le=1)
    fragile: bool
    rigidity: str
    confidence: float = Field(ge=0, le=1)


@dataclass
class Estimate:
    index: int
    label: str
    properties: ObjectProperties
    material: str
    dims_raw: np.ndarray
    hull_volume_raw: float


def scale_vote(expected_m: np.ndarray, measured_raw: np.ndarray) -> float:
    """Meters per VGGT unit implied by one object, matching dimensions by rank.

    Matching sorted dimensions sidesteps disagreement about which side is the
    "length". The median of the three ratios ignores one badly measured side
    (usually the depth of a thin object).
    """
    expected = np.sort(np.asarray(expected_m, dtype=float))
    measured = np.sort(np.asarray(measured_raw, dtype=float))
    return float(np.median(expected / np.maximum(measured, 1e-9)))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def estimate_scale(estimates: list[Estimate]) -> tuple[float, list[float], str]:
    votes, weights, names = [], [], []
    for e in estimates:
        if e.properties.confidence >= MIN_SCALE_CONFIDENCE:
            votes.append(scale_vote(np.array(e.properties.typical_dimensions_cm) / 100, e.dims_raw))
            weights.append(e.properties.confidence)
            names.append(e.properties.label)
    if not votes:  # nothing recognized confidently: use everything, flagged in scale_source
        votes = [scale_vote(np.array(e.properties.typical_dimensions_cm) / 100, e.dims_raw) for e in estimates]
        weights = [1.0] * len(votes)
        names = ["low-confidence"]
    scale = weighted_median(np.array(votes), np.array(weights))
    return scale, [round(v, 5) for v in votes], "known_objects:" + ",".join(names)


def mass_from_volume(volume_m3: float, density: float, fill_fraction: float) -> float:
    return volume_m3 * density * fill_fraction


def resolve_mass(computed: float, typical: float) -> tuple[float, str | None]:
    """Prefer the geometric estimate; fall back to the typical mass if they disagree badly."""
    ratio = computed / typical
    if 1 / MASS_TOLERANCE <= ratio <= MASS_TOLERANCE:
        return computed, None
    note = (
        f"volume x density gave {computed * 1000:.0f} g, {ratio:.1f}x the typical "
        f"{typical * 1000:.0f} g; using the typical mass"
    )
    return typical, note


def sanitize(props: ObjectProperties) -> tuple[ObjectProperties, str]:
    """Clamp to physically sensible ranges and normalize the material name."""
    material = materials.normalize(props.material)
    fixed = props.model_copy(
        update={
            "friction": float(np.clip(props.friction, 0.05, 1.5)),
            "restitution": float(np.clip(props.restitution, 0.0, 0.95)),
            "fill_fraction": float(np.clip(props.fill_fraction, 0.02, 1.0)),
            "rigidity": props.rigidity if props.rigidity in ("rigid", "soft") else "rigid",
            "typical_dimensions_cm": [max(0.5, float(v)) for v in props.typical_dimensions_cm],
        }
    )
    return fixed, material


def object_crops(job_dir: Path, index: int, count: int = 3) -> list[Image.Image]:
    """Crops of the object from the frames where its mask is largest, spread out in time,
    with everything outside the mask dimmed so the model knows which object we mean."""
    labels = load_arrays(job_dir / "masks.npz")["labels"]
    areas = (labels == index).reshape(len(labels), -1).sum(axis=1)
    ranked = [int(i) for i in np.argsort(-areas) if areas[i] > 0]
    picks: list[int] = []
    for frame in ranked:
        if all(abs(frame - p) >= max(2, len(labels) // 8) for p in picks):
            picks.append(frame)
        if len(picks) == count:
            break
    crops = []
    for frame in picks:
        image = Image.open(job_dir / "frames" / f"{frame:03d}.jpg").convert("RGB")
        mask = Image.fromarray(((labels[frame] == index) * 255).astype(np.uint8)).resize(image.size)
        mask_np = np.asarray(mask) > 127
        ys, xs = np.nonzero(mask_np)
        pixels = np.asarray(image).astype(np.float32)
        pixels[~mask_np] *= 0.35
        pad = int(0.25 * max(np.ptp(xs), np.ptp(ys))) + 8
        box = (
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(image.width, xs.max() + pad),
            min(image.height, ys.max() + pad),
        )
        crops.append(Image.fromarray(pixels.astype(np.uint8)).crop(box))
    return crops


def run(job_dir: Path, hint_proportions: bool = False, output: str = "physics.json") -> dict:
    from concurrent.futures import ThreadPoolExecutor

    lift = json.loads((job_dir / "lift.json").read_text())
    arrays = load_arrays(job_dir / "lift.npz")

    def estimate(obj: dict) -> tuple[Estimate, dict]:
        points = arrays[f"obj{obj['index']}_points"]
        dims_raw = np.array(obj["dims_raw"])
        hull = geometry.convex_hull(geometry.complete_bottom(points, geometry.SUPPORT_BAND * dims_raw[1]))
        proportions = " : ".join(f"{d / dims_raw.max():.2f}" for d in dims_raw)
        hint = PROPORTION_HINT.format(proportions) if hint_proportions else ""
        prompt = PROPERTIES_PROMPT.format(n=3, hint=hint, materials=", ".join(materials.MATERIAL_NAMES))
        props, usage = meta_api.ask_structured(prompt, object_crops(job_dir, obj["index"]), ObjectProperties)
        props, material = sanitize(props)
        return Estimate(obj["index"], obj["label"], props, material, dims_raw, hull.volume), usage

    # One Muse Spark call per object, all at once: they are independent and network-bound.
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(lift["objects"])))) as pool:
        results = list(pool.map(estimate, lift["objects"]))
    estimates = [e for e, _ in results]
    usage_total = {
        key: sum(usage.get(key, 0) for _, usage in results) for key in ("prompt_tokens", "completion_tokens")
    }

    if not estimates:
        raise RuntimeError("no objects survived 3D separation")
    scale, votes, scale_source = estimate_scale(estimates)

    out = []
    for e in estimates:
        table = materials.lookup(e.material)
        volume = e.hull_volume_raw * scale**3
        computed = mass_from_volume(volume, table["density_kg_m3"], e.properties.fill_fraction)
        mass, note = resolve_mass(computed, e.properties.typical_mass_kg)
        out.append(
            {
                "index": e.index,
                # The detection label is kept: it was chosen while seeing every object at
                # once, so it tells similar objects apart ("white mug", "purple mug").
                "label": e.label,
                "category": e.properties.label,
                "confidence": e.properties.confidence,
                "material": e.material,
                "volume_m3": volume,
                "hollow": e.properties.hollow,
                "fill_fraction": e.properties.fill_fraction,
                "density_kg_m3": table["density_kg_m3"],
                "mass_kg": mass,
                "mass_computed_kg": computed,
                "typical_mass_kg": e.properties.typical_mass_kg,
                "typical_dimensions_cm": e.properties.typical_dimensions_cm,
                "measured_dimensions_cm": (e.dims_raw * scale * 100).round(1).tolist(),
                "friction": e.properties.friction,
                "restitution": e.properties.restitution,
                "fragile": e.properties.fragile,
                "rigidity": e.properties.rigidity,
                "physics_notes": note,
            }
        )
    result = {
        "scale_m_per_unit": scale,
        "scale_votes": votes,
        "scale_source": scale_source,
        "proportion_hint": hint_proportions,
        "muse_spark_usage": usage_total,
        "objects": out,
    }
    (job_dir / output).write_text(json.dumps(result, indent=2) + "\n")
    return result
