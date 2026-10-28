"""Material lookup table (darius/data/materials.json), shared with the simulator's material editor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class Material(TypedDict):
    density_kg_m3: float
    friction: float
    restitution: float
    fragile: bool
    rigidity: str


_TABLE: dict[str, Material] = {
    name: value
    for name, value in json.loads((Path(__file__).resolve().parents[1] / "data" / "materials.json").read_text()).items()
    if not name.startswith("_")
}

MATERIAL_NAMES: list[str] = list(_TABLE)

# Distinct, colorblind-friendly hues for per-object overlays and fallback colors.
PALETTE: list[tuple[int, int, int]] = [
    (228, 87, 46),
    (47, 93, 138),
    (243, 198, 35),
    (108, 195, 185),
    (155, 89, 182),
    (46, 204, 113),
    (231, 76, 160),
    (52, 73, 94),
    (230, 126, 34),
    (26, 188, 156),
    (192, 57, 43),
    (41, 128, 185),
    (142, 68, 173),
    (39, 174, 96),
    (211, 84, 0),
    (127, 140, 141),
]


def normalize(material: str) -> str:
    """Map free-form material names onto the table ("Ceramic mug" -> "ceramic")."""
    text = material.strip().lower()
    if text in _TABLE:
        return text
    aliases = {
        "porcelain": "ceramic",
        "stoneware": "ceramic",
        "clay": "ceramic",
        "steel": "metal",
        "iron": "metal",
        "brass": "metal",
        "tin": "metal",
        "aluminium": "aluminum",
        "cloth": "fabric",
        "cotton": "fabric",
        "plush": "fabric",
        "textile": "fabric",
        "wool": "fabric",
        "silicone": "rubber",
        "latex": "rubber",
        "food": "organic",
        "fruit": "organic",
        "vegetable": "organic",
        "plant": "organic",
        "marble": "stone",
        "concrete": "stone",
        "granite": "stone",
        "book": "paper",
        "cardstock": "cardboard",
        "acrylic": "plastic",
        "polymer": "plastic",
    }
    for word in text.replace("-", " ").split():
        if word in _TABLE:
            return word
        if word in aliases:
            return aliases[word]
    return "unknown"


def lookup(material: str) -> Material:
    return _TABLE[normalize(material)]
