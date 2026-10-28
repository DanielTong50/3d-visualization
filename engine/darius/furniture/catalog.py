"""The furniture library: four flat-pack products, designed as boxes and compiled into plans.

Each design places every part in the finished piece and lists the build the way a
manual would: which parts come in at each step, which joints they make, and how the
assembly lies on the floor meanwhile (a bookcase is built on its side, nailed face
down, then stood up). Joints, fastener positions, hardware counts and the floor
layout are computed by `FurnitureBuilder`. The plans are exact, so they serve as
ground truth for reading manuals and tracking builds.

Hardware numbers follow the six-digit style of flat-pack manuals but are this
library's own.
"""

from __future__ import annotations

from collections.abc import Callable

from ..assembly.schema import Action, AssemblyPlan
from .builder import FurnitureBuilder

WHITE = "#efece6"
BACK_BOARD = "#ddd8ce"
OAK = "#d6bc93"
PINE = "#dcb98a"
BLACK = "#36353a"


def hardware(builder: FurnitureBuilder) -> None:
    builder.item("101350", "wooden dowel 8 x 30 mm", "dowel", [8, 30])
    builder.item("118331", "cam bolt 34 mm", "cam_bolt", [6, 34])
    builder.item("116952", "cam lock 15 mm", "cam_lock", [15, 12])
    builder.item("101351", "nail 1.5 x 20 mm", "nail", [1.5, 20])
    builder.item("115753", "shelf pin 5 mm", "shelf_pin", [5, 16])
    builder.item("100214", "screw 3.5 x 25 mm", "screw", [3.5, 25])
    builder.item("109041", "bolt M6 x 70 mm", "bolt", [6, 70])


def halden_bookcase() -> AssemblyPlan:
    """An 80 x 106 cm bookcase: two sides, top, bottom, a fixed shelf, a plinth, a nailed back and
    two shelves on pins. Built on its side, nailed face down, stood up."""
    b = FurnitureBuilder("halden-bookcase", "Halden bookcase", "bookcase", color=WHITE)
    hardware(b)
    w, h, d, t, plinth = 800.0, 1060.0, 280.0, 18.0, 60.0
    inner = w - 2 * t
    b.part("A1", "A", "side panel", "side", [t, h, d], [-(w / 2 - t / 2), h / 2, 0])
    b.part("A2", "A", "side panel", "side", [t, h, d], [w / 2 - t / 2, h / 2, 0])
    b.part("B", "B", "top panel", "top", [inner, t, d], [0, h - t / 2, 0])
    b.part("C", "C", "bottom panel", "bottom", [inner, t, d], [0, plinth + t / 2, 0])
    middle = (plinth + t + h - t) / 2
    b.part("D", "D", "fixed shelf", "shelf", [inner, t, d - 20], [0, middle, -10])
    b.part("E", "E", "plinth", "plinth", [inner, plinth, t], [0, plinth / 2, d / 2 - t / 2 - 20])
    b.part("F", "F", "back panel", "back", [w - 4, h - plinth - 4, 3], [0, plinth + (h - plinth) / 2, -d / 2 - 1.5],
           color=BACK_BOARD, material="hardboard")  # fmt: skip
    b.part("G1", "G", "shelf", "shelf", [inner, t, d - 20], [0, (plinth + t + middle) / 2, -10])
    b.part("G2", "G", "shelf", "shelf", [inner, t, d - 20], [0, (middle + h - t) / 2, -10])

    b.step(
        "Top and bottom onto the left side",
        "Lay side panel A flat, inside face up. Push dowels and cam bolts into it, set top B and bottom C "
        "on them, and turn the cam locks clockwise until tight.",
        ["A1", "B", "C"],
        [b.join("B", "A1", "cam_lock"), b.join("C", "A1", "cam_lock")],
        orientation="on_side",
        actions=[
            Action(kind="lay_down", target="A1", text="Lay side panel A flat, inside face up"),
            Action(kind="insert", target="A1", text="Push 4 dowels and 4 cam bolts into side A"),
            Action(kind="tighten", target="assembly", text="Turn the cam locks clockwise, a half turn"),
        ],
        tools=["screwdriver"],
    )
    b.step(
        "Fixed shelf and plinth",
        "Fit fixed shelf D at mid-height and plinth E at the front, under bottom C.",
        ["D", "E"],
        [b.join("D", "A1", "cam_lock"), b.join("E", "A1", "dowel")],
        orientation="on_side",
        actions=[Action(kind="align", target="E", text="Plinth E goes at the front, its face flush with the sides")],
        tools=["screwdriver"],
    )
    b.step(
        "Close with the right side",
        "Lower side panel A onto the dowels and cam bolts, then tighten all eight cam locks.",
        ["A2"],
        [
            b.join("B", "A2", "cam_lock"),
            b.join("C", "A2", "cam_lock"),
            b.join("D", "A2", "cam_lock"),
            b.join("E", "A2", "dowel"),
        ],  # fmt: skip
        orientation="on_side",
        actions=[Action(kind="tighten", target="assembly", text="Tighten all cam locks clockwise")],
        tools=["screwdriver"],
    )
    b.step(
        "Face down, nail on the back",
        "Turn the bookcase face down. Check it is square, lay back panel F on it and nail it to the sides, "
        "the top and the bottom.",
        ["F"],
        [b.join("F", "A1", "nail"), b.join("F", "A2", "nail"), b.join("F", "B", "nail"), b.join("F", "C", "nail")],
        orientation="on_front",
        actions=[
            Action(kind="flip", target="assembly", axis="x", degrees=90, text="Turn the bookcase face down"),
            Action(kind="align", target="F", text="Measure both diagonals: equal means square"),
        ],
        tools=["hammer"],
    )
    b.step(
        "Stand it up, add the shelves",
        "Stand the bookcase upright. Push shelf pins into the holes at the heights you want and rest shelves G "
        "on them.",
        ["G1", "G2"],
        [b.join("G1", "A1", "pin"), b.join("G1", "A2", "pin"), b.join("G2", "A1", "pin"), b.join("G2", "A2", "pin")],
        orientation="upright",
        actions=[Action(kind="stand_up", target="assembly", axis="x", degrees=-90, text="Stand the bookcase upright")],
    )
    b.notes.append("Two people make steps 4 and 5 easier.")
    return b.build()


def varde_nightstand() -> AssemblyPlan:
    """A 45 x 56 cm nightstand with a drawer: carcass built on its back, drawer box built
    separately, then slid in."""
    b = FurnitureBuilder("varde-nightstand", "Varde nightstand", "nightstand", color=OAK, material="oak veneer")
    hardware(b)
    w, h, d, t = 450.0, 560.0, 400.0, 16.0
    inner = w - 2 * t
    b.part("A1", "A", "side panel", "side", [t, h - t, d], [-(w / 2 - t / 2), (h - t) / 2, 0])
    b.part("A2", "A", "side panel", "side", [t, h - t, d], [w / 2 - t / 2, (h - t) / 2, 0])
    b.part("B", "B", "top", "top", [w, t, d], [0, h - t / 2, 0])
    b.part("C", "C", "bottom", "bottom", [inner, t, d], [0, 60 + t / 2, 0])
    b.part("D", "D", "back panel", "back", [w - 4, 480, 3], [0, 302, -d / 2 - 1.5], color=BACK_BOARD,
           material="hardboard")  # fmt: skip
    # The drawer box runs from the carcass front edge back 360 mm; the front overlays the carcass.
    side_x = inner / 2 - 6
    box_back = d / 2 - 360
    b.part("F1", "F", "drawer side", "drawer_side", [12, 140, 360], [-side_x, 450, d / 2 - 180], color=WHITE)
    b.part("F2", "F", "drawer side", "drawer_side", [12, 140, 360], [side_x, 450, d / 2 - 180], color=WHITE)
    b.part("G", "G", "drawer back", "drawer_back", [inner - 24, 140, 12], [0, 450, box_back + 6], color=WHITE)
    b.part("H", "H", "drawer bottom", "drawer_bottom", [inner - 24, 3, 348], [0, 384.5, box_back + 12 + 174],
           color=BACK_BOARD, material="hardboard")  # fmt: skip
    b.part("E", "E", "drawer front", "drawer_front", [w - 4, 160, t], [0, 450, d / 2 + t / 2])

    b.step(
        "Bottom between the sides",
        "Lay the parts on a soft surface. Fit bottom C between side panels A with dowels and cam locks.",
        ["A1", "A2", "C"],
        [b.join("C", "A1", "cam_lock"), b.join("C", "A2", "cam_lock")],
        orientation="on_back",
        tools=["screwdriver"],
    )
    b.step(
        "Fit the top",
        "Set top B on the sides' upper edges and lock it.",
        ["B"],
        [b.join("B", "A1", "cam_lock"), b.join("B", "A2", "cam_lock")],
        orientation="on_back",
        actions=[Action(kind="tighten", target="assembly", text="Tighten the four cam locks")],
        tools=["screwdriver"],
    )
    b.step(
        "Face down, nail on the back",
        "Turn the carcass face down and nail back panel D to the sides and the bottom.",
        ["D"],
        [b.join("D", "A1", "nail"), b.join("D", "A2", "nail"), b.join("D", "C", "nail")],
        orientation="on_front",
        actions=[Action(kind="flip", target="assembly", axis="x", degrees=180, text="Turn the carcass face down")],
        tools=["hammer"],
    )
    b.step(
        "Build the drawer box",
        "Screw drawer sides F to drawer back G.",
        ["F1", "F2", "G"],
        [b.join("F1", "G", "screw"), b.join("F2", "G", "screw")],
        orientation="on_front",
        tools=["screwdriver"],
    )
    b.step(
        "Drawer bottom and front",
        "Slide bottom H into the grooves of the drawer sides, then screw front E to the sides.",
        ["H", "E"],
        [b.join("H", "F1", "slot"), b.join("H", "F2", "slot"), b.join("E", "F1", "screw"), b.join("E", "F2", "screw")],
        orientation="on_front",
        actions=[Action(kind="slide", target="H", text="Slide bottom H in from the front")],
        tools=["screwdriver"],
    )
    b.step(
        "Stand it up, slide the drawer in",
        "Stand the nightstand upright and slide the drawer onto its runners until it closes flush.",
        [],
        [b.join("F1", "A1", "slot"), b.join("F2", "A2", "slot")],
        orientation="upright",
        actions=[
            Action(kind="stand_up", target="assembly", axis="x", degrees=90, text="Stand the nightstand upright"),
            Action(kind="slide", target="E", text="Slide the drawer in"),
        ],
    )
    return b.build()


def lund_side_table() -> AssemblyPlan:
    """A 50 x 50 cm side table: four legs, four rails bolted between them, a top screwed on."""
    b = FurnitureBuilder("lund-side-table", "Lund side table", "table", color=PINE, material="solid pine")
    hardware(b)
    top, height, leg, rail_h, rail_t = 500.0, 450.0, 40.0, 80.0, 18.0
    corner = top / 2 - 30 - leg / 2  # leg centers, 30 mm in from the edges
    span = 2 * corner - leg  # rail length between legs
    rail_y = height - 20 - rail_h / 2
    legs = {"B1": (-corner, corner), "B2": (corner, corner), "B3": (corner, -corner), "B4": (-corner, -corner)}
    for name, (x, z) in legs.items():
        b.part(name, "B", "leg", "leg", [leg, height - 20, leg], [x, (height - 20) / 2, z])
    offset = corner + leg / 2 - rail_t / 2 - 6
    b.part("C1", "C", "rail", "rail", [span, rail_h, rail_t], [0, rail_y, offset])
    b.part("C2", "C", "rail", "rail", [rail_t, rail_h, span], [offset, rail_y, 0])
    b.part("C3", "C", "rail", "rail", [span, rail_h, rail_t], [0, rail_y, -offset])
    b.part("C4", "C", "rail", "rail", [rail_t, rail_h, span], [-offset, rail_y, 0])
    b.part("A", "A", "table top", "top", [top, 20, top], [0, height - 10, 0])

    b.step(
        "Front frame",
        "Bolt rail C between two legs B. Hand-tight for now.",
        ["B1", "B2", "C1"],
        [b.join("C1", "B1", "bolt"), b.join("C1", "B2", "bolt")],
        tools=["allen key"],
    )
    b.step(
        "Back frame",
        "Build the second frame the same way.",
        ["B3", "B4", "C3"],
        [b.join("C3", "B3", "bolt"), b.join("C3", "B4", "bolt")],
        tools=["allen key"],
    )
    b.step(
        "Join the frames",
        "Connect the two frames with the side rails C, then tighten all eight bolts.",
        ["C2", "C4"],
        [
            b.join("C2", "B2", "bolt"),
            b.join("C2", "B3", "bolt"),
            b.join("C4", "B4", "bolt"),
            b.join("C4", "B1", "bolt"),
        ],  # fmt: skip
        actions=[Action(kind="tighten", target="assembly", text="Tighten all bolts")],
        tools=["allen key"],
    )
    b.step(
        "Screw on the top",
        "Lay top A face down, set the frame on it upside down and centered, and screw through the rails. "
        "Turn the table over.",
        ["A"],
        [
            b.join("A", "C1", "screw"),
            b.join("A", "C2", "screw"),
            b.join("A", "C3", "screw"),
            b.join("A", "C4", "screw"),
        ],  # fmt: skip
        orientation="upside_down",
        actions=[
            Action(
                kind="flip", target="assembly", axis="z", degrees=180, text="Turn the frame upside down onto the top"
            ),
        ],
        tools=["screwdriver"],
    )
    return b.build()


def ostra_desk() -> AssemblyPlan:
    """A 120 x 60 cm desk: sides and a modesty panel locked to the top, built upside down, with a
    footrest shelf."""
    b = FurnitureBuilder("ostra-desk", "Ostra desk", "desk", color=BLACK, material="black-brown foil")
    hardware(b)
    w, d, h, top_t, side_t = 1200.0, 600.0, 750.0, 25.0, 25.0
    side_x = w / 2 - 50 - side_t / 2
    inner = 2 * (side_x - side_t / 2)
    b.part("B", "B", "desktop", "top", [w, top_t, d], [0, h - top_t / 2, 0], color=OAK, material="oak veneer")
    b.part("A1", "A", "side panel", "side", [side_t, h - top_t, d - 20], [-side_x, (h - top_t) / 2, 0])
    b.part("A2", "A", "side panel", "side", [side_t, h - top_t, d - 20], [side_x, (h - top_t) / 2, 0])
    b.part("C", "C", "modesty panel", "panel", [inner, 345, 18], [0, h - top_t - 345 / 2, -220])
    b.part("E", "E", "footrest shelf", "shelf", [inner, 18, 280], [0, 109, 40])

    b.step(
        "Sides to the desktop",
        "Lay desktop B face down on the box. Fit side panels A to it with dowels and cam locks.",
        ["B", "A1", "A2"],
        [b.join("A1", "B", "cam_lock"), b.join("A2", "B", "cam_lock")],
        orientation="upside_down",
        actions=[Action(kind="lay_down", target="B", text="Lay the desktop face down on the cardboard")],
        tools=["screwdriver"],
    )
    b.step(
        "Modesty panel",
        "Fit modesty panel C between the sides and screw it to the desktop.",
        ["C"],
        [b.join("C", "A1", "cam_lock"), b.join("C", "A2", "cam_lock"), b.join("C", "B", "screw", fasteners=3)],
        orientation="upside_down",
        tools=["screwdriver"],
    )
    b.step(
        "Stand it up, fit the footrest",
        "With a helper, turn the desk upright. Fit footrest shelf E between the sides.",
        ["E"],
        [b.join("E", "A1", "dowel"), b.join("E", "A2", "dowel")],
        orientation="upright",
        actions=[Action(kind="stand_up", target="assembly", axis="z", degrees=180, text="Turn the desk upright")],
    )
    return b.build()


CATALOG: dict[str, Callable[[], AssemblyPlan]] = {
    "halden-bookcase": halden_bookcase,
    "varde-nightstand": varde_nightstand,
    "lund-side-table": lund_side_table,
    "ostra-desk": ostra_desk,
}


def build(plan_id: str) -> AssemblyPlan:
    return CATALOG[plan_id]()
