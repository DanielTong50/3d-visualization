import json
from pathlib import Path

import numpy as np
import pytest

from scan2sim import geometry, meta_api, segment
from scan2sim.package import write_obj, write_ply
from scan2sim.schema import Scene

REPO = Path(__file__).resolve().parents[2]
SCENES = sorted((REPO / "scenes").glob("*/scene.json"))


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.parent.name)
def test_shipped_scenes_match_schema(path):
    scene = Scene.model_validate_json(path.read_text())
    for obj in scene.objects:
        if obj.collider.type == "convex_hull":
            assert (path.parent / obj.collider.file).is_file()
        if obj.visual:
            assert (path.parent / obj.visual.file).is_file()
        if scene.provenance is None:  # hand-authored: mass must follow from the stated inputs
            assert obj.mass_kg == pytest.approx(obj.volume_m3 * obj.density_kg_m3 * obj.fill_fraction, rel=0.05)


def test_ply_round_trip(tmp_path):
    points = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    colors = np.random.default_rng(1).integers(0, 255, (100, 3), dtype=np.uint8)
    write_ply(tmp_path / "x.ply", points, colors)
    raw = (tmp_path / "x.ply").read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert b"element vertex 100" in header
    data = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    assert np.array_equal(data["p"], points) and np.array_equal(data["c"], colors)


def test_obj_writer_is_one_indexed(tmp_path):
    hull = geometry.convex_hull(np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float))
    write_obj(tmp_path / "h.obj", hull, "cube")
    faces = [line for line in (tmp_path / "h.obj").read_text().splitlines() if line.startswith("f ")]
    indices = [int(i) for line in faces for i in line.split()[1:]]
    assert min(indices) == 1 and max(indices) == 8 and len(faces) == 12


def test_grid_boxes_are_clamped_and_ordered():
    assert segment.grid_to_pixels([100, 200, 300, 1001], (1000, 500)) == (100, 100, 300, 500)
    assert segment.grid_to_pixels([300, 200, 100, 400], (1000, 1000)) == (100, 200, 300, 400)
    assert segment.grid_to_pixels([10, 10, 11, 11], (1000, 1000)) is None


def test_label_maps_prefer_confident_masks():
    a = np.zeros((4, 4), bool)
    a[:, :3] = True
    b = np.zeros((4, 4), bool)
    b[:, 2:] = True
    masks = [meta_api.FrameMask(0, 1, a, 0.9), meta_api.FrameMask(0, 2, b, 0.4)]
    labels = segment.label_maps(masks, frames=1, frame_size=(4, 4), out_size=(4, 4))
    assert labels[0, 0, 2] == 1 and labels[0, 0, 3] == 2 and labels[0, 0, 0] == 1


def test_box_local_masks_are_pasted_at_their_box():
    class Mask:
        width, height = 2, 2

    class Bounds:
        left, top, right, bottom = 3, 1, 5, 3

    class Record:
        mask, bounds = Mask(), Bounds()

    import scan2sim.meta_api as api

    original = api.decode_mask_to_raster
    api.decode_mask_to_raster = lambda mask: bytes([1, 0, 1, 1])
    try:
        full = meta_api.raster_from_record(Record(), (6, 4))
    finally:
        api.decode_mask_to_raster = original
    assert full.sum() == 3 and full[1, 3] and not full[1, 4] and full[2, 4]


def test_schema_file_is_current():
    committed = json.loads((REPO / "docs" / "scene.schema.json").read_text())
    assert committed == Scene.model_json_schema()
