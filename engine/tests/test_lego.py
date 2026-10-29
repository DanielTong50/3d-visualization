"""LEGO geometry/plan invariants and official-source retrieval, with no network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from darius.cli import main
from darius.lego import compile_assembly, get_assembly
from darius.lego.catalog import CATALOG, export_library
from darius.lego.instructions import InstructionLookupError, lookup_instructions, official_url, parse_official_page
from darius.lego.schema import LegoAssembly
from darius.service.api import create_app


@pytest.mark.parametrize("build_id", list(CATALOG))
def test_original_curriculum_is_valid_and_contacts_are_exact(build_id: str) -> None:
    plan = compile_assembly(get_assembly(build_id))
    assert plan.validation.valid, plan.validation.errors
    assert plan.execution.status == "simulation_only" and len(plan.execution.blockers) >= 4
    assert len(plan.steps) == len(plan.assembly.bricks) == 6
    assert plan.metrics.primitives == 60
    seen = set()
    bricks = {b.id: b for b in plan.assembly.bricks}
    for step in plan.steps:
        assert set(step.dependencies) <= seen
        assert step.primitives[-1].kind == "verify"
        assert step.source_center_mm[1] == bricks[step.brick_id].height_plates * 1.6
        seen.add(step.brick_id)
    for contact in plan.connections:
        upper = bricks[contact.upper]
        expected = upper.cells if contact.lower == "baseplate" else upper.cells & bricks[contact.lower].cells
        assert {tuple(v) for v in contact.studs} == expected
        if contact.lower != "baseplate":
            lower = bricks[contact.lower]
            assert lower.position[1] + lower.height_plates == upper.position[1]


def test_cross_bond_and_bridge_have_multiple_support_dependencies() -> None:
    tower = compile_assembly(get_assembly("signal-tower"))
    assert tower.steps[2].dependencies == ["b01", "b02"]
    bridge = compile_assembly(get_assembly("bridge"))
    assert bridge.steps[4].dependencies == ["b03", "b04"]
    assert bridge.steps[4].support_ratio == 0.5
    assert any("50%" in w for w in bridge.validation.warnings)


def edited(**changes) -> LegoAssembly:
    data = get_assembly("signal-tower").model_dump()
    for key, value in changes.items():
        data[key] = value
    return LegoAssembly.model_validate(data)


def test_collision_floating_fixture_bounds_and_unsupported_parts_are_refused() -> None:
    data = get_assembly("signal-tower").model_dump()
    data["bricks"][1]["position"] = data["bricks"][0]["position"]
    plan = compile_assembly(LegoAssembly.model_validate(data))
    assert not plan.validation.valid and any("intersecting" in e for e in plan.validation.errors)
    data = get_assembly("signal-tower").model_dump()
    data["bricks"][0]["position"][1] = 20
    data["bricks"][0]["position"][0] = 16
    data["bricks"][0]["part_id"] = "9999"
    errors = compile_assembly(LegoAssembly.model_validate(data)).validation.errors
    assert any("no stud support" in e for e in errors)
    assert any("outside" in e for e in errors)
    assert any("unsupported" in e for e in errors)


def test_wrong_schedule_and_inventory_dimensions_are_caught() -> None:
    assembly = get_assembly("signal-tower")
    data = assembly.model_dump()
    data["steps"][0]["brick_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="every brick"):
        LegoAssembly.model_validate(data)
    data = assembly.model_dump()
    data["steps"][0]["brick_ids"], data["steps"][5]["brick_ids"] = (
        data["steps"][5]["brick_ids"],
        data["steps"][0]["brick_ids"],
    )
    assert not compile_assembly(LegoAssembly.model_validate(data)).validation.valid
    data = assembly.model_dump()
    data["bricks"][0]["studs"] = [2, 2]
    assert any("dimensions" in e for e in compile_assembly(LegoAssembly.model_validate(data)).validation.errors)


def test_export_is_deterministic_and_checked_in_fixtures_match(tmp_path: Path) -> None:
    export_library(tmp_path)
    fixture = Path(__file__).resolve().parents[2] / "assemblies" / "lego"
    for build_id in CATALOG:
        assert json.loads((tmp_path / build_id / "plan.json").read_text()) == json.loads(
            (fixture / build_id / "plan.json").read_text()
        )


def page_for(data: dict) -> str:
    return '<script id="__NEXT_DATA__" type="application/json">' + json.dumps({"props": {"data": data}}) + "</script>"


def test_official_metadata_matches_set_and_filters_nonofficial_assets() -> None:
    page = page_for(
        {
            "setNumber": "12345",
            "name": "Test",
            "setPieceCount": "20",
            "buildingInstructions": [
                {
                    "pdf": {"pdfUrl": "https://www.lego.com/cdn/test.pdf", "fileSize": "1024"},
                    "sequence": {"element": "1", "total": "2"},
                },
                {"pdf": {"pdfUrl": "https://attacker.example/private.pdf"}},
                {"pdf": {"pdfUrl": "https://www.lego.com.attacker.example/private.pdf"}},
            ],
        }
    )
    result = parse_official_page("12345", page)
    assert len(result.booklets) == 1
    assert result.booklets[0].size_bytes == 1024
    assert result.semantic_status == "not_parsed" and result.training_rights == "permission-required"
    with pytest.raises(InstructionLookupError):
        parse_official_page("99999", page)
    with pytest.raises(InstructionLookupError):
        parse_official_page("12345", "page changed")


@pytest.mark.parametrize("set_number", ["../etc", "https://evil", "123", "12345678", "12345?x", "１２３４５"])
def test_instruction_lookup_cannot_fetch_arbitrary_urls(set_number: str) -> None:
    with pytest.raises(ValueError):
        official_url(set_number)


def test_cached_official_metadata_is_explicit_and_does_not_need_network(monkeypatch) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("cache must not fetch")

    monkeypatch.setattr("urllib.request.build_opener", no_network)
    result = lookup_instructions("10698")
    assert result.retrieval == "cached" and len(result.booklets) == 9
    assert result.set_number == "10698" and result.training_rights == "permission-required"


def test_api_compiles_curriculum_and_bounds_training_input() -> None:
    root = Path(__file__).resolve().parents[2]
    client = TestClient(create_app(root / "assemblies"))
    assert len(client.get("/lego/library").json()) == 3
    plan = client.get("/lego/builds/bridge").json()
    assert plan["validation"]["valid"] and plan["metrics"]["stud_connections"] == 28
    assert client.get("/lego/builds/missing").status_code == 404
    response = client.post("/lego/compile", json=get_assembly("stairway").model_dump())
    assert response.status_code == 200 and response.json()["validation"]["valid"]
    assert client.post("/lego/compile", json={}).status_code == 422
    assert client.get("/lego/instructions/123").status_code == 422
    assert client.get("/lego/instructions/10698").json()["retrieval"] == "cached"
    for query in ("episodes=1", "episodes=65", "seed=-1", "seed=4294967296", "fps=0", "fps=31"):
        assert client.get(f"/lego/builds/bridge/training?{query}").status_code == 422
    result = client.get("/lego/builds/bridge/training?episodes=2&seed=7&fps=1")
    assert result.status_code == 200
    assert len(result.json()["episodes"]) == 2


def test_cli_compile_export_and_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    assert main(["lego", "compile", "bridge", "--out", str(path)]) == 0
    assert json.loads(path.read_text())["validation"]["valid"]
    assert main(["lego", "compile", "missing"]) == 1
    assert main(["lego", "instructions", "123"]) == 1


@pytest.mark.parametrize("build_id", list(CATALOG))
def test_indexed_tray_slots_have_clearance_and_do_not_overlap_fixture(build_id: str) -> None:
    plan = compile_assembly(get_assembly(build_id))
    boxes = []
    for step in plan.steps:
        brick = next(b for b in plan.assembly.bricks if b.id == step.brick_id)
        width, depth = (n * 8 for n in brick.studs)
        x, _, z = step.source_center_mm
        box = (x - width / 2, x + width / 2, z - depth / 2, z + depth / 2)
        assert box[0] >= plan.assembly.baseplate.studs[0] * 8 + 24
        for other in boxes:
            assert (
                box[1] + 16 <= other[0] or other[1] + 16 <= box[0] or box[3] + 16 <= other[2] or other[3] + 16 <= box[2]
            )
        boxes.append(box)


def test_grid_does_not_coerce_boolean_or_float_coordinates() -> None:
    for invalid in (True, 1.0, "1"):
        data = get_assembly("bridge").model_dump()
        data["bricks"][0]["position"][0] = invalid
        with pytest.raises(ValidationError):
            LegoAssembly.model_validate(data)


def test_import_permissions_cannot_silently_become_project_original() -> None:
    from darius.lego.schema import Source

    with pytest.raises(ValidationError, match="permission review"):
        Source(kind="reviewed-import")
    with pytest.raises(ValidationError, match="permission_reference"):
        Source(kind="reviewed-import", training_rights="permission-documented", review_status="reviewed")
    source = Source(
        kind="reviewed-import",
        training_rights="permission-documented",
        review_status="reviewed",
        permission_reference="A documented source permission reference",
    )
    assert source.review_status == "reviewed"


def test_malformed_official_page_fields_cannot_escape_asset_filter() -> None:
    data = {
        "setNumber": "12345",
        "name": "Test",
        "theme": "bad",
        "setImage": [],
        "buildingInstructions": [
            None,
            {"pdf": []},
            {"pdf": {"pdfUrl": "https://www.lego.com:bad/test.pdf"}},
            {"sequence": [], "pdf": {"pdfUrl": "https://www.lego.com/cdn/test.pdf", "coverImage": "bad"}},
        ],
    }
    result = parse_official_page("12345", page_for(data))
    assert len(result.booklets) == 1 and result.theme is None and result.image_url is None


def test_rotated_brick_keeps_source_orientation_until_clearance_then_rotates_to_target() -> None:
    plan = compile_assembly(get_assembly("bridge"))
    step = plan.steps[4]
    assert step.source_yaw_deg == 0
    assert step.target_yaw_deg == 90
    assert [p.yaw_deg for p in step.primitives[:4]] == [0, 0, 0, 0]
    assert [p.yaw_deg for p in step.primitives[4:]] == [90] * 6
    assert step.primitives[3].position_mm[1] == step.primitives[4].position_mm[1]
