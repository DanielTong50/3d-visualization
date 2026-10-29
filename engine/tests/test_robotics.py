"""The real training-data boundary: reproducibility, units, provenance and episode separation."""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from darius.lego import compile_assembly, get_assembly
from darius.robotics.dataset import export_dataset, robot_position, training_bundle
from darius.robotics.mujoco import export_scene, scene_xml
from darius.robotics.train import predict_delta, train_baseline


@pytest.fixture
def plan():
    return compile_assembly(get_assembly("signal-tower"))


def test_dataset_is_seeded_and_split_by_episode(plan):
    first = training_bundle(plan, episodes=3, seed=17, fps=5)
    assert first == training_bundle(plan, episodes=3, seed=17, fps=5)
    assert first != training_bundle(plan, episodes=3, seed=18, fps=5)
    manifest = first["manifest"]
    assert manifest["source"] == "synthetic_kinematic_reference"
    assert manifest["execution"] == "simulation_only"
    assert set(manifest["split"]["train"]).isdisjoint(manifest["split"]["validation"])
    assert manifest["frames"] == sum(len(e["records"]) for e in first["episodes"])
    for episode in first["episodes"]:
        rows = episode["records"]
        assert rows[-1]["done"]
        assert not any(row["done"] for row in rows[:-1])
        assert all(row["measured_force_n"] is None and row["physical_success"] is None for row in rows)
        assert len(rows[0]["observation.state"]) == len(rows[0]["action"]) == 5
        for previous, following in zip(rows, rows[1:], strict=False):
            predicted = np.add(previous["observation.state"], previous["action"])
            np.testing.assert_allclose(predicted, following["observation.state"], atol=2e-9)
            assert following["timestamp"] - previous["timestamp"] == pytest.approx(1 / 5)


def test_reference_frame_is_right_handed_and_si():
    assert robot_position([8, 3.2, 16]) == [0.008, -0.016, 0.0032]
    basis = np.array([robot_position([1000, 0, 0]), robot_position([0, 1000, 0]), robot_position([0, 0, 1000])])
    assert np.linalg.det(basis) == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [{"episodes": 1}, {"episodes": 65}, {"episodes": True}, {"fps": 0}, {"fps": 31}, {"seed": -1}, {"seed": 2**32}],
)
def test_dataset_rejects_invalid_generation_parameters(plan, kwargs):
    with pytest.raises(ValueError):
        training_bundle(plan, **kwargs)


def test_invalid_plan_cannot_become_training_demonstrations(plan):
    invalid = copy.deepcopy(plan.model_dump(mode="json"))
    invalid["validation"]["valid"] = False
    with pytest.raises(ValueError, match="valid compiled"):
        training_bundle(invalid)
    with pytest.raises(ValueError, match="valid compiled"):
        scene_xml(invalid)


def test_manual_draft_and_missing_rights_cannot_enter_training_corpus(plan):
    candidate = plan.model_dump(mode="json")
    candidate["assembly"]["source"].update(
        kind="reviewed-import", review_status="pending", training_rights="permission-required"
    )
    with pytest.raises(ValueError, match="explicit geometric review"):
        training_bundle(candidate)
    candidate["assembly"]["source"].update(review_status="reviewed", training_rights="permission-required")
    with pytest.raises(ValueError, match="documented permission"):
        training_bundle(candidate)
    candidate["assembly"]["source"].update(training_rights="permission-documented", permission_reference="")
    with pytest.raises(ValueError, match="nonempty permission_reference"):
        training_bundle(candidate)


def test_mujoco_loads_and_steps_all_original_scenes_when_installed():
    mujoco = pytest.importorskip("mujoco", reason="optional simulation extra is not installed")
    for build_id in ("signal-tower", "bridge", "stairway"):
        compiled = compile_assembly(get_assembly(build_id))
        model = mujoco.MjModel.from_xml_string(scene_xml(compiled))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data, nstep=100)
        assert np.isfinite(data.qpos).all()
        assert data.time == pytest.approx(0.2)


def test_export_checksums_and_training_run_end_to_end(plan, tmp_path):
    dataset = tmp_path / "dataset"
    manifest = export_dataset(plan, dataset, episodes=4, seed=29, fps=5)
    checksums = json.loads((dataset / "checksums.json").read_text())
    for relative, digest in checksums.items():
        assert hashlib.sha256((dataset / relative).read_bytes()).hexdigest() == digest
    assert len(manifest["episode_files"]) == 4
    output = tmp_path / "baseline"
    report = train_baseline(dataset, output)
    assert report["normalization_fit"] == "training episodes only"
    assert report["split"] == manifest["split"]
    assert np.isfinite(report["evaluation"]["validation"]["ridge"]["translation_delta_rmse_mm"])
    assert report["evaluation"]["validation"]["ridge"]["samples"] > 0
    row = json.loads((dataset / manifest["episode_files"][-1]["path"]).read_text().splitlines()[0])
    assert len(predict_delta(output / "model.npz", row)) == 5
    with np.load(output / "model.npz", allow_pickle=False) as model:
        assert model["weights"].shape == (17, 5)
    with pytest.raises(ValueError, match="not empty"):
        export_dataset(plan, dataset)


def test_dataset_tampering_fails_before_training(plan, tmp_path):
    dataset = tmp_path / "dataset"
    export_dataset(plan, dataset, episodes=2, fps=2)
    path = dataset / "episodes/0000.jsonl"
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        train_baseline(dataset, tmp_path / "model")
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("duplicate_field", ["episode_index", "path"])
def test_repacked_manifest_cannot_alias_train_and_validation_episodes(plan, tmp_path, duplicate_field):
    dataset = tmp_path / "dataset"
    manifest = export_dataset(plan, dataset, episodes=2, fps=2)
    manifest["episode_files"][1][duplicate_field] = manifest["episode_files"][0][duplicate_field]
    path = dataset / "manifest.json"
    path.write_text(json.dumps(manifest))
    checksums = json.loads((dataset / "checksums.json").read_text())
    checksums["manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dataset / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="unique index and file path"):
        train_baseline(dataset, tmp_path / "model")


def test_dataset_rejects_checksum_path_escape(plan, tmp_path):
    dataset = tmp_path / "dataset"
    export_dataset(plan, dataset, episodes=2, fps=2)
    checksums = json.loads((dataset / "checksums.json").read_text())
    checksums["../../outside"] = "0" * 64
    (dataset / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="leaves its directory"):
        train_baseline(dataset, tmp_path / "model")


def test_mjcf_uses_si_dynamic_loose_bricks_and_reference_targets(plan, tmp_path):
    xml = scene_xml(plan)
    root = ET.fromstring(xml)
    assert root.tag == "mujoco"
    assert root.find("option").attrib["gravity"] == "0 0 -9.81"
    assert len(root.findall(".//freejoint")) == len(plan.assembly.bricks)
    assert len(root.findall(".//site")) == len(plan.assembly.bricks)
    assert root.find(".//body[@name='reference_tcp']").attrib["mocap"] == "true"
    assert root.find("actuator") is None  # No pretend robot controller.
    path = export_scene(plan, tmp_path / "scene.xml")
    assert path.read_text() == xml


def test_rotated_bridge_source_and_target_orientations_are_explicit():
    plan = compile_assembly(get_assembly("bridge"))
    root = ET.fromstring(scene_xml(plan))
    source_quat = [float(v) for v in root.find(".//body[@name='b05']").attrib["quat"].split()]
    target_quat = [float(v) for v in root.find(".//site[@name='goal_b05']").attrib["quat"].split()]
    assert source_quat == [1, 0, 0, 0]
    np.testing.assert_allclose(target_quat, [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)], atol=1e-8)
