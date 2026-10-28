import json
from pathlib import Path

import numpy as np
import pytest

from darius import cli
from darius.tools import gallery
from darius.tools.validate import validate

REPO = Path(__file__).resolve().parents[2]
SANDBOX = REPO / "scenes" / "sandbox"
SHIPPED = sorted(p.parent for p in (REPO / "scenes").glob("*/scene.json"))


@pytest.mark.parametrize("scene_dir", SHIPPED, ids=lambda p: p.name)
def test_shipped_scenes_validate(scene_dir):
    assert validate(scene_dir) == []


def test_info_lists_every_object(capsys):
    assert cli.main(["info", str(SANDBOX)]) == 0
    out = capsys.readouterr().out
    assert "rubber ball" in out and "Physics sandbox" in out


def test_schema_command_prints_the_json_schema(capsys):
    assert cli.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "Scene" and "inertial" in json.dumps(schema)


def test_gallery_comparison_tolerates_last_digit_noise():
    assert gallery.same_text('pos="0.1 0.200001"', 'pos="0.1 0.2"')
    assert not gallery.same_text('pos="0.1 0.21"', 'pos="0.1 0.2"')
    assert not gallery.same_text('name="a"', 'name="b"')


def test_push_env_pushes_the_target_toward_the_goal():
    pytest.importorskip("mujoco")
    from darius.robotics.sim import PushEnv

    env = PushEnv(SANDBOX, target="obj_04", max_steps=80)  # the rubber ball
    obs, info = env.reset(seed=0)
    assert obs["pusher"].shape == (2,) and obs["objects"].shape == (5, 7) and obs["target"].shape == (7,)
    start = np.linalg.norm(obs["target"][:2] - obs["goal"])
    for _ in range(80):
        # Stay just behind the ball, on the line to the goal, and lean toward the goal.
        target, goal, pusher = obs["target"][:2], obs["goal"], obs["pusher"]
        heading = (goal - target) / max(np.linalg.norm(goal - target), 1e-6)
        velocity = 0.15 * heading + 2.0 * (target - 0.045 * heading - pusher)
        obs, reward, terminated, truncated, info = env.step(velocity)
        if terminated or truncated:
            break
    assert -reward < start  # the ball got closer to the goal
    assert info["fell"] == []


def test_settle_reports_a_resting_scene():
    pytest.importorskip("mujoco")
    from darius.robotics.sim import PushEnv

    report = PushEnv(SANDBOX).settle(4.0)  # the rubber ball is dropped in and bounces for about 2.6 s
    assert set(report) == {f"obj_0{k}" for k in range(1, 6)}
    assert all(v["motion_m"] < 0.003 and v["drift_m"] < 0.03 for v in report.values())
