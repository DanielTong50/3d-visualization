"""A scanned scene as a MuJoCo environment, with a Gymnasium-style API.

    from darius.robotics.sim import PushEnv
    env = PushEnv("scenes/kitchen-counter", target="obj_03")
    obs, info = env.reset(seed=0)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(policy(obs))

The task is planar pushing, the classic real-to-sim benchmark: a capsule fingertip,
driven kinematically (a MuJoCo mocap body) at a fixed height above the table, must
push the target object to a goal on the table. Everything else is the scene as
scanned: every object is a free rigid body with its estimated mass, inertia,
friction and restitution, so pushing a heavy book and pushing a light cup differ
the way they do in reality (to the accuracy of the estimates).

The environment is a thin layer over the bundle's MJCF (`robotics/scene.xml`, rebuilt
from scene.json when the environment starts), which works in any other MuJoCo-based
stack too. It needs `mujoco` (`uv sync --extra sim`), but not
Gymnasium: `observation_space` and `action_space` are plain bounds; wrap the class
if a Gymnasium registry is wanted.

Observation (dict of float arrays, world frame):
    pusher    (2,)    fingertip x, y
    target    (7,)    target position and orientation quaternion [x, y, z, w]
    velocity  (6,)    target linear and angular velocity
    goal      (2,)    goal x, y
    objects   (N, 7)  every object's position and quaternion, in scene order

Action: (2,) fingertip velocity in the table plane, m/s, clipped to `max_speed`.
Reward: minus the target's distance to the goal, in meters. Terminated when it is
within `success_radius`; truncated after `max_steps`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..scene.io import ScenePackage, load_scene
from .export import bundle

PUSHER_RADIUS_M = 0.008
PUSHER_HALF_LENGTH_M = 0.03


class PushEnv:
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(
        self,
        scene: str | Path | ScenePackage,
        target: str | None = None,
        goal_distance_m: float = 0.12,
        control_hz: float = 20.0,
        max_speed: float = 0.25,
        max_steps: int = 200,
        success_radius: float = 0.02,
    ):
        import mujoco

        self._mujoco = mujoco
        self.package = scene if isinstance(scene, ScenePackage) else load_scene(scene)
        with tempfile.TemporaryDirectory() as tmp:  # always the current scene.json, never a stale bundle
            folder = bundle.write(self.package, Path(tmp) / bundle.FOLDER)
            spec = mujoco.MjSpec.from_file(str(folder / "scene.xml"))
            self._add_pusher(spec)
            self.model = spec.compile()
        self.pusher_height = PUSHER_RADIUS_M + 0.012
        self.data = mujoco.MjData(self.model)
        self.objects = [o.id for o in self.package.objects]
        if not self.objects:
            raise ValueError("the scene has no objects to push")
        self.target = target or self.objects[0]
        if self.target not in self.objects:
            raise ValueError(f"no object {self.target!r}; choose from {', '.join(self.objects)}")
        self._body = {name: self.model.body(name).id for name in self.objects}
        self._joint = {name: self.model.joint(name).id for name in self.objects}
        self._key = self.model.key("capture").id
        self.substeps = max(1, round(1 / (control_hz * self.model.opt.timestep)))
        self.dt = self.substeps * self.model.opt.timestep
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.goal_distance_m = goal_distance_m
        outline = np.array(self.package.table_outline(), float)
        self.table_xy = np.column_stack([outline[:, 0], -outline[:, 1]])  # scene [x, z] -> world [x, y]
        self.action_space = (-np.full(2, max_speed), np.full(2, max_speed))
        self.goal = np.zeros(2)
        self._steps = 0

    # ---------- Gymnasium-style API ----------

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Restore the captured scene, let it settle, pick a goal, and park the fingertip behind
        the target (on the side away from the goal)."""
        mujoco = self._mujoco
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        self._park_pusher(np.array([0.0, 0.0]), height=1.0)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.5 / self.model.opt.timestep)):  # settle the scan's tiny overlaps
            mujoco.mj_step(self.model, self.data)
        start = self.target_xy()
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], float)
        else:
            center = self.table_xy.mean(axis=0)
            toward = center - start
            base = np.arctan2(toward[1], toward[0]) if np.linalg.norm(toward) > 1e-3 else rng.uniform(-np.pi, np.pi)
            angle = base + rng.uniform(-0.6, 0.6)
            self.goal = start + self.goal_distance_m * np.array([np.cos(angle), np.sin(angle)])
        away = start - self.goal
        away /= max(np.linalg.norm(away), 1e-9)
        radius = 0.5 * float(np.hypot(*self.package.object(self.target).dimensions_m[::2]))
        self._park_pusher(start + away * (radius + PUSHER_RADIUS_M + 0.02))
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observation(), {"goal": self.goal.copy(), "target": self.target}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        mujoco = self._mujoco
        velocity = np.clip(np.asarray(action, float).reshape(2), -self.max_speed, self.max_speed)
        position = self.data.mocap_pos[0, :2].copy()
        for k in range(self.substeps):
            self._park_pusher(position + velocity * self.model.opt.timestep * (k + 1))
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        distance = float(np.linalg.norm(self.target_xy() - self.goal))
        terminated = distance < self.success_radius
        truncated = self._steps >= self.max_steps
        info = {"distance_m": distance, "fell": self.fallen_objects()}
        return self._observation(), -distance, terminated, truncated, info

    # ---------- Helpers ----------

    def target_xy(self) -> np.ndarray:
        return self.data.xpos[self._body[self.target], :2].copy()

    def object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World position and [x, y, z, w] quaternion of every object."""
        poses = {}
        for name, body in self._body.items():
            w, x, y, z = self.data.xquat[body]
            poses[name] = (self.data.xpos[body].copy(), np.array([x, y, z, w]))
        return poses

    def fallen_objects(self) -> list[str]:
        """Objects that left the table (more than 5 cm below its top)."""
        return [name for name, body in self._body.items() if self.data.xpos[body, 2] < -0.05]

    def settle(self, seconds: float = 4.0, window: float = 0.5) -> dict[str, dict[str, float]]:
        """Simulate from the captured state with the fingertip out of the way, and report how far
        each object drifted, and how much it still moved over the last `window` seconds.

        Rest is judged by motion, not by velocity: a resting convex body can chatter between
        contact points with a velocity that flips sign every step while going nowhere.
        """
        mujoco = self._mujoco
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        self._park_pusher(np.zeros(2), height=1.0)
        mujoco.mj_forward(self.model, self.data)
        start = {name: self.data.xpos[body].copy() for name, body in self._body.items()}
        steps = int(seconds / self.model.opt.timestep)
        tail = steps - int(window / self.model.opt.timestep)
        before: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k in range(steps):
            mujoco.mj_step(self.model, self.data)
            if k == tail:
                before = {n: (self.data.xpos[b].copy(), self.data.xquat[b].copy()) for n, b in self._body.items()}
        report = {}
        for name, body in self._body.items():
            position, quat = before.get(name, (self.data.xpos[body], self.data.xquat[body]))
            turn = 2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(quat, self.data.xquat[body]))), -1, 1)))
            report[name] = {
                "drift_m": float(np.linalg.norm(self.data.xpos[body, :2] - start[name][:2])),
                "motion_m": float(np.linalg.norm(self.data.xpos[body] - position)),
                "turn_deg": float(turn),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(self.data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))),
            }
        return report

    def _add_pusher(self, spec) -> None:
        pusher = spec.worldbody.add_body(name="pusher", mocap=True, pos=[0, 0, 1.0])
        pusher.add_geom(
            name="pusher",
            type=self._mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[PUSHER_RADIUS_M, PUSHER_HALF_LENGTH_M, 0],
            pos=[0, 0, PUSHER_HALF_LENGTH_M],  # the capsule's lower tip sits at the body origin
            rgba=[1.0, 0.31, 0.0, 1.0],
            friction=[0.5, 0.005, 0.0001],
        )

    def _park_pusher(self, xy: np.ndarray, height: float | None = None) -> None:
        self.data.mocap_pos[0] = [xy[0], xy[1], self.pusher_height if height is None else height]

    def _observation(self) -> dict[str, np.ndarray]:
        poses = self.object_poses()
        position, quat = poses[self.target]
        address = self.model.jnt_dofadr[self._joint[self.target]]
        return {
            "pusher": self.data.mocap_pos[0, :2].copy(),
            "target": np.concatenate([position, quat]),
            "velocity": self.data.qvel[address : address + 6].copy(),
            "goal": self.goal.copy(),
            "objects": np.stack([np.concatenate(poses[name]) for name in self.objects]),
        }
