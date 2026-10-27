"""Load every exported scene in MuJoCo, simulate 4 s, and check it comes to rest.

uv run --with mujoco python scripts/check_mujoco.py
"""

import glob
import sys

import mujoco
import numpy as np

failures = 0
for path in sorted(glob.glob("../scenes/*/scene.xml")):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.xpos.copy()
    for _ in range(int(4 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    drift = np.linalg.norm((data.xpos - start)[1:, :2], axis=1).max()
    speed = np.abs(data.qvel).max()
    tilt = np.degrees(np.arccos(np.clip(data.xmat[1:].reshape(-1, 3, 3)[:, 2, 2], -1, 1))).max()
    ok = drift < 0.03 and speed < 0.02
    failures += not ok
    print(
        f"{'ok  ' if ok else 'FAIL'} {path}: drift {drift * 100:.1f} cm, max speed {speed:.4f}, max tilt {tilt:.1f} deg"
    )
sys.exit(1 if failures else 0)
