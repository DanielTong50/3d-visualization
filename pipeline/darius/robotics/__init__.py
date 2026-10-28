"""The robotics layer: everything a robot learning or evaluation stack needs from a scene.

    mass        mass, center of mass and inertia tensor of each object (shell model for hollow ones)
    grasps      antipodal parallel-jaw grasp candidates on each object's collision geometry
    export      MuJoCo MJCF, URDF and OpenUSD (UsdPhysics) writers, plus the bundle that holds them
    sim         a MuJoCo-backed environment with a Gymnasium-style reset/step API (needs `mujoco`)

Everything here works in the robotics world frame (+z up, meters; see
`darius.core.frames`), and reads only scene packages, never pipeline internals.
"""
