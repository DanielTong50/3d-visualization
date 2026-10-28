# Frontend (web app)

The Darius web app (React 19 + Vite + TypeScript): the home page with a live, rotating 3D hero, the interactive
pipeline replay and the 3D scene graph; upload and the live processing screen; and the simulator (three.js,
Spark for Gaussian splats, Rapier for physics). The scene format it reads is in
[`docs/scene-format.md`](../docs/scene-format.md).

```bash
npm ci               # once
npm run dev          # development server (also reachable from a phone on the same Wi-Fi)
npm test             # Vitest: physics on the reference scene and real scans, robotics math, replay parsing
npm run typecheck
npm run build        # type-check and build into dist/
```

- `?scene=<id>` opens `/scenes/<id>/scene.json`; add `&view=robot` (or `debug`, `physics`) to open in that view.
  Scene folders come from the repo's `/scenes` directory, served in development and copied into the build.
- `?job=<id>` follows a processing job on the API (`VITE_API_URL`, default: the deployed Modal app).
- `?debug` exposes `window.__darius` (object positions, tilt, screen points, `sway(on)` for frame-rate tests).

| Module | What it does |
|---|---|
| `src/sim/Simulator.ts` | the 3D view: splats and meshes, on-demand rendering, adaptive resolution, view modes, pointer input |
| `src/sim/physics.ts` | the Rapier world: table slab from the scanned outline, mass-scaled grab spring, carry, throw, "what if" actions |
| `src/sim/robotics.ts` | robot-view marks: centers of mass, equivalent inertia ellipsoids (Jacobi eigen-solver), grasp glyphs |
| `src/sim/scene.ts` | scene types (mirror of `pipeline/darius/scene/schema.py`), loading and validation |
| `src/sim/splatCleanup.ts` | client-side cleanup for scenes packaged before the pipeline did it |
| `src/Pipeline.tsx`, `src/replay.ts` | the stage graph with each stage's real output (finished scenes or live jobs) |
| `src/SceneGraphView.tsx` | the interactive 3D scene graph (`robotics/scene_graph.json`) |
| `src/PointCloudView.tsx` | VGGT's point cloud with the camera path |
| `src/Home.tsx`, `src/Processing.tsx`, `src/SimulatorView.tsx`, `src/Inspector.tsx` | pages and panels |
