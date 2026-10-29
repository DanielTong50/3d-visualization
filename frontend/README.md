# Darius LEGO workbench

A React 19 + TypeScript + Three.js interface for inspecting the LEGO assembly compiler, playing kinematic pick-and-place previews, discovering official instructions by set number, and exporting plans and synthetic training handoffs.

```bash
npm ci
npm run dev       # http://localhost:5173
npm run build     # TypeScript + production build
npm test          # deterministic tests, no model requests
```

The default page is the LEGO lab. `?build=bridge` and `?build=stairway` deep-link the curriculum. `?view=legacy` retains the earlier furniture research interface, including its manual/video demos.

## Real data, explicit boundaries

`src/pages/LegoLab.tsx` fetches the engine-generated `assemblies/lego/<id>/plan.json` files. Vite serves the repository's assemblies folder in development and copies it into production builds. The three default assemblies are original Darius fixtures, not reconstructions of official LEGO sets.

`src/lego/LegoStage.ts` renders stud-grid bricks, exact source tray positions, target ghosts and a kinematic robot. Playback uses a 5.1-second presentation cycle per brick. It illustrates grasp, transfer, alignment, pressing and verification; it does not simulate clutch force, solve robot IK, execute hardware or run a learned robot policy. The graph, inventory and JSON panels inspect actual compiler artifacts.

Instruction discovery links directly to LEGO's official set pages. Set **10698** also has bundled, dated official booklet metadata. Other sets use the API when configured, with an official-page fallback. Public availability does not establish a license for model training. Automatic image interpretation and geometric reconstruction require the engine's separate draft/review pipeline.

The training export downloads an engine synthetic bundle when available; otherwise it downloads an explicitly labelled planning draft. It is not native LeRobot v3 data or evidence of successful robot learning.

## Connect the local API

```bash
# Terminal one, repository root
cd engine
uv run uvicorn darius.service.api:local_app --factory --port 8787

# Terminal two, frontend/
VITE_API_URL=http://127.0.0.1:8787 npm run dev
```

The LEGO interface uses `/lego/instructions/{set_number}` and `/lego/builds/{id}/training`. No model credentials are shipped to the browser. Model-assisted manual drafting is an engine workflow documented in the root README.

## Controls

- Select an original build in the library.
- Play/pause, jump to the next brick, restart, or drag the timeline.
- Drag the scene to orbit and scroll to zoom; reset the camera or enter fullscreen.
- Toggle exploded geometry and the robot preview.
- Select a sequence step or graph node to seek to that brick.
- Inspect the physical inventory or the raw compiled plan, then download JSON.
- Import a compiler-generated `plan.json` to preview your own build (desktop library panel). The browser checks bounded geometry and shape; support semantics are validated by the engine compiler.

The interface is responsive and keyboard accessible; WebGL failures leave compiler inspection and downloads available. The legacy furniture renderer remains independent from the LEGO stage.
