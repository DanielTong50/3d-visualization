# Frontend (web app)

The web app: home screen with upload and demo gallery, processing screen, and the 3D simulator
(React + Vite + TypeScript + three.js + Rapier).
The scene file format is in [`docs/scene-format.md`](../docs/scene-format.md).

```bash
npm install          # once
npm run dev          # development server (also reachable from a phone on the same Wi-Fi)
npm test             # physics tests on the sandbox scene in /scenes/sandbox
npm run build        # type-check and build into dist/
npm run preview      # serve the built app (also on the Wi-Fi network)
```

- `?scene=<folder>` opens `/scenes/<folder>/scene.json` (default: `fake`). Scene folders come from the
  repo's `/scenes` directory, served in development and copied into the build.
- `?debug` exposes `window.__scan2sim` (object positions, tilt, screen points) for browser tests.

Code map: `src/sim/scene.ts` (scene types, loading, checks), `src/sim/physics.ts` (Rapier world, grabbing),
`src/sim/Simulator.ts` (three.js view, camera, pointer input), `src/App.tsx` and `src/InfoCard.tsx` (UI).
