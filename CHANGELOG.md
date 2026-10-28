# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-28: the assembly copilot

Darius becomes an AR copilot for flat-pack furniture. The real-to-sim scene compiler is retired and preserved at
`v0.2.0`.

### Added
- The assembly plan (`darius.assembly`): parts, hardware, joints and steps as a graph, one Pydantic schema exported as
  JSON Schema and mirrored in TypeScript; validation, subassemblies, and graph scoring by letter (`compare`).
- Progress tracking: a hidden Markov model over the number of steps done (stay .86, advance .11, skip .02, back .01;
  evidence capped to [.15, .85] and tempered by .7), with forward filtering live and Viterbi plus forward-backward for
  a whole video, in Python and TypeScript. Guidance for the next step from the plan and the belief.
- A flat-pack compiler (`darius.furniture`): products declared as boxes, contacts found from geometry, hardware laid
  out per joint kind, floor layouts; a library of four products (bookcase, nightstand, side table, desk).
- Perception with Muse Spark: manuals read into plans in one schema-constrained request (`to_plan` resolves repeated
  letters and implicit base parts and validates the result); frames checked against a plan with the manual's step
  pages as references; whole videos decoded into a dated build history.
- `darius eval`: reads every library manual and watches every rendered session, stores the outputs with their scores,
  and summarizes them in `assemblies/benchmarks.json`; `--replay` re-scores without the API and `--check` fails if a
  stored score drifts. Results: parts F1 100% and hardware 100% on all four manuals, edges F1 99.2% on average;
  18/18 steps found in video, 89.7% of frames right, 0.80 s mean timing error.
- HTTP API (`/library`, `/manuals`, `/frames`, `/videos`) on Modal, gated by `DARIUS_ACCESS_CODE`.
- Web app: a home page with the copilot building in a studio; the copilot session (tracked parts, ghosts, arcs,
  turns, fastener markers, the assembly graph and the tracker's belief over simulated perception); the manual reader
  and the video analyzer, which replay the stored runs; live mode on a camera, a video file or a filmed build.
- Generated test data: hidden-line manuals in the flat-pack style drawn from each plan (parts slide in along a free
  way, subassemblies come in as one piece) and handheld sessions with frame-exact ground truth
  (`frontend/scripts/capture.mjs`).
- Tests: 28 Python, 28 TypeScript; CI also validates the library and re-scores the stored results.

### Changed
- The Python project moved from `pipeline/` to `engine/`, with lighter dependencies (no GPU stack, no SAM client).
- The web app is formatted with Prettier.

### Removed
- The scan pipeline (reconstruction, segmentation, physics, splats), the scene format, the robotics exporters, the
  simulator and the scene library. All of it remains at `v0.2.0`.

## [0.2.0] - 2026-10-27: Darius, a real-to-sim scene compiler

The project is renamed **Darius** and becomes a real-to-sim scene compiler for robotics.

### Added
- Robotics layer (`darius.robotics`): exact mass properties from the collision meshes with a closed-form shell
  model for hollow objects; antipodal parallel-jaw grasp synthesis; a 3D scene graph (`supports`, `near`,
  `part_of`, `grasp_of`, `observes`); writers for MuJoCo MJCF, URDF and OpenUSD with UsdPhysics; `world.json` with
  poses, mass properties, grasps and camera intrinsics; a Gymnasium-style MuJoCo pushing environment (`PushEnv`).
- Scene format: optional `inertial` and `grasps` per object, `capture.image_size`, `background.backdrop`.
- `darius` command line: `info`, `validate`, `export`, `graph`, `sim`, `gallery [--check]`, `schema`.
- Ten real demo scenes from Objectron clips (kitchen counter with five objects, mugs, cups, shoes, books, a camera,
  a cereal box), each with its robotics bundle and a replay of how it was built.
- Web app: rotating live hero across the scans, an interactive pipeline replay (stage graph, keyframes, VGGT point
  cloud with the camera path, masks, separated objects, physics table, held-out render), an interactive 3D scene
  graph, a Robot view (centers of mass, inertia ellipsoids, grasps), a live stage graph on the processing screen.
- Simulator: carry upright with Q / E (or the wheel) to turn, throws at the hand's velocity, a landing marker.
- Pipeline: preview artifacts after each stage and an API endpoint that serves them while a job runs.

### Changed
- The Python package is layered: `core`, `scene`, `models`, `robotics`, `pipeline`, `calibration`, `service`,
  `eval`, `tools`. Modal app, volumes and secret are `darius`; the access code is `DARIUS_ACCESS_CODE`.
- Splat training uses MCMC densification (450k Gaussian budget) with anti-aliased rasterization: held-out PSNR
  rose on every retrained clip, by 0.4–3.8 dB (median +2.2 dB).
- The orchestrator runs the stage graph concurrently (reconstruct ∥ segment, physics ∥ splat).
- The browser renders splats straight to the canvas, only when something changes, at an adaptive pixel ratio; physics
  stays real-time down to 5 fps (up to 24 steps per frame).
- The grab spring's stiffness scales with mass, so heavy objects no longer hang far below the pointer.
- MJCF contacts use geom priority so table contacts take each object's own friction, and a time constant derived
  from a fixed natural period, so bouncy objects no longer gain energy.
- The default grasp gripper opens to 140 mm.

### Fixed
- Close-up scans gave every initial Gaussian to the object, leaving the table without splats.
- Object splats over 5 mm were dropped, which made MCMC-trained objects see-through (now 1 cm).
- The table outline now reaches under every object; objects at the edge of a sparsely seen counter fell off.
- With Rapier's 0.1 length unit, the table's internal-edge fix let a thin book fall through the table.

### Added
- Photo-real appearance: a GPU splat stage trains 3D Gaussian splats with gsplat and keeps each
  Gaussian's object label through densification, so every object's splats move with its rigid body.
  VGGT's cameras are refined jointly (per-view rigid correction, and a focal scale learned through a
  differentiable zoom). Held-out views are scored after test-time pose alignment.
- Export-side splat cleanup, so nothing is left behind when an object moves: drifted, needle-shaped and
  oversized object splats are dropped; the background is cut out around each object; and each
  footprint is cleared and re-covered with table-colored splats.
- Scene format: `splats` visuals (SPZ v3), `fallback_visual`, `convex_parts` colliders, `pick_hull`,
  `table_plane.polygon_m` (the outline of the visible table top), and a capture `view`.
- Compound colliders from approximate convex decomposition (CoACD), used only when they remove at
  least 15% of the single hull's empty volume.
- Dark redesign of the web app: home page with a live 3D preview, how it works, demos and a filming
  guide; processing screen with a progress bar; simulator with Move, Lift and Push tools, a "What if"
  bar (Cmd-K) with suggestions, an object inspector, a physics settings sheet and keyboard shortcuts.
- Splat rendering in the browser (Spark) with table-only backdrop, ghost and needle clean-up, footprint
  patches, and scans opening from the filmed side.
- Gallery thumbnails are clean keyframes, and the sandbox scene has one.
- Scenes open from the camera they were filmed from (`view`), backed off to fit every object on narrow
  screens.
- Hand-authored scenes get a soft dark floor, and the camera frames the whole table.

### Changed
- Grabbing uses a spring joint.
- The 3D engine is lazy-loaded: the home page ships 250 KB of JavaScript instead of 7.9 MB.

### Fixed
- The table collider was smaller than the visible table (0.52 × 0.47 m vs. 0.76 × 0.68 m on the mugs
  scene), so objects could fall through what looked like solid table. The simulator now builds it from
  the outline as a closed triangle-mesh slab, which also fixes sinking and drift on scanned tables.
- Jobs created before a stage was added now pick it up instead of failing, and a rerun clears a
  previous failure from the job status.

## [0.1.0] - 2026-10-27

Initial release.

### Pipeline
- Staged processing pipeline on Modal (`pipeline/scan2sim`, now `pipeline/darius`): sharp-keyframe ingest, VGGT
  reconstruction in a warm GPU worker, Muse Spark detection handed to SAM 3.1 as box prompts for
  video tracking, multi-view voxel voting with gravity alignment, Muse Spark physical priors with
  object-voted metric scale and volume-based mass, and scene packaging.
- Resumable runner: every stage is a function of a job folder with a marker file, and `--start
  <stage>` re-runs from any stage without repeating earlier work.
- Reality check: friction measured from a video of a flicked object by fitting a constant-deceleration
  slide, written back into the scene as `physics_source: "measured"`.
- MuJoCo MJCF export of any scene package, with deterministic export bundles.

### API
- FastAPI service on Modal: video upload, job status, scene files, and "what if" scenarios through
  Muse Spark tool calling. Endpoints that spend GPU time or API credits accept an optional shared
  access code.

### Simulator
- Browser simulator (React, three.js, Rapier) at a fixed 120 Hz: orbit camera; carry, stack and throw
  with mouse or touch (the held object's weight is cancelled and a spring pulls its center of mass);
  info cards with each value's provenance; reset.
- Real-scan rendering with per-object and background point clouds, invisible hull meshes for picking,
  and a bounding-box fallback for degenerate hulls.
- Home screen with video upload and client-side checks, access-code prompt, demo gallery, and a
  processing screen that explains which stage failed and what to do.
- Gravity presets, slow motion, material editing from the shared material table, MuJoCo bundle
  download, a "what if" box whose model actions are validated client-side, and a reality-check panel.

### Scene format
- Scene package format v1: Pydantic models as the source of truth, a generated JSON Schema, a
  TypeScript mirror, and a hand-authored sandbox scene covering every collider type.
- First real scan, `scenes/objectron-mugs`, processed from a 7-second phone clip.

### Evaluation and tooling
- Objectron evaluation against metric ground truth (24 clips, 6 classes) with a scale-prompt ablation
  and a generated report (`docs/results.md`).
- CI: lint, tests and build for both halves, freshness of generated scene files, and a check that
  exported scenes settle in MuJoCo. GitHub Pages deployment of the simulator.

[Unreleased]: https://github.com/DanielTong50/3d-visualization/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/DanielTong50/3d-visualization/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/DanielTong50/3d-visualization/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DanielTong50/3d-visualization/releases/tag/v0.1.0
