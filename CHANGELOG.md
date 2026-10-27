# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-27

Initial release.

### Pipeline
- Staged processing pipeline on Modal (`pipeline/scan2sim`): sharp-keyframe ingest, VGGT
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

[Unreleased]: https://github.com/DanielTong50/3d-visualization/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DanielTong50/3d-visualization/releases/tag/v0.1.0
