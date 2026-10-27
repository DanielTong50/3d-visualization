# Design: Scan2Sim

**Status:** v0.3 · **Scope:** single tabletop scenes, rigid bodies

## Problem

Robots and embodied AI systems learn and are tested in simulation, but building a *physically
accurate* digital copy of a real place still takes days of manual work per scene: modeling each
object, guessing its mass and friction, and tuning by hand. Phone-scanning apps capture appearance
only; the result is a frozen shell with no physics. Geometry from video is largely solved.
**Per-object physics is not**, and it is the bottleneck for real-to-sim.

## Goals

| Goal | Measure |
|---|---|
| Phone video to playable scene with zero manual modeling | End-to-end processing time, target < 5 min |
| Every object becomes its own rigid body | Objects detected and separated / ground-truth objects |
| Objects are the right size | Height error against metric ground truth (Objectron) |
| Objects behave like what they are | Mass, friction, restitution plausible per material; heavy, light, bouncy and fragile objects visibly differ |
| Simulation can be corrected from reality | Friction measured from a real push closes the sim-vs-real slide-distance gap |
| Usable by non-experts, useful to researchers | Works in a browser on a phone; exports to MuJoCo |

## Non-goals (v1)

- **Whole rooms.** One tabletop with a handful of well-separated objects.
- **Deformables, liquids, cloth, articulation.** Everything is a rigid body; soft objects get a `soft`
  flag and low restitution.
- **Transparent or mirror-like objects.** They break multi-view reconstruction.
- **Photorealistic rendering.** Objects render as colored point clouds; Gaussian splats are future work.
- **Accounts and multi-user sharing.**

## Users

- **Curious users:** film a desk, then grab, throw and stack the real objects in a browser; tap one to
  see what the system thinks it is and how heavy it is; type "what if someone bumps the table?".
- **Robotics and ML researchers:** inspect estimated physical parameters and their provenance, compare
  simulated and real outcomes with numbers, and export the scene to their simulator.

## Experience

1. **Upload:** one button, filming tips, and a gallery of pre-processed scenes that always works.
2. **Processing:** each stage in plain language with progress and timing; failures say which stage
   failed and what to do about it (too dark, too short, too blurry).
3. **Simulator:** orbit and zoom; grab, drag and throw by mouse or touch; tap for an info card (label,
   material, mass, friction, bounciness, fragility, and whether each value was estimated or measured);
   reset; gravity presets; slow motion; change an object's material; a "what if" box.
4. **Reality check:** a real push next to the simulated one, with slide-distance error before and after
   correction.

## Architecture

See [architecture.md](architecture.md). In short: a staged pipeline on Modal (VGGT for cameras and
depth, Muse Spark for detection and physical priors, SAM 3.1 for tracking), a versioned scene package
as the only contract, and a browser simulator on three.js and Rapier.

## Key decisions

| Decision | Alternatives | Why |
|---|---|---|
| VGGT feed-forward reconstruction | COLMAP / GLOMAP SfM | Seconds instead of minutes, robust on short phone clips, no feature-matching failures on low texture |
| Muse Spark boxes → SAM 3.1 box prompts | SAM concept (text) prompts | Text prompts were wording-sensitive in testing and can't express "any object" |
| Majority vote per voxel across views | Per-frame mask projection with depth tests | Each point already has a label from its own pixel; voting removes mask bleed without visibility reasoning |
| Scale from recognized objects' typical sizes | Reference card in every scene | No setup for the user; multiple objects vote, so one misrecognition doesn't decide it |
| Mass = hull volume × density × fill fraction | Ask the model for mass | Geometry is measured; the model only supplies material and hollowness; its mass guess is a sanity check |
| Point-cloud visuals | Meshes, Gaussian splats | Exact correspondence with what was reconstructed; per-object motion is trivial |
| Rapier (WASM) in the browser | Server-side simulation | Interactive at 120 Hz on phones, no server cost per user |

## Risks

| Risk | Mitigation |
|---|---|
| Wrong "up" makes everything slide away | Plane fit restricted to the objects' neighborhood; normal oriented toward cameras; diagnostics checked per scene |
| Wrong scale makes toy-sized or giant objects | Confidence-weighted median over objects; measured on Objectron ground truth (see results) |
| Messy masks produce junk objects | Box-prompt tracking, multi-view voting, largest-component filter, minimum point count |
| Underside never observed; objects tip over | Collider base from the footprint of the lower half of the object |
| Model output malformed or implausible | JSON-schema constrained decoding, range clamps, material-table fallbacks |
| Transient API failures | Bounded retries with backoff; stage-level resumability |
| Public endpoints spend credits | Optional shared access code on spending endpoints |

## Future work

- Gaussian-splat visuals per object (VGGT poses initialize gsplat directly).
- Hole filling under moved objects (Muse Image edits of keyframes).
- Convex decomposition (CoACD) for strongly concave objects.
- Articulated objects; full rooms; a simulated robot arm practicing in the scanned scene.
