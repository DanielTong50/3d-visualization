# Design: Darius

**Status:** v0.2 (Darius) · **Scope:** single tabletop scenes, rigid bodies, robot-ready exports

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
| Usable by non-experts, useful to researchers | Works in a browser on a phone; exports to MuJoCo, Isaac Sim (USD) and URDF |
| A scan is a robot environment | Every scene settles in MuJoCo; exported mass, inertia and contact parameters match the manifest; grasp candidates and a scene graph for every object |

## Non-goals (v1)

- **Whole rooms.** One tabletop with a handful of well-separated objects.
- **Deformables, liquids, cloth, articulation.** Everything is a rigid body; soft objects get a `soft`
  flag and low restitution.
- **Transparent or mirror-like objects.** They break multi-view reconstruction.
- **Relighting.** Splats bake in the capture's lighting; moving an object doesn't update its shading or shadows.
- **Accounts and multi-user sharing.**

## Users

- **Curious users:** film a desk, then grab, throw and stack the real objects in a browser; tap one to
  see what the system thinks it is and how heavy it is; type "what if someone bumps the table?".
- **Robotics and ML researchers:** inspect estimated physical parameters and their provenance, compare
  simulated and real outcomes with numbers, and load the scene in their simulator (MuJoCo, Isaac Sim,
  PyBullet), or train in it directly with the Gymnasium-style pushing environment.

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
| Gaussian splats, separated by label | Textured meshes | Photo-real from the same cameras; carrying labels through densification makes per-object motion trivial |
| Convex parts only when they help | Always decompose | Scanned interiors are filled by depth noise; a measured gate avoids paying for parts that change nothing |
| Rapier (WASM) in the browser | Server-side simulation | Interactive at 120 Hz on phones, no server cost per user |
| MCMC densification with a fixed budget | Clone-and-split with opacity resets | +2.2 dB median held-out PSNR on the same clips, and a splat count the browser can afford |
| One robotics model, many writers | A writer per format reading scene.json | MJCF, URDF, USD and the scene graph cannot disagree about a pose, a mass or a grasp |
| Shell model for hollow objects | Uniform density over the hull | A mug carries its mass in its wall; closed form, no extra geometry |
| Contact time constant from a fixed natural period | MuJoCo's customary 0.01 s | Low damping ratios (bouncy materials) otherwise gain energy at a 2 ms step |
| 140 mm parallel-jaw gripper by default | Franka Hand (80 mm) | Most tabletop objects (mugs, shoes, cameras) are 9–14 cm across |

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

- Relightable appearance, so moved objects cast and receive shadows.
- Hole filling under moved objects (Muse Image edits of keyframes).
- Convex decomposition (CoACD) for strongly concave objects.
- Articulated objects; full rooms.
- A simulated arm (MuJoCo Menagerie) in the exported scenes, with grasps executed and verified in simulation.
- Grasp synthesis on the convex parts instead of the single hull, so handles and rims become graspable.
