# Scene package format (v1)

A scene package is a folder: one `scene.json` manifest plus the asset files it names. It is the only
interface between the compiler and its consumers (the browser simulator, the robotics exporters, the MuJoCo
environment), which is why the simulator can be developed and tested against a hand-written scene
(`scenes/sandbox`) without any GPU.

- **Source of truth:** [`pipeline/darius/scene/schema.py`](../pipeline/darius/scene/schema.py) (Pydantic,
  `extra="forbid"`); TypeScript mirror in `frontend/src/sim/scene.ts`.
- **Machine-readable schema:** [`scene.schema.json`](scene.schema.json), regenerated with
  `uv run darius schema > ../docs/scene.schema.json`. A test fails if it drifts.
- **Every shipped scene is validated in CI** (`darius validate`: the schema, every referenced file, closed meshes,
  physically valid inertia tensors, centers of mass inside their objects).
- **Layout:** short arrays and objects stay on one line (`darius.scene.io.dumps`), so hand-authored scenes stay
  hand-editable and generated ones diff cleanly.

```
scenes/objectron-mugs/
├── scene.json          manifest (below)
├── obj_01.spz          object Gaussian splats, object-local (SPZ v3)
├── obj_01.ply          object points, object-local: fallback visual
├── obj_01_hull.obj     single convex hull: collider or pick hull
├── obj_02_part0.obj …  convex parts, when the object uses a compound collider
├── background.spz      the table layer's splats, world coordinates, objects cut out, footprints filled
├── backdrop.spz        the surroundings beyond the table (only reliable near the filmed path)
├── background.ply      background points: fallback visual
├── thumbnail.jpg       a clean keyframe (gallery)
├── pipeline/           what each stage produced, for the web app's replay (keyframes, points.bin, pipeline.json, …)
├── robotics/           the robotics bundle: MJCF, URDF, USD, world.json, scene_graph.json (docs/robotics.md)
├── export.zip          scene.json + robotics/
└── reality_check.json  optional measured-friction reports
```

## Conventions

| | |
|---|---|
| Units | SI: meters, kilograms, seconds |
| Axes | Right-handed, **+y up**. The table top is the plane y = 0. Robotics outputs use +z up ([frames](robotics.md#frames)) |
| Gravity | `gravity` vector, normally `[0, -9.81, 0]` |
| Rotations | Quaternions `[x, y, z, w]` |
| Object-local frame | Origin at the center of the object's axis-aligned bounding box at capture time |

Gravity is not an afterthought: the pipeline defines "up" by fitting the table plane and orienting
its normal toward the cameras. If that is wrong, every object slides off the table, so the lift stage
records `cameras_above_table` (should be 1.0) and the table's RMS residual in its diagnostics.

## Manifest

| Field | Type | Meaning |
|---|---|---|
| `version` | `1` | Format version |
| `scene_id`, `title` | string | Identifier and display name |
| `scale_source` | string | How metric scale was obtained, e.g. `known_objects:coffee mug,book` or `hand-authored` |
| `background` | object | See below |
| `objects` | array | See below |
| `view` | object, optional | A capture camera: `position`, `target`, vertical `fov_deg`. Splats look best near filmed viewpoints, so viewers should open here |
| `capture` | object, optional | The filmed path after camera refinement: `cameras` (`position`, `rotation_quat`, OpenGL convention), `fov_deg`, `aspect`, `image_size` |
| `provenance` | object, optional | Input video, license, frames used, model IDs, stage timings, per-object scale votes, appearance quality (held-out PSNR) |

### `background`

| Field | Meaning |
|---|---|
| `table_plane.point`, `.normal`, `.size_m` | The support surface; `size_m` is the bounding box of the outline |
| `table_plane.polygon_m` | Optional convex outline of the visible table top, `[x, z]` pairs counter-clockwise seen from +y. Colliders should prefer it over `size_m` |
| `visual` | Optional `splats` or `points` visual in world coordinates. When present, it replaces a solid table |
| `backdrop` | Optional `splats` visual of the surroundings beyond the table; viewers fade it out away from `capture.cameras` |
| `fallback_visual` | Optional `points` visual for devices that can't render splats |
| `table_color` | Hex color for a solid table when there is no visual |
| `floor_depth_m` | A floor this far below the table, so dropped objects don't fall forever; `null` for none |

### `objects[]`

| Field | Meaning |
|---|---|
| `id` | Stable id (`obj_01`…), used by the "what if" action API |
| `label`, `confidence` | Name and 0–1 recognition confidence from Muse Spark |
| `material` | A key of [`materials.json`](../pipeline/darius/data/materials.json) |
| `dimensions_m` | Gravity-aligned extents `[length, height, width]`: height is vertical, length the long horizontal axis |
| `volume_m3` | Convex-hull volume |
| `hollow`, `fill_fraction` | Whether the hull encloses mostly air, and the solid fraction of it (≈0.25 for a mug) |
| `density_kg_m3`, `mass_kg` | `mass = volume × density × fill_fraction`, unless that disagrees with a typical mass by more than 2.5×, in which case `physics_notes` explains the fallback |
| `friction` | Coulomb coefficient against the table |
| `restitution` | Coefficient of restitution on a hard surface |
| `rigidity`, `fragile` | `rigid`/`soft`; whether a hard impact would break it |
| `collider` | `box {half_extents}`, `cylinder {radius, half_height}` (y-aligned), `sphere {radius}`, `convex_hull {file}`, or `convex_parts {files}` (a compound of convex pieces in the object's frame) |
| `pick_hull` | Optional single convex hull OBJ, for picking and raycasts |
| `visual` | Optional `splats {file, count, sh_degree}` or `points {file, count, point_size_m}`; without it the simulator draws the collider shape in `color` |
| `fallback_visual` | Optional `points` visual for devices that can't render splats |
| `pose` | Initial `position` and `rotation_quat` of the object-local frame |
| `physics_source` | `estimated` (pipeline), `measured` (corrected by a reality check), or `hand-authored` |
| `color` | Hex color: the drawn color without a visual, and the median color of a scanned object (robot simulators draw colliders in it) |
| `inertial` | Optional `{com_m, inertia_kg_m2: [ixx, iyy, izz, ixy, ixz, iyz], model: "solid" \| "shell"}` in the body frame ([mass properties](robotics.md#mass-properties)) |
| `grasps` | Optional list of `{position, rotation_quat, width_m, score}` parallel-jaw candidates in the body frame, best first; grasp frame +z approach, +y closing ([grasps](robotics.md#grasp-candidates)) |

### How simulators should apply the coefficients

`friction` and `restitution` describe the object *against the table*, so a table contact must use the
object's value unchanged. The simulator gives the table and floor zero friction and restitution and
uses Rapier's `Max` combine rule, so the pair takes the object's value (between two objects, the
larger one applies). Giving the table 1.0 with the `Multiply` rule is equivalent for table contacts.
Either way, a pushed box slides `v0² / (2 μ g)`, which is the relation the reality check inverts.
MuJoCo has no restitution coefficient; the MJCF exporter maps it to a contact damping ratio
(`ζ = −ln e / √(π² + ln² e)`) with a time constant chosen so the contact stays resolvable
([details](robotics.md#mujoco-mjcf)).

## Asset formats

- **SPZ:** [Niantic SPZ](https://github.com/nianticlabs/spz) version 3 (gzip), written with the reference
  library. Coordinates are the scene frame, declared to SPZ as RUB (x right, y up, z toward the viewer).
  Object files are object-local; the background and backdrop are in world coordinates. `antialiased: true` on a
  splat visual means it was trained with anti-aliased rasterization (render with a 0.3 px² blur and opacity
  compensation); otherwise the dilation is baked into each splat (classic 3DGS).
- **PLY:** `binary_little_endian 1.0`, one `vertex` element with `float x, y, z` and `uchar red, green, blue`.
  Up to 60k points per object and 150k for the background.
- **OBJ:** `v` and `f` lines only, 1-indexed, faces wound counter-clockwise seen from outside.
  At most 32 vertices (farthest-point subsampled): denser hulls with many tiny faces made scanned objects
  rock indefinitely in Rapier instead of coming to rest.

## `reality_check.json`

A list of reports, one per analyzed push video:

```json
{
  "object_id": "obj_02",
  "fit": { "v0": 1.21, "friction": 0.38, "distance_m": 0.196, "rms_residual_m": 0.002 },
  "friction_estimated": 0.45, "friction_measured": 0.38,
  "real_distance_m": 0.196,
  "predicted_distance_before_m": 0.166, "predicted_distance_after_m": 0.196,
  "error_before_m": 0.030, "error_after_m": 0.000,
  "track": { "t": [0.0, 0.033], "s_m": [0.0, 0.04] }
}
```

The numbers above illustrate the shape only. See `pipeline/darius/calibration/reality_check.py` for the model.
