# Scene package format (v1)

A scene package is a folder: one `scene.json` manifest plus the asset files it names. It is the only
interface between the pipeline and the simulator, which is why the simulator can be developed and
tested against a hand-written scene (`scenes/sandbox`) without any GPU.

- **Source of truth:** `pipeline/scan2sim/schema.py` (Pydantic, `extra="forbid"`).
- **Machine-readable schema:** [`scene.schema.json`](scene.schema.json), regenerated with
  `uv run python -m scan2sim.schema > ../docs/scene.schema.json`. A test fails if it drifts.
- **Every shipped scene is validated in CI** (`pipeline/tests/test_packaging.py`).

```
scenes/objectron-mugs/
├── scene.json          manifest (below)
├── obj_01.ply          object points, object-local, binary PLY
├── obj_01_hull.obj     object collider, object-local, convex
├── background.ply      table and surroundings, world coordinates
├── thumbnail.jpg       keyframe with segmentation overlay (gallery)
├── scene.xml           optional MuJoCo export (python -m scan2sim.export)
└── reality_check.json  optional measured-friction reports
```

## Conventions

| | |
|---|---|
| Units | SI: meters, kilograms, seconds |
| Axes | Right-handed, **+y up**. The table top is the plane y = 0. |
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
| `provenance` | object, optional | Input video, license, frames used, model IDs, stage timings, per-object scale votes |

### `background`

| Field | Meaning |
|---|---|
| `table_plane.point`, `.normal`, `.size_m` | The support surface. The simulator builds a fixed box collider of this size with its top at y = 0 |
| `visual` | Optional `points` visual in world coordinates. When present, the simulator draws it instead of a solid table |
| `table_color` | Hex color for a solid table when there is no visual |
| `floor_depth_m` | A floor this far below the table, so dropped objects don't fall forever; `null` for none |

### `objects[]`

| Field | Meaning |
|---|---|
| `id` | Stable id (`obj_01`…), used by the "what if" action API |
| `label`, `confidence` | Name and 0–1 recognition confidence from Muse Spark |
| `material` | A key of [`materials.json`](../pipeline/scan2sim/data/materials.json) |
| `dimensions_m` | Gravity-aligned extents `[length, height, width]`: height is vertical, length the long horizontal axis |
| `volume_m3` | Convex-hull volume |
| `hollow`, `fill_fraction` | Whether the hull encloses mostly air, and the solid fraction of it (≈0.25 for a mug) |
| `density_kg_m3`, `mass_kg` | `mass = volume × density × fill_fraction`, unless that disagrees with a typical mass by more than 2.5×, in which case `physics_notes` explains the fallback |
| `friction` | Coulomb coefficient against the table |
| `restitution` | Coefficient of restitution on a hard surface |
| `rigidity`, `fragile` | `rigid`/`soft`; whether a hard impact would break it |
| `collider` | `box {half_extents}`, `cylinder {radius, half_height}` (y-aligned), `sphere {radius}`, or `convex_hull {file}` |
| `visual` | Optional `points {file, count, point_size_m}`; without it the simulator draws the collider shape in `color` |
| `pose` | Initial `position` and `rotation_quat` of the object-local frame |
| `physics_source` | `estimated` (pipeline), `measured` (corrected by a reality check), or `hand-authored` |

### How simulators should apply the coefficients

`friction` and `restitution` describe the object *against the table*, so a table contact must use the
object's value unchanged. The simulator gives the table and floor zero friction and restitution and
uses Rapier's `Max` combine rule, so the pair takes the object's value (between two objects, the
larger one applies). Giving the table 1.0 with the `Multiply` rule is equivalent for table contacts.
Either way, a pushed box slides `v0² / (2 μ g)`, which is the relation the reality check inverts.
MuJoCo has no restitution coefficient; the MJCF exporter maps it to a contact damping ratio
(`ζ = −ln e / √(π² + ln² e)`).

## Asset formats

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

The numbers above illustrate the shape only. See `pipeline/scan2sim/reality_check.py` for the model.
