# LEGO assembly as a compilable robot task

Darius separates **intent**, **geometry**, **motion references**, **observations**, and **measured outcomes**. This separation makes an assembly inspectable before a policy or physical robot exists, and gives future learned components a precise contract.

```text
LegoAssembly                     CompiledPlan                    Synthetic dataset
  original/reviewed source          explicit stud contacts          seeded whole episodes
  nominal brick inventory    →      support dependencies       →    task-space state/action
  discrete target poses            validation diagnostics          manifest + checksums
  manual step schedule             nominal TCP primitives          offline baseline
                                         ↓
                               Three.js kinematic preview
                                         ↓
                               MJCF scene scaffold
```

An instruction-discovery path resolves official set-number metadata. Optional acquisition/rasterization and a vision-language adapter can produce an explicitly unapproved `LegoAssembly` candidate; human correction, geometric checks and source-permission review remain separate gates. No extraction accuracy has been measured.

## Source representation

`engine/darius/lego/schema.py` defines Pydantic models that forbid unknown fields and nonfinite values. The current curriculum supports rectangular bricks/plates with quarter-turn rotations and upright stud connections. It intentionally rejects unsupported part IDs and dimension mismatches.

```json
{
  "schema_version": "1.0",
  "id": "one-brick",
  "title": "First placement",
  "description": "An original calibration task.",
  "source": {
    "kind": "original",
    "description": "Original Darius task",
    "url": null,
    "training_rights": "project-original",
    "permission_reference": null,
    "review_status": "original"
  },
  "baseplate": {"studs": [16, 16], "color_hex": "#d7dccf", "height_mm": 3.2},
  "bricks": [
    {
      "id": "b01",
      "part_id": "3001",
      "color": "coral",
      "color_hex": "#f4553b",
      "studs": [2, 4],
      "height_plates": 3,
      "position": [6, 0, 4],
      "rotation": 0
    }
  ],
  "steps": [{"index": 1, "title": "Seat the foundation", "brick_ids": ["b01"]}],
  "notes": []
}
```

`Source` distinguishes `original` from `reviewed-import`, records `review_status` (`original`, `pending`, `reviewed`), and separates `project-original`, `permission-required` and `permission-documented` training rights. Documented permission requires a nonempty `permission_reference`. The schema prevents an imported manual candidate from claiming project-original rights or original review status. These declarations record a review decision; the software cannot itself establish the underlying rights.

Each physical brick is a distinct instance, even when several share part ID and color. `steps` use consecutive indices starting at one and schedule every instance exactly once. A source step can contain multiple bricks; compilation produces a separate placement step for each, preserving the source step reference.

### Units, origins and rotations

| Quantity | Convention |
|---|---|
| Assembly axes | Right-handed, **+Y up** |
| `position[0]`, `position[2]` | Minimum footprint corner, integer stud-grid coordinates |
| `position[1]` | Body-bottom height in plate-grid coordinates above the baseplate top |
| Stud pitch | 8 mm nominal |
| Plate pitch | 3.2 mm nominal |
| Brick body height | `height_plates × 3.2 mm` |
| Rotation | 0/90/180/270°, clockwise viewed from above |
| Primitive TCP | Millimeters in the assembly frame, distinct from brick center |
| Dataset robot reference | Right-handed, +Z up; meters and radians |

For rotated footprint width/depth `(w,d)` and grid corner `(x,y,z)`, the body center is:

```text
c_mm = [8(x + w/2), baseplate_height_mm + 3.2(y + h/2), 8(z + d/2)]
```

The current grasp TCP lies above the brick top with a nominal 6 mm jaw allowance. `source_center_mm` and `target_center_mm` explicitly represent body centers so rendering and scene export do not mistake the TCP for the brick centroid. Real tool geometry must replace this nominal offset. Source positions are laid out in nonoverlapping indexed tray slots using the largest unrotated footprint and nominal clearance; target centers include the declared baseplate height.

`CompiledStep.source_yaw_deg` is explicitly `0`, while `target_yaw_deg` is the brick's final clockwise grid rotation. Approach, descend, grasp and lift retain source yaw. The remaining six primitives use target yaw, so nominal rotation occurs during the elevated transfer rather than before pickup. The stage and MJCF use source orientation for loose bricks and target orientation for assembled ghosts/sites. This remains a task-space reference; a physical adapter must validate the entire rotating gripper/brick sweep.

These dimensions are idealized. Stud pitch is a useful discrete representation; it does not encode manufacturing tolerance or clutch mechanics. LDraw's corresponding geometry conventions are 20/24/8 LDU for stud pitch/brick height/plate height, with native −Y up. A future importer must transform coordinates explicitly. [LDraw specification](https://www.ldraw.org/article/218.html).

## Deterministic contact compilation

`engine/darius/lego/compiler.py::compile_assembly` performs the following checks and derivations.

1. Resolve each part against the small supported rectangular-part table and check dimensions.
2. Check every target footprint lies within the baseplate.
3. Reject pairs whose X/Z occupied cells overlap while their open body-height intervals intersect.
4. Traverse placements in the declared schedule. A brick at layer zero contacts the fixture. Higher bricks contact earlier bricks whose top height equals its bottom height and whose stud cells overlap.
5. Record each nonempty lower/upper intersection as a connection, with exact overlapping stud-grid cells.
6. Require at least one support cell, compute support coverage, and warn when it is partial.
7. Detect earlier solids that obstruct the nominal straight-down insertion column.
8. Emit explicit dependencies and the nominal primitive sequence; retain execution blockers.

For an upper brick footprint `F`, the support coverage is:

```text
support_ratio = |union of eligible lower contact cells| / |F|
```

This is a geometric coverage ratio. It is not a probability, center-of-mass stability test, clutch force estimate, or proof of a viable grasp. The bridge fixture intentionally demonstrates partial support.

Pairwise intersection checks are quadratic in the brick count; occupied-cell operations are bounded by the small rectangular footprints and the schema's maximum task size. This is appropriate for an inspectable curriculum, not a claim of industrial-scale assembly planning.

### Output diagnostics

`CompiledPlan.validation` contains `valid`, `errors`, and `warnings`. `POST /lego/compile` returns this diagnostic plan even when geometric validation fails; consumers must check `validation.valid`. Schema violations receive FastAPI's validation error. The CLI returns a failure exit code for an invalid compiled geometry. Dataset and scene exporters reject invalid plans.

The compiler still describes nominal primitives for diagnostic inspection. No plan is authorized for physical execution: `execution.status` is always `simulation_only`, and each plan names missing calibration, IK/collision checks, force validation and robot connectivity.

## The primitive program

Each brick compiles to ten primitives:

```text
1  move     open jaws above indexed tray slot
2  move     descend to nominal grasp TCP
3  grasp    request jaw closure
4  move     lift above the fixture
5  move     transfer over the target
6  move     align just above the target
7  press    nominal seating reference; uncalibrated force parameter
8  release  request jaw opening
9  move     retract to observation clearance
10 verify   require evidence before physical advancement
```

`duration_s` is authored timing. Summed duration is a nominal preview duration, not measured robot cycle time. `grasp`, `press`, and `verify` are intended operations; their presence does not prove possession, contact, or success. A real adapter must turn them into guarded controller calls and record outcomes.

The current collision checks cover target brick volumes and the nominal vertical insertion relationship. They do not perform full arm/gripper swept-volume collision checking, solve IK, plan grasps, or model elastic deformation.

## Official instruction discovery

`engine/darius/lego/instructions.py` accepts a set number of four to seven digits. It constructs the official LEGO instructions URL itself. The server never accepts an arbitrary URL to fetch.

The adapter reads the official page's embedded structured metadata and selects the matching `setNumber`. It returns booklet sequence, main/additional status, official PDF URL, optional size/cover, retrieval timestamp and provenance labels. HTTPS asset links are restricted to known official hosts. Network reads have a 15-second timeout, a 3 MiB size cap, and redirects disabled.

The repository includes a metadata snapshot for set `10698` to support a reproducible demo. A default lookup can return `retrieval: "cached"`; `--refresh` requests a live official lookup. Request handlers do not write the cache. The snapshot contains links and metadata, not copied PDF pages.

Upstream HTML is not a stable public API. Parsing failure should surface a readable error and the official page link; it must never fabricate booklet data. Multiple booklets remain visible. Every response carries `semantic_status: "not_parsed"` and `training_rights: "permission-required"`.

LEGO describes thousands of freely downloadable instruction PDFs. That supports automatic discovery without requiring the user to upload a file; complete coverage and semantic reconstruction remain separate problems. [LEGO download guidance](https://www.lego.com/en-us/service/help-topics/article/how-to-download-building-instructions-online).

## Selected booklet acquisition and candidate inference

`engine/darius/lego/documents.py::fetch_booklet` accepts a set number and the one-based position of a booklet from official discovery. It downloads only the selected official HTTPS PDF with a 30-second timeout, 32 MiB response cap, redirects disabled, content-type checks and PDF signature checks. It rasterizes at most 16 explicitly selected pages (or the first bounded selection) using the optional `manuals` dependency. Page dimensions and document/page counts are bounded.

The output contains `source.pdf`, `page-NNN.png`, and `acquisition.json` with source URLs, retrieval metadata, selected original page numbers and hashes. Selecting fewer than all pages sets `partial_document: true`. These local copyrighted source artifacts belong in ignored outputs, not the checked-in curriculum. Acquisition is assembled in a temporary sibling directory and committed by rename only after download, rendering and manifest creation succeed, preventing partial output from appearing as a completed acquisition.

`engine/darius/lego/manual.py` uses the configured structured-output VLM to read ordered images into `ManualDraftReading`. Per-brick fields include part identity, grid pose, source-page reference, evidence, uncertainties and model self-assessed confidence. That score is not calibrated probability. Unsupported parts, unresolved poses, invalid values and unsupported assembly features are exposed rather than silently replaced with boxes.

When every required part/pose is available, `draft_from_reading` builds a candidate and invokes the deterministic compiler. The result always carries `semantic_status: "candidate_draft"` and `requires_review: true`. The candidate source is imported, pending review and permission-required. A valid geometry result does not approve it or establish training rights. Tests stub the VLM. One live smoke run used official 10698 alternate camera booklet 4, selected pages 3–5, and returned 10 draft brick entries with no complete candidate because of unsupported geometry and uncertainty. This exercises inference and abstention; no extraction accuracy benchmark is claimed.

```bash
# From engine/; network + optional PDF renderer required.
uv sync --locked --extra manuals
uv run darius lego fetch 10698 --booklet 4 --pages 1,2,3 --out outputs/camera-manual

# Separate optional model call, requiring MODEL_API_KEY.
uv run darius lego manual outputs/camera-manual --id camera-draft \
  --source-url https://www.lego.com/en-us/service/building-instructions/10698 \
  --out outputs/camera-candidate.json

# Or compose acquisition, rasterization and inference in one command.
uv run darius lego draft-set 10698 --booklet 4 --pages 1,2,3 --out outputs/camera-draft
```

`source_page` in the VLM reading indexes the ordered image selection. The composed CLI result retains `source_page_map` and the acquisition manifest so a noncontiguous selection can be joined back to original PDF page numbers. A sparse selection is not a complete official build. Review must resolve identities, coordinate choices, all omitted steps, source-page alignment, unsupported features and permission before treating it as a dataset source.

## Synthetic episode generation

`engine/darius/robotics/dataset.py` revalidates the complete `CompiledPlan` schema, including nested source-review/permission invariants, before converting validated primitives into deterministic state/action records. Supplying a plain dictionary does not bypass the source schema. A local RNG is seeded per episode. The current variation is a planar scene translation within ±12 mm and a duration multiplier in `[0.8, 1.2]`. It does not randomize cameras, materials, mass or friction.

For normalized primitive time `u`, state interpolation uses smoothstep:

```text
s(u) = 3u² − 2u³
state_next = start + s(u) · (goal − start)
action = state_next − state_current
```

The action is a per-frame Cartesian/yaw/gripper delta. It is not velocity, torque, measured joint motion or a motor command. Yaw is mapped into the robot reference frame and wrapped to use the short angular difference.

Features:

| Key | Shape | Meaning |
|---|---:|---|
| `observation.state` | 5 | EE XYZ meters, yaw radians, gripper open fraction |
| `observation.goal` | 5 | Known primitive target from the compiler |
| `observation.phase` | 5 | One-hot `move`, `grasp`, `release`, `press`, `verify` |
| `observation.progress` | 1 | Fraction of placements already completed |
| `action` | 5 | One-frame delta of the state coordinates |
| `measured_force_n` | nullable | `null`, because there is no force sensor |
| `physical_success` | nullable | `null`, because there is no physical trial |

Episode boundaries are explicit, with only the final record marked `done`. Whole episodes are assigned to train/validation; validation holds out scene translation and duration samples for the **same build**. This does not establish generalization to unseen sets.

The disk export is intentionally simple:

```text
dataset/
  assembly.json             source task
  compiled-plan.json        exact compiler output
  manifest.json             schema, features, provenance, split and limitations
  episodes/0000.jsonl        one episode per file
  episodes/0001.jsonl
  ...
  checksums.json            SHA-256 for the written dataset files
```

Pending manual drafts and unresolved training permission are rejected before episode generation. Exports refuse a nonempty destination so an experiment does not silently replace another. Hashes detect unintended dataset changes; they do not prove sensor authenticity or grant source permissions. The JSONL format is a Darius interchange contract, not a LeRobotDataset claim.

## Offline training baseline

`engine/darius/robotics/train.py` loads and checks the manifest, disjoint split membership, unique episode identities/indices/file paths, boundaries, dimensions, finite values and hashes. Every split episode must resolve to exactly one in-directory episode file; IDs alone cannot disguise reuse of one file across splits. It standardizes input features using training episodes only, appends an intercept and fits a regularized linear model:

```text
W = (XᵀX + λI)⁻¹XᵀY
```

The implementation uses a linear solve and leaves the intercept unpenalized. It writes `model.npz` and `report.json`, with training/validation errors and zero/mean-action baselines. Translation, yaw and gripper errors are reported separately because their units differ.

This is a reproducible data-to-model exercise and regression check for the representation. It is not learned perception, closed-loop manipulation, physical insertion, or an ACT/VLA policy. Its goal/phase inputs are privileged compiler conditioning and must remain disclosed.

## Simulation boundary

`engine/darius/robotics/mujoco.py` exports an MJCF fixture containing free primitive brick bodies, visual-only studs, target sites and a noncolliding mocap TCP reference. Scene geometry uses meters and +Z up.

The scene's masses and friction are scaffold values. The model has no actual manipulator, controller, realistic under-tube geometry, stud interlock, compliant clutch model, or task reward/reset environment. All three original scenes were loaded and stepped for 500 × 0.002 s (1 s) with finite state in MuJoCo 3.17.0. This verifies XML/runtime compatibility, not assembly success. Install the optional runtime with `uv sync --locked --extra simulation`.

## Where learning belongs next

1. **Manual image → assembly candidates:** part identity, additions, camera/view parameters, pose hypotheses and confidence, linked to source pages.
2. **Observation → actual state:** instance pose, possession, missing/occluded pieces and assembly progress.
3. **Task goal → local manipulation:** learned action chunks inside a calibrated controller/fixture envelope.
4. **Outcome → recovery:** independent seating verifier and a bounded corrective policy.

Keep these boundaries measurable. A manually corrected pose can unblock a motor-control experiment without implying the manual parser solved it. A perfect rendered target cannot validate grasping. A good synthetic loss cannot replace physical trials. The complete proposed experiment is in [robot-training.md](robot-training.md).
