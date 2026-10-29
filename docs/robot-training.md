# From the assembly lab to a learned robot

**Engineering draft, 2026-10-29.** Darius currently compiles original LEGO-compatible assemblies, previews nominal motions, exports synthetic task-space episodes, and fits a small offline sanity baseline. This document specifies the work needed to train and evaluate a physical robot. It does not report a trained manipulation policy or a hardware demonstration.

## 1. Choose the first physical task

Start with one rectangular 2×4 brick, one indexed tray slot, one fixed baseplate location, and one upright placement. A camera observes the whole fixture; a second camera can observe the gripper. Bolt or clamp the baseplate so the robot does not move it while pressing.

The first milestone is **pick → carry → align → seat → independently verify**. A complete six-brick build is a later milestone. A constrained fixture makes failures attributable and provides a useful baseline before adding arbitrary pile picking or natural-language goals.

Candidate hardware routes:

| Route | Initial purpose | Required evidence before insertion |
|---|---|---|
| SO-101 leader/follower setup | Low-cost teleoperation, recording, and coarse pick/place | Repeatability at the fixture, gripper possession checks, current/load behavior, recovery and stop behavior |
| Arm with supported Cartesian and force/torque control | Guarded insertion experiments | Calibrated TCP, force bias, tool mass, collision envelope, force-limited controller behavior |
| Existing lab arm | Fastest route when already calibrated and supported | Reproduce all fixture and sensor checks on the actual end effector |

SO-101 is a documented open hardware route; suitability for LEGO clutch insertion must be established experimentally. Do not choose an arm on nominal reach alone. Gripper jaw clearance, repeatability under load, sensing, and access to a guarded controller are decisive. Follow the hardware vendor's setup and operating procedure. [SO-101 documentation](https://huggingface.co/docs/lerobot/v0.6.1/so101).

## 2. Define the robot adapter

The browser's animated manipulator is an explanatory kinematic model. The compiler's Cartesian waypoints are references. A hardware adapter must provide:

```text
RobotAdapter
  observe() -> timestamped cameras, q, dq, gripper, optional wrench
  transform_task_to_base(pose) -> pose in calibrated robot base frame
  plan_free_motion(goal, collision_scene) -> validated joint trajectory
  execute_guarded_approach(goal, limits) -> controller outcome
  grasp(width, limits) -> possession evidence
  press(goal, limits) -> force/displacement trace and outcome
  stop(reason) -> stopped state
```

These are interface requirements, not implemented robot methods. The adapter owns joint limits, IK, self-collision, fixture collision, tool offsets, real-time control, timeouts, and stop semantics. Do not forward JSON waypoints directly to a motor bus.

### Coordinate calibration

The assembly frame is right-handed and **+Y up**, in millimeters. The dataset's robot frame is right-handed and **+Z up**, in meters:

```text
p_robot_m = 0.001 · [x_assembly_mm, −z_assembly_mm, y_assembly_mm]
```

This fixed axis conversion is not the measured camera-to-robot transform. The physical chain still requires:

```text
T_robot_base←target = T_robot_base←fixture · T_fixture←target
T_robot_base←camera · p_camera ≈ T_robot_base←fixture · p_fixture
```

Record the transform matrices, calibration images, timestamp, residuals, tool geometry, and software revision. Verify them at several held-out fixture points with the actual tool. Report median and tail position/orientation errors under load, not only a calibration fit residual.

The compiler's nominal `12 N` press parameter is an **uncalibrated placeholder**. It is not a safe universal insertion limit. Derive operating limits from the chosen arm, gripper, fixture, parts, and supervised measurements. Disable insertion until those limits exist.

## 3. Record the right data

Manuals specify desired assembly states. They do not contain the robot's observations, grasp mechanics, control actions, contact forces or corrective behavior. A learned manipulation policy needs those additional signals.

### Physical episode contract

| Channel | Suggested record | Purpose |
|---|---|---|
| Time | Monotonic sensor/control timestamps; synchronization offset | Detect dropped frames and action lag |
| Images | Fixed camera plus wrist/oblique camera when available | Workspace state, gripper possession and insertion visibility |
| Proprioception | Joint positions/velocities, gripper position; units and joint ordering | State conditioning and replay validity |
| Action | Executed controller command with representation/version | Learning target tied to an actual interface |
| Contact | Measured wrench or calibrated load proxy, with validity flag | Identify contact, seating and excessive load |
| Task | Assembly hash, brick instance, target pose, local support graph | Goal and subtask conditioning |
| Outcome | Independent success/failure, error category, intervention | Evaluation and recovery learning |
| Provenance | Robot/tool/fixture IDs, calibration ID, operator, source permission | Reproducibility and data governance |

Choose a recording frequency that the hardware and cameras can sustain and verify synchronization. **30 Hz is a proposed starting setting**, not a requirement or a measured Darius rate. Preserve higher-rate contact/control traces if the controller produces them; do not imply interpolated RGB frames are new measurements.

Collect successful demonstrations and controlled recovery examples. Keep `success`, `failure`, `aborted`, and `unknown` distinct. A robot reaching the nominal endpoint does not prove the brick snapped into place.

### Independent completion evidence

Use at least two suitable signals: geometry after gripper retraction, stable pose after a short observation interval, a calibrated force/displacement signature, or a supervised physical inspection. Retain the raw evidence. The policy's own confidence should not be the only success label.

Useful failure codes include `empty_grasp`, `wrong_instance`, `dropped_part`, `pose_misalignment`, `incomplete_seating`, `fixture_moved`, `collision_stop`, `perception_uncertain`, and `verification_failed`.

## 4. Understand the current synthetic export

```bash
cd engine
uv run darius lego dataset signal-tower --out ../outputs/tower-data --episodes 24 --seed 7
uv run python -m darius.robotics.train ../outputs/tower-data --out ../outputs/tower-baseline
```

The export contains source assembly and compiled plan, a manifest, per-episode JSONL records, and checksums. Episodes describe a nominal Cartesian teacher with seeded variation. Their source is synthetic. Goal information is available as plan conditioning.

The small NumPy baseline fits task-space behavior cloning as a pipeline sanity check. Its held-out error is **open-loop synthetic prediction error**. It is not a closed-loop rollout score, an insertion success rate, a vision policy, a force-control result, or evidence of sim-to-real transfer.

Do not mix those files with physical episodes without a `source` field and explicit sampling policy. Do not label requested force limits as measured wrench. Do not label compiler-generated `verify` primitives as observed success.

### LeRobot conversion boundary

Darius JSONL is not LeRobotDataset v3.0. A future adapter should use the version-pinned `LeRobotDataset` API to create the dataset, write frames with matching sensor/action definitions, and finalize metadata and encoding through that library. Merely renaming a manifest or directory does not produce a valid LeRobot dataset. The official format separates tabular signals, video, and episode/feature metadata. [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/v0.6.1/lerobot-dataset-v3).

Before training, inspect an episode with camera and action overlays, verify units, check every episode boundary, recompute statistics on the training split only, and reconstruct several source timestamps from the final encoded artifact.

## 5. Train baselines in a deliberate order

### B0 — deterministic task-space teacher, hardware integration proposed

Execute calibrated task primitives in the constrained fixture with a verified hardware adapter. This establishes whether the hardware and controller can perform the task before a learned policy is blamed for failures.

### B1 — offline numeric sanity baseline, implemented

Fit the repository's small ridge-regression policy to the synthetic export. Check disjoint episode validation and artifact loading. Use it to detect schema/unit issues and exercise a train/evaluate artifact flow. The model receives privileged plan information; it is not comparable to an image-only policy.

### B2 — ACT on real demonstrations, proposed

ACT is the first imitation-learning baseline: fixed-camera image(s), joint state and task context produce short action chunks. Start with a single brick/target and fixed fixture. Keep the robot command representation identical between recorded data and inference. ACT is an established fine-manipulation method; no LEGO performance transfers automatically from its original paper. [ACT paper](https://arxiv.org/abs/2304.13705), [LeRobot ACT documentation](https://huggingface.co/docs/lerobot/main/act).

After recording, converting and validating a real dataset, an example external training invocation is:

```bash
# Separate, version-pinned LeRobot environment; requires your real dataset.
# Replace the dataset ID and choose a device for your training machine.
lerobot-train \
  --dataset.repo_id=YOUR_ACCOUNT/darius-lego-real-v1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/act-lego-v1 \
  --job_name=act-lego-v1 \
  --steps=20000
```

This command is a draft experiment, not an executed Darius training run. Pin the LeRobot commit/release and verify its CLI before starting. `training/act-real-draft.yaml` records project-specific assumptions; it is a planning document, not a LeRobot config accepted by this command. The official workflow describes separate recording, training and evaluation. [LeRobot imitation-learning guide](https://huggingface.co/docs/lerobot/v0.6.1/il_robots).

### B3 — Diffusion Policy on matched data, proposed

Use the same real episodes, held-out resets, cameras, action representation, compute reporting and trial budget as ACT. Multiple feasible grasps and approach paths make a distribution over actions interesting, but the experiment must establish whether it helps here. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).

### B4 — graph-conditioned policy, proposed

Condition the policy on the target brick's part/color embedding, target transform, local support subgraph, assembly progress, and current uncertainty. A graph encoder can pool neighboring supports into a task vector. Compare against a baseline with the same parameter/compute budget and target pose but no graph, and against an oracle perception baseline.

Do not expose full future trajectories as observations during evaluation. Distinguish information available at deployment from privileged simulator state.

### B5 — corrective data and residual insertion, proposed

Collect expert takeover at uncertain grasps and failed alignments. Train a recovery policy only after the system can identify a failed placement. For force-sensitive insertion, a learned residual can adjust a proven guarded controller inside a small workspace envelope; the controller retains hard operating limits and timeout behavior.

## 6. Curriculum and collection budget

Numbers below are **initial experiment allocations**, not evidence-backed sufficiency guarantees.

| Stage | Task family | Proposed initial data | Exit evidence |
|---|---|---|---|
| C0 | One brick from an indexed slot into a free tray | 30–50 supervised episodes | Possession and release are observable; calibrated replay stays inside fixture |
| C1 | One brick seated at one fixed baseplate target | 50–100 successful demos plus labeled failures | Independent seating checks; contact traces understood |
| C2 | Several targets, tray offsets, and brick colors | 100–200 demos across reset strata | Success on held-out reset seeds and a different recording session |
| C3 | Tower, stairway and bridge motifs | Balanced per motif; include support/order changes | Local support correctness and complete-build success measured |
| C4 | New structures of known parts | Hold out entire assemblies, not random frames | Generalization with no assembly identity leakage |
| C5 | Novel part shapes and nonvertical insertion | New geometry, grippers and annotated demos | Updated collision/contact models and separate qualification |

Advance when a gate passes, not when a target episode count is reached. A large set reconstructed from a manual is not the right first motor-learning task.

## 7. Simulation that can support a claim

The optional MJCF export gives an inspectable scene with primitive brick bodies, visual studs, target sites and a mocap gripper. It is useful for unit checks, coordinate inspection and building a future simulator adapter. It does not implement realistic clutch engagement, gripper friction, robot actuation or a manipulation environment.

```bash
cd engine
uv run darius lego sim signal-tower --out ../outputs/signal-tower.xml
```

For a training simulator, implement a reset/step/observation contract, actual arm and jaw geometry, contacts, actuators, calibrated camera models, and a seating model. Treat a distance-triggered weld as an **abstract snap model** and label it accordingly. Calibrate that model using measured physical force/displacement traces; show sensitivity to contact and friction parameters. MJCF is an expressive model format, but loadable XML alone is not physical validation. [MuJoCo modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).

Randomize lighting, camera extrinsics, fixture pose, source slot position, friction and part mass within measured or disclosed ranges. Keep a clean nominal evaluation suite. Isaac Lab can later provide parallel rendered environments; that scale-up should follow a validated single-environment task. [Isaac Lab](https://isaac-sim.github.io/IsaacLab/v2.0.0/index.html).

## 8. Evaluation protocol

Split by **whole episode**, then by **recording session**, then by **assembly family** for stronger tests. Never put adjacent frames from one trajectory in both train and test. For manual learning, separate manual/booklet and near-duplicate rendered views by build identity. Report which split a number uses.

| Metric | Definition | Why it matters |
|---|---|---|
| Part selection accuracy | Correct physical part instance selected / attempts | Separates perception from motor control |
| Placement pose error | Translation and rotation error after retract | Detects alignment errors before labeling success |
| Seating success | Independently verified engaged placements / attempted placements | The actual contact task |
| Structural accuracy | Correct intended support edges / target support edges, plus false edges | Distinguishes plausible render from correct assembly |
| Full-build success | All target placements and structural checks pass / build trials | Exposes compounding error |
| Recovery success | Failed placement corrected within fixed retry budget / recoverable failures | Tests closed-loop behavior |
| Interventions | Human takeovers and stops per build | Prevents hidden supervision from inflating success |
| Force/time | Peak measured load, limit trips, cycle time distributions | Reveals operating cost and contact behavior |

For each result publish numerator, denominator, confidence interval, robot/tool/calibration IDs, policy checkpoint, source manifest hashes and all retry/intervention rules. Count aborted trials. Report per-brick and complete-build scores separately: even independent 95% placement success compounds to approximately `0.95^20 ≈ 36%` over twenty placements. That calculation is illustrative, not a Darius result.

The central ablations are: no graph conditioning; no independent verification; no recovery; real-only versus synthetic-pretrained; nominal versus randomized sim; and oracle versus predicted brick pose. Use identical held-out trials across policies when possible.

## 9. Manual-to-policy research pipeline

The optional acquisition and VLM-draft adapter is implemented as an unapproved candidate path. It can fetch selected official booklet pages, retain provenance and propose a constrained reading; mocked tests validate the adapter contract. One live smoke run on three selected pages of an official alternate camera booklet returned ten uncertain/unsupported brick entries and withheld a complete candidate. No reconstruction accuracy score or official-set success result is reported. Candidate geometry and model self-assessed confidence do not replace human review. Pending imports or unresolved permission are blocked from synthetic episode export.


```text
Official set-number lookup
  → permission/provenance record
  → booklet selection + PDF page extraction
  → optional VLM candidate reading (unapproved; quality unevaluated)
  → per-step added-part and camera inference research
  → discrete pose hypotheses on known part library
  → geometric compiler checks + human correction
  → validated assembly graph
  → calibrated task primitives / demonstrations
  → learned local policy + independent verifier
  → physical evaluation with disclosed failures
```

Manual interpretation should emit candidate part IDs, pose alternatives, confidence and page references. Reject unsupported shapes and ambiguous orientations. Learn the image-to-graph module separately from the graph-to-motion module so their errors can be measured independently. MEPNet establishes closely related manual-to-pose work; JARVIS is direct prior art for manual-guided robotic assembly and feedback-driven primitive execution. [MEPNet](https://arxiv.org/abs/2207.12572), [JARVIS](https://arxiv.org/abs/2610.27618).

## 10. Deliverables for the first real experiment

- A calibrated rig record and a working adapter that can stop, time out and expose execution outcomes.
- Original task definitions, source hashes, a documented rights basis, synchronized demonstrations and a data card.
- A deterministic controller baseline and a separately trained ACT checkpoint.
- A fixed held-out trial manifest, raw video/contact evidence, failure taxonomy and complete result table.
- A short reproducible demo that shows a failed placement being detected and recovered, if recovery actually works.

The most compelling result is a robot that can explain the intended connection, perform one real insertion, prove it seated, and recover when it did not. Every additional brick then tests whether the architecture scales.
