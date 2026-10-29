# Contributing to the LEGO branch

Darius treats a LEGO assembly as a typed program that can be inspected, compiled, previewed and exported. Keep the difference between intended motion, simulated behavior, measured observation and verified outcome explicit in code and documentation.

## Setup

```bash
# Repository root
cd engine
uv sync --locked

cd ../frontend
npm ci
npm run dev
```

The LEGO curriculum, compiler, synthetic exporter and numeric training baseline need no model key. Optional manual drafts require `MODEL_API_KEY`; `uv sync --locked --extra manuals` enables PDF rasterization. Run the optional API in another terminal:

```bash
cd engine
uv run uvicorn darius.service.api:local_app --factory --port 8787
```

Restart the frontend with `VITE_API_URL=http://127.0.0.1:8787 npm run dev` for the local API. The `.env.example` model settings apply to optional LEGO manual drafts and retained furniture perception endpoints, not the deterministic compiler/data path.

## Code ownership by layer

| Area | Responsibility |
|---|---|
| `engine/darius/lego/schema.py` | Source and compiled-task contracts |
| `engine/darius/lego/compiler.py` | Grid geometry, validation, connections and nominal task primitives |
| `engine/darius/lego/catalog.py` | Original curriculum and deterministic fixture exports |
| `engine/darius/lego/instructions.py` | Bounded official set-number metadata discovery |
| `engine/darius/lego/documents.py`, `manual.py` | Selected official PDF/page acquisition and optional unapproved VLM drafts |
| `engine/darius/robotics/dataset.py` | Reproducible task-space episodes, manifest, split and hashes |
| `engine/darius/robotics/train.py` | CPU sanity baseline and offline evaluation |
| `engine/darius/robotics/mujoco.py` | Nominal MJCF scene scaffold |
| `engine/darius/service/api.py`, `cli.py` | HTTP and command-line entry points |
| `frontend/src/lego/`, `pages/LegoLab.tsx` | LEGO rendering, workbench and data contract |
| `assemblies/lego/` | Generated source/compiled fixtures and explicit metadata snapshots |
| `docs/`, `training/` | Architecture, research provenance and marked experiment drafts |

The prior furniture implementation remains under `assembly/`, `furniture/`, `perception/`, `models/`, `eval/`, and legacy frontend pages. Preserve its working commands and stored benchmarks unless a change deliberately updates them.

## Change the representation first

A new part or operation affects several contracts. Update the Python source schema/compiler, TypeScript mirror, original fixtures, renderer, episode export and meaningful tests together. If the semantics change, version the schema and document the migration.

Conventions:

- The assembly frame is right-handed, +Y up, millimeters after grid conversion. X/Z grid units are nominal 8 mm studs; Y grid units are nominal 3.2 mm plates.
- `position` is the target footprint corner/body bottom. Body centers and grasp TCPs are different quantities.
- Loose tray bricks have explicit source yaw 0; the target yaw is the assembled quarter turn. Pickup/lift retain source yaw, with rotation during nominal elevated transfer. Keep the stage, scene and dataset consistent.
- The dataset robot reference is +Z up: `[x, -z, y] / 1000`; angles are radians. It is not a measured hardware calibration.
- Every physical brick is an instance. Shared part ID/color does not make two bricks the same instance.
- `support_ratio` measures geometric contact-cell coverage. Do not treat it as confidence or physical stability.
- `execution.status` stays `simulation_only` until a separate calibrated hardware adapter is implemented and qualified.

Full definitions live in [the LEGO architecture](docs/lego-architecture.md).

## Add an original curriculum build

1. Define the build in `lego/catalog.py` with unique instances and a complete schedule.
2. Compile it and inspect every diagnostic. Unsupported geometry and floating or intersecting bricks should fail; partial support should be understandable.
3. Add tests that check intended structure and a meaningful invalid variant.
4. Regenerate the fixtures from the engine directory:

   ```bash
   uv run darius lego export --out ../assemblies/lego
   uv run darius lego schema --out ../docs/lego.schema.json
   ```

5. Update frontend build selection and test playback, support graph, inventory and plan download.
6. Document the source. Official set reconstructions must not be labeled `original`.

The current compiler is deliberately restricted to supported rectangular upright parts. Do not stretch the schema to pretend hinges, SNOT, Technic or compliant insertion are solved. Introduce their geometry, connection semantics and tests explicitly.

## Verify the change

```bash
# From engine/
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run darius lego export --out ../outputs/lego-fixture-check

# From frontend/
npm run format:check
npm run typecheck
npm test
npm run build
```

Use a fresh `outputs/` path when a command refuses a nonempty destination. The old furniture checks remain useful regression checks:

```bash
# From engine/
uv run darius validate ../assemblies/*/plan.json
uv run darius eval --check
```

Browser verification should cover play/pause, one-step advancement, scrubbing to completion, reset, build changes, inspectors, export, lookup failure, and a narrow viewport. Check console/runtime errors. Capture screenshots only after the actual representative scene has rendered.

For changes to datasets or baseline training, run a complete export/train cycle and inspect both manifest and report:

```bash
# From engine/
uv run darius lego dataset bridge --out ../outputs/bridge-data-check --episodes 8 --seed 7
uv run python -m darius.robotics.train ../outputs/bridge-data-check --out ../outputs/bridge-model-check
```

Optional MuJoCo checks should load and step the emitted XML with a recorded MuJoCo version. Passing that check establishes scene/runtime compatibility, not a functioning robot task or LEGO clutch model.

## Data and research integrity

- Keep synthetic and physical provenance distinct. Generated `verify` primitives are not success labels; requested limits are not measured forces.
- Split by whole episode with unique IDs, episode indices and file paths across splits. For stronger claims hold out recording sessions, assemblies or part families. Fit normalization on training data only.
- Report numeric synthetic loss as such. Do not convert it into a manipulation success claim.
- Never hand-edit generated benchmarks or fabricate model outputs. Preserve the actual input/manifest/checkpoint hashes for published results.
- Record source URLs, retrieval time, hashes, licenses and permission basis for third-party materials. Publicly downloadable manuals are not automatically a redistributable training corpus.
- Keep official lookup restricted to validated set numbers and fixed official endpoints. Do not replace it with arbitrary server-side URL fetching.
- External page structure can change. Preserve a readable failure and official-link fallback; do not fabricate parsed metadata.
- Keep credentials out of source, command output, fixtures and screenshots.

Read [robot-training.md](docs/robot-training.md) before adding physical-learning claims. `training/*-draft.yaml` are project experiment specifications, not runnable LeRobot or simulator configurations.

## Documentation and dependencies

Pin dependency versions exactly and update lockfiles with them. Keep API/model details tied to verified primary documentation. The [research ledger](docs/research-sources.md) records the LEGO branch's sources; [api-notes.md](docs/api-notes.md) retains earlier perception API findings.

A good change explains the concrete behavior, the evidence that verifies it, and the boundary that remains. If the frontend shows a capability, its label should agree with the backend artifact that supports it.
