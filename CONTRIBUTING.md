# Contributing

## Layout

```
pipeline/            Python 3.11 processing pipeline (Modal) and HTTP API
  scan2sim/          the package: one module per stage + shared geometry/schema
  modal_app.py       Modal functions, GPU class, web API, CLI entrypoint
  scripts/           evaluation, figures, API smoke checks
  tests/             pytest
frontend/            React + three.js + Rapier simulator (Vite, TypeScript)
scenes/              scene packages; scenes/sandbox is hand-authored and must always load
docs/                architecture, scene format, results, verified API notes
```

## Setup

```bash
cp .env.example .env            # MODEL_API_KEY, HF_TOKEN (see comments in the file)
cd pipeline && uv sync          # Python 3.11, exact pins from uv.lock
cd ../frontend && npm ci
```

Cloud runs need a Modal account (`uv run modal token new`) and the `scan2sim` secret:
`uv run modal secret create scan2sim --from-dotenv ../.env`.

## Checks

| | Command (from its folder) |
|---|---|
| Pipeline format, lint | `uv run ruff format --check . && uv run ruff check .` |
| Pipeline tests | `uv run pytest` (no network, no GPU) |
| Frontend types, tests, build | `npm run typecheck && npm test && npm run build` |
| Schema in sync | `uv run python -m scan2sim.schema > ../docs/scene.schema.json` (a test fails if it drifts) |
| MuJoCo export | `uv run python -m scan2sim.export ../scenes/<id>` |

CI runs all of these on every push and pull request.

## Conventions

- **The scene format is a contract.** Change `pipeline/scan2sim/schema.py` first, then regenerate
  `docs/scene.schema.json`, update `docs/scene-format.md`, the TypeScript mirror
  (`frontend/src/sim/scene.ts`) and `scenes/sandbox`, all in the same change.
- **Stages stay pure functions of a job folder.** A stage may read any earlier stage's outputs and must
  finish by writing its marker file (`jobs.STAGES`). Don't keep state anywhere else.
- **Verify external APIs from documentation.** Meta Model API and VGGT details live in
  [`docs/api-notes.md`](docs/api-notes.md), with dates and sources. Add to it when a live call teaches
  something new.
- **Pin dependencies exactly** (`==` in `pyproject.toml`, exact versions in `package.json`).
- **Model output is untrusted input.** Constrain it with a JSON schema, validate ranges, and fall back to
  material defaults rather than failing.
- **Cost.** Debug GPU work on small inputs. Every Modal function has a timeout. A full evaluation run
  (24 clips) costs about $1–2.

## Running the pipeline

```bash
cd pipeline
uv run modal run modal_app.py --video ../test_videos/desk.mov --name desk     # -> scenes/desk
uv run modal run modal_app.py --job <job_id> --name desk --start physics      # re-run later stages
uv run modal deploy modal_app.py                                              # web API
uv run python scripts/evaluate_objectron.py --per-class 4                     # docs/results.md numbers
```

Filming tips for good scans: matte objects 3–10 cm apart on a plain table, bright indirect light, one
slow full circle at chest height, 20–60 s. Glass, mirrors and polished metal break reconstruction.
