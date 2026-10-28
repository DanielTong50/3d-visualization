# Contributing

## Layout

```
engine/              Python 3.11 (uv): the darius package, the HTTP API and its Modal deployment
  darius/            layered: core, assembly, furniture, models, perception, eval, service
  modal_app.py       the API on Modal, and a cloud entrypoint that decodes a video
  tests/             pytest: graph, tracker, library, perception (model stubbed), API
frontend/            React 19 + three.js (Vite, TypeScript): home, copilot, manual reader, video analyzer, live
  scripts/capture.mjs   renders manual pages, session videos and posters (Playwright, ffmpeg)
assemblies/          the plan library and everything measured on it (see "The library" below)
docs/                plan.schema.json (generated), verified API notes, README media
```

## Setup

```bash
cp .env.example .env            # MODEL_API_KEY (see the comments in the file)
cd engine && uv sync            # exact pins from uv.lock
cd ../frontend && npm ci
```

The API runs on Modal: `uv run modal token new` once, then
`uv run modal secret create darius --from-dotenv ../.env` and `uv run modal deploy modal_app.py`.

## Checks

| | Command (from its folder) |
|---|---|
| Engine format, lint | `uv run ruff format --check . && uv run ruff check .` |
| Engine tests | `uv run pytest` (no network: model calls are stubbed) |
| Library plans are valid | `uv run darius validate ../assemblies/*/plan.json` |
| The exported library matches the catalog | `uv run darius export` leaves `git status` clean |
| Stored model results re-score to the published numbers | `uv run darius eval --check` |
| Schema in sync | `uv run darius schema > ../docs/plan.schema.json` leaves `git status` clean |
| Web format, types, tests, build | `npm run format:check && npm run typecheck && npm test && npm run build` |

CI runs these on every push and pull request.

## The library

`assemblies/<id>/` holds everything about one product:

| File | Written by |
|---|---|
| `plan.json` | `darius export` (from `engine/darius/furniture/catalog.py`) |
| `manual/page-NN.png` | `node scripts/capture.mjs --manuals` |
| `session/session.mp4`, `session/truth.json` | `node scripts/capture.mjs --sessions` |
| `manual/reading.json` | `darius eval --manuals` (calls the model) |
| `session/analysis.json` | `darius eval --sessions` (calls the model) |
| `../benchmarks.json` | `darius eval` (any run) |

Change the catalog, then re-export, re-capture and re-evaluate in that order. `darius eval --replay` re-scores the
stored model outputs after a change to `to_plan`, the tracker or the scoring, without calling the model.

## Conventions

- **The plan schema is a contract.** Change `engine/darius/assembly/schema.py` first, then regenerate
  `docs/plan.schema.json` and update the TypeScript mirror (`frontend/src/engine/plan.ts`) in the same change.
- **The tracker exists twice.** `engine/darius/assembly/tracking.py` and `frontend/src/engine/tracking.ts` must keep
  the same constants and math; both have tests.
- **Respect the layers** (see `engine/darius/__init__.py`): a layer imports only from the layers below it, and
  `__init__` modules stay import-free.
- **Model output is untrusted input.** Constrain it with a JSON schema, then build and validate the structure in
  code (`to_plan`, `to_observation`); never let a reading become a plan without `validate`.
- **Every published number is a stored run.** Do not hand-edit `reading.json`, `analysis.json` or
  `benchmarks.json`; `darius eval --check` fails if a stored score no longer matches its outputs.
- **Verify external APIs from documentation.** Meta Model API details live in [`docs/api-notes.md`](docs/api-notes.md)
  with dates and sources. Add to it when a live call teaches something new.
- **Pin dependencies exactly** (`==` in `pyproject.toml`, exact versions in `package.json`).
- **Cost.** A full `darius eval` is about 56 model calls (4 manuals, 52 frame batches of 4 frames plus the step
  pages) and costs about $2 at Muse Spark 1.3 prices.

## Filming a build

Live mode and `darius track` take real video. Film from chest height with the whole work area in view, keep the
phone mostly still, and make sure each part's letter-side is visible when it goes on. Bright, even light helps.
