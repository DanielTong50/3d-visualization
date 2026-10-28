# Physics sandbox

A hand-authored scene package: a table with a heavy hardcover book, a light tissue box, a ceramic mug
(cylinder collider), a rubber ball (sphere) and a rubber doorstop (convex hull from `obj_05_hull.obj`, the
same kind of collider file the pipeline writes).

It is the simulator's reference scene and covers every collider type, so it must keep loading whenever the
format in [`docs/scene-format.md`](../../docs/scene-format.md) changes. Frontend and pipeline tests both
load it. Rebuild its MuJoCo export and bundle with `cd pipeline && uv run darius gallery`.
