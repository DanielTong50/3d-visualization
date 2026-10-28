"""Benchmarks: the perception models against the library's ground truth.

`darius eval` reads every generated manual and watches every rendered session in
`assemblies/`, scores the results against the procedural plans, and writes each result next to
its inputs (`manual/reading.json`, `session/analysis.json`) plus a summary in
`assemblies/benchmarks.json`. The web app replays these files, so every number and every
model output on the site is one of these runs.
"""
