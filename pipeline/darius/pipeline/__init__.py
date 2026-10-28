"""The compiler: phone video in, scene package out.

Seven stages form a dependency graph (`jobs.STAGES`); a stage starts as soon as
the stages it reads from are done:

    ingest ──┬── reconstruct ──┬── lift ──┬── physics ──┬── package
             └── segment ──────┘          └── splat ────┘

Each stage is a pure function of a job folder (`stages.py`), so any stage can be
re-run alone, locally or in the cloud (`modal_app.py`).
"""
