"""Darius: LEGO assembly intelligence, from stud-grid plans to robot-learning artifacts.

The LEGO branch compiles exact original builds, discovers public instructions,
produces reviewable vision-language drafts and exports synthetic training data.
The furniture perception pipeline remains available as a historical baseline.

Layers, lowest first. A layer imports only from the layers listed before it:

    darius.core         cross-process errors
    darius.lego         stud-grid IR, compiler, curriculum, official instructions and manual drafts
    darius.robotics     synthetic Cartesian episodes, numeric imitation baseline, MJCF scene scaffold
    darius.assembly     the assembly plan (parts, hardware, joints, steps) as a graph; validation,
                        comparison, plan-conditioned progress tracking, guidance
    darius.furniture    a procedural flat-pack library with exact geometry: ground truth
    darius.models       the client for the foundation model perception calls (Muse Spark)
    darius.perception   manual pages -> plan; camera frames -> evidence; video -> timeline
    darius.eval         benchmarks against the furniture library's ground truth
    darius.service      the HTTP API: library, manuals, frames, videos

    >>> from darius.furniture import catalog
    >>> plan = catalog.build("halden-bookcase")
    >>> [step.title for step in plan.steps]
"""

__version__ = "1.3.0"
