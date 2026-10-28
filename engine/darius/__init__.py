"""Darius: an assembly copilot. It reads a manual into an assembly graph, tracks a build from a
phone camera, and shows the next move in AR.

Layers, lowest first. A layer imports only from the layers listed before it:

    darius.core         cross-process errors
    darius.assembly     the assembly plan (parts, hardware, joints, steps) as a graph; validation,
                        comparison, plan-conditioned progress tracking, guidance
    darius.furniture    a procedural flat-pack library with exact geometry: ground truth
    darius.models       clients for the foundation models perception calls
    darius.perception   manual pages -> plan; camera frame -> observation; video -> timeline
    darius.pipeline     job folders and the stage graph that runs perception in the cloud
    darius.service      the HTTP API: manuals, live sessions, videos
    darius.eval         benchmarks against the furniture library's ground truth

    >>> from darius.furniture import catalog
    >>> plan = catalog.build("halden-bookcase")
    >>> [step.title for step in plan.steps]
"""

__version__ = "1.0.0"
