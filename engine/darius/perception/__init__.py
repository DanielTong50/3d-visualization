"""Vision: reading a manual into a plan, and camera frames into progress through it.

- `manual`: manual pages -> `ManualReading` (what the pages say) -> `AssemblyPlan` (a graph).
- `frames`: camera frames + a plan -> per-frame readings of which steps are done.
- `video`: a whole video -> sampled frames -> readings -> a decoded timeline and the assembly
  graph as it was built, joint by joint.

The models are called through `darius.models.meta_api`; everything after the model call
(turning readings into plans and observations, decoding) is deterministic and tested.
"""
