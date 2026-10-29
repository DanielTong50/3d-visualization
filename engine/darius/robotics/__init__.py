"""Robot-learning scaffolding built from exact LEGO plans.

Exports contain synthetic Cartesian reference trajectories, never measurements or
commands for a physical arm. Hardware execution intentionally has no adapter here.
"""

from .dataset import export_dataset, training_bundle

__all__ = ["export_dataset", "training_bundle"]
