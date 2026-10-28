"""Collision shapes for scanned objects: one convex hull, or a compound of convex parts.

A single convex hull fills in every concavity: a mug's hull is a solid cylinder,
so nothing can sit inside it and its handle is a slab. Approximate convex
decomposition (CoACD, Wei et al., SIGGRAPH 2022) splits a concave shape into a
few convex pieces that a physics engine can combine into one rigid body.

CoACD needs a closed mesh, and the reconstruction gives points, so the object is
first voxelized and its exposed voxel faces become a watertight surface.
"""

from __future__ import annotations

import numpy as np

from ..core import geometry

MAX_PARTS = 8
CONCAVITY_THRESHOLD = 0.08  # CoACD's stopping criterion; higher = fewer, coarser parts
MIN_PART_VOLUME_SHARE = 0.01
# Use parts only if they leave out at least this share of the single hull's volume.
MIN_EMPTY_SPACE_REMOVED = 0.15


def convex_parts(points: np.ndarray, voxel: float) -> list[geometry.Hull]:
    """Convex pieces covering `points` (object-local meters), at most MAX_PARTS of them.

    Returns a single hull when decomposition fails or doesn't help: a part count of
    one means the object is already close to convex.
    """
    import coacd

    coacd.set_log_level("error")
    vertices, faces = geometry.voxel_surface_mesh(points, voxel, dilate=1)
    try:
        parts = coacd.run_coacd(
            coacd.Mesh(vertices, faces.astype(np.int32)),
            threshold=CONCAVITY_THRESHOLD,
            max_convex_hull=MAX_PARTS,
            preprocess_mode="off",  # the voxel mesh is already manifold
            seed=0,
        )
    except Exception:  # CoACD raises plain RuntimeErrors on degenerate input
        parts = []
    # Voxelization and dilation grow the shape by about a voxel; clamping every part to
    # the object's own bounding box keeps it from poking through the table or its neighbors.
    lo, hi = points.min(axis=0), points.max(axis=0)
    hulls = []
    for part_vertices, _ in parts:
        try:
            hulls.append(geometry.convex_hull(np.clip(np.asarray(part_vertices), lo, hi)))
        except ValueError:
            continue
    total = sum(h.volume for h in hulls)
    hulls = [h for h in hulls if total and h.volume >= MIN_PART_VOLUME_SHARE * total]
    return hulls or [geometry.convex_hull(points)]


def hull_coverage(parts: list[geometry.Hull], hull: geometry.Hull, samples: int = 20_000, seed: int = 0) -> float:
    """Fraction of the single hull's volume that the parts cover (Monte Carlo)."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    lo, hi = hull.vertices.min(axis=0), hull.vertices.max(axis=0)
    probes = rng.uniform(lo, hi, (samples, 3))
    probes = probes[Delaunay(hull.vertices).find_simplex(probes) >= 0]
    covered = np.zeros(len(probes), dtype=bool)
    for part in parts:
        covered |= Delaunay(part.vertices).find_simplex(probes) >= 0
    return float(covered.mean()) if len(probes) else 1.0


def best_collider(points: np.ndarray, voxel: float) -> tuple[list[geometry.Hull], dict]:
    """Convex parts when they remove real empty space from the hull, else the single hull.

    Scanned interiors are often filled in by depth noise (a mug reconstructed from
    outside looks solid), in which case a decomposition only adds cost. The decision
    and the measured coverage go into the diagnostics.
    """
    hull = geometry.convex_hull(points)
    parts = convex_parts(points, voxel)
    coverage = hull_coverage(parts, hull) if len(parts) > 1 else 1.0
    use_parts = len(parts) > 1 and coverage <= 1 - MIN_EMPTY_SPACE_REMOVED
    return (parts if use_parts else [hull]), {
        "parts": len(parts),
        "hull_coverage": round(coverage, 3),
        "used_parts": use_parts,
    }
