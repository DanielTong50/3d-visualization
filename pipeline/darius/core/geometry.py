"""Point-cloud geometry: support-plane fitting, gravity alignment, voxel ops, hulls.

Everything here is plain NumPy/SciPy and deterministic given a seed, so it is
unit-tested directly (see tests/test_geometry.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class Plane:
    """The plane n·x + d = 0 with unit normal n."""

    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset

    def flipped(self) -> Plane:
        return Plane(-self.normal, -self.offset)


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[Plane, np.ndarray]:
    """Fit the dominant plane with RANSAC, then refine on the inliers by least squares.

    Returns the plane and a boolean inlier mask. `threshold` is the inlier distance
    in the cloud's own units.
    """
    if len(points) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    rng = np.random.default_rng(seed)
    best_count, best_plane = -1, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        offset = -float(normal @ sample[0])
        count = int(np.count_nonzero(np.abs(points @ normal + offset) < threshold))
        if count > best_count:
            best_count, best_plane = count, Plane(normal, offset)
    if best_plane is None:
        raise ValueError("points are degenerate (collinear)")
    inliers = np.abs(best_plane.signed_distance(points)) < threshold
    return refine_plane(points[inliers]), inliers


def refine_plane(points: np.ndarray) -> Plane:
    """Total-least-squares plane: the normal is the smallest principal axis."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return Plane(normal, -float(normal @ centroid))


def rotation_aligning(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit vector `source` onto unit vector `target`."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    axis = np.cross(source, target)
    sin, cos = np.linalg.norm(axis), float(source @ target)
    if sin < 1e-9:
        if cos > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to source.
        helper = np.array([1.0, 0, 0]) if abs(source[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    axis /= sin
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sin * k + (1 - cos) * (k @ k)


def gravity_frame(plane: Plane, camera_centers: np.ndarray, table_points: np.ndarray) -> np.ndarray:
    """4x4 transform into a frame where the table is y = 0 and +y points at the cameras.

    The plane normal's sign is ambiguous; cameras filming a tabletop are above it,
    so the normal is oriented toward the mean camera center. The origin is the
    centroid of the table inliers projected onto the plane.
    """
    if np.mean(plane.signed_distance(camera_centers)) < 0:
        plane = plane.flipped()
    rotation = rotation_aligning(plane.normal, np.array([0.0, 1.0, 0.0]))
    center = table_points.mean(axis=0)
    center = center - plane.signed_distance(center[None])[0] * plane.normal
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ center
    return transform


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Integer voxel coordinates, one row per point."""
    return np.floor(points / voxel).astype(np.int64)


def majority_vote(
    keys: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    min_share: float,
    min_votes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel label by majority vote.

    Each point votes for its voxel with its label (0 = background). A voxel takes
    the winning label only if that label has at least `min_share` of the voxel's
    votes and `min_votes` votes; otherwise it is background.

    Returns (unique voxel keys, winning label per voxel, voxel index per point).
    """
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.zeros((len(unique), num_labels), dtype=np.int32)
    np.add.at(counts, (inverse, labels), 1)
    winner = counts.argmax(axis=1)
    total = counts.sum(axis=1)
    top = counts[np.arange(len(unique)), winner]
    accepted = (top >= min_votes) & (top >= min_share * total)
    return unique, np.where(accepted, winner, 0), inverse


def largest_component(points: np.ndarray, voxel: float) -> np.ndarray:
    """Mask of points in the largest 26-connected cluster of occupied voxels."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    keys = voxel_keys(points, voxel)
    keys -= keys.min(axis=0)
    grid = np.zeros(keys.max(axis=0) + 1, dtype=bool)
    grid[tuple(keys.T)] = True
    components, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if count <= 1:
        return np.ones(len(points), dtype=bool)
    point_components = components[tuple(keys.T)]
    sizes = np.bincount(point_components)
    sizes[0] = 0
    return point_components == sizes.argmax()


def remove_statistical_outliers(points: np.ndarray, neighbors: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """Mask of points whose mean neighbor distance is within `std_ratio` sigmas."""
    from scipy.spatial import cKDTree

    if len(points) <= neighbors:
        return np.ones(len(points), dtype=bool)
    distances, _ = cKDTree(points).query(points, k=neighbors + 1)
    mean = distances[:, 1:].mean(axis=1)
    return mean <= mean.mean() + std_ratio * mean.std()


def farthest_point_sample(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Indices of `count` points spread evenly over the cloud."""
    if len(points) <= count:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = rng.integers(len(points))
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, count):
        chosen[i] = int(distance.argmax())
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[i]], axis=1))
    return chosen


# Fraction of an object's height whose footprint is taken as its base (complete_bottom).
SUPPORT_BAND = 0.5


@dataclass(frozen=True)
class Hull:
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3), counter-clockwise seen from outside
    volume: float


HULL_VERTICES = 32


def convex_hull(points: np.ndarray, max_vertices: int = HULL_VERTICES) -> Hull:
    """Convex hull with at most `max_vertices` vertices, outward-wound faces.

    The budget is a physics choice, not a file-size one: a scanned mug with a
    96-vertex hull (many tiny, nearly coplanar faces) never came to rest in Rapier,
    rocking at up to 0.1 m/s indefinitely, while the same mug at 32 vertices settled
    and slept within 3 s. Farthest-point subsampling keeps the silhouette.
    """
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise ValueError(f"cannot build a hull: {error}") from error
    if len(hull.vertices) > max_vertices:
        subset = points[hull.vertices]
        hull = ConvexHull(subset[farthest_point_sample(subset, max_vertices)])
        points = hull.points
    used = np.unique(hull.simplices)
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    vertices = points[used]
    faces = remap[hull.simplices]
    center = vertices.mean(axis=0)
    for i, (a, b, c) in enumerate(faces):
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if normal @ (vertices[a] - center) < 0:
            faces[i] = (a, c, b)
    return Hull(vertices, faces, float(hull.volume))


def footprint_extents(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Gravity-aligned box: (dimensions [x, y, z], yaw) from PCA on the xz footprint.

    y is the height above the table; x and z are the long and short horizontal
    axes after rotating by `yaw` about +y.
    """
    xz = points[:, [0, 2]] - points[:, [0, 2]].mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(xz.T))
    major = vectors[:, -1]
    yaw = float(np.arctan2(-major[1], major[0]))
    cos, sin = np.cos(yaw), np.sin(yaw)
    local_x = xz @ np.array([cos, -sin])
    local_z = xz @ np.array([sin, cos])
    dims = np.array([np.ptp(local_x), np.ptp(points[:, 1]), np.ptp(local_z)])
    return dims, yaw


def complete_bottom(points: np.ndarray, band: float) -> np.ndarray:
    """Add the projection onto y = 0 of every point within `band` of the object's base.

    A camera circling a table barely sees an object's lower edge, so lifted clouds
    float with a sparse, often one-sided bottom. Using only the lowest ring as the
    base makes scanned objects tip over (a mug whose bottom was seen from one side
    gets a base the size of a coin). Callers pass about half the object's height:
    an object resting on a table is supported by roughly the footprint of its
    lower half, which gives the collider a full, flat base at the table.
    """
    base = points[:, 1].min()
    ring = points[points[:, 1] <= base + band].copy()
    ring[:, 1] = 0.0
    return np.concatenate([points, ring])


def table_outline(xz: np.ndarray, cell: float, min_density: int = 3) -> np.ndarray:
    """Counter-clockwise (seen from +y) convex outline of the table top from its inlier points.

    Points are binned on a 2D grid; only the largest 8-connected region of cells with
    at least `min_density` points is kept, so stray coplanar points (a far wall's
    base, floor patches) can't stretch the outline.
    """
    keys = np.floor(xz / cell).astype(np.int64)
    origin = keys.min(axis=0)
    keys -= origin
    counts = np.zeros(keys.max(axis=0) + 1, dtype=np.int32)
    np.add.at(counts, tuple(keys.T), 1)
    dense = counts >= min_density
    regions, count = ndimage.label(dense, structure=np.ones((3, 3)))
    if count == 0:
        raise ValueError("no dense table region")
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    cells = np.argwhere(regions == sizes.argmax())
    corners = np.concatenate([cells + offset for offset in ([0, 0], [1, 0], [0, 1], [1, 1])])
    outline = (corners + origin) * cell
    hull = ConvexHull(outline)
    polygon = outline[hull.vertices]
    # ConvexHull lists 2D vertices counter-clockwise in (x, z). Seen from +y (looking down
    # the -y axis) with right-handed axes, x right and z toward the viewer, that order
    # is clockwise, so reverse it.
    return polygon[::-1]


def voxel_surface_mesh(points: np.ndarray, voxel: float, dilate: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh of the occupied voxels (exposed cube faces), for convex decomposition.

    Occupancy is dilated to close gaps between reconstructed points; cavities larger
    than the dilation (a mug's inside) stay open.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    origin = keys.min(axis=0) - dilate - 1
    keys -= origin
    grid = np.zeros(keys.max(axis=0) + dilate + 2, dtype=bool)
    grid[tuple(keys.T)] = True
    if dilate:
        grid = ndimage.binary_dilation(grid, iterations=dilate)
    vertices: dict[tuple[int, int, int], int] = {}
    faces: list[tuple[int, int, int]] = []

    def vid(corner: tuple[int, int, int]) -> int:
        if corner not in vertices:
            vertices[corner] = len(vertices)
        return vertices[corner]

    # For each axis and direction, emit the face between an occupied and an empty cell.
    for axis in range(3):
        for direction in (-1, 1):
            neighbor = np.roll(grid, -direction, axis=axis)
            exposed = np.argwhere(grid & ~neighbor)
            u, v = [a for a in range(3) if a != axis]
            for cell in exposed:
                base = cell.copy()
                if direction == 1:
                    base[axis] += 1
                quad = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = base.copy()
                    corner[u] += du
                    corner[v] += dv
                    quad.append(vid(tuple(int(c) for c in corner)))
                a, b, c, d = quad
                # Orient outward: the cyclic order (u, v) gives normal +axis when u < v cyclically.
                outward = direction if (v - u) % 3 == 1 else -direction
                faces += [(a, b, c), (a, c, d)] if outward > 0 else [(a, c, b), (a, d, c)]
    corners = np.array(list(vertices), dtype=float)
    return (corners + origin) * voxel, np.array(faces, dtype=np.int64)
