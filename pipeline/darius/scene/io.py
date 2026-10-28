"""Reading and writing scene packages and the asset files they hold.

    >>> from darius.scene.io import load_scene
    >>> scene = load_scene("scenes/sandbox")
    >>> mug = scene.object("obj_03")
    >>> [m.vertices.shape for m in scene.collision_meshes(mug)]

Collision geometry is always returned as closed, outward-wound triangle meshes in
the object's body frame (scene axes). Primitive colliders (box, sphere, cylinder)
are tessellated, so mass properties and grasp synthesis need one code path; the
exporters still write primitives as primitives.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .schema import Scene, SceneObject


@dataclass(frozen=True)
class Mesh:
    """A closed triangle mesh: faces wound counter-clockwise seen from outside."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) vertex indices

    def transformed(self, rotation: np.ndarray, translation: np.ndarray | None = None) -> Mesh:
        """The mesh with every vertex mapped to rotation @ v + translation."""
        vertices = self.vertices @ np.asarray(rotation, float).T
        if translation is not None:
            vertices = vertices + np.asarray(translation, float)
        return Mesh(vertices, self.faces)


# ---------- Asset files ----------


def read_obj(path: Path) -> Mesh:
    """`v` and `f` lines of a Wavefront OBJ (1-indexed; `f a/b/c` texture and normal refs ignored).
    Polygons are fanned into triangles."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            ring = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            faces += [[ring[0], ring[k], ring[k + 1]] for k in range(1, len(ring) - 1)]
    return Mesh(np.array(vertices, dtype=np.float64).reshape(-1, 3), np.array(faces, dtype=np.int64).reshape(-1, 3))


def write_obj(path: Path, mesh, comment: str) -> None:
    """Write `mesh` (anything with `vertices` and `faces`) as an OBJ with 1-indexed faces."""
    lines = [f"# {line}" for line in comment.splitlines()] + ["# Body-frame coordinates in meters."]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    Path(path).write_text("\n".join(lines) + "\n")


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY: float32 x y z, uint8 red green blue."""
    vertex = np.empty(
        len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertex["x"], vertex["y"], vertex["z"] = np.asarray(points).T.astype(np.float32)
    vertex["red"], vertex["green"], vertex["blue"] = np.asarray(colors).T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex.tobytes())


# ---------- Primitive shapes as meshes ----------


def box_mesh(half_extents) -> Mesh:
    hx, hy, hz = half_extents
    corners = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float64)
    # Corner index = 4 * (x > 0) + 2 * (y > 0) + (z > 0); two outward triangles per side.
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]
    return Mesh(corners, np.array(faces, dtype=np.int64))


def cylinder_mesh(radius: float, half_height: float, segments: int = 64) -> Mesh:
    """A cylinder along +y (scene.json's convention), as a closed prism."""
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), np.zeros(segments), -radius * np.sin(angles)], axis=1)
    bottom, top = ring + [0, -half_height, 0], ring + [0, half_height, 0]
    vertices = np.concatenate([bottom, top, [[0, -half_height, 0], [0, half_height, 0]]])
    b_center, t_center = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(i, j, segments + j), (i, segments + j, segments + i)]  # side
        faces += [(t_center, segments + i, segments + j), (b_center, j, i)]  # caps
    return Mesh(vertices, np.array(faces, dtype=np.int64))


def sphere_mesh(radius: float, subdivisions: int = 3) -> Mesh:
    """An icosphere: an icosahedron with each face split in four, `subdivisions` times."""
    t = (1 + math.sqrt(5)) / 2
    vertices = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t]]
    vertices += [[0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2)]
    faces += [(10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11)]
    faces += [(6, 2, 10), (8, 6, 7), (9, 8, 1)]
    points = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in vertices]

    def midpoint(a: int, b: int, cache: dict[tuple[int, int], int]) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            middle = points[a] + points[b]
            points.append(middle / np.linalg.norm(middle))
            cache[key] = len(points) - 1
        return cache[key]

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        split = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b, cache), midpoint(b, c, cache), midpoint(c, a, cache)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = split
    return Mesh(np.array(points) * radius, np.array(faces, dtype=np.int64))


# ---------- Scene packages ----------


class ScenePackage:
    """A scene package on disk: the validated manifest plus access to its asset files."""

    def __init__(self, root: Path, manifest: Scene):
        self.root = Path(root)
        self.manifest = manifest

    @classmethod
    def load(cls, root: str | Path) -> ScenePackage:
        root = Path(root)
        return cls(root, Scene.model_validate_json((root / "scene.json").read_text()))

    @property
    def objects(self) -> list[SceneObject]:
        return self.manifest.objects

    def object(self, object_id: str) -> SceneObject:
        for obj in self.manifest.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(f"no object {object_id!r} in {self.root}")

    def collision_meshes(self, obj: SceneObject) -> list[Mesh]:
        """The object's collision geometry in its body frame (scene axes)."""
        collider = obj.collider
        if collider.type == "box":
            return [box_mesh(collider.half_extents)]
        if collider.type == "sphere":
            return [sphere_mesh(collider.radius)]
        if collider.type == "cylinder":
            return [cylinder_mesh(collider.radius, collider.half_height)]
        if collider.type == "convex_hull":
            return [self._mesh(collider.file)]
        return [self._mesh(file) for file in collider.files]

    def hull(self, obj: SceneObject) -> Mesh:
        """One convex hull of the whole object: the pick hull if the package has one."""
        if obj.pick_hull:
            return self._mesh(obj.pick_hull)
        from ..core import geometry

        points = np.concatenate([m.vertices for m in self.collision_meshes(obj)])
        hull = geometry.convex_hull(points, max_vertices=256)
        return Mesh(hull.vertices, hull.faces)

    def table_outline(self) -> np.ndarray:
        """The table top's outline as (N, 2) [x, z] scene points, counter-clockwise seen from +y."""
        plane = self.manifest.background.table_plane
        if plane.polygon_m:
            return np.array(plane.polygon_m, dtype=np.float64)
        (px, _, pz), (sx, sz) = plane.point, plane.size_m
        return np.array(
            [
                [px - sx / 2, pz + sz / 2],
                [px + sx / 2, pz + sz / 2],
                [px + sx / 2, pz - sz / 2],
                [px - sx / 2, pz - sz / 2],
            ]
        )

    @cached_property
    def _meshes(self) -> dict[str, Mesh]:
        return {}

    def _mesh(self, file: str) -> Mesh:
        if file not in self._meshes:
            self._meshes[file] = read_obj(self.root / file)
        return self._meshes[file]

    def save(self) -> Path:
        """Write the manifest back to scene.json (in the package's compact layout)."""
        path = self.root / "scene.json"
        path.write_text(dumps(self.manifest))
        return path


def load_scene(path: str | Path) -> ScenePackage:
    """Load a scene package folder (or its scene.json)."""
    path = Path(path)
    return ScenePackage.load(path.parent if path.name == "scene.json" else path)


# ---------- scene.json layout ----------

INLINE_WIDTH = 100
_NUMBER = re.compile(r"-?\d+\.0(?![\d])")


def dumps(manifest: Scene) -> str:
    """scene.json text: two-space indents, with short arrays and objects kept on one line.

    Lists of numbers (positions, quaternions, colliders) read as one line each, which
    keeps hand-authored scenes hand-editable and makes diffs of generated ones short.
    Whole numbers are written without a trailing ".0".
    """
    data = manifest.model_dump(mode="json", exclude_none=True)
    return _format(data, 0) + "\n"


def _scalar(value) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return json.dumps(value, ensure_ascii=False)


def _format(value, indent: int) -> str:
    inline = _inline(value)
    if inline is not None and indent * 2 + len(inline) <= INLINE_WIDTH:
        return inline
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k)}: {_format(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        items = [f"{inner}{_format(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return _scalar(value)


def _inline(value) -> str | None:
    """One-line form of scalars, flat lists and small flat objects; None for nested ones."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = _inline(v)
            if text is None or isinstance(v, dict):
                return None
            parts.append(f"{json.dumps(k)}: {text}")
        return "{ " + ", ".join(parts) + " }" if parts else "{}"
    if isinstance(value, list):
        if any(isinstance(v, dict) for v in value):
            return None
        parts = [_inline(v) for v in value]
        if any(p is None for p in parts):
            return None
        return "[" + ", ".join(parts) + "]"  # type: ignore[arg-type]
    return _scalar(value)
