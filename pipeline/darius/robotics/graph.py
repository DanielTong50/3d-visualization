"""A 3D scene graph: what is in the scene, what holds what up, what is near what, and where
each object can be grasped.

3D scene graphs are how robot planners and language-driven agents reason about a place: nodes
for surfaces, objects and their parts, edges for the relations between them (Armeni et al.,
2019; Hughes et al., Hydra, 2022; Gu et al., ConceptGraphs, 2023). Darius derives one from the
compiled scene, in the robotics world frame (+z up, meters, table top at z = 0):

nodes
    frame     the world frame
    surface   the table top
    object    one per rigid body, at its center of mass: label, material, mass, size, friction
    part      each convex collision part, at its centroid (the object's collision skeleton)
    grasp     each grasp candidate, at the midpoint between the fingertips
    camera    the viewpoints the scene was filmed from

edges
    contains   frame -> surface
    supports   surface -> object, or object -> object when stacked: the upper one's base is
               within SUPPORT_GAP_M of the lower one's top (or the table) and their footprints overlap
    near       object -- object: the gap between their collision hulls is under NEAR_M
               (distance_m: 0 when the hulls interpenetrate, else the closest pair of hull
               vertices, an upper bound on the true gap)
    part_of    part -> object
    grasp_of   grasp -> object
    observes   camera -> object: the object's center projects into the camera's image

The graph is written to robotics/scene_graph.json and drawn by the web app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import frames
from .model import Body, RobotScene

FORMAT = "darius.scene_graph/1"
SUPPORT_GAP_M = 0.025
NEAR_M = 0.10


@dataclass
class SceneGraph:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, id: str, type: str, position, **attributes) -> None:
        self.nodes.append({"id": id, "type": type, "position": _r(position), **attributes})

    def add_edge(self, source: str, target: str, relation: str, **attributes) -> None:
        self.edges.append({"source": source, "target": target, "relation": relation, **attributes})

    def relations(self, relation: str) -> list[tuple[str, str]]:
        return [(e["source"], e["target"]) for e in self.edges if e["relation"] == relation]

    def as_dict(self, scene: RobotScene) -> dict:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node["type"]] = counts.get(node["type"], 0) + 1
        return {
            "format": FORMAT,
            "scene_id": scene.scene_id,
            "title": scene.title,
            "frame": "world: +z up, meters, the table top is z = 0",
            "counts": {"nodes": counts, "edges": len(self.edges)},
            "nodes": self.nodes,
            "edges": self.edges,
        }


def _r(values, digits: int = 5) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in np.ravel(values)]


def _world_vertices(body: Body) -> np.ndarray:
    """The body's collision geometry as world-frame points (primitives sampled on their surface)."""
    rotation = frames.quat_to_matrix(body.quat)
    points = []
    for shape in body.shapes:
        if shape.kind == "mesh":
            points.append(shape.mesh.vertices)
        elif shape.kind == "box":
            hx, hy, hz = shape.size
            points.append(np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]))
        elif shape.kind == "sphere":
            r = shape.size[0]
            points.append(np.vstack([np.eye(3) * r, -np.eye(3) * r]))
        else:
            r, h = shape.size
            angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ring = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(16)])
            points.append(np.vstack([ring + [0, 0, -h], ring + [0, 0, h]]))
    return np.vstack(points) @ rotation.T + body.position


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Gap between two convex hulls given by their vertices: 0 if one reaches inside the other,
    else the smallest vertex-to-vertex distance (an upper bound on the true gap)."""
    from scipy.spatial import Delaunay, QhullError, cKDTree

    try:
        if (Delaunay(a).find_simplex(b) >= 0).any() or (Delaunay(b).find_simplex(a) >= 0).any():
            return 0.0
    except QhullError:  # a flat hull has no inside
        pass
    distance, _ = cKDTree(a).query(b, k=1)
    return float(distance.min())


def _footprints_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    (alo, ahi), (blo, bhi) = (a[:, :2].min(0), a[:, :2].max(0)), (b[:, :2].min(0), b[:, :2].max(0))
    return bool(np.all(alo <= bhi) and np.all(blo <= ahi))


def _sees(camera, point: np.ndarray) -> bool:
    """Whether `point` projects into the camera's image (OpenGL camera: looking along -z)."""
    local = frames.quat_to_matrix(camera.quat).T @ (point - camera.position)
    depth = -local[2]
    if depth <= 0.01:
        return False
    tan_y = math.tan(math.radians(camera.fovy_deg) / 2)
    tan_x = tan_y * camera.width / camera.height
    return abs(local[0] / depth) <= tan_x and abs(local[1] / depth) <= tan_y


def build(scene: RobotScene) -> SceneGraph:
    graph = SceneGraph()
    table_center = np.append(scene.table_outline.mean(axis=0), 0.0)
    area = 0.5 * abs(
        float(
            np.sum(
                scene.table_outline[:, 0] * np.roll(scene.table_outline[:, 1], -1)
                - np.roll(scene.table_outline[:, 0], -1) * scene.table_outline[:, 1]
            )
        )
    )
    graph.add_node("world", "frame", [0, 0, 0], label="world frame")
    graph.add_node("table", "surface", table_center, label="table top", area_m2=round(area, 4))
    graph.add_edge("world", "table", "contains")

    hulls = {body.id: _world_vertices(body) for body in scene.bodies}
    for body in scene.bodies:
        rotation = frames.quat_to_matrix(body.quat)
        com = body.position + rotation @ body.mass.com_m
        graph.add_node(
            body.id,
            "object",
            com,
            label=body.label,
            material=body.material,
            mass_kg=round(body.mass.mass_kg, 4),
            dimensions_m=_r(body.dimensions_m, 4),
            friction=body.friction,
            restitution=body.restitution,
            mesh=f"meshes/{body.id}_visual.obj" if body.visual is not None else None,
            pose={"position": _r(body.position), "rotation_quat": _r(body.quat, 6)},
        )
        for k, shape in enumerate(body.shapes):
            if shape.kind != "mesh" or len(body.shapes) == 1:
                continue
            centroid = body.position + rotation @ shape.mesh.vertices.mean(axis=0)
            part = f"{body.id}/part_{k}"
            graph.add_node(part, "part", centroid, label=f"convex part {k}", mesh=f"meshes/{shape.name}.obj")
            graph.add_edge(part, body.id, "part_of")
        for k, grasp in enumerate(body.grasps):
            node = f"{body.id}/grasp_{k}"
            graph.add_node(
                node,
                "grasp",
                body.position + rotation @ grasp.position,
                label=f"grasp {k}",
                width_m=round(grasp.width_m, 4),
                score=round(grasp.score, 3),
                approach=_r(rotation @ grasp.approach, 4),
                closing=_r(rotation @ grasp.closing, 4),
            )
            graph.add_edge(node, body.id, "grasp_of")

    # Support: the table, or a lower object whose top meets this one's base.
    for body in scene.bodies:
        points = hulls[body.id]
        base = float(points[:, 2].min())
        supporters = [
            other.id
            for other in scene.bodies
            if other.id != body.id
            and base > SUPPORT_GAP_M
            and abs(base - float(hulls[other.id][:, 2].max())) <= SUPPORT_GAP_M
            and _footprints_overlap(points, hulls[other.id])
        ]
        if supporters:
            for supporter in supporters:
                graph.add_edge(supporter, body.id, "supports")
        elif base <= SUPPORT_GAP_M:
            graph.add_edge("table", body.id, "supports")

    for i, a in enumerate(scene.bodies):
        for b in scene.bodies[i + 1 :]:
            gap = _gap(hulls[a.id], hulls[b.id])
            if gap < NEAR_M:
                graph.add_edge(a.id, b.id, "near", distance_m=round(gap, 4))

    for camera in scene.cameras:
        graph.add_node(camera.name, "camera", camera.position, label=camera.name.replace("_", " "))
        for body in scene.bodies:
            if _sees(camera, body.position):
                graph.add_edge(camera.name, body.id, "observes")
    return graph
