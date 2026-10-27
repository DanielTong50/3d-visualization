"""Read Objectron ground truth without the protobuf toolchain.

Objectron (Ahmadyan et al., CVPR 2021) annotates phone videos with metric 3D
boxes from ARKit. We need three things from a `Sequence` message
(objectron/schema/annotation_data.proto): each object's box size
(`Object.scale`, meters) and category, and per frame each object's projected
box center (`ObjectAnnotation.keypoints[0].point_2d`, normalized image
coordinates). A ~40-line wire-format reader covers that.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def fields(data: bytes):
    """Yield (field_number, wire_type, value) for one message."""
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            value, pos = data[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = data[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


def _floats(value: bytes, wire: int) -> list[float]:
    if wire == 2:  # packed
        return list(struct.unpack(f"<{len(value) // 4}f", value))
    return [struct.unpack("<f", value)[0]]


@dataclass
class GroundTruthObject:
    id: int
    category: str
    scale_m: list[float] = field(default_factory=list)  # box size; Objectron boxes are y-up


@dataclass
class Annotation:
    objects: list[GroundTruthObject]
    # frame_id -> {object_id: (x, y)} normalized box-center projection
    centers: dict[int, dict[int, tuple[float, float]]]


def parse_sequence(data: bytes) -> Annotation:
    objects, centers = [], {}
    for number, _wire, value in fields(data):
        if number == 1:  # Object
            obj = GroundTruthObject(0, "")
            for n, w, v in fields(value):
                if n == 1:
                    obj.id = v
                elif n == 2:
                    obj.category = v.decode()
                elif n == 6:
                    obj.scale_m += _floats(v, w)
            objects.append(obj)
        elif number == 2:  # FrameAnnotation
            frame_id, per_object = 0, {}
            for n, _, v in fields(value):
                if n == 1:
                    frame_id = v
                elif n == 2:  # ObjectAnnotation
                    object_id, center = 0, None
                    for n2, _, v2 in fields(v):
                        if n2 == 1:
                            object_id = v2
                        elif n2 == 2 and center is None:  # first AnnotatedKeyPoint = box center
                            for n3, _, v3 in fields(v2):
                                if n3 == 3:  # point_2d
                                    xy = {n4: struct.unpack("<f", v4)[0] for n4, _, v4 in fields(v3) if n4 in (1, 2)}
                                    center = (xy.get(1, 0.0), xy.get(2, 0.0))
                    if center is not None:
                        per_object[object_id] = center
            centers[frame_id] = per_object
    return Annotation(objects, centers)
