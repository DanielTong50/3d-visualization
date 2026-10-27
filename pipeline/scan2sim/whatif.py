""" "What if" scenarios: plain English in, simulator actions out, via Muse Spark tool calling.

The model never touches the simulation. It sees a compact description of the
scene and returns function calls against a small action API; the browser
validates and executes them. The tool schemas below are that API's contract,
mirrored by frontend/src/sim/actions.ts.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from pydantic import BaseModel, Field

from . import meta_api

SYSTEM = """You control a rigid-body physics simulation of a real tabletop that was scanned from video.
Turn the user's scenario into calls to the provided tools. Coordinates: meters, +y is up,
+x is the viewer's right, -z is away from the viewer, the table top is y = 0.
Use the object ids exactly as given. Prefer physically plausible magnitudes: a flick is about
1 m/s, a hard shove 3 m/s, a throw 5 m/s. If the scenario needs several things at once, call
several tools. After the tool calls, add one short sentence describing what will happen."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "apply_impulse",
        "description": "Give one object a sudden change in velocity, like a push, flick, or throw.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "Object id, e.g. 'obj_02'"},
                "direction": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Direction [x, y, z]; it is normalized",
                },
                "speed_m_s": {"type": "number", "description": "Velocity change in m/s, 0 to 8"},
            },
            "required": ["object_id", "direction", "speed_m_s"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_property",
        "description": "Change one physical property of an object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "property": {"type": "string", "enum": ["mass_kg", "friction", "restitution", "material"]},
                "value": {"type": "string", "description": "A number, or a material name for 'material'"},
            },
            "required": ["object_id", "property", "value"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "bump_table",
        "description": "Jolt the whole table sideways, as if someone bumped into it.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "strength": {"type": "string", "enum": ["light", "medium", "hard"]},
            },
            "required": ["direction", "strength"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "tilt_table",
        "description": "Tilt the table, so objects may slide or roll downhill.",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {"type": "number", "description": "-45 to 45"},
                "axis": {
                    "type": "string",
                    "enum": ["x", "z"],
                    "description": "'z' tilts left/right, 'x' tilts toward/away",
                },
            },
            "required": ["degrees", "axis"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_gravity",
        "description": "Switch gravity to a preset.",
        "parameters": {
            "type": "object",
            "properties": {"preset": {"type": "string", "enum": ["earth", "moon", "mars", "jupiter", "zero"]}},
            "required": ["preset"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "drop_object",
        "description": "Lift an object straight up and let it fall.",
        "parameters": {
            "type": "object",
            "properties": {"object_id": {"type": "string"}, "height_m": {"type": "number", "description": "0.05 to 2"}},
            "required": ["object_id", "height_m"],
            "additionalProperties": False,
        },
    },
]


class SceneObjectSummary(BaseModel):
    id: str
    label: str
    material: str
    mass_kg: float
    position: list[float] = Field(min_length=3, max_length=3)


class WhatIfRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    objects: list[SceneObjectSummary] = Field(max_length=32)


class Action(BaseModel):
    name: str
    arguments: dict[str, Any]


class WhatIfResponse(BaseModel):
    actions: list[Action]
    narration: str | None


TOOL_NAMES = {tool["name"] for tool in TOOLS}


def plan(request: WhatIfRequest) -> WhatIfResponse:
    client = openai.OpenAI(base_url=meta_api.BASE_URL, api_key=meta_api.api_key(), max_retries=2)
    scene = "\n".join(
        f"- {o.id}: {o.label} ({o.material}, {o.mass_kg * 1000:.0f} g) at [{', '.join(f'{v:.2f}' for v in o.position)}]"
        for o in request.objects
    )
    response = client.responses.create(
        model=meta_api.MUSE_SPARK_MODEL,
        instructions=SYSTEM,
        input=f"Objects on the table:\n{scene}\n\nScenario: {request.prompt}",
        tools=TOOLS,
        reasoning={"effort": "low"},
    )
    known_ids = {o.id for o in request.objects}
    actions = []
    for item in response.output:
        if item.type != "function_call" or item.name not in TOOL_NAMES:
            continue
        try:
            arguments = json.loads(item.arguments)
        except json.JSONDecodeError:
            continue
        if "object_id" in arguments and arguments["object_id"] not in known_ids:
            continue
        actions.append(Action(name=item.name, arguments=arguments))
    narration = (response.output_text or "").strip() or None
    return WhatIfResponse(actions=actions, narration=narration)
