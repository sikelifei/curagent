"""Strict response parsing and JSON-schema validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from curagent.core.errors import StrictToolCallError, ToolSchemaError
from curagent.core.types import ModelResponse, ToolCall, ToolSchema


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _strict_json_loads(raw: str) -> Any:
    return json.loads(
        raw,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def strict_parse_response(response: ModelResponse) -> ToolCall:
    """Parse exactly one native call or one whole-document JSON call."""
    if response.protocol == "json":
        if response.tool_calls:
            raise StrictToolCallError("JSON protocol response must not contain native tool calls")
        if not isinstance(response.raw_response, str):
            raise StrictToolCallError("JSON protocol response must be a string")
        try:
            value = _strict_json_loads(response.raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StrictToolCallError(f"response is not one strict JSON object: {exc}") from exc
        return _parse_normalized(value)

    if response.protocol != "native":
        raise StrictToolCallError(f"unsupported model protocol: {response.protocol!r}")
    if len(response.tool_calls) != 1:
        raise StrictToolCallError(
            f"native response must contain exactly one tool call; got {len(response.tool_calls)}"
        )
    value = response.tool_calls[0]
    if not isinstance(value, Mapping):
        raise StrictToolCallError("native tool call must be an object")
    if "function" in value:
        allowed = {"id", "type", "function", "index"}
        extras = set(value) - allowed
        if extras:
            raise StrictToolCallError(f"unknown native tool call fields: {sorted(extras)}")
        if "index" in value and (
            not isinstance(value["index"], int)
            or isinstance(value["index"], bool)
            or value["index"] < 0
        ):
            raise StrictToolCallError("native tool call index must be a non-negative integer")
        function = value.get("function")
        if not isinstance(function, Mapping):
            raise StrictToolCallError("native tool call function must be an object")
        if set(function) != {"name", "arguments"}:
            raise StrictToolCallError("native function must contain only name and arguments")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = _strict_json_loads(arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StrictToolCallError(f"native function arguments are not strict JSON: {exc}") from exc
        call = _parse_normalized({"name": function["name"], "arguments": arguments})
        return ToolCall(
            name=call.name,
            arguments=call.arguments,
            provider_call_id=str(value["id"]) if value.get("id") is not None else None,
        )
    return _parse_normalized(value)


def _parse_normalized(value: Any) -> ToolCall:
    if not isinstance(value, Mapping):
        raise StrictToolCallError("tool call must be a JSON object")
    if set(value) != {"name", "arguments"}:
        raise StrictToolCallError("tool call must contain only name and arguments")
    name = value["name"]
    arguments = value["arguments"]
    if not isinstance(name, str) or not name:
        raise StrictToolCallError("tool name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise StrictToolCallError("tool arguments must be an object")
    _validate_json_value(arguments, path="arguments")
    return ToolCall(name=name, arguments=dict(arguments))


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise StrictToolCallError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictToolCallError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise StrictToolCallError(f"{path} contains non-JSON value {type(value).__name__}")


def validate_tool_call(call: ToolCall, schemas: Sequence[ToolSchema]) -> ToolSchema:
    matches = [schema for schema in schemas if schema.name == call.name]
    if not matches:
        raise ToolSchemaError(f"unknown tool: {call.name}")
    if len(matches) != 1:
        raise ToolSchemaError(f"tool name is not unique: {call.name}")
    schema = matches[0]
    _validate_value(call.arguments, schema.parameters, path="arguments")
    return schema


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ToolSchemaError(f"{path} must be one of {schema['enum']!r}")
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            raise ToolSchemaError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolSchemaError(f"{path} missing required fields: {missing}")
        if schema.get("additionalProperties", True) is False:
            extras = set(value) - set(properties)
            if extras:
                raise ToolSchemaError(f"{path} has unknown fields: {sorted(extras)}")
        for name, item in value.items():
            if name in properties:
                _validate_value(item, properties[name], path=f"{path}.{name}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise ToolSchemaError(f"{path} must be an array")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolSchemaError(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolSchemaError(f"{path} must contain at most {schema['maxItems']} items")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_value(item, item_schema, path=f"{path}[{index}]")
        return
    if expected == "string" and not isinstance(value, str):
        raise ToolSchemaError(f"{path} must be a string")
    if expected == "string" and "minLength" in schema and len(value) < schema["minLength"]:
        raise ToolSchemaError(f"{path} must contain at least {schema['minLength']} characters")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ToolSchemaError(f"{path} must be an integer")
    if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ToolSchemaError(f"{path} must be a number")
    if expected == "boolean" and not isinstance(value, bool):
        raise ToolSchemaError(f"{path} must be a boolean")
    if expected == "null" and value is not None:
        raise ToolSchemaError(f"{path} must be null")


def framework_tool_schemas(*, python_enabled: bool) -> list[ToolSchema]:
    spec = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "context": {},
            "expected_output": {"type": "string"},
        },
        "required": ["task", "context"],
        "additionalProperties": False,
    }
    schemas = [
        ToolSchema(
            name="spawn_agent",
            description="Run one recursive child and wait for its final SubagentResult.",
            parameters=spec,
        ),
        ToolSchema(
            name="spawn_agents",
            description="Run children sequentially in input order and return their results.",
            parameters={
                "type": "object",
                "properties": {"specs": {"type": "array", "items": spec, "minItems": 1}},
                "required": ["specs"],
                "additionalProperties": False,
            },
        ),
        ToolSchema(
            name="finish",
            description="Return this node's final serializable result.",
            parameters={
                "type": "object",
                "properties": {"result": {}},
                "required": ["result"],
                "additionalProperties": False,
            },
        ),
    ]
    if python_enabled:
        schemas.append(
            ToolSchema(
                name="python_exec",
                description="Execute the exact supplied Python as isolated pure computation.",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            )
        )
    return schemas
