from __future__ import annotations

from typing import Any

from schemas import DatasetExample
from tools import ToolRegistry


class DatasetValidationError(ValueError):
    pass


def validate_example(example: DatasetExample, registry: ToolRegistry) -> None:
    for tool in example.available_tools:
        if tool.get("name") not in {s.name for s in registry.list_specs()}:
            raise DatasetValidationError(f"Unknown tool in available_tools: {tool.get('name')}")

    for call in example.assistant_tool_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in {s.name for s in registry.list_specs()}:
            raise DatasetValidationError(f"Unknown tool in assistant_tool_calls: {name}")
        if not registry.validate_arguments(name, args):
            raise DatasetValidationError(f"Invalid arguments for tool: {name}")

    if not example.assistant_final:
        raise DatasetValidationError("Missing final answer")

    for result in example.tool_results:
        if "name" not in result or "result" not in result:
            raise DatasetValidationError("Malformed tool result item")
