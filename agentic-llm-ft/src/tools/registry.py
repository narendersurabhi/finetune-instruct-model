from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from schemas import ToolSpec

ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._impls: dict[str, ToolFn] = {}

    def register(self, spec: ToolSpec, implementation: ToolFn) -> None:
        self._specs[spec.name] = spec
        self._impls[spec.name] = implementation

    def get_spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        return self._specs[name]

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def render_for_prompt(self) -> str:
        as_json = [spec.model_dump() for spec in self.list_specs()]
        return json.dumps(as_json, indent=2)

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> bool:
        spec = self.get_spec(name)
        required = spec.parameters.get("required", [])
        properties = spec.parameters.get("properties", {})

        for field in required:
            if field not in arguments:
                return False
        for key, value in arguments.items():
            if key not in properties:
                return False
            expected = properties[key].get("type")
            if expected == "string" and not isinstance(value, str):
                return False
            if expected == "number" and not isinstance(value, (int, float)):
                return False
        return True

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.validate_arguments(name, arguments):
            raise ValidationError.from_exception_data("ToolArguments", [])
        return self._impls[name](arguments)
