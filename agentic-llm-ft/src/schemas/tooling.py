from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ToolSpec(BaseModel):
    """Declarative tool specification used for prompting and validation."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters_schema(self) -> "ToolSpec":
        schema_type = self.parameters.get("type")
        if schema_type != "object":
            raise ValueError("Tool parameters schema must be a JSON object schema with type='object'.")
        if "properties" not in self.parameters:
            raise ValueError("Tool parameters schema must include properties.")
        return self


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ParsedAssistantOutput(BaseModel):
    plan: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None

    @classmethod
    def model_validate(cls, data: dict[str, Any]):
        calls = [ToolCall.model_validate(c) for c in data.get("tool_calls", [])]
        return cls(
            plan=data.get("plan", []),
            tool_calls=calls,
            final_answer=data.get("final_answer"),
        )
