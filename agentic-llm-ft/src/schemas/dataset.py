from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetExample(BaseModel):
    id: str
    system_prompt: str
    user_prompt: str
    assistant_final: str
    available_tools: list[dict[str, Any]] = Field(default_factory=list)
    assistant_plan: list[str] = Field(default_factory=list)
    assistant_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
