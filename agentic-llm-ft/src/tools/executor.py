from __future__ import annotations

from tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, name: str, arguments: dict) -> dict:
        return self.registry.execute(name=name, arguments=arguments)
