from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent.output_parser import OutputParser
from tools import ToolExecutor, ToolRegistry

ModelFn = Callable[[list[dict]], str]


class AgentRuntime:
    def __init__(self, registry: ToolRegistry, model_fn: ModelFn, run_dir: Path) -> None:
        self.registry = registry
        self.model_fn = model_fn
        self.parser = OutputParser()
        self.executor = ToolExecutor(registry)
        self.trace_path = run_dir / "agent_traces.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, system_prompt: str, user_prompt: str, max_steps: int = 3) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": f"Available tools: {self.registry.render_for_prompt()}"},
        ]

        for _ in range(max_steps):
            raw_output = self.model_fn(messages)
            parsed = self.parser.parse(raw_output)
            tool_results = []
            for call in parsed.tool_calls:
                result = self.executor.run(call.name, call.arguments)
                tool_results.append({"name": call.name, "result": result})
                messages.append({"role": "tool", "content": json.dumps({"name": call.name, "result": result})})

            self._trace(raw_output, parsed.model_dump(), tool_results)
            if parsed.final_answer:
                return parsed.final_answer

        return "Unable to produce final answer within step budget."

    def _trace(self, raw_output: str, parsed_json: dict, tool_results: list[dict]) -> None:
        record = {
            "raw_model_output": raw_output,
            "parsed_json": parsed_json,
            "tool_calls": parsed_json.get("tool_calls", []),
            "tool_results": tool_results,
            "final_answer": parsed_json.get("final_answer"),
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
