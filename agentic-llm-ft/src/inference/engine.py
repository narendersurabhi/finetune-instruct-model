from __future__ import annotations

import json
from pathlib import Path

from agent import AgentRuntime
from tools import ToolRegistry


class StubModel:
    """Deterministic local model stub used for tests and offline demos."""

    def __call__(self, messages: list[dict]) -> str:
        latest_user = ""
        for message in reversed(messages):
            if message["role"] == "user":
                latest_user = message["content"]
                break
        if "weather" in latest_user.lower():
            return json.dumps(
                {
                    "plan": ["Need weather info", "Call tool", "Answer user"],
                    "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
                    "final_answer": None,
                }
            )
        if any(m["role"] == "tool" for m in messages):
            return json.dumps(
                {"plan": ["Summarize tool results"], "tool_calls": [], "final_answer": "Done."}
            )
        return json.dumps({"plan": ["Answer directly"], "tool_calls": [], "final_answer": "Done."})


def run_agent_inference(registry: ToolRegistry, prompt: str, run_dir: Path) -> str:
    runtime = AgentRuntime(registry=registry, model_fn=StubModel(), run_dir=run_dir)
    return runtime.run(system_prompt="You are an agentic assistant.", user_prompt=prompt)
