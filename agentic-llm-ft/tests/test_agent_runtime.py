from pathlib import Path

from agent import AgentRuntime
from tools import build_default_registry


class TwoStepModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[dict]) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"plan": ["call weather"], "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}], "final_answer": null}'
        return '{"plan": ["done"], "tool_calls": [], "final_answer": "Seattle is sunny."}'


def test_agent_runtime_loop(tmp_path: Path) -> None:
    runtime = AgentRuntime(build_default_registry(), TwoStepModel(), tmp_path)
    answer = runtime.run("sys", "weather?")
    assert "Seattle" in answer
    assert (tmp_path / "agent_traces.jsonl").exists()
