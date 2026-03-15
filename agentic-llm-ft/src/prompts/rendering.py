from __future__ import annotations

from schemas import DatasetExample


def render_training_messages(example: DatasetExample) -> list[dict[str, str]]:
    assistant_json = {
        "plan": example.assistant_plan,
        "tool_calls": example.assistant_tool_calls,
        "final_answer": example.assistant_final,
    }
    return [
        {"role": "system", "content": example.system_prompt},
        {"role": "user", "content": example.user_prompt},
        {"role": "system", "content": f"Available tools: {example.available_tools}"},
        {"role": "assistant", "content": str(assistant_json).replace("'", '"')},
    ]


def render_eval_messages(example: DatasetExample) -> list[dict[str, str]]:
    """Build input messages for a single eval example (no assistant turn).
    Matches the agent/inference format so the same model interface can be used."""
    return [
        {"role": "system", "content": example.system_prompt},
        {"role": "user", "content": example.user_prompt},
        {"role": "system", "content": f"Available tools: {example.available_tools}"},
    ]
