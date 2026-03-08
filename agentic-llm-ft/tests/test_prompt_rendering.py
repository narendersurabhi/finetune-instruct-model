from schemas import DatasetExample
from prompts import render_training_messages


def test_render_training_messages() -> None:
    ex = DatasetExample(
        id="x",
        system_prompt="sys",
        user_prompt="user",
        available_tools=[{"name": "get_weather"}],
        assistant_plan=["plan"],
        assistant_tool_calls=[{"name": "get_weather", "arguments": {"city": "Seattle"}}],
        tool_results=[],
        assistant_final="final",
    )
    msgs = render_training_messages(ex)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "assistant"
