from agent import OutputParser


def test_json_output_parsing() -> None:
    parser = OutputParser()
    parsed = parser.parse('{"plan": ["p"], "tool_calls": [], "final_answer": "ok"}')
    assert parsed.final_answer == "ok"
