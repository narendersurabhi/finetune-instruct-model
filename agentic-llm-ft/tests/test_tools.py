from tools import build_default_registry


def test_tool_schema_validation() -> None:
    registry = build_default_registry()
    assert registry.validate_arguments("get_weather", {"city": "Seattle"})
    assert not registry.validate_arguments("get_weather", {})


def test_mock_tool_execution() -> None:
    registry = build_default_registry()
    result = registry.execute("calculator", {"operation": "multiply", "a": 6, "b": 7})
    assert result["result"] == 42
