import pytest

from tools import ToolArgumentsError, build_default_registry


def test_tool_schema_validation() -> None:
    registry = build_default_registry()
    assert registry.validate_arguments("get_weather", {"city": "Seattle"})
    assert not registry.validate_arguments("get_weather", {})


def test_mock_tool_execution() -> None:
    registry = build_default_registry()
    result = registry.execute("calculator", {"operation": "multiply", "a": 6, "b": 7})
    assert result["result"] == 42


def test_execute_raises_tool_arguments_error_for_invalid_args() -> None:
    registry = build_default_registry()
    with pytest.raises(ToolArgumentsError) as exc_info:
        registry.execute("get_weather", {})
    assert "Missing required" in str(exc_info.value)
    with pytest.raises(ToolArgumentsError) as exc_info:
        registry.execute("get_weather", {"city": "Seattle", "unknown_key": 1})
    assert "Unknown argument" in str(exc_info.value)
