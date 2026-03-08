from __future__ import annotations

from schemas import ToolSpec
from tools.mock_tools import (
    calculator,
    calendar_lookup,
    get_stock_price,
    get_weather,
    retrieve_faq,
    search_docs,
)
from tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        ),
        get_weather,
    )
    registry.register(
        ToolSpec(
            name="get_stock_price",
            description="Get stock quote",
            parameters={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        ),
        get_stock_price,
    )
    registry.register(
        ToolSpec(
            name="calculator",
            description="Perform arithmetic",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        ),
        calculator,
    )
    registry.register(
        ToolSpec(
            name="search_docs",
            description="Search internal docs",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        search_docs,
    )
    registry.register(
        ToolSpec(
            name="retrieve_faq",
            description="Retrieve FAQ answer",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        ),
        retrieve_faq,
    )
    registry.register(
        ToolSpec(
            name="calendar_lookup",
            description="Get calendar events for date",
            parameters={
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        ),
        calendar_lookup,
    )
    return registry
