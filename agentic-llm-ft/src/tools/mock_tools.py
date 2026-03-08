from __future__ import annotations

from typing import Any


def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    city = args["city"]
    unit = args.get("unit", "celsius")
    temp = {"celsius": 22, "fahrenheit": 71}.get(unit, 22)
    return {"city": city, "temperature": temp, "unit": unit, "condition": "sunny"}


def get_stock_price(args: dict[str, Any]) -> dict[str, Any]:
    ticker = args["ticker"].upper()
    prices = {"AAPL": 213.42, "MSFT": 427.15, "GOOG": 176.88}
    return {"ticker": ticker, "price": prices.get(ticker, 100.00), "currency": "USD"}


def calculator(args: dict[str, Any]) -> dict[str, Any]:
    operation = args["operation"]
    a, b = args["a"], args["b"]
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else None}
    return {"operation": operation, "result": ops.get(operation)}


def search_docs(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    return {"query": query, "hits": [{"title": "Agent Tools Guide", "score": 0.91}]}


def retrieve_faq(args: dict[str, Any]) -> dict[str, Any]:
    topic = args["topic"]
    faqs = {
        "billing": "Billing cycles start on the first day of each month.",
        "security": "All data is encrypted at rest and in transit.",
    }
    return {"topic": topic, "answer": faqs.get(topic, "No FAQ found")}


def calendar_lookup(args: dict[str, Any]) -> dict[str, Any]:
    date = args["date"]
    return {"date": date, "events": ["Team standup 10:00", "Roadmap review 14:00"]}
