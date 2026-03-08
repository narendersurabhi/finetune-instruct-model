from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


def Field(default: Any = None, default_factory: Callable[[], Any] | None = None):
    if default_factory is not None:
        return field(default_factory=default_factory)
    return field(default=default)


def model_validator(*, mode: str = "after"):
    def decorator(fn):
        fn.__model_validator__ = True
        return fn

    return decorator


class ValidationError(ValueError):
    @classmethod
    def from_exception_data(cls, title: str, line_errors: list[Any]):
        return cls(f"{title}: {line_errors}")


class BaseModel:
    def __init_subclass__(cls) -> None:
        dataclass(cls)

    @classmethod
    def model_validate(cls, data: dict[str, Any]):
        obj = cls(**data)
        for name in dir(obj):
            attr = getattr(obj, name)
            if callable(attr) and getattr(attr, "__model_validator__", False):
                obj = attr()
        return obj

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)
