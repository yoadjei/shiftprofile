"""Name -> callable registries.

Every model, dataset, shift, explainer and metric is registered by name so that
configs can refer to them as strings and results can record exactly what ran.
"""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


class Registry:
    """A namespace of named callables that refuses silent overwrites."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            if name in self._items:
                raise ValueError(
                    f"{self.kind} {name!r} is already registered. "
                    "Registering twice silently changes what a config means."
                )
            self._items[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}. Available: {', '.join(self.names())}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


METRICS = Registry("metric")
MODELS = Registry("model")
DATASETS = Registry("dataset")
SHIFTS = Registry("shift")
EXPLAINERS = Registry("explainer")
