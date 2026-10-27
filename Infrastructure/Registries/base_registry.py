"""Small explicit registry used for dependency injection and configuration lookup."""

from collections.abc import Callable
from typing import Generic, TypeVar


ComponentT = TypeVar("ComponentT")


class ComponentRegistry(Generic[ComponentT]):
    """Maps stable names to constructors; duplicate registration is an error."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._constructors: dict[str, Callable[..., ComponentT]] = {}

    def register(self, name: str, constructor: Callable[..., ComponentT]) -> None:
        if not name or name.strip() != name:
            raise ValueError(f"Invalid {self.category} registry name: {name!r}")
        if name in self._constructors:
            raise ValueError(f"{self.category} component is already registered: {name}")
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: object) -> ComponentT:
        try:
            constructor = self._constructors[name]
        except KeyError as error:
            choices = ", ".join(sorted(self._constructors)) or "<none>"
            raise KeyError(f"Unknown {self.category} component {name!r}; available: {choices}") from error
        return constructor(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))

    def contains(self, name: str) -> bool:
        return name in self._constructors
