"""Errors used to keep deferred research components explicit."""


class ComponentUnavailableError(NotImplementedError):
    """Raised instead of returning fabricated output from a deferred component."""

    def __init__(self, component: str, stage: str, reason: str) -> None:
        self.component = component
        self.stage = stage
        self.reason = reason
        super().__init__(f"{component} is unavailable in {stage}: {reason}")
