"""Current-input semantic evidence contract. Stage 1B implementation is deferred."""

from INFRA.errors import ComponentUnavailableError


class Stage1BParameterExtractor:
    def __init__(self) -> None:
        raise ComponentUnavailableError(
            "TransformerParam extractor",
            "Stage 1A",
            "Current-input semantic parameter inference begins in Stage 1B.",
        )
