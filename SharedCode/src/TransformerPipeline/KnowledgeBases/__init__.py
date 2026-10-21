"""Historical/context evidence contract. This is a sibling, not a child, of TransformerParam."""

from INFRA.errors import ComponentUnavailableError


class Stage1BKnowledgeBase:
    def __init__(self) -> None:
        raise ComponentUnavailableError(
            "KnowledgeBases subsystem",
            "Stage 1A",
            "Redis, graph, vector and event-history stores are deferred to Stage 1B.",
        )
