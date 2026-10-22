from INFRA.errors import ComponentUnavailableError


class GroupedVQ:
    def __init__(self) -> None:
        raise ComponentUnavailableError("GroupedVQ", "Stage 1A", "Only ordinary K=256 MiniBatchKMeans VQ is locked.")
