from INFRA.errors import ComponentUnavailableError


class ProductVQ:
    def __init__(self) -> None:
        raise ComponentUnavailableError("ProductVQ", "Stage 1A", "Only ordinary K=256 MiniBatchKMeans VQ is locked.")
