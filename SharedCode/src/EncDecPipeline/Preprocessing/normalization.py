"""Stage 1A uses RGB tensors in [0, 1], explicitly without ImageNet normalization."""


def identity_normalize(tensor: object) -> object:
    return tensor
