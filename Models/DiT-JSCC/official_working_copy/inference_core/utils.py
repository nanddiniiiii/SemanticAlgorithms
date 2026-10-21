"""Small model factory retained for checkpoint-compatible inference."""


def get_nnet(name, **kwargs):
    if name != "uvit":
        raise NotImplementedError(f"Unsupported network for public inference: {name}")

    from libs.uvit import UViT

    return UViT(**kwargs)
