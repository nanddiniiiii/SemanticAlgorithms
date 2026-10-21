"""Reconstruction clipping is an explicit evaluation operation."""


def clamp_reconstruction(tensor: object) -> object:
    return tensor.clamp(0.0, 1.0)
