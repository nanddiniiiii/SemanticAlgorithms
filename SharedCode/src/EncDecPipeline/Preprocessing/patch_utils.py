"""Dimension guards shared by training and high-resolution evaluation."""


def validate_divisible(height: int, width: int, divisor: int = 16) -> None:
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(f"Image dimensions {(height, width)} must be divisible by {divisor}.")
