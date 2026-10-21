"""JPEG byte codec with no hidden image resizing or quality adaptation."""

from __future__ import annotations

from io import BytesIO
from typing import Any


class JPEGCodec:
    def __init__(self, quality: int) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100].")
        self.quality = quality

    def encode(self, image: Any) -> bytes:
        stream = BytesIO()
        image.convert("RGB").save(
            stream,
            format="JPEG",
            quality=self.quality,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        return stream.getvalue()

    @staticmethod
    def decode(payload: bytes) -> Any:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            return image.convert("RGB").copy()
