"""Download and validate the complete Kodak Lossless True Color suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path


BASE_URL = "https://r0k.us/graphics/kodak/kodak"
EXPECTED_BYTES = (
    736501, 617995, 502888, 637432, 785610, 618959,
    566322, 788470, 582899, 593463, 621023, 531024,
    822712, 692201, 612582, 534247, 602078, 780947,
    671476, 492462, 637051, 701970, 557596, 706397,
)
EXPECTED_SHA256 = (
    "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd",
    "4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd",
    "e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db",
    "e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa",
    "10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53",
    "363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18",
    "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4",
    "ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7",
    "6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d",
    "9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23",
    "7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6",
    "d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980",
    "bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08",
    "55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e",
    "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683",
    "a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54",
    "37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b",
    "1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa",
    "b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc",
    "3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a",
    "ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b",
    "1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29",
    "e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf",
    "1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_image(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path.name}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{path.name}: SHA-256 mismatch")
    with path.open("rb") as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path.name}: invalid PNG/IHDR signature")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if (bit_depth, color_type, compression, filtering) != (8, 2, 0, 0):
        raise ValueError(f"{path.name}: expected 8-bit RGB PNG IHDR")
    if (width, height) not in {(768, 512), (512, 768)}:
        raise ValueError(f"{path.name}: unexpected size {(width, height)}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "width": width,
        "height": height,
        "interlace": interlace,
        "url": f"{BASE_URL}/{path.name}",
    }


def download_one(
    destination: Path, expected_bytes: int, expected_sha256: str, retries: int = 3
) -> None:
    if destination.exists():
        try:
            validate_image(destination, expected_bytes, expected_sha256)
            return
        except (OSError, ValueError):
            destination.unlink()
    url = f"{BASE_URL}/{destination.name}"
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Kodak24Benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            validate_image(partial, expected_bytes, expected_sha256)
            os.replace(partial, destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(float(attempt))


def download_kodak24(output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (expected_bytes, expected_sha256) in enumerate(
        zip(EXPECTED_BYTES, EXPECTED_SHA256, strict=True), start=1
    ):
        path = output_dir / f"kodim{index:02d}.png"
        print(f"[{index:02d}/24] {path.name}")
        download_one(path, expected_bytes, expected_sha256)
        records.append(validate_image(path, expected_bytes, expected_sha256))
    manifest = {
        "dataset": "Kodak Lossless True Color Image Suite (PhotoCD PCD0992)",
        "source": "https://r0k.us/graphics/kodak/",
        "image_count": len(records),
        "files": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/Kodak24"))
    args = parser.parse_args()
    records = download_kodak24(args.output.resolve())
    print(f"Validated {len(records)} images in {args.output.resolve()}")


if __name__ == "__main__":
    main()
