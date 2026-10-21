"""Tensor-state-only Stage 1A export and strict inference manifest validation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Channels.channel_utils import require_torch
from EncDecPipeline.Models.SwinJSCC.swin_config import SwinJSCCConfig


FULL_FILENAME = "stage1_swinjscc_full.pt"
ENCODER_FILENAME = "stage1_swinjscc_encoder.pt"
DECODER_FILENAME = "stage1_swinjscc_decoder.pt"
MANIFEST_FILENAME = "stage1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_fingerprint() -> dict[str, str]:
    torch = require_torch()
    return {
        "python_full": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pytorch_full": str(torch.__version__),
        "pytorch_major_minor": ".".join(str(torch.__version__).split("+")[0].split(".")[:2]),
        "cuda_runtime": str(torch.version.cuda),
    }


@dataclass(slots=True)
class Stage1ArtifactManifest:
    """Validated manifest required before loading a Stage 1A inference artifact."""

    model_dir: Path
    raw: dict[str, Any]

    @property
    def config(self) -> SwinJSCCConfig:
        return SwinJSCCConfig.from_manifest_dict(self.raw["architecture"])

    @property
    def upstream_root(self) -> Path:
        source_path = Path(self.raw["upstream"]["relative_path"])
        if source_path.is_absolute():
            return source_path
        return Path.cwd() / source_path

    @classmethod
    def load_and_validate(
        cls,
        model_dir: str | Path,
        strict_runtime: bool = True,
        validate_source: bool = True,
    ) -> "Stage1ArtifactManifest":
        directory = Path(model_dir)
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Required inference manifest is absent: {manifest_path}")
        manifest = cls(model_dir=directory, raw=json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest._validate_schema()
        manifest._validate_checkpoint_hashes()
        if strict_runtime:
            manifest._validate_runtime()
        if validate_source:
            manifest._validate_upstream_source()
        return manifest

    def _validate_schema(self) -> None:
        required = {"format_version", "architecture", "upstream", "runtime", "checkpoint_hashes", "data_provenance"}
        absent = required.difference(self.raw)
        if absent:
            raise ValueError(f"Manifest is incomplete; missing fields: {sorted(absent)}")
        if self.raw["format_version"] != 1:
            raise ValueError(f"Unsupported manifest format: {self.raw['format_version']}")
        config = self.config
        if config.channel_type != "awgn" or config.objective.lower() != "mse":
            raise ValueError("This Stage 1A loader accepts only the locked AWGN/MSE configuration.")
        if tuple(config.rate_grid) != (32, 64, 96, 128, 192):
            raise ValueError("Rate grid in manifest differs from the locked Stage 1A grid.")
        if tuple(config.snr_db_grid) != (1, 4, 7, 10, 13):
            raise ValueError("SNR grid in manifest differs from the locked Stage 1A grid.")

    def _validate_checkpoint_hashes(self) -> None:
        required = {FULL_FILENAME, ENCODER_FILENAME, DECODER_FILENAME}
        recorded = self.raw["checkpoint_hashes"]
        missing = required.difference(recorded)
        if missing:
            raise ValueError(f"Manifest does not hash required checkpoint files: {sorted(missing)}")
        for filename in required:
            actual_path = self.model_dir / filename
            if not actual_path.exists():
                raise FileNotFoundError(f"Required checkpoint missing: {actual_path}")
            if sha256_file(actual_path) != recorded[filename]:
                raise ValueError(f"Checkpoint hash mismatch: {filename}")

    def _validate_runtime(self) -> None:
        recorded = self.raw["runtime"]
        current = runtime_fingerprint()
        for key in ("python_major_minor", "pytorch_major_minor"):
            if recorded.get(key) != current.get(key):
                raise RuntimeError(
                    f"Inference runtime mismatch for {key}: artifact={recorded.get(key)!r}, current={current.get(key)!r}."
                )

    def _validate_upstream_source(self) -> None:
        patch_file = self.upstream_root / ".stage1a_patch.json"
        if not patch_file.exists():
            raise FileNotFoundError(f"Patched upstream source is unavailable: {patch_file}")
        source_info = json.loads(patch_file.read_text(encoding="utf-8"))
        expected = self.raw["upstream"]
        if source_info.get("upstream_commit") != expected["commit"]:
            raise RuntimeError("Loaded upstream commit does not match the exported artifact.")
        if source_info.get("patch_version") != expected["patch_version"]:
            raise RuntimeError("Loaded upstream patch version does not match the exported artifact.")
        if source_info.get("source_hashes") != expected["source_hashes"]:
            raise RuntimeError("Patched upstream source hashes do not match the exported artifact.")

    def load_tensor_states(self, adapter: Any) -> None:
        torch = require_torch()
        full_payload = torch.load(self.model_dir / FULL_FILENAME, map_location="cpu", weights_only=True)
        if not isinstance(full_payload, dict) or set(full_payload) != {"encoder_state_dict", "decoder_state_dict"}:
            raise ValueError("Full checkpoint has an unexpected non-tensor-state schema.")
        encoder, decoder = adapter._require_built()
        encoder.load_state_dict(full_payload["encoder_state_dict"], strict=True)
        decoder.load_state_dict(full_payload["decoder_state_dict"], strict=True)


def export_stage1_artifacts(
    adapter: Any,
    model_dir: str | Path,
    data_provenance: dict[str, Any],
    training_summary: dict[str, Any],
    upstream_relative_path: str = "external/SwinJSCC",
) -> Stage1ArtifactManifest:
    """Export explicit tensor state dictionaries and their validation manifest."""

    torch = require_torch()
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    encoder, decoder = adapter._require_built()
    full_path = destination / FULL_FILENAME
    encoder_path = destination / ENCODER_FILENAME
    decoder_path = destination / DECODER_FILENAME

    # Only dictionaries of tensors are serialized; whole Python model objects are never saved.
    full_payload = {"encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict()}
    torch.save(full_payload, full_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    upstream_root = Path(upstream_relative_path)
    patch_path = upstream_root / ".stage1a_patch.json"
    if not patch_path.exists():
        raise FileNotFoundError(f"Cannot export without verified patched source: {patch_path}")
    upstream_info = json.loads(patch_path.read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "architecture": adapter.config.to_manifest_dict(),
        "upstream": {
            "relative_path": upstream_relative_path,
            "commit": upstream_info["upstream_commit"],
            "patch_version": upstream_info["patch_version"],
            "source_hashes": upstream_info["source_hashes"],
        },
        "runtime": runtime_fingerprint(),
        "checkpoint_hashes": {
            FULL_FILENAME: sha256_file(full_path),
            ENCODER_FILENAME: sha256_file(encoder_path),
            DECODER_FILENAME: sha256_file(decoder_path),
        },
        "data_provenance": data_provenance,
        "training_summary": training_summary,
        "research_status": "corrected modern PyTorch port; not a paper-exact replication",
    }
    (destination / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return Stage1ArtifactManifest(model_dir=destination, raw=manifest)
