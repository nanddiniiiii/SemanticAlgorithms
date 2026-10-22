"""Strict manifest-validated model loader."""

from pathlib import Path

from EncDec.Models.SwinJSCC.adapter import SwinJSCCAdapter
from EncDec.Models.SwinJSCC.checkpoint_manager import Stage1ArtifactManifest


def load_stage1_swinjscc(model_dir: str | Path, device: str | None = None, upstream_root: str | Path | None = None) -> SwinJSCCAdapter:
    manifest = Stage1ArtifactManifest.load_and_validate(model_dir, upstream_root=upstream_root)
    adapter = SwinJSCCAdapter(config=manifest.config, upstream_root=manifest.upstream_root)
    adapter.build(device=device)
    manifest.load_tensor_states(adapter)
    return adapter
