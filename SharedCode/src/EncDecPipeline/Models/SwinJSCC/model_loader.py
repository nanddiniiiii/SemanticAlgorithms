"""Strict manifest-validated model loader."""

from pathlib import Path

from EncDecPipeline.Models.SwinJSCC.adapter import SwinJSCCAdapter
from EncDecPipeline.Models.SwinJSCC.checkpoint_manager import Stage1ArtifactManifest


def load_stage1_swinjscc(model_dir: str | Path, device: str | None = None) -> SwinJSCCAdapter:
    manifest = Stage1ArtifactManifest.load_and_validate(model_dir)
    adapter = SwinJSCCAdapter(config=manifest.config, upstream_root=manifest.upstream_root)
    adapter.build(device=device)
    manifest.load_tensor_states(adapter)
    return adapter
