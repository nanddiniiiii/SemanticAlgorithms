"""Corrected modern-runtime SwinJSCC adapter."""

from EncDecPipeline.Models.SwinJSCC.adapter import SwinJSCCAdapter, register_swinjscc
from EncDecPipeline.Models.SwinJSCC.swin_config import SwinJSCCConfig

__all__ = ["SwinJSCCAdapter", "SwinJSCCConfig", "register_swinjscc"]
