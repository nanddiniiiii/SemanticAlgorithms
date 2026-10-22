"""Corrected modern-runtime SwinJSCC adapter."""

from EncDec.Models.SwinJSCC.adapter import SwinJSCCAdapter, register_swinjscc
from EncDec.Models.SwinJSCC.swin_config import SwinJSCCConfig

__all__ = ["SwinJSCCAdapter", "SwinJSCCConfig", "register_swinjscc"]
