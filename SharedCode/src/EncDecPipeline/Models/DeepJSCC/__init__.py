"""DeepJSCC integration components."""

from EncDecPipeline.Models.DeepJSCC.adapter import (
    DeepJSCCAdapter,
    register_deepjscc,
)
from EncDecPipeline.Models.DeepJSCC.channel_wrapper import (
    DeepJSCCChannelWrapper,
)

__all__ = [
    "DeepJSCCAdapter",
    "DeepJSCCChannelWrapper",
    "register_deepjscc",
]