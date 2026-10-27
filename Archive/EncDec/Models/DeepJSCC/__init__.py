"""DeepJSCC integration components."""

from Archive.EncDec.Models.DeepJSCC.adapter import (
    DeepJSCCAdapter,
    register_deepjscc,
)
from Archive.EncDec.Models.DeepJSCC.channel_wrapper import (
    DeepJSCCChannelWrapper,
)

__all__ = [
    "DeepJSCCAdapter",
    "DeepJSCCChannelWrapper",
    "register_deepjscc",
]