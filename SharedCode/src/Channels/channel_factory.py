"""Concrete channel factory; no channel implementation is hosted inside INFRA."""

from Channels.awgn import AWGNChannel
from Channels.rayleigh import RayleighChannel
from Channels.rician import RicianChannel
from INFRA.Registries import CHANNEL_REGISTRY


def register_builtin_channels() -> None:
    entries = {"awgn": AWGNChannel, "rayleigh_perfect_csi": RayleighChannel, "rician": RicianChannel}
    for name, constructor in entries.items():
        if not CHANNEL_REGISTRY.contains(name):
            CHANNEL_REGISTRY.register(name, constructor)


def create_channel(name: str):
    register_builtin_channels()
    return CHANNEL_REGISTRY.create(name)
