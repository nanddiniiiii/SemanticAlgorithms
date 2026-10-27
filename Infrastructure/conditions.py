"""Explicit channel conditions for training and evaluation.

An adapter must opt in before using a condition; configuring a channel here
does not replace an author's hardcoded channel inside a model.
"""
from dataclasses import dataclass
from math import isfinite

from Infrastructure.Channels.channel_factory import create_channel


@dataclass(frozen=True)
class ChannelCondition:
    channel: str = 'awgn'
    snr_db: float = 10.0
    coherence: str = "symbol"

    def __post_init__(self):
        if self.channel not in {'awgn', 'rayleigh_perfect_csi'}:
            raise ValueError('Supported channels are awgn and rayleigh_perfect_csi')
        if self.coherence not in {'symbol','image'}:
            raise ValueError('Rayleigh coherence must be symbol or image')
        if not isfinite(self.snr_db):
            raise ValueError('SNR must be finite')

    def build(self):
        result = create_channel(self.channel)
        if self.channel == 'rayleigh_perfect_csi':
            result.coherence = self.coherence
        return result


@dataclass(frozen=True)
class ChannelSetup:
    train: ChannelCondition = ChannelCondition()
    test: ChannelCondition = ChannelCondition()

    def for_stage(self, stage: str) -> ChannelCondition:
        if stage == 'train':
            return self.train
        if stage in {'validation', 'test'}:
            return self.test
        raise ValueError('Stage must be train, validation or test')


def from_options(options):
    """One parser shared by adapters and notebook preflight."""
    fallback=float(options.get('snr_db',10))
    return ChannelSetup(**{stage: ChannelCondition(
        options.get(stage+'_channel','awgn'), float(options.get(stage+'_snr_db',fallback)),
        options.get(stage+'_coherence','symbol')) for stage in ('train','test')})


def validate_model_conditions(name, options):
    setup=from_options(options)
    backend=options.get('channel_backend')
    if backend not in {None,'native','infrastructure'}:
        raise ValueError('channel_backend must be native or infrastructure')
    if backend=='infrastructure' and name not in {'deepjscc','adjscc','swinjscc'}:
        raise ValueError(name+' has no Infrastructure transport adapter')
    if backend=='native' and name in {'deepjscc','adjscc'}:
        raise ValueError(name+' uses the documented Infrastructure transport adaptation')
    flexible = name in {'deepjscc','adjscc'} or (name=='swinjscc' and options.get('channel_backend')=='infrastructure')
    if name=='swinjscc' and options.get('channel_backend')=='infrastructure' and options.get('swin_variant','fixed_c32')!='fixed_c32':
        raise ValueError('Infrastructure Swin transport currently supports fixed_c32 only; native adaptive masks need a separate integration')
    if not flexible and (setup.train.channel!='awgn' or setup.test.channel!='awgn'
                         or setup.train.snr_db!=setup.test.snr_db):
        raise ValueError(name+' does not yet support these channel conditions; use its native AWGN contract')
    if name=='hjscc' and setup.train.snr_db!=10:
        raise ValueError('Native HJSCC requires AWGN at 10 dB')
    if name=='sqgan' and (setup.train.snr_db!=10 or setup.test.snr_db!=10):
        raise ValueError('SQ-GAN is a source codec; SNR is not an evaluated metric')
    return setup
