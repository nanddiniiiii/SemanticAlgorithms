"""Diagnostic supported-operation FLOPs on retained pretrained model interfaces.

CPU times are diagnostic, not comparable with GPU results. Run on an otherwise
idle machine for timing work; this script does not certify complete FLOP coverage.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['jcm', 'adjscc', 'deepjscc', 'sqgan'], default=['jcm', 'adjscc'])
    args = parser.parse_args()
    import torch
    from semantic_lossless.config import Config
    from semantic_lossless.codecs import build_codec
    from semantic_lossless.metrics import profile_forward
    from semantic_lossless.artifacts import write_json
    torch.set_num_threads(2)
    for name in args.models:
        size = {'jcm': 32, 'adjscc': 256, 'deepjscc': 128, 'sqgan': 256}[name]
        config = Config(name, device='cpu', dataset='cifar10' if name == 'jcm' else 'image_folder',
            crop_size=size, snr_db=12 if name == 'jcm' else 10,
            checkpoint=str(ROOT/'Models/pretrained_downloads/ADJSCC/ADJSCC_C=2.pth.tar') if name == 'adjscc' else '',
            model_options={'latent_channels': 2})
        torch.manual_seed(config.seed)
        model = build_codec(config).eval()
        image = torch.full((1, 3, size, size*2 if name == 'sqgan' else size), .5)
        profiled = model
        if name == 'sqgan':
            semantic = torch.zeros(1, 20, 256, 512); semantic[:, 0] = 1
            class Paired(torch.nn.Module):
                def __init__(self): super().__init__(); self.core = model
                def forward(self, image): return self.core(image, semantic)
            profiled = Paired()
        profile = profile_forward(profiled, image, warmup=0 if name == 'sqgan' else 3,
            repeats=1 if name == 'sqgan' else 10)
        profile.update(algorithm=name, input_shape=list(image.shape),
            input_scope='synthetic constant input for computational profiling only; no quality benchmark',
            hardware='local CPU; threads2', torch=torch.__version__, model_metadata=model.metadata,
            timing_comparison_eligible=False, timing_limitation='diagnostic CPU timing; separate GPU/idle-host profiling required')
        if name == 'sqgan':
            profile['input_scope'] = 'synthetic constant RGB and valid single-class one-hot map for computational profiling only'
            profile['flops_scope'] = 'ten-component SQ01 packet encode/decode adapter, not a native GAN training loop; partial supported operations'
        write_json(ROOT/'Docs/validation'/(name+'_cpu_profile.json'), profile)
        print(name, profile['forward_flops'], profile['flops_coverage'], flush=True)


if __name__ == '__main__':
    main()
