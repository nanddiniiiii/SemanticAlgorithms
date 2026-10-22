"""Pinned author SwinJSCC factory, separate from the retained adapted port."""
import ast
from datetime import datetime
import importlib
from pathlib import Path
from types import SimpleNamespace

from .artifacts import sha256
from .codecs import upstream_import
from .sources import fetch_source, source_path

CHECKPOINTS = {
    'fixed_c32': ('SwinJSCC_w/o_SAandRA', '32',
        '864d52a64281b3fcd4db34f7696b225b96c998b040b81d96cc556ecb837a17ff'),
    'sa_ra': ('SwinJSCC_w/_SAandRA', '32,64,96,128,192',
        'eaa56bc7ba82c2d145afced4b54f563253e7e9c59dba2f868076ee8a0066887c'),
}


def build_native_swin(checkpoint, variant='sa_ra', device='cuda:0'):
    import torch
    checkpoint = Path(checkpoint)
    receipt = fetch_source('swinjscc')
    source = source_path('swinjscc')
    model_name, channels, expected = CHECKPOINTS[variant]
    if sha256(checkpoint) != expected:
        raise ValueError('Checkpoint does not match the verified native Swin variant')
    args = SimpleNamespace(trainset='DIV2K', testset='kodak', model_size='base',
        model=model_name, C=channels, distortion_metric='MSE', channel_type='awgn',
        multiple_snr='10' if variant == 'fixed_c32' else '1,4,7,10,13')
    tree = ast.parse((source/'main.py').read_text())
    config_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'config')
    namespace = {'torch': torch, 'nn': torch.nn, 'datetime': datetime, 'args': args}
    exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(source/'main.py'), 'exec'), namespace)
    config = namespace['config']
    config.CUDA = str(device).startswith('cuda')
    config.device = torch.device(device)
    with upstream_import(source, ['net', 'loss']):
        model = importlib.import_module('net.network').SwinJSCC(args, config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.to(device).eval()
    metadata = {'source_commit': receipt['commit'], 'variant': variant,
        'checkpoint_sha256': expected, 'strict_load': True, 'native_math_unmodified': True,
        'training_reproduced': False, 'channel': 'native author AWGN',
        'snr_definition': 'complex Es/N0; adaptive normalization uses active feature mask',
        'model_size': 'base', 'config_shape': [3, 256, 256],
        'parameters': sum(p.numel() for p in model.parameters())}
    return model, metadata
