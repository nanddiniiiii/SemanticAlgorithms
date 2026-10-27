"""Strictly test author Swin checkpoints against the pinned native configuration.

CPU construction/load only. Native CUDA forward is a separate acceptance step.
"""
import ast
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from src.codecs import upstream_import
from src.sources import fetch_source, source_path
from src.artifacts import sha256, write_json


def main():
    fetch_source('swinjscc')
    source = source_path('swinjscc')
    tree = ast.parse((source / 'main.py').read_text())
    config_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'config')
    reports = []
    variants = [
        ('SwinJSCC_w/o_SAandRA', '32', 'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C32.model'),
        ('SwinJSCC_w/_SAandRA', '32,64,96,128,192', 'SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model'),
    ]
    with upstream_import(source, ['net', 'loss']):
        model_class = importlib.import_module('net.network').SwinJSCC
        for variant, channels, filename in variants:
            args = SimpleNamespace(trainset='DIV2K', testset='kodak', model_size='base',
                model=variant, C=channels, distortion_metric='MSE', channel_type='awgn', multiple_snr='1,4,7,10,13')
            namespace = {'torch': torch, 'nn': torch.nn, 'datetime': datetime, 'args': args}
            exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(source/'main.py'), 'exec'), namespace)
            config = namespace['config']
            config.CUDA = False
            config.device = torch.device('cpu')
            model = model_class(args, config)
            path = ROOT / 'Models/pretrained_downloads/SwinJSCC' / filename
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            reports.append({'variant': variant, 'checkpoint': filename, 'sha256': sha256(path),
                'strict_load': True, 'parameters': sum(p.numel() for p in model.parameters()),
                'model_size': 'base', 'input_config': list(config.image_dims), 'forward_verified': False})
            del model, state
    receipt = {'scope': 'native author configuration and strict CPU state loading; CUDA forward pending', 'models': reports}
    write_json(ROOT/'Docs/validation/swin_native_checkpoint_load.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

