"""Install isolated model environments on Kaggle without replacing base packages.

Environment creation is not runtime/model validation. Legacy native-op assets
remain explicit gates. Every successful environment records pip freeze and GPU
package versions; no unpinned author source is installed from HEAD.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RECIPES = {
    'DeepJSCC': ('3.9', '1.12.1+cu113', '0.13.1+cu113', []),
    'SwinJSCC': ('3.9', '1.12.1+cu113', '0.13.1+cu113', ['timm==0.6.13']),
    'DeepJSCC_Q': ('3.9', '1.12.1+cu113', '0.13.1+cu113', ['timm==0.6.13', 'einops==0.6.1', 'matplotlib==3.7.5', 'PyYAML==6.0.1', 'tqdm==4.66.1']),
    'NTSCC_plus': ('3.8', '1.9.0+cu111', '0.10.0+cu111', ['timm==0.6.13', 'compressai==1.2.4', 'configargparse==1.7', 'wandb==0.15.12', 'einops==0.6.1']),
    'SQGAN': ('3.10', '1.12.1+cu113', '0.13.1+cu113', ['pytorch-lightning==1.5.6', 'torchmetrics==0.7.3', 'omegaconf==2.1.2', 'einops==0.3.2', 'albumentations==0.5.2', 'opencv-python-headless==4.8.1.78', 'wandb==0.15.12', 'pandas==1.5.3', 'matplotlib==3.7.5', 'scikit-learn==1.3.2', 'easydict==1.11', 'blobfile==2.1.1', 'pytz==2023.3.post1', 'pytorch-fid==0.3.0'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=list(RECIPES)+['ADJSCC'])
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'model': args.model, 'status': 'PROVISIONING', 'runtime_validated': False,
              'gpu_training_started': False, 'python': None}
    receipt = args.output / 'environment.json'
    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    def run(command):
        subprocess.run([str(x) for x in command], check=True)
    try:
        if platform.system() != 'Linux':
            raise RuntimeError('CUDA environment provisioning must run inside Kaggle Linux, not this Windows CPU host')
        if args.model == 'ADJSCC':
            raise RuntimeError('ADJSCC requires an audited Linux build of the bundled TensorFlow Compression native ops matching TF2.1. The supplied binary is macOS-only; pip tensorflow-compression==2.1 instead requires TF2.5 and is NOT a valid fix. See model_runtime/adjscc_tf.py.')
        pyversion, torchversion, visionversion, packages = RECIPES[args.model]
        env = args.output.resolve() / 'venv'
        interpreter = env / 'bin/python'
        if not interpreter.exists():
            run([sys.executable, '-m', 'pip', 'install', 'uv==0.9.5'])
            run([sys.executable, '-m', 'uv', 'venv', '--python', pyversion, '--seed', str(env)])
        report['python'] = str(interpreter)
        report['recipe'] = {'python': pyversion, 'torch': torchversion,
                             'torchvision': visionversion, 'packages': packages}
        save()
        run([interpreter, '-m', 'pip', 'install', 'pip==23.3.2', 'setuptools==59.5.0', 'wheel==0.41.3'])
        run([interpreter, '-m', 'pip', 'install', 'torch=='+torchversion, 'torchvision=='+visionversion,
             '--extra-index-url', 'https://download.pytorch.org/whl/'+('cu111' if args.model == 'NTSCC_plus' else 'cu113')])
        common = ['numpy==1.23.5', 'Pillow==9.5.0', 'pytorch-msssim==1.0.0', 'lpips==0.1.4', 'pytest==7.4.4']
        run([interpreter, '-m', 'pip', 'install', *common, *packages])
        if args.model == 'NTSCC_plus':
            run([interpreter, '-m', 'pip', 'install', 'natten==0.14.6+torch190cu111',
                 '-f', 'https://shi-labs.com/natten/wheels'])
        run([interpreter, '-m', 'pip', 'check'])
        freeze = subprocess.check_output([str(interpreter), '-m', 'pip', 'freeze'], text=True)
        (args.output / 'requirements.resolved.txt').write_text(freeze)
        report.update(status='READY_ENVIRONMENT_UNVALIDATED', freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest())
    except Exception as exc:
        report.update(status='BLOCKED_ENVIRONMENT', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
