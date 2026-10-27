"""Explicit local setup; git clone itself cannot safely execute an environment hook."""
from pathlib import Path
import argparse
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',type=Path,default=ROOT/'.venv-team')
    parser.add_argument('--managed-runtime',action='store_true',help='Inherit installed platform Torch/CUDA instead of replacing it')
    parser.add_argument('--skip-install',action='store_true')
    args=parser.parse_args()
    destination=args.path.resolve()
    print('Preparing project environment:',destination,flush=True)
    if not (destination/'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=True,system_site_packages=args.managed_runtime).create(destination)
    python=destination/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not args.skip_install:
        dependencies=['-e',str(ROOT)]
        if not args.managed_runtime:dependencies+=['torch','torchvision']
        subprocess.run([str(python),'-m','pip','install',*dependencies],check=True)
    subprocess.run([str(python),'-c','import sys; print("Actual interpreter:",sys.executable); import torch; print("Torch:",torch.__version__,"CUDA:",torch.cuda.is_available())'],check=True)
    print('Environment ready. Model-specific dependencies are installed by the notebook initialization cells.',flush=True)


if __name__=='__main__':main()
