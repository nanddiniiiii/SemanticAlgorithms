"""Run Kaggle commands with an identity check; B uses isolated WSL credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ACCOUNTS={'A':'satyapaladugu','B':'satyapaladuguplus'}
B_CLI='/home/Satya/.venvs/semantic-kaggle/bin/kaggle'
B_CONFIG='/home/Satya/.config/semantic-lossless/kaggle/account_b'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('account',choices=['A','B'])
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command:
        parser.error('Provide a Kaggle command, for example kernels list --mine')
    expected=ACCOUNTS[args.account]
    if args.account=='A':
        credential=Path.home()/'.kaggle/credentials.json'
        identity=json.loads(credential.read_text())['username'] if credential.is_file() else None
        command=['kaggle']+args.command
    else:
        probe="import json,pathlib; p=pathlib.Path.home()/'.kaggle/credentials.json'; print(json.loads(p.read_text()).get('username','') if p.is_file() else '')"
        result=subprocess.run(['wsl','--exec','/home/Satya/.venvs/semantic-kaggle/bin/python','-c',probe],capture_output=True,text=True,check=True)
        identity=result.stdout.strip()
        command=['wsl','--exec','env','KAGGLE_CONFIG_DIR='+B_CONFIG,B_CLI]+args.command
    if identity!=expected:
        raise SystemExit(f'Account {args.account}: expected {expected}, authenticated identity is {identity or "missing"}; sign in locally first')
    os.environ['PYTHONIOENCODING']='utf-8'
    print(f'Using account {args.account}: {expected}',flush=True)
    return subprocess.call(command)

if __name__=='__main__':
    sys.exit(main())
