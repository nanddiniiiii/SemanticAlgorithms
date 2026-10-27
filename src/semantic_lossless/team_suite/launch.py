"""Submit one team notebook from the user's own CLI and collect saved outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from .monitor import collect, verify_download
from .runtime import write_json

ROOT=Path(__file__).resolve().parents[3]
DATASETS=['soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24']


def main(team):
    parser=argparse.ArgumentParser(description='Submit this team notebook and retrieve its completed outputs.')
    parser.add_argument('--username',help='Your Kaggle username; prompted if omitted')
    parser.add_argument('--config-dir',help='Optional Kaggle CLI credential directory')
    parser.add_argument('--asset',action='append',default=[],help='Additional Kaggle dataset owner/slug containing model assets; repeat as needed')
    parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    username=(args.username or input('Your Kaggle username: ')).strip()
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in username):
        raise ValueError('Invalid Kaggle username')
    if args.interval<10:raise ValueError('Polling interval must be at least 10 seconds')
    if shutil.which('kaggle') is None:raise RuntimeError('Install and authenticate the Kaggle CLI in this Python environment first')
    env=os.environ.copy()
    if args.config_dir:env['KAGGLE_CONFIG_DIR']=str(Path(args.config_dir).expanduser().resolve())
    token=uuid.uuid4().hex[:10];slug=f'semantic-{team.replace("_","-")}-{token}'
    ref=f'{username}/{slug}'
    destination=ROOT/'Results'/'team_downloads'/team/(username+'__'+slug)/'1'
    destination.mkdir(parents=True,exist_ok=False)
    metadata={'id':ref,'title':slug,'code_file':'notebook.ipynb','language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':DATASETS+args.asset,'kernel_sources':[],
        'competition_sources':[],'model_sources':[]}
    write_json(destination/'submission.json',{'kernel':ref,'team':team,'datasets':DATASETS+args.asset,
        'source_revision':'05164d6170e601ac9a43dcdd767fd5282baa6d9a',
        'note':'Fresh unique kernel; run version must not be updated before retrieval'})
    print('Submitting private notebook:',ref,flush=True)
    print('The three common image datasets will be attached automatically. Additional private model assets may require setup.',flush=True)
    with tempfile.TemporaryDirectory(prefix='semantic-kaggle-') as tmp:
        folder=Path(tmp)
        shutil.copy2(ROOT/'Notebooks'/'TeamSuite'/(team+'.ipynb'),folder/'notebook.ipynb')
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        subprocess.run(['kaggle','kernels','push','-p',str(folder)],env=env,check=True,timeout=180)
    print('Notebook:',f'https://www.kaggle.com/code/{ref}',flush=True)
    print('Do not save a new version of this notebook while this monitor is running.',flush=True)
    previous=None
    while True:
        status=subprocess.run(['kaggle','kernels','status',ref],env=env,capture_output=True,text=True,timeout=90)
        message=(status.stdout+'\n'+status.stderr).strip()
        if message!=previous:print(message,flush=True);previous=message
        if status.returncode:raise RuntimeError('Status check failed; notebook may still be running: '+message)
        upper=message.upper()
        if any(word in upper for word in ('COMPLETE','ERROR','CANCELLED')):break
        time.sleep(args.interval)
    raw=destination/'published_output'
    subprocess.run(['kaggle','kernels','output',ref,'-p',str(raw)],env=env,check=True,timeout=3600)
    receipt=verify_download(raw)
    models=ROOT/'Models'/'team_runs'/team/slug
    for manifest in raw.rglob('artifacts.json'):
        base=manifest.parent
        for checkpoint in base.rglob('checkpoints/*'):
            if checkpoint.is_file():
                target=models/checkpoint.relative_to(raw)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(checkpoint,target)
    write_json(destination/'retrieval_verified.json',{**receipt,'kernel':ref,'status':message,
        'identity_scope':'unique new kernel; user instructed not to update before retrieval',
        'scientific_validity_certified':False})
    rows=collect(ROOT/'Results'/'team_downloads')
    print('Saved outputs:',raw,'\nModel files:',models,'\nComparison CSV:',ROOT/'Results'/'team_downloads'/'comparison.csv',
          '\nRows:',rows,flush=True)
    if 'COMPLETE' not in upper:raise RuntimeError('Notebook terminated unsuccessfully; retained artifacts for inspection')
