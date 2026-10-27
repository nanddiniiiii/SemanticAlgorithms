"""Retrieve a saved Kaggle version without changing jobs or Git branches."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

from .dataset import digest
from .runtime import write_json


def verify_download(directory):
    """Validate exported per-model files using their own relative roots."""
    manifests=list(Path(directory).rglob('artifacts.json'))
    verified=0
    for manifest in manifests:
        data=json.loads(manifest.read_text())
        for item in data['files']:
            root=manifest.parent.resolve();path=(root/item['path']).resolve()
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('Downloaded artifact failed verification: '+str(path))
            verified+=1
    if not manifests:raise FileNotFoundError('No exported artifact manifests found in downloaded outputs')
    return {'manifests':len(manifests),'verified_files':verified}


def collect(root):
    """Rebuild the merged table under an exclusive lock; never duplicate rows."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=root/'comparison.csv.lock'
    started=time.monotonic()
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);break
        except FileExistsError:
            if time.monotonic()-started>=30:
                raise TimeoutError('Comparison collector lock remained busy; inspect '+str(lock))
            time.sleep(.1)
    try:
        rows=[]
        paths=[]
        # A concurrently downloading teammate's partial files are not evidence.
        for marker in sorted(root.rglob('retrieval_verified.json')):
            tree=marker.parent/'published_output'
            for manifest in tree.rglob('artifacts.json'):
                for item in json.loads(manifest.read_text())['files']:
                    if item['path'].endswith('/per_image.csv'):
                        path=(manifest.parent/item['path']).resolve()
                        path.relative_to(tree.resolve())
                        if digest(path)!=item['sha256']:
                            raise ValueError('Previously verified metrics changed: '+str(path))
                        paths.append(path)
        for path in sorted(set(paths)):
            # Keep tiny CPU pilots out of the final comparison table.
            if 'pilot' in path.parts:continue
            profile_path=path.parent/'profile.json'
            profile=json.loads(profile_path.read_text()) if profile_path.exists() else {}
            summary_path=path.parent/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            fields={key:profile.get(key) for key in ('parameters','registered_kernel_flops',
                'flops_coverage','latency_p50_seconds','latency_p95_seconds','peak_cuda_memory_bytes',
                'incremental_cuda_memory_bytes','device','precision')}
            fields.update({('evaluation_'+key):summary.get(key) for key in
                ('requested_trials','successful_trials','failed_trials','mean_over_successful_trials_only')})
            with path.open(newline='',encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    rows.append({**row,**fields,'artifact_source':path.relative_to(root).as_posix()})
        keys=sorted(set().union(*(row.keys() for row in rows))) if rows else ['artifact_source']
        temporary=root/'comparison.csv.tmp'
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        temporary.replace(root/'comparison.csv')
        return len(rows)
    finally:
        os.close(fd);lock.unlink()


def main(team):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True,help='owner/slug of the teammate notebook')
    parser.add_argument('--version',required=True,type=int,help='Expected saved notebook version; keep this kernel unchanged until retrieval')
    parser.add_argument('--output',type=Path,default=Path('Results/team_downloads'))
    parser.add_argument('--config-dir',type=Path,help='Optional isolated Kaggle credential directory; never print its contents')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--max-hours',type=float,default=20)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<10 or args.max_hours<=0 or args.version<=0:
        parser.error('interval>=10, max-hours>0 and a positive saved version are required')
    parts=args.kernel.split('/')
    if len(parts)!=2 or any(not p or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in p) for p in parts):
        parser.error('kernel must be owner/slug')
    target=args.output.resolve()/team/(parts[0]+'__'+parts[1])/str(args.version)
    target.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ)
    if args.config_dir:environment['KAGGLE_CONFIG_DIR']=str(args.config_dir.resolve())
    ref=args.kernel+'/'+str(args.version)
    start=time.monotonic();previous=None
    while time.monotonic()-start<args.max_hours*3600:
        status=subprocess.run(['kaggle','kernels','status',ref],env=environment,capture_output=True,text=True,timeout=90)
        output=(status.stdout+'\n'+status.stderr).strip()
        # Kaggle CLI status resolves the latest run even when a version is
        # supplied. Never treat that response as proof of an exact version.
        record={'kernel':args.kernel,'expected_version':args.version,'returncode':status.returncode,'status':output,
                'status_scope':'latest kernel run; CLI does not establish version identity'}
        write_json(target/'remote_status.json',record)
        if output!=previous:print(output,flush=True);previous=output
        terminal=status.returncode==0 and any(word in output.upper() for word in ('COMPLETE','ERROR','CANCELLED'))
        if terminal:
            raise RuntimeError('Automatic retrieval disabled: Kaggle CLI status is latest-run scoped and cannot verify the requested version. Verify remote notebook identity and retrieve manually.')
        if args.once:return
        time.sleep(args.interval)
