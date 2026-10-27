"""Persisted stages, environment identity and bounded subprocess execution."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def revision(root):
    result = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    dirty = subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True)
    return {'revision':result.stdout.strip() if result.returncode==0 else None,
            'working_tree_dirty':bool(dirty.stdout.strip())}


def environment(root):
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ('torch','torchvision','numpy','Pillow','timm','compressai','natten','lpips','pytorch-lightning'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    value = {'interpreter':sys.executable, 'python':sys.version, 'platform':platform.platform(),
             'repository':str(Path(root).resolve()), **revision(root), 'packages':packages,
             'kaggle_managed_parent':bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))}
    paths=list((Path(root)/'src').rglob('*.py'))+list((Path(root)/'Infrastructure').rglob('*.py'))
    paths += [Path(root)/'scripts/run.py',Path(root)/'configs/models.json',Path(root)/'configs/environment.json']
    value['project_source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(paths) if p.is_file()}
    try:
        import torch
        value.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                     devices=[{'id':i,'name':torch.cuda.get_device_name(i),
                               'total_memory':torch.cuda.get_device_properties(i).total_memory}
                              for i in range(torch.cuda.device_count())])
    except ImportError:
        value.update(cuda_available=False, devices=[])
    return value


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text); self.log.write(text); self.flush()
        return len(text)

    def flush(self):
        self.original.flush(); self.log.flush()


@contextmanager
def stage(run, name):
    """Python stdout/stderr + status; subprocesses must inherit these via pipes."""
    run = Path(run); (run/'logs').mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {'stage':name,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
    write_json(run/f'{name}_status.json', record)
    with (run/'logs'/f'{name}-{uuid.uuid4().hex[:8]}.log').open('w',encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
            print(f'START {name}: {run}',flush=True)
            try:
                yield record
                record['status']='completed'
            except Exception as exc:
                record.update(status='failed',error=str(exc),error_type=type(exc).__name__)
                traceback.print_exc()
                raise
            finally:
                record['elapsed_seconds']=time.monotonic()-started
                write_json(run/f'{name}_status.json',record)
                print(f"END {name}: {record['status']}, {record['elapsed_seconds']:.1f}s",flush=True)


def new_session(output, team):
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    path=Path(output).resolve()/team/identifier
    path.mkdir(parents=True,exist_ok=False)
    return path


def export(run):
    from .dataset import digest
    run=Path(run)
    files=[{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
           for p in sorted(run.rglob('*')) if p.is_file() and p.name not in {'artifacts.json','export_cell.log'}
           and not p.name.endswith('.tmp')]
    write_json(run/'artifacts.json',{'files':files,'verified_on_local_filesystem':True,
        'remote_persistence_verified':False,
        'excluded_live_file':'export_cell.log continues writing after this receipt; other logs are hashed',
        'note':'Kaggle output becomes downloadable after a saved version publishes it'})
    return files
