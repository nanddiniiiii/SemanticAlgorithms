"""Generate a CPU-only Kaggle check for the agreed common dataset membership."""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'src/semantic_lossless/team_suite/dataset.py').read_bytes()
identity=hashlib.sha256(source).hexdigest()
cell=f'''import base64, hashlib, json, os, platform, sys, time
from pathlib import Path
print('START dataset preflight: verify mounts and freeze image membership',flush=True)
data=base64.b64decode({base64.b64encode(source).decode()!r})
assert hashlib.sha256(data).hexdigest()=={identity!r}
module=Path('/kaggle/working/team_dataset_module.py');module.write_bytes(data)
import importlib.util
spec=importlib.util.spec_from_file_location('team_dataset_module',module)
dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
out=Path('/kaggle/working/team_dataset_preflight');out.mkdir(exist_ok=True)
print('Python:',sys.version,'platform:',platform.platform(),flush=True)
print('Source SHA256:',{identity!r},'output:',out,flush=True)
started=time.monotonic()
try:
    roots=dataset.discover('/kaggle/input')
    print('Discovered roots:',json.dumps(roots,indent=2),flush=True)
    manifest=dataset.freeze(roots,out/'common_dataset.json')
    result={{'status':'passed','identity':manifest['identity'],'counts':manifest['counts'],
             'roots':roots,'source_sha256':{identity!r},'elapsed_seconds':time.monotonic()-started,
             'scope':'decoded RGB/file hashes, membership and exact duplicates; no training or evaluation'}}
except Exception as error:
    import traceback
    traceback.print_exc()
    result={{'status':'failed','error':repr(error),'source_sha256':{identity!r},
             'elapsed_seconds':time.monotonic()-started}}
(out/'summary.json').write_text(json.dumps(result,indent=2))
print('END dataset preflight:',json.dumps(result),flush=True)
if result['status']!='passed':raise RuntimeError('Dataset preflight failed; inspect summary')
'''
compile(cell,'dataset_preflight','exec')
folder=ROOT/'Kaggle/team_dataset_preflight';folder.mkdir(parents=True,exist_ok=True)
notebook={'cells':[{'cell_type':'code','id':'dataset-preflight','metadata':{},
    'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}],
    'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
    'nbformat':4,'nbformat_minor':5}
(folder/'preflight.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata={'id':'satyapaladuguplus/team-dataset-membership-preflight','title':'Team dataset membership preflight',
    'code_file':'preflight.ipynb','language':'python','kernel_type':'notebook','is_private':True,
    'enable_gpu':False,'enable_internet':False,
    'dataset_sources':['soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24'],
    'kernel_sources':[],'competition_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps({'folder':str(folder),'dataset_source_sha256':identity}))
