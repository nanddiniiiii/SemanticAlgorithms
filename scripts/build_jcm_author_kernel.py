"""Package the modular source for an isolated pretrained evaluation on account B."""
import base64
import hashlib
import io
import json
import argparse
from pathlib import Path
import zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--account',choices=['A','B'],default='A')
args=parser.parse_args()
owner={'A':'satyapaladugu','B':'satyapaladuguplus'}[args.account]
root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root/'src').rglob('*.py')):
        archive.writestr(path.relative_to(root).as_posix(),path.read_bytes())
    for name in ['sources.lock.json','scripts/run_jcm_author_evaluation.py']:
        archive.writestr(name,(root/name).read_bytes())
payload=buffer.getvalue()
script=f'''import base64, hashlib, io, os, runpy, sys, zipfile
from pathlib import Path
blob=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
root=Path('/kaggle/working/Semantic_Lossless')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():
        (root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root)
os.chdir(root)
runpy.run_path(str(root/'scripts/run_jcm_author_evaluation.py'),run_name='__main__')
'''
compile(script,'kernel.py','exec')
folder=root/'Kaggle/jcm_author_eval'
folder.mkdir(parents=True,exist_ok=True)
(folder/'kernel.py').write_text(script,encoding='utf-8')
metadata={'id':owner+'/jcm-author-checkpoint-evaluation','title':'JCM Author Checkpoint Evaluation',
          'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,
          'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
          'dataset_sources':['pankrzysiu/cifar10-python','satyapaladugu/semantic-lossless-private-jcm-assets'],'competition_sources':[], 'kernel_sources':[], 'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built native-checkpoint evaluation; source bundle bytes:',len(payload))
