"""Package separate native-checkpoint evaluation and missing-weight training jobs."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('sources.lock.json', (ROOT/'sources.lock.json').read_bytes())
    for path in sorted((ROOT/'src').rglob('*.py')):
        archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    for name in ['run_native_swin_evaluation.py', 'train_shared_native_classifier.py']:
        archive.writestr('scripts/'+name, (ROOT/'scripts'/name).read_bytes())
    for name in ['swinjscc', 'shared_semantic_codebook']:
        source = ROOT/'.upstream'/name
        receipt = json.loads((source/'source_receipt.json').read_text())
        included = [item for item in receipt['files'] if Path(item['path']).suffix.lower() in {'.py', '.yaml', '.yml', '.json', '.md', '.txt'}]
        for item in included:
            path = source/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
        transported = {**receipt, 'files': included,
            'transport_scope': 'native Python/config/docs; non-executable figure assets omitted'}
        archive.writestr(source.relative_to(ROOT).as_posix()+'/source_receipt.json', json.dumps(transported).encode())
payload = buffer.getvalue()
revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
bootstrap = f'''import base64,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
blob=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
root=Path('/kaggle/working/Semantic_Lossless');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():(root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.chdir(root);os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root);os.environ['PYTHONPATH']=str(root/'src')
assert (root/'sources.lock.json').is_file(),'Source bundle lacks required root marker'
provenance={{'source_bundle_sha256':hashlib.sha256(blob).hexdigest(),'root':str(root),
    'python':sys.version,'interpreter':sys.executable,'git_revision':{revision!r},'working_tree_dirty':{dirty!r},
    'scope':'Stage1 existing-method experiments'}}
(root/'source_bundle.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({{'stage':'source_preflight','status':'passed',**provenance}}),flush=True)
import torch
assert torch.cuda.is_available(),'Actual GPU hardware was not allocated'
print(json.dumps({{'stage':'environment','torch':torch.__version__,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}),flush=True)
subprocess.run([sys.executable,'-c','from semantic_lossless.paths import project_root; print("VERIFIED_PROJECT_ROOT",project_root(),flush=True)'],check=True)
logs=root/'run_logs';logs.mkdir(exist_ok=True)
def run_logged(stage,command,timeout):
    start=time.monotonic();path=logs/(stage+'.log')
    event={{'stage':stage,'status':'started','command':command,'log':str(path),'source_bundle_sha256':provenance['source_bundle_sha256']}}
    with (logs/'stages.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\\n')
    print(json.dumps(event),flush=True)
    offset=0
    with path.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
        timed_out=False
        while process.poll() is None:
            with path.open(encoding='utf-8',errors='replace') as reader:
                reader.seek(offset);text=reader.read();offset=reader.tell()
            if text:print(text,end='',flush=True)
            if time.monotonic()-start>timeout:
                timed_out=True;process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(1)
    with path.open(encoding='utf-8',errors='replace') as reader:
        reader.seek(offset);text=reader.read()
        if text:print(text,end='',flush=True)
    result={{'stage':stage,'status':'timeout' if timed_out else 'completed' if process.returncode==0 else 'failed',
        'returncode':process.returncode,'elapsed_seconds':time.monotonic()-start,'log':str(path)}}
    with (logs/'stages.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\\n')
    print(json.dumps(result),flush=True)
    if timed_out or process.returncode:raise RuntimeError('Stage failed: '+stage+'; see '+str(path))
'''
image = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'


def write_job(folder_name, owner, slug, title, datasets, body):
    folder = ROOT/'Kaggle'/folder_name; folder.mkdir(parents=True, exist_ok=True)
    script = bootstrap+body
    compile(script, title, 'exec')
    (folder/'kernel.py').write_text(script)
    metadata = {'id': owner+'/'+slug, 'title': title, 'code_file': 'kernel.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
        'docker_image_pinning_type': 'original', 'docker_image': image,
        'dataset_sources': datasets, 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


write_job('stage1_swin', 'satyapaladuguplus', 'stage1-native-swin-evaluation',
    'Stage1 Native Swin Evaluation',
    ['sherylmehta/kodak-dataset', 'satyapaladuguplus/semantic-lossless-private-swin-assets'], '''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','timm==0.4.12','pytorch-msssim==1.0.0','lpips==0.1.4'],300)
for variant,filename in [('fixed_c32','SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C32.model'),('sa_ra','SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model')]:
    weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob(filename)})
    pictures=sorted({p.resolve().parent for p in Path('/kaggle/input').rglob('kodim01.png')})
    assert len(weights)==len(pictures)==1
    command=[sys.executable,'scripts/run_native_swin_evaluation.py','--checkpoint',str(weights[0]),'--variant',variant,'--images',str(pictures[0]),'--output',str(root/'Results/stage1/swinjscc'/variant)]
    run_logged('evaluation_'+variant,command,3600)
''')

write_job('stage1_shared_training', 'satyapaladugu', 'stage1-shared-codebook-classifier',
    'Stage1 Shared Codebook Classifier', ['pratt3000/stl10-binary-files'], '''
files=sorted({p.resolve() for p in Path('/kaggle/input').rglob('train_X.bin')})
assert len(files)==1,files
data=root/'Datasets/STL10';data.mkdir(parents=True)
(data/'stl10_binary').symlink_to(files[0].parent,target_is_directory=True)
base=[sys.executable,'scripts/train_shared_native_classifier.py','--data-root',str(data)]
# A real-data pilot must pass before the 200-epoch auxiliary training starts.
run_logged('classifier_pilot',base+['--output',str(root/'Results/stage1/shared_semantic_codebook/classifier_pilot'),'--pilot-steps','2','--epochs','200','--max-seconds','600'],900)
run_logged('classifier_training',base+['--output',str(root/'Results/stage1/shared_semantic_codebook/classifier_training'),'--epochs','200','--max-seconds','7200'],7800)
''')

assets = ROOT/'Kaggle/assets/swin_native'; assets.mkdir(parents=True, exist_ok=True)
for path in (ROOT/'Models/pretrained_downloads/SwinJSCC').glob('*.model'):
    target = assets/path.name
    if not target.exists(): shutil.copyfile(path, target)
(assets/'dataset-metadata.json').write_text(json.dumps({'title': 'Semantic Lossless Private Swin Assets',
    'id': 'satyapaladuguplus/semantic-lossless-private-swin-assets', 'licenses': [{'name': 'other'}]}, indent=2))
print(json.dumps({'bundle_bytes': len(payload), 'jobs': ['stage1_swin', 'stage1_shared_training']}))
