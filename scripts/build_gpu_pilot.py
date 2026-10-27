"""Build a self-contained short Kaggle pilot from the current modular sources."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root/'src').rglob('*.py')):
        archive.writestr(path.relative_to(root).as_posix(),path.read_bytes())
    for name in ['semcom.py','sources.lock.json']:
        archive.writestr(name,(root/name).read_bytes())
payload=buffer.getvalue()
encoded=base64.b64encode(payload).decode()
code=f'''# Source transport bundle; all implementation modules remain separate files.
import base64, hashlib, io, zipfile, os, sys, json, gc, queue, time
from pathlib import Path
blob=base64.b64decode({encoded!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
ROOT=Path('/kaggle/working/Semantic_Lossless')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():
        (ROOT/name).resolve().relative_to(ROOT.resolve())
    archive.extractall(ROOT)
os.chdir(ROOT)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(ROOT)
sys.path.insert(0,str(ROOT/'src'))
'''
body='''import torch
from src.config import Config
from src.codecs import build_codec
from src.sources import fetch_source
from src.artifacts import write_json, export_manifest
from src.events import Events
from src.training import train
assert torch.cuda.is_available(), 'No CUDA device allocated'
assert torch.cuda.device_count()>=2, 'This pilot requires T4 x2'
write_json(ROOT/'pilot_environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'scope':'GPU runtime checks; no quality benchmark'})
print((ROOT/'pilot_environment.json').read_text(),flush=True)
data_roots=sorted({path.parent.resolve() for path in Path('/kaggle/input').rglob('cifar-10-batches-py') if (path/'data_batch_1').is_file()})
assert len(data_roots)==1, f'Expected one mounted native CIFAR-10 Python directory, got {data_roots}'
DATA_ROOT=str(data_roots[0])
for algorithm in ['adjscc','jcm']:
    fetch_source('diffcom' if algorithm=='adjscc' else algorithm)
    config=Config(algorithm,dataset='cifar10',data_root=DATA_ROOT,crop_size=32,
                  device='cuda:0',batch_size=8,workers=2,multi_gpu=True,epochs=1,max_seconds=600,
                  max_steps=2,download_dataset=False,max_images=2,
                  model_options={'from_scratch':True,'latent_channels':2})
    run=ROOT/'Results/experiments'/algorithm/'gpu_pilot'
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',config.as_dict())
    codec=build_codec(config)
    seen=queue.Queue()
    encoder=codec.core.jscc_encoder if algorithm=='adjscc' else codec.core.encoder
    hook=encoder.register_forward_hook(lambda module,inputs,output: seen.put(str(inputs[0].device)))
    train(config,codec,run,Events(run,config),pilot=True)
    hook.remove()
    work=[]
    while not seen.empty(): work.append(seen.get())
    assert {'cuda:0','cuda:1'}.issubset(work),f'Did not observe work on both GPUs: {work}'
    write_json(run/'gpu_work.json',{'observed_devices':work,'peak_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]})
    # Only ADJSCC passed the AMP qualification in version 2. Keep JCM FP32.
    if algorithm=='adjscc':
        codec.train()
        parallel=torch.nn.DataParallel(codec)
        image=torch.rand(2,3,256,256,device='cuda:0')
        labels=torch.tensor([1,2],device='cuda:0')
        optimizer=codec.optimizer()
        optimizer.zero_grad(set_to_none=True)
        scaler=torch.amp.GradScaler('cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            output=parallel(image)
            loss=codec.loss(output,{'image':image,'label':labels})
        assert torch.isfinite(loss), 'AMP loss was nonfinite'
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in codec.parameters()),'AMP gradient overflow'
        scaler.step(optimizer);scaler.update()
        write_json(run/'amp_check.json',{'finite':True,'shape':list(image.shape),'both_gpus':True,'scope':'synthetic AMP step after real CIFAR training pilot; no convergence claim'})
        del parallel,optimizer,output,loss,image
    else:
        write_json(run/'precision_policy.json',{'training_precision':'float32','reason':'JCM initial AMP gradient overflow in pilot version 2; native FP32 two-GPU training passed'})
    export_manifest(run)
    del codec
    gc.collect();torch.cuda.empty_cache()
from src.events import collect
collect(ROOT/'Results/experiments',ROOT/'Results/metrics.csv')
write_json(ROOT/'pilot_status.json',{'status':'completed','algorithms':['adjscc','jcm'],'scope':'real-data two-step GPU compatibility pilot plus synthetic target-size AMP checks'})
print('DUAL_GPU_PILOT_COMPLETED',flush=True)
'''
notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Short dual-T4 runtime pilot\nNot a training-quality benchmark.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':body.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
for index,cell in enumerate(notebook['cells']):
    cell['id']=f'pilot-cell-{index}'
    if cell['cell_type']=='code':compile(''.join(cell['source']),'pilot','exec')
folder=root/'Kaggle/pilot';folder.mkdir(exist_ok=True)
(folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
metadata={'id':'satyapaladugu/semantic-lossless-dual-t4-pilot','title':'Semantic Lossless Dual T4 Pilot','code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['pankrzysiu/cifar10-python'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Pilot bundle bytes:',len(payload))
