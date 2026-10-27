"""Build a bounded author-version NATTEN GPU compatibility diagnostic."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source='''import json, os, platform, subprocess, sys, time, traceback
from pathlib import Path
out=Path('/kaggle/working/ntscc_legacy_compat');out.mkdir(exist_ok=True)
started=time.monotonic();log=out/'commands.log'
print('START isolated Python 3.9 / torch 1.9 / NATTEN 0.14.6 diagnostic',flush=True)
print('Kaggle interpreter:',sys.executable,sys.version,'output:',out,flush=True)
def run(cmd, timeout=900):
    print('START command:',cmd,flush=True)
    with log.open('a',encoding='utf-8') as stream:
        stream.write('COMMAND: '+repr(cmd)+'\\n');stream.flush()
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,text=True)
        start=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-start>timeout:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise TimeoutError('Command exceeded bound; see '+str(log))
            print('Command active, elapsed:',round(time.monotonic()-start),'seconds; log:',log,flush=True)
            time.sleep(15)
        if process.returncode:raise RuntimeError('Command failed '+str(process.returncode)+'; see '+str(log))
    print('END command:',round(time.monotonic()-start,1),'seconds',flush=True)
try:
    archive=out/'micromamba.tar.bz2'
    run(['curl','-fL','--max-time','120','https://micro.mamba.pm/api/micromamba/linux-64/latest','-o',str(archive)],150)
    run(['tar','-xjf',str(archive),'-C',str(out),'bin/micromamba'],120)
    mm=out/'bin/micromamba';env=out/'environment'
    run([str(mm),'create','-y','-p',str(env),'-c','conda-forge','python=3.9','pip'],900)
    python=env/'bin/python'
    run([str(python),'-m','pip','install','numpy<2','torch==1.9.0+cu111','torchvision==0.10.0+cu111',
         '-f','https://download.pytorch.org/whl/torch_stable.html'],900)
    run([str(python),'-m','pip','install','natten==0.14.6+torch190cu111',
         '-f','https://shi-labs.com/natten/wheels'],600)
    check="""import json, torch, natten
from natten import NeighborhoodAttention2D
assert torch.cuda.is_available(), 'CUDA unavailable'
layer=NeighborhoodAttention2D(dim=32,kernel_size=7,num_heads=4).cuda()
x=torch.randn(1,8,8,32,device='cuda',requires_grad=True)
y=layer(x); assert y.shape==x.shape and torch.isfinite(y).all()
y.square().mean().backward(); torch.cuda.synchronize()
assert x.grad is not None and torch.isfinite(x.grad).all()
print(json.dumps({'torch':torch.__version__,'natten':getattr(natten,'__version__','unknown'),
 'device':torch.cuda.get_device_name(0),'shape':list(y.shape),'finite_forward_backward':True}),flush=True)
"""
    check_path=out/'check.py';check_path.write_text(check)
    run([str(python),str(check_path)],120)
    result={'status':'passed','scope':'isolated author-version NATTEN finite GPU forward/backward only',
            'elapsed_seconds':time.monotonic()-started,'log':str(log)}
except Exception as error:
    traceback.print_exc();result={'status':'failed','error':repr(error),'elapsed_seconds':time.monotonic()-started,'log':str(log)}
(out/'summary.json').write_text(json.dumps(result,indent=2))
print('END diagnostic:',json.dumps(result),flush=True)
if result['status']!='passed':raise RuntimeError('Legacy compatibility not established')
'''
compile(source,'ntscc_legacy_pilot','exec')
folder=ROOT/'Kaggle/ntscc_legacy_pilot';folder.mkdir(parents=True,exist_ok=True)
notebook={'cells':[{'cell_type':'code','id':'legacy-compat','metadata':{},'execution_count':None,
                    'outputs':[],'source':source.splitlines(keepends=True)}],
          'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
          'nbformat':4,'nbformat_minor':5}
(folder/'pilot.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata={'id':'satyapaladuguplus/ntscc-legacy-natten-compat','title':'NTSCC legacy NATTEN compatibility',
          'code_file':'pilot.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
          'dataset_sources':[],'kernel_sources':[],'competition_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(folder)
