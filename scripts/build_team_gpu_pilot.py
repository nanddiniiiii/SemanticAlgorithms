"""Bounded synthetic two-T4 pilot of the first integrated source paths."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['deepjscc','adjscc','swinjscc','isec','jcm','hjscc','mdjcm','ntscc_plus','shared_semantic_codebook','sqgan'],
                        default=['deepjscc','adjscc','swinjscc','isec','jcm'])
    parser.add_argument('--real-manifest',type=Path,help='Verified Kaggle dataset manifest for a small train/dev pilot')
    args=parser.parse_args()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT/'src').rglob('*.py')):
            archive.writestr(path.relative_to(ROOT).as_posix(),path.read_bytes())
        archive.writestr('sources.lock.json',(ROOT/'sources.lock.json').read_bytes())
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest()
    setup=f'''import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
print('START source/environment setup',flush=True)
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
ROOT=Path('/kaggle/working/Semantic_Lossless');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():(ROOT/name).resolve().relative_to(ROOT.resolve())
    archive.extractall(ROOT)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(ROOT)
sys.path.insert(0,str(ROOT/'src'))
subprocess.run([sys.executable,'-m','pip','install','timm==0.4.12','PyYAML==6.0.3','einops==0.8.2','gdown==5.2.0','pytorch-msssim==1.0.0'],check=True)
from src.runtime import environment,write_json
OUT=Path('/kaggle/working/team_gpu_pilot');OUT.mkdir(exist_ok=True)
write_json(OUT/'source_bundle.json',{{'sha256':{sha!r},'scope':'mutable local snapshot, identified by bundled bytes'}})
write_json(OUT/'environment.json',environment(ROOT))
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()>=2, 'Actual two-GPU allocation required'
print(json.dumps(environment(ROOT),indent=2),flush=True)
print('END source/environment setup',flush=True)
'''
    body='MODELS='+repr(args.models)+'\n'+'''import gc, numpy as np, traceback, torch
from PIL import Image
from src.dataset import freeze
from src.models import build
from src.train import train
from src.evaluate import evaluate
from src.runtime import stage,export
from src.assets import obtain
print('START synthetic data generation; no benchmark images used',flush=True)
rng=np.random.default_rng(22);roots={}
for name,count in [('div2k_train',2),('flickr2k_train',2),('div2k_val',1),('kodak_test',1)]:
    folder=OUT/'synthetic'/name;folder.mkdir(parents=True,exist_ok=True);roots[name]=str(folder)
    for index in range(count):
        Image.fromarray(rng.integers(0,256,(256,512,3),dtype=np.uint8)).save(folder/f'{index}.png')
manifest=OUT/'synthetic_manifest.json';freeze(roots,manifest,strict_counts=False)
results=[]
for name in MODELS:
    run=OUT/name;run.mkdir()
    options={'algorithm':name,'from_scratch':True,'device':'cuda:0','data_parallel':True,
        'crop':128 if name=='isec' else 256,'batch_size':2,'epochs':3,'max_seconds':120,
        'latent_channels':16 if name=='isec' else 2 if name=='adjscc' else 32,'iterations':1}
    try:
        with stage(run,'numerical_pilot'):
            if name in {'hjscc','mdjcm','ntscc_plus'}:
                print('Installing author entropy-model dependencies for',name,flush=True)
                subprocess.run([sys.executable,'-m','pip','install','compressai==1.2.6','scikit-learn'],check=True,timeout=600)
            if name=='ntscc_plus':
                buildenv={**os.environ,'MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'7.5'}
                print('Building author NATTEN0.14.6 for actual T4, bounded two compile workers',flush=True)
                subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','natten==0.14.6'],
                    env=buildenv,check=True,timeout=600)
            if name=='swinjscc':
                paths=list(Path('/kaggle/input').rglob('SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C32.model'))
                if len(paths)!=1:raise FileNotFoundError('Mount the verified native SwinC32 checkpoint dataset')
                options.update(checkpoint=str(paths[0]),swin_variant='fixed_c32',rate=32,from_scratch=False)
            if name=='jcm':options.update(checkpoint=str(obtain(name,ROOT/'Models/pretrained_downloads')),from_scratch=False)
            if name=='shared_semantic_codebook':
                from src.sources import fetch_source
                from src.shared import construct
                fetch_source(name)
                candidates=list(Path('/kaggle/input').rglob('classifier.pt'))
                if len(candidates)!=1:raise RuntimeError('Mount exactly one verified frozen classifier.pt')
                options.update(classifier=str(candidates[0]),manifest=str(manifest),codebook_size=2,
                               construction_images=2,construction_seconds=120,latent_channels=4,
                               evaluation_tile=[256,256])
                options['codebook']=str(construct(options,run))
            if name=='sqgan':
                subprocess.run([sys.executable,'-m','pip','install','pytorch-lightning==1.9.5',
                    'torchmetrics==0.11.4','transformers==4.46.3','scipy','omegaconf'],check=True,timeout=600)
                from src.sources import fetch_source
                from src.special import prepare
                from src.sqgan_train import execute
                fetch_source(name)
                options.update(checkpoint=str(obtain(name,ROOT/'Models/pretrained_downloads')),
                               manifest=str(manifest),learning_rate=1.25e-5)
                options=prepare(name,options,run)
                execute('pilot',options,run)
                results.append({'algorithm':name,'status':'passed','scope':'native objective synthetic pilot; source codec only'})
            else:
                model=build(name,options,'cuda:0')
                write_json(run/'model.json',model.metadata)
                summary=train(model,manifest,run,options,pilot=True)
                evaluation=evaluate(model,manifest,run,options,pilot=True)
                if evaluation['successful_trials']!=1 or evaluation['failed_trials']:
                    raise RuntimeError('Development inference did not complete')
                results.append({'algorithm':name,**summary,'training_status':summary['status'],'status':'passed'})
                del model
    except Exception as error:
        traceback.print_exc();results.append({'algorithm':name,'status':'failed','error':str(error)})
    write_json(OUT/'summary.json',{'scope':'synthetic shape/gradient/checkpoint/dualGPU pilots; no quality benchmark',
        'models':results,'all_passed':all(item['status']=='passed' for item in results)})
    gc.collect();torch.cuda.empty_cache()
    export(run)
print('END bounded GPU pilots',json.dumps(results),flush=True)
'''
    if args.real_manifest:
        manifest_bytes=args.real_manifest.read_bytes()
        frozen=json.loads(manifest_bytes)
        expected='add977eb3277909e5ad1447770abdb368458445ab42b2f48a91c24f4c31f7fa0'
        actual=hashlib.sha256(json.dumps(frozen['images'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if frozen['identity']!=expected or actual!=expected:raise ValueError('Unexpected or altered dataset manifest')
        start=body.index("print('START synthetic data generation")
        end=body.index('results=[]',start)
        compressed=zlib.compress(manifest_bytes,9)
        replacement=('import zlib\n'
            'print("START real-data pilot using verified train/dev membership; Kodak untouched",flush=True)\n'
            'manifest=OUT/"real_manifest.json"\n'
            f'frozen=json.loads(zlib.decompress(base64.b64decode({base64.b64encode(compressed).decode()!r})))\n'
            'from src.dataset import discover\n'
            'roots=discover("/kaggle/input")\n'
            'print("Current dataset mounts:",json.dumps(roots),flush=True)\n'
            'if set(roots)!=set(frozen["roots"]): raise FileNotFoundError("All four verified dataset roots must be mounted")\n'
            'frozen["roots"]=roots\n'
            'write_json(manifest,frozen)\n')
        body=body[:start]+replacement+body[end:]
        body=body.replace('synthetic shape/gradient/checkpoint/dualGPU pilots; no quality benchmark',
                          'real train/development pilot; no Kodak test or convergence claim')
    cells=[]
    for index,source in enumerate((setup,body)):
        compile(source,'team_gpu_pilot','exec')
        cells.append({'cell_type':'code','id':f'pilot-{index}','metadata':{},'execution_count':None,
                      'outputs':[],'source':source.splitlines(keepends=True)})
    folder=ROOT/'Kaggle/team_suite_pilot';folder.mkdir(parents=True,exist_ok=True)
    notebook={'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
              'nbformat':4,'nbformat_minor':5}
    (folder/'pilot.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    write={'id':'satyapaladuguplus/team-suite-dual-t4-contract-pilot','title':'Team suite dual T4 contract pilot',
        'code_file':'pilot.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':['satyapaladuguplus/semantic-lossless-private-swin-assets']+(
            ['satyapaladuguplus/semantic-lossless-pseudo-label-assets'] if 'shared_semantic_codebook' in args.models else []),
        'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    if args.real_manifest:
        write['dataset_sources']+=['soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24']
    (folder/'kernel-metadata.json').write_text(json.dumps(write,indent=2),encoding='utf-8')
    print(json.dumps({'folder':str(folder),'source_bundle_sha256':sha,'bytes':len(payload)}))


if __name__=='__main__':main()
