"""Build an isolated benchmark using an immutable successful SQ-GAN output."""
import ast
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'Notebooks/project_notebooks/Training/sqgan_div2k_channel/sqgan_div2k_awgn_rayleigh_training.ipynb'
DEST=ROOT/'Notebooks/project_notebooks/Training/SQGAN_speed_benchmark'
nb=json.loads(SOURCE.read_text())
cells=copy.deepcopy(nb['cells'][:4])
cells[0]['source']=['# SQ-GAN isolated speed benchmark\n\nNo benchmark updates count toward training. The active training run is not modified.\nEach candidate restores the same successful 98-epoch checkpoint. Architecture, batch 1/GPU,\n256x512 geometry, 800-image dataset and 150-epoch scheduler horizon remain fixed.\n40 batches per candidate are timing probes, not full epochs or comparison results.\n']
config=''.join(cells[1]['source']).replace('/kaggle/working/sqgan_div2k_channel_results','/tmp/sqgan_probe_source')
cells[1]['source']=config.splitlines(keepends=True)
code8=''.join(nb['cells'][8]['source'])
tree=ast.parse(code8)
shim=next(ast.literal_eval(n.args[0]) for n in ast.walk(tree)
          if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
          and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)
          and isinstance(n.args[0].value,str) and 'from session_control import install' in n.args[0].value)
shim=shim.replace('from session_control import install','from speed_probe import install')
runtime='(COMPAT_DIR/"speed_probe.py").write_text('+repr((Path(__file__).parent/'speed_probe.py').read_text())+')\n'
runtime+='(COMPAT_DIR/"sitecustomize.py").write_text('+repr(shim)+')\n'
runtime+=r'''
import gc
import threading
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
sys.path.insert(0,str(REPO))
os.chdir(REPO)
from adaptations.sqgan_div2k_channel.dataset import DIV2KWithPseudoLabels
from input_paths import resolve_div2k
DIV2K_TRAIN,DIV2K_VAL=resolve_div2k()
REPORT=Path('/kaggle/working/sqgan_speed_benchmark'); REPORT.mkdir(exist_ok=True)
checkpoint=next((REPO/'logs').glob('*sqgan_awgn_ssm*/checkpoints/last.ckpt'))
raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
expected={'global_step':raw['global_step'],'lr':[[g['lr'] for g in opt['param_groups']] for opt in raw['optimizer_states']]}
assert len(raw['optimizer_states'])==len(raw['lr_schedulers'])==2
source_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
del raw; gc.collect()
source_cfg=OmegaConf.load(OUTPUT/'config_awgn_ssm.yaml')
cache=Path('/tmp/sqgan_resized_cache'); cache.mkdir(exist_ok=True)
mapping={}; equivalence=[]
for split in ('train','validation'):
    params=source_cfg.data.params[split].params
    params.image_root=str(DIV2K_TRAIN if split=='train' else DIV2K_VAL)
    params.label_root=str(OUTPUT/'pseudo_labels'/('div2k_train' if split=='train' else 'div2k_val'))
    dataset=DIV2KWithPseudoLabels(**OmegaConf.to_container(params,resolve=True))
    assert len(dataset)==(800 if split=='train' else 100)
    originals=list(dataset.pairs)
    for i,(image_path,mask_path) in enumerate(originals):
        with Image.open(image_path) as im, Image.open(mask_path) as ma:
            image,mask=dataset._resize_cover(im.convert('RGB'),ma.convert('L'))
        dest=cache/split; dest.mkdir(exist_ok=True)
        cp,mp=dest/f'{i}_rgb.png',dest/f'{i}_label.png'
        image.save(cp,compress_level=1); mask.save(mp,compress_level=1)
        mapping[str(image_path)]=[str(cp),str(mp)]
    # Exact equality after the same random crop/flip proves cache equivalence.
    for i in (0,len(dataset)//2,len(dataset)-1):
        np.random.seed(20261020+i); a=dataset[i]
        old=dataset.pairs[i]; dataset.pairs[i]=tuple(Path(v) for v in mapping[str(old[0])])
        np.random.seed(20261020+i); b=dataset[i]
        dataset.pairs[i]=old
        assert all(torch.equal(a[k],b[k]) for k in ('image','label','semantic'))
        equivalence.append({'split':split,'index':i,'all_tensors_exactly_equal':True})
    del dataset
(cache/'mapping.json').write_text(json.dumps(mapping))
(REPORT/'cache_equivalence.json').write_text(json.dumps(equivalence,indent=2))

variants=[('baseline_fp32',0,False,False,32),
          ('pinned_fp32',0,True,False,32),
          ('cached_fp32',0,True,True,32),
          ('workers2_cached_fp32',2,True,True,32),
          ('cached_fp16',0,True,True,16)]
results=[]
for name,workers,pin,cached,precision in variants:
    branch=REPO/'logs'/f'probe_{name}'
    (branch/'checkpoints').mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkpoint,branch/'checkpoints/last.ckpt')
    cfg=copy.deepcopy(source_cfg)
    cfg.data.params.num_workers=workers
    cfgpath=Path('/tmp')/f'{name}.yaml'; OmegaConf.save(cfg,cfgpath)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(COMPAT_DIR)+os.pathsep+str(REPO),
               PROBE_VARIANT=name,PROBE_OUTPUT=str(REPORT/name),
               PROBE_PIN='1' if pin else '0',PROBE_EXPECTED=json.dumps(expected))
    env.pop('PROBE_CACHE_MAP',None)
    if cached: env['PROBE_CACHE_MAP']=str(cache/'mapping.json')
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    args=[sys.executable,'train.py','--base',str(cfgpath),'--mode','ssm',
          '--max_epochs','150','--gpus','0,1','--precision',str(precision),
          '--logtype','tensorboard','--num_sanity_val_steps','0','--enable_progress_bar','false',
          '--limit_val_batches','0','--resume',str(branch)]
    samples=[]; stop=threading.Event()
    def sample_gpus():
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True)
            samples.append({'unix':time.time(),'csv':p.stdout.strip()})
            stop.wait(1)
    monitor=threading.Thread(target=sample_gpus,daemon=True); monitor.start()
    started=time.time()
    try:
        with (REPORT/f'{name}.log').open('w') as log:
            p=subprocess.run(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        report_files=sorted((REPORT/name).glob('rank_*.json'))
        reports=[json.loads(f.read_text()) for f in report_files]
        result={'variant':name,'exit_code':p.returncode,'wall_seconds':time.time()-started,
                'rank_reports':reports,'passed':p.returncode==0 and len(reports)==2
                and all(r['batches_observed']==40 and r['all_model_parameters_finite'] for r in reports)}
    except subprocess.TimeoutExpired:
        # Killing DDP descendants is unsafe here; end this isolated Kaggle job.
        raise RuntimeError(f'{name} exceeded its probe budget; do not apply this candidate')
    finally:
        stop.set(); monitor.join(timeout=5)
    (REPORT/f'{name}_gpu_samples.json').write_text(json.dumps(samples))
    results.append(result)
    (REPORT/'results.json').write_text(json.dumps(results,indent=2))
    print('VARIANT RESULT',json.dumps(result),flush=True)
    shutil.rmtree(branch)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_hash
passing=[r for r in results if r['passed']]
baseline=next(r for r in passing if r['variant']=='baseline_fp32')
baseline_seconds=max(r['mean_batch_seconds'] for r in baseline['rank_reports'])
for r in passing:
    r['speedup_vs_baseline']=baseline_seconds/max(v['mean_batch_seconds'] for v in r['rank_reports'])
report={'source_checkpoint_sha256':source_hash,'source_unchanged':True,'results':results,
        'production_settings_changed':False,'benchmark_epochs_counted':0,
        'policy':'No automatic adoption. FP16 needs a full-epoch stability and quality check; workers need a full epoch coverage/resume check.'}
(REPORT/'benchmark_report.json').write_text(json.dumps(report,indent=2))
print('BENCHMARK COMPLETE',json.dumps(report),flush=True)
'''
runtime='(COMPAT_DIR/"input_paths.py").write_text('+repr((Path(__file__).parent/'input_paths.py').read_text())+')\nsys.path.insert(0,str(COMPAT_DIR))\n'+runtime
cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':runtime.splitlines(keepends=True)})
for c in cells:
    if c['cell_type']=='code':
        c['outputs']=[]; c['execution_count']=None; ast.parse(''.join(c['source']))
nb['cells']=cells
DEST.mkdir(parents=True,exist_ok=True)
(DEST/'sqgan_speed_benchmark.ipynb').write_text(json.dumps(nb,indent=1))
metadata={'id':'hmmkk20/sqgan-speed-benchmark-20261020','title':'SQGAN Speed Benchmark 20261020',
          'code_file':'sqgan_speed_benchmark.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
          'machine_shape':'NvidiaTeslaT4','dataset_sources':['sherylmehta/kodak-dataset'],
          'kernel_sources':['jagan028/div2k-dataset-generation-for-isr','hmmkk20/sqgan-checkpoint-publication-test-20261020'],
          'competition_sources':[],'model_sources':[]}
(DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(DEST)
