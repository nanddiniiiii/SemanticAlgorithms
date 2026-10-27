"""Explicit environment setup and protocol defaults used by generated notebooks."""
import json
from pathlib import Path
import os
import subprocess
import sys
import venv

from .runtime import write_json

BASE=['gdown==5.2.0','pytorch-msssim==1.0.0','lpips==0.1.4','PyYAML==6.0.3','einops==0.8.2']
EXTRA={
    'deepjscc':[], 'adjscc':[], 'swinjscc':['timm==0.4.12'],
    'isec':[],
    'ntscc_plus':['timm==0.4.12','compressai==1.2.6','natten==0.14.6'],
    'hjscc':['compressai==1.2.6','scipy'],
    'mdjcm':['timm==0.4.12','compressai==1.2.6','scikit-learn'],
    'jcm':[], 'shared_semantic_codebook':['timm==0.4.12'],
    'sqgan':['pytorch-lightning==1.9.5','torchmetrics==0.11.4','transformers==4.46.3','scipy','omegaconf'],
    'deepjscc_wz':['lightning==2.0.6','torchmetrics==1.0.1','hydra-core==1.3.2','hydra-colorlog==1.2.0',
        'pyrootutils','rich','kornia','compressai','lmdb','tensorboard','diffusers','scikit-learn','opencv-python','mlflow'],
    'gbsed':[]
}


def setup_model(root,name,run,install=True,python_override=None,command_runner=None):
    require_integrated(name)
    if name not in EXTRA:raise ValueError(name)
    root=Path(root);run=Path(run)
    if python_override:
        python=Path(python_override)
        if not python.is_file():raise FileNotFoundError(python)
        print('Using explicitly supplied model environment:',python,flush=True)
        return python
    destination=root/'.venv-team'/name
    if not (destination/'pyvenv.cfg').is_file():
        print('Creating isolated model environment:',destination,flush=True)
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(destination)
    python=destination/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    dependencies=BASE+EXTRA[name]
    def execute(command,timeout=1200):
        if command_runner:return command_runner(command,timeout)
        return subprocess.run(command,check=True,timeout=timeout)
    write_json(run/'environment_plan.json',{'python':str(python),'inherits_platform_torch':True,
        'dependencies':dependencies,'requested_model':name,'torch_stack_replaced':False})
    if name=='gbsed':
        # Never install an entire frozen multi-framework CUDA environment over
        # Kaggle's driver stack without a successful compatibility pilot.
        print('Checking GBSED author framework stack before any GPU work',flush=True)
        execute([str(python),'-c','import torch, tensorflow, sionna, roadscene2vec; print(torch.__version__, tensorflow.__version__)'])
    if install:
        print('Installing model dependencies in subprocess environment:',dependencies,flush=True)
        regular=[item for item in dependencies if item!='natten==0.14.6']
        execute([str(python),'-m','pip','install','-e',str(root),*regular])
        if name=='ntscc_plus':
            # Legacy setup imports the installed Torch headers while building.
            # Build isolation hides those headers; newer NATTEN is not equivalent.
            old={key:os.environ.get(key) for key in ('MAX_JOBS','TORCH_CUDA_ARCH_LIST')}
            try:
                import torch
                if not torch.cuda.is_available():raise RuntimeError('Native NTSCC+ qualification needs CUDA')
                os.environ['MAX_JOBS']='2'
                os.environ['TORCH_CUDA_ARCH_LIST']=';'.join(sorted({
                    '.'.join(map(str,torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())}))
                execute([str(python),'-m','pip','install','--no-build-isolation','natten==0.14.6'],timeout=600)
            finally:
                for key,value in old.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
    return python


def default_options(name,manifest):
    value={'algorithm':name,'manifest':str(manifest),'device':'cuda:0','data_parallel':True,
           'snr_db':10,'train_channel':'awgn','test_channel':'awgn',
           'train_snr_db':10,'test_snr_db':10,'seed':22,'batch_size':2,'crop':256,'epochs':150,'max_seconds':14400,
           'learning_rate':1e-4,'from_scratch':True,'trials':3,'workers':2,
           'dataset_adaptation':True,'paper_convergence_reproduced':False}
    if name=='isec':value.update(crop=128,learning_rate=1e-3,iterations=50,latent_channels=16)
    if name=='adjscc':value.update(latent_channels=2,from_scratch=False)
    if name=='swinjscc':value.update(swin_variant='fixed_c32',rate=32,from_scratch=False)
    if name=='jcm':value.update(from_scratch=False,variant='approved_spatial_reconstruction_only')
    if name=='shared_semantic_codebook':value.update(evaluation_tile=[256,256],latent_channels=4)
    if name=='sqgan':value.update(learning_rate=1.25e-5,evaluation_tile=[256,512])
    if name=='mdjcm':value.update(modulation_order=64,clip_norm=.5)
    return value


def environment_options(name, manifest, environment, profile=None, overrides=None):
    """Resolve authoritative stage-specific channel settings before costly setup."""
    from Infrastructure.conditions import validate_model_conditions
    profile=profile or environment['default_channel_profile']
    options=default_options(name,manifest)
    options.update(environment['channel_profiles'][profile])
    if name=='swinjscc' and (options['train_channel']!='awgn' or options['test_channel']!='awgn'):
        options['channel_backend']='infrastructure'
    overrides=overrides or {}
    if 'snr_db' in overrides and not {'train_snr_db','test_snr_db'} & overrides.keys():
        options.update(train_snr_db=overrides['snr_db'],test_snr_db=overrides['snr_db'])
    options.update(overrides)
    options['snr_db']=options['train_snr_db']
    validate_model_conditions(name,options)
    return options


def record_completion(session, name, trained, evaluated, packaged):
    """Saved-output ledger, not a live or runtime-reset-safe external queue."""
    from .runtime import write_json
    path=Path(session)/'completed_models.json'
    ledger=json.loads(path.read_text()) if path.is_file() else {'models':{},'delivery':'saved notebook outputs only'}
    ledger['models'][name]={'training_completed':bool(trained),'evaluation_completed':bool(evaluated),
                          'model_packaged':bool(packaged),'local_retrieval_acknowledged':False}
    write_json(path,ledger)


def require_integrated(name):
    """Keep source-only candidates visible without claiming they can train."""
    roster=json.loads((Path(__file__).resolve().parents[1]/'configs/models.json').read_text())
    reason=roster.get('integration_gates',{}).get(name)
    if reason:
        raise NotImplementedError(name+': '+reason+'. Other assigned models may continue; this slot is not implemented.')
