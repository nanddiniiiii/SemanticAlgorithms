"""Build the public master notebook without executing GPU work."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
cells=[md('''# SemanticAlgorithms master runner
Reuse this notebook for every Kaggle version. Set the public repository URL and private dataset paths below. Each run checks out main into a fresh source directory and records its SHA; upstream model manifests stay pinned. No credentials belong in this notebook. Runtime blockers stop execution and are recorded, never replaced with fabricated results.
'''),code('''# FIRST CODE CELL: fresh latest public benchmark source; preserve all prior artifacts.
import os, re, json, subprocess, hashlib, sys
from pathlib import Path
from datetime import datetime, timezone
PUBLIC_REPOSITORY_URL = os.environ.get("PUBLIC_REPOSITORY_URL", "") # https://github.com/<owner>/SemanticAlgorithms
if not re.fullmatch(r"https://github\\.com/[A-Za-z0-9_.-]+/SemanticAlgorithms(?:\\.git)?", PUBLIC_REPOSITORY_URL):
    raise ValueError("Set PUBLIC_REPOSITORY_URL to the reviewed public SemanticAlgorithms repository")
RUN = Path('/kaggle/working/semanticlossless_runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN.mkdir(parents=True, exist_ok=False)
CODE=RUN/'code'
subprocess.run(['git','clone','--depth','1','--branch','main','--',PUBLIC_REPOSITORY_URL,str(CODE)],check=True)
SHA=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
(RUN/'checkout_receipt.json').write_text(json.dumps({'repository':PUBLIC_REPOSITORY_URL,'branch':'main','commit':SHA,'previous_artifacts_deleted':False},indent=2))
print(json.dumps({'event':'source_checkout','sha':SHA,'run':str(RUN)}))
'''),code('''# Parameters: attach DIV2K train, validation, original Kodak24 and private checkpoints.
MODEL = os.environ.get('MODEL', 'SwinJSCC')
STAGE = os.environ.get('STAGE','semantic' if MODEL=='SQGAN' else 'main') # SQGAN: semantic, image, finetune
CHANNEL = os.environ.get('CHANNEL','awgn') # separate job for each channel and rate
MODE = os.environ.get('MODE', 'preflight') # preflight, train, evaluate
TRAIN_ROOT = Path(os.environ.get('TRAIN_ROOT','/kaggle/input/div2k/DIV2K_train_HR'))
VAL_ROOT = Path(os.environ.get('VAL_ROOT','/kaggle/input/div2k/DIV2K_valid_HR'))
KODAK_ROOT = Path(os.environ.get('KODAK_ROOT','/kaggle/input/kodak24'))
CHUNK_SECONDS = int(os.environ.get('CHUNK_SECONDS','1800'))
RESUME = os.environ.get('RESUME','') # explicit checkpoint file under a private attached dataset
ALLOWED=['DeepJSCC','SwinJSCC','ADJSCC','NTSCC_plus','DeepJSCC_Q','SQGAN']
assert MODEL in ALLOWED and MODE in ['preflight','train','evaluate'] and CHANNEL in ['awgn','rayleigh']
assert 60 <= CHUNK_SECONDS <= 10800
ARTIFACTS=RUN/'artifacts'; ARTIFACTS.mkdir()
CONTRACT=CODE/'SharedCode/reproduction/experiment_contract.json'
contract=json.loads(CONTRACT.read_text())
t=contract['training']; e=contract['evaluation']; stop=t['early_stopping']
assert t['global_batch_size']==16 and t['snr_db']==10
assert (stop['minimum_epochs'],stop['maximum_epochs'],stop['patience'],stop['relative_min_delta'])==(50,150,20,.001)
assert e['snr_db']==[0,5,10,15,20] and e['independent_channel_trials_per_image']==10
assert abs(e['primary_cbr_budget']-1/6)<1e-12
assert t['sqgan_stages']==['semantic','image','finetune']
print(json.dumps({'event':'parameters','model':MODEL,'mode':MODE,'chunk_seconds':CHUNK_SECONDS,'contract_sha256':hashlib.sha256(CONTRACT.read_bytes()).hexdigest()}))
'''),code('''# Fail before training if data, hardware, pinned source or runtime is incomplete.
# Provision owns isolated model dependencies; all image/model imports run there.
ENV_ROOT=ARTIFACTS/'environment'
subprocess.run([sys.executable,str(CODE/'SharedCode/reproduction/provision.py'),
                '--model',MODEL,'--output',str(ENV_ROOT)],cwd=CODE,check=True)
environment=json.loads((ENV_ROOT/'environment.json').read_text())
assert environment.get('status')=='READY_ENVIRONMENT_UNVALIDATED', 'Environment provision has not completed'
MODEL_PYTHON=environment['python']
assert Path(MODEL_PYTHON).is_file(), 'Provisioned model interpreter missing'
subprocess.run([MODEL_PYTHON,str(CODE/'SharedCode/reproduction/delivery/validate_inputs.py'),
                '--train-root',str(TRAIN_ROOT),'--val-root',str(VAL_ROOT),'--kodak-root',str(KODAK_ROOT)],check=True)
LOCK=CODE/'sources.lock.json'
assert LOCK.is_file(), 'Published pinned sources.lock.json missing'
subprocess.run([sys.executable,str(CODE/'SharedCode/reproduction/source_lock.py'),'--lock-file',str(LOCK)],cwd=CODE,check=True)
(RUN/'sources.lock.json').write_bytes(LOCK.read_bytes())
manifests=list((CODE/'Models/author_releases').glob('*/*.manifest.json'))
assert manifests, 'Pinned upstream source manifests missing'
for manifest in manifests:
    m=json.loads(manifest.read_text()); snapshot=CODE/m['source_directory']
    assert len(m['commit'])==40
    for name,digest in m['files_sha256'].items():
        p=(snapshot/name).resolve(); assert p.is_relative_to(snapshot.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Pinned source mismatch: {p}'
RUNTIME=CODE/'SharedCode/reproduction'
assert (RUNTIME/'model_runtime/__main__.py').is_file(), 'model_runtime missing; training blocked'
if RESUME: assert Path(RESUME).is_file(), 'Explicit resume checkpoint not found'
print(json.dumps({'event':'input_validation_passed','train':800,'validation':100,'kodak':24,'gpus':2}))
'''),code('''# Runtime owns full epochs, optimizer/RNG resume, stages, measured eligibility and JSON progress.
# Contract: effective global batch16, SNR10, epochs50..150, patience20, relative min_delta .001.
# Test SNR0/5/10/15/20, 10 channel trials, primary CBR1/6 plus measured curves.
# SQGAN trains its three codec stages once and evaluates AWGN and Rayleigh; DiT/DDM are audit only.
base=[MODEL_PYTHON,'-m','model_runtime','--model',MODEL,'--channel',CHANNEL,'--stage',STAGE,'--contract',str(CONTRACT),
      '--train-root',str(TRAIN_ROOT),'--val-root',str(VAL_ROOT),'--kodak-root',str(KODAK_ROOT),
      '--output',str(ARTIFACTS),'--chunk-seconds',str(CHUNK_SECONDS)]
if RESUME: base+=['--resume',RESUME]
def invoke(mode):
    env=os.environ.copy(); env['PYTHONPATH']=str(RUNTIME)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run(base+['--mode',mode],cwd=CODE,env=env)
    if result.returncode: raise RuntimeError(f'{MODEL} {mode} failed with exit {result.returncode}; inspect JSON progress')
try:
    invoke('preflight')
    preflight=json.loads((ARTIFACTS/'preflight.json').read_text())
    assert preflight.get('status')=='RUNTIME_PREFLIGHT_PASSED_NOT_TRAINED', 'Runtime preflight receipt has not passed'
    if MODE!='preflight': invoke(MODE)
finally:
    # Hash after either success or failure so available diagnostic artifacts can be checked on download.
    files={str(p.relative_to(ARTIFACTS)).replace('\\\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in ARTIFACTS.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    (ARTIFACTS/'artifact_sha256.json').write_text(json.dumps(files,indent=2))
'''),md('''## Consolidation
Only eligible measured results may enter the final comparison. A successful preflight is not a trained model or a result. Incomplete chunks and unsupported operators remain disclosed. Keep checkpoints and outputs private; publish reviewed code only. Save notebook versions against the same notebook id, and attach the prior private checkpoint dataset to resume.
''')]
cells.append(code('''# Optional final consolidation: attach downloaded completed run directories explicitly.
COMPLETED_MEASUREMENTS = [] # paths to actual completed measurement directories, never preflight outputs
if COMPLETED_MEASUREMENTS:
    subprocess.run([sys.executable,str(CODE/'SharedCode/reproduction/delivery/consolidate.py'),
                    '--contract',str(CONTRACT),'--inputs',*map(str,COMPLETED_MEASUREMENTS),
                    '--output',str(RUN/'consolidated')],check=True)
# This exports validated complete measured grids as unranked partial evidence.
# Ranking stays blocked pending checkpoint/provenance/ledger review and matched rates/FLOP coverage.
'''))
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'Notebooks/AuthorReproduction/SemanticAlgorithms_Master.ipynb';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
