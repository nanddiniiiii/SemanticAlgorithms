"""Generate three transparent launchers; model logic stays in source modules."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PINNED_REVISION='d48af5be4330ac7ed1b04f2f2130c517e99201ed'
ROSTER=json.loads((ROOT/'configs/models.json').read_text())


def cell(source,kind='code',identifier=''):
    value={'cell_type':kind,'id':identifier,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def build(team,names):
    cells=[cell(f'# Semantic Lossless: {team}\n\nAssigned models, in order: {", ".join(names)}.\n\n'
        'Assignment is not implementation readiness. See configs/models.json integration_gates; blocked slots report an explicit error and do not train. '
        'Before running: enable Internet and T4 x2 in Kaggle settings and attach these datasets using Add Input: '
        '`soumikrakshit/div2k-high-resolution-images`, `daehoyang/flickr2k`, and `drxinchengzhu/kodak24`. '
        'A notebook cannot attach datasets to itself after it starts. The setup checks the attached files and prints errors. '
        'Shared codebook requires a separately attached classifier checkpoint; see the overrides in the setup cell.\n\n'
        'Initialization → two-step pilot and bounded training → evaluation/export, three cells per model. '
        'Failures are saved and displayed; later models continue, but a failed model is never reported as trained. '
        'CPU pilots are not T4 tests. Native GBSED/WZ tasks remain separate. SQ-GAN is an ideal source codec '
        'unless an additional digital link is explicitly implemented. Each training call is capped at four hours '
        'between optimizer steps; setup, pilots and evaluation add time.\n',kind='markdown',identifier='overview')]
    setup='''from pathlib import Path
import json, os, subprocess, sys, time, uuid
print('START setup: public Git clone, project environment, source identity', flush=True)
URL = 'https://github.com/nanddiniiiii/Semantic_Lossless.git'
REVISION = '__PINNED_REVISION__'  # Published implementation commit.
REPO = Path('/tmp/Semantic_Lossless')  # Disposable checkout and environments; results live under /kaggle/working.
if not REVISION:
    raise RuntimeError('The flattened source revision is not published yet; set REVISION before running')
if not REPO.exists():
    subprocess.run(['git','clone','--filter=blob:none',URL,str(REPO)],check=True,
                   env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'})
if REVISION:
    subprocess.run(['git','-C',str(REPO),'fetch','origin',REVISION],check=True)
    subprocess.run(['git','-C',str(REPO),'checkout','--detach',REVISION],check=True)
assert (REPO/'scripts/run.py').is_file(), 'The reviewed implementation must be published before this launcher can clone it'
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(REPO)
sys.path.insert(0,str(REPO))
os.chdir(REPO)
from src.runtime import new_session, write_json, environment, export
print(json.dumps(environment(REPO),indent=2),flush=True)
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()<2:
    raise RuntimeError('This notebook requires two CUDA GPUs; select Kaggle T4 x2')
SESSION=new_session('/kaggle/working/team_results',TEAM)
write_json(SESSION/'delivery_identity.json',{'delivery_run_id':DELIVERY_RUN_ID,'team':TEAM,'source_revision':REVISION})
print('Persistent output prefix:',SESSION,flush=True)
print('Use Save & Run All to publish outputs. Live /kaggle/working is not a CLI-download service.',flush=True)
READY={};TRAINED={};EVALUATED={};PACKAGED={};EXPORTED={};PYTHONS={};RUNS={}
MODELS_TO_RUN = ASSIGNED[:]  # Select a subset when four maximum-length runs exceed your session budget.
if len(MODELS_TO_RUN)>4 or len(set(MODELS_TO_RUN))!=len(MODELS_TO_RUN) or set(MODELS_TO_RUN)-set(ASSIGNED):
    raise ValueError('Choose up to four distinct assigned models in the desired order')
# Attach the same four dataset folders to every teammate notebook.
# Folder names can differ across Kaggle datasets: enter their actual paths here.
DATASET_ROOTS={
    'div2k_train':'', 'flickr2k_train':'', 'div2k_val':'', 'kodak_test':''}
EXPECTED_DATASET_IDENTITY = 'add977eb3277909e5ad1447770abdb368458445ab42b2f48a91c24f4c31f7fa0'
# Identity is the verified account-B v2 manifest; any change requires a new protocol.
MANIFEST=SESSION/'common_dataset.json'
OVERRIDES={}  # Per-model native data paths/checkpoints; see configs/README.md.
ENVIRONMENT=json.loads((REPO/'configs/environment.json').read_text())
CHANNEL_PROFILE=ENVIRONMENT['default_channel_profile']  # awgn, rayleigh_symbol, rayleigh_block, awgn_to_rayleigh
# Stage-specific SNR values are authoritative. Rayleigh currently supports DeepJSCC,
# ADJSCC and fixed-C32 SwinJSCC; native digital/source codecs reject unsupported profiles.
# Select a compatible subset or use OVERRIDES to retain native channels per model.
classifiers=list(Path('/kaggle/input').rglob('classifier.pt'))
if len(classifiers)==1:
    OVERRIDES['shared_semantic_codebook']={'classifier':str(classifiers[0])}
elif 'shared_semantic_codebook' in MODELS_TO_RUN:
    print('Shared codebook: attach exactly one dataset containing classifier.pt, or set OVERRIDES manually.',flush=True)
# Examples:
# OVERRIDES['sqgan']={'checkpoint':'/kaggle/input/author-assets/Final_ckpt_parts'}
# OVERRIDES['shared_semantic_codebook']={'classifier':'/kaggle/input/author-assets/classifier.pt'}
# OVERRIDES['deepjscc_wz']={'paired_data_root':'/kaggle/input/native-stereo'}
# OVERRIDES['gbsed']={'extraction_config':'...', 'learning_config':'...', 'receiver_weights':'...'}
from src.notebook_support import environment_options
for selected in MODELS_TO_RUN:
    environment_options(selected,MANIFEST,ENVIRONMENT,CHANNEL_PROFILE,OVERRIDES.get(selected))
print('Channel preflight passed:',CHANNEL_PROFILE,flush=True)
write_json(SESSION/'environment_config.json',ENVIRONMENT)
from src.dataset import freeze, discover
detected=discover('/kaggle/input',keys=[key for key,value in DATASET_ROOTS.items() if not value])
DATASET_ROOTS={key:(value or detected.get(key,'')) for key,value in DATASET_ROOTS.items()}
print('Resolved common dataset roots:',json.dumps(DATASET_ROOTS,indent=2),flush=True)
frozen=freeze(DATASET_ROOTS,MANIFEST)
print('Frozen dataset identity:',frozen['identity'],flush=True)
if EXPECTED_DATASET_IDENTITY and frozen['identity']!=EXPECTED_DATASET_IDENTITY:
    raise ValueError('Dataset identity differs from the agreed first-team manifest')
write_json(SESSION/'environment.json',environment(REPO))

def run_command(command,log,timeout=1800):
    import threading
    started=time.monotonic()
    with Path(log).open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def drain():
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
        reader=threading.Thread(target=drain,daemon=True);reader.start()
        try:
            code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            reader.join(timeout=10)
            raise TimeoutError('Stage timeout; logs and earlier checkpoints retained: '+str(log))
        reader.join(timeout=10)
    print('Command finished:',code,'elapsed',round(time.monotonic()-started,1),'seconds',flush=True)
    if code:raise RuntimeError('Command failed; inspect '+str(log))

def initialize(name):
    run=SESSION/name;run.mkdir(exist_ok=False);RUNS[name]=run
    # Per-model process environments prevent upstream modules/dependencies colliding.
    from src.notebook_support import setup_model, environment_options
    python=setup_model(REPO,name,run,python_override=OVERRIDES.get(name,{}).get('python'),
        command_runner=lambda command,timeout:run_command(command,run/'dependencies.log',timeout))
    PYTHONS[name]=python
    config=environment_options(name,MANIFEST,ENVIRONMENT,CHANNEL_PROFILE,OVERRIDES.get(name))
    config.update(teammate=TEAM,run_id=SESSION.name,models_root='/kaggle/working/models')
    path=run/'requested_config.json';write_json(path,config)
    return path

def invoke(name,stage):
    run=RUNS[name]
    run_command([str(PYTHONS[name]),'-u',str(REPO/'scripts/run.py'),stage,
        '--config',str(run/'requested_config.json'),'--run',str(run)],
        run/(stage+'_cell.log'),timeout=17400 if stage=='train' else 1800)

print('END setup: membership frozen; use per-model initialization cells next',flush=True)
'''
    setup=setup.replace('__PINNED_REVISION__',PINNED_REVISION)
    setup='DELIVERY_RUN_ID = None  # Bound by the local launcher; manual Kaggle runs may leave None\n'+'TEAM='+repr(team)+'\nASSIGNED='+repr(names)+'\n'+setup
    cells.append(cell(setup,identifier='setup'))
    for position,name in enumerate(names):
        cells.append(cell(f'## Slot {position+1}\nDefault: {name}. The model comes from MODELS_TO_RUN[{position}]; changing that list changes execution order.\n',kind='markdown',identifier=name+'-description'))
        selection=f"name=MODELS_TO_RUN[{position}] if len(MODELS_TO_RUN)>{position} else None\n"
        for index,label,source in [
            (1,'initialize',selection+'''if name is not None:
READY[name]=False
print('START initialization:',name,flush=True)
try:
    if name in MODELS_TO_RUN:
        initialize(name)
        invoke(name,'prepare')
        READY[name]=True
    else:
        print('NOT SELECTED for this session:',name,flush=True)
except Exception as error:
    print('INITIALIZATION FAILED:',name,type(error).__name__,str(error),flush=True)
    write_json(SESSION/(name+'_initialization_failure.json'),{'error':str(error),'status':'failed'})
print('END initialization:',name,'ready=',READY[name],flush=True)
'''),
            (2,'pilot-train',selection+'''if name is not None:
TRAINED[name]=False
print('START pilot/training:',name,flush=True)
if READY.get(name):
    try:
        invoke(name,'pilot')
        invoke(name,'train')
        TRAINED[name]=True
    except Exception as error:
        print('PILOT/TRAINING FAILED:',name,type(error).__name__,str(error),flush=True)
else:
    print('BLOCKED: initialization did not pass',flush=True)
print('END training:',name,'trained=',TRAINED[name],flush=True)
'''),
            (3,'evaluate-export',selection+'''if name is not None:
EVALUATED[name]=False
PACKAGED[name]=False
EXPORTED[name]=False
print('START evaluation/export:',name,flush=True)
if TRAINED.get(name):
    try:
        invoke(name,'evaluate')
        EVALUATED[name]=True
    except Exception as error:
        print('EVALUATION FAILED:',name,type(error).__name__,str(error),flush=True)
if TRAINED.get(name) and EVALUATED.get(name):
    try:
        invoke(name,'package')
        PACKAGED[name]=json.loads((RUNS[name]/'package_result.json').read_text())['produced']
    except Exception as error:print('MODEL PACKAGING FAILED:',str(error),flush=True)
if name in RUNS:
    try:
        invoke(name,'export')
        EXPORTED[name]=True
        from src.notebook_support import record_completion
        record_completion(SESSION,name,TRAINED.get(name,False),EVALUATED.get(name,False),PACKAGED.get(name,False))
    except Exception as error:print('EXPORT FAILED:',str(error),flush=True)
print('END evaluation/export:',name,'output=',str(RUNS.get(name)),flush=True)
''')]:
            # The optional slot guard wraps the whole stage, without hidden
            # dependence on the name left by a previously executed cell.
            prefix,body=source.split('if name is not None:\n',1)
            source=prefix+'if name is not None:\n'+''.join('    '+line if line.strip() else line for line in body.splitlines(keepends=True))
            cells.append(cell(source,identifier=f'{name}-{index}-{label}'))
    cells.append(cell("""print('START session summary',flush=True)
write_json(SESSION/'session_summary.json',{'team':TEAM,'initialized':READY,'training_completed':TRAINED,
    'evaluation_completed':EVALUATED,'selected_models':MODELS_TO_RUN,
    'model_packaging_completed':PACKAGED,
    'artifact_export_completed':EXPORTED,
    'source':environment(REPO),'published_output_retrieval_verified':False})
print(json.dumps({'initialized':READY,'training_completed':TRAINED,'evaluation_completed':EVALUATED},indent=2),flush=True)
print('END session summary. Save this executed notebook version and retain every output.',SESSION,flush=True)
export(SESSION)
failed=[name for name in MODELS_TO_RUN if not (EVALUATED.get(name) and PACKAGED.get(name) and EXPORTED.get(name))]
if failed:raise RuntimeError('Selected models did not complete evaluation/package/export: '+', '.join(failed))
""",identifier='summary'))
    for item in cells:
        if item['cell_type']=='code':ast.parse(''.join(item['source']))
    return {'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'experiment':{'team':team,'models':names,'status':'requires model/data GPU pilots'}},
        'nbformat':4,'nbformat_minor':5}


if __name__=='__main__':
    folder=ROOT/'Notebooks';folder.mkdir(parents=True,exist_ok=True)
    for team,names in ROSTER['teams'].items():
        target=folder/(team+'.ipynb')
        target.write_text(json.dumps(build(team,names),indent=1)+'\n',encoding='utf-8')
        print(target)
