"""Generate two thin Kaggle launchers; all model logic stays in src/."""
import json
from pathlib import Path


def cell(text, kind='code'):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':
        value.update(execution_count=None,outputs=[])
    return value


def notebook(account, names):
    cells=[cell(f'# Semantic Lossless — account {account}\n\nRun the setup cell, configure native dataset/checkpoint paths, then run three cells per model. Pending integrations are skipped explicitly. GPU pilots must pass before training. No cell deletes checkpoints or earlier results.','markdown'),cell('''from pathlib import Path
import os, subprocess, sys, json
# Mount a checkout containing the new src/semantic_lossless package.
# The historical remote commit alone does not contain this implementation yet.
REPO = Path('/kaggle/working/Semantic_Lossless')
assert (REPO / 'src/semantic_lossless/cli.py').is_file(), 'Mount/clone the updated implementation checkout first'
os.chdir(REPO)
os.environ['SEMANTIC_LOSSLESS_ROOT'] = str(REPO)
sys.path.insert(0,str(REPO/'src'))
from semantic_lossless.catalog import get_model
# Set only native paths here. Credentials never belong in notebooks.
CONFIGS = {}  # e.g. {'jcm': '/kaggle/working/jcm.json'}
TRAIN_MODELS = set()  # Explicitly select audited native recipes; generic examples are adapted runs.
READY = {}
TRAINED = {}
def command(stage, config, inference_only=False):
    extra = ['--inference-only'] if inference_only else []
    subprocess.run([sys.executable,'-u','semcom.py',stage,'--config',str(config)]+extra,check=True)
def latest_run(config):
    data=json.loads(Path(config).read_text())
    root=Path(data.get('output_root','Results/experiments'))/data['algorithm']
    return max(root.iterdir(),key=lambda p:p.stat().st_mtime_ns)
''')]
    for name in names:
        cells.append(cell(f'## {name}\nPrepare/pilot → train or retain → evaluate/profile/export.','markdown'))
        cells.append(cell(f'''name = {name!r}
config = CONFIGS.get(name)
spec = get_model(name)
READY[name] = False
if not config:
    print(name, 'SKIPPED: configure its native dataset and checkpoint paths')
elif spec.backend in ('pending','unavailable'):
    print(name, 'BLOCKED:',spec.note)
else:
    command('prepare',config)
    if name == 'shared_semantic_codebook':
        data=json.loads(Path(config).read_text())
        if not data.get('model_options', {{}}).get('codebook'):
            command('construct',config)
            construction_run=latest_run(config)
            book=construction_run/'codebook.pt'
            if not book.is_file():
                raise RuntimeError('Codebook construction did not complete')
            data.setdefault('model_options', {{}})['codebook']=str(book.resolve())
            prepared_config=construction_run/'codec_config.json'
            prepared_config.write_text(json.dumps(data,indent=2))
            CONFIGS[name]=config=str(prepared_config.resolve())
    command('pilot',config,inference_only=name not in TRAIN_MODELS)
    READY[name] = True
'''))
        cells.append(cell(f'''name = {name!r}
if READY.get(name) and get_model(name).trainable and name in TRAIN_MODELS:
    command('train',CONFIGS[name])
    TRAINED[name] = latest_run(CONFIGS[name])
    # Evaluate the newly trained best checkpoint rather than original weights.
    best = TRAINED[name]/'best.pt'
    if not best.is_file():
        raise RuntimeError('No fully validated best checkpoint; inspect run status before evaluation')
    data=json.loads(Path(CONFIGS[name]).read_text())
    data['checkpoint']=str(best.resolve())
    data.setdefault('model_options', {{}})['from_scratch']=False
    evaluation_config=TRAINED[name]/'evaluation_config.json'
    evaluation_config.write_text(json.dumps(data,indent=2))
    CONFIGS[name]=str(evaluation_config.resolve())
elif READY.get(name):
    print(name, 'Retaining pretrained weights; long training not selected in this launcher')
else:
    print(name, 'SKIPPED: pilot has not passed')
'''))
        cells.append(cell(f'''name = {name!r}
if READY.get(name):
    command('evaluate',CONFIGS[name])
    evaluation_run=latest_run(CONFIGS[name])
    command('profile',CONFIGS[name])
    subprocess.run([sys.executable,'semcom.py','export','--run-dir',str(evaluation_run)],check=True)
    subprocess.run([sys.executable,'semcom.py','collect'],check=True)
    # Kaggle retains /kaggle/working as notebook output after Save & Run All.
    # Results remain in algorithm/run folders; collection merges them by event ID.
else:
    print(name, 'SKIPPED: no verified pilot')
'''))
    for index,item in enumerate(cells):
        item['id']=f'{account.lower()}-cell-{index}'
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}

if __name__=='__main__':
    target=Path('Notebooks/Current');target.mkdir(parents=True,exist_ok=True)
    queues={'A':['deepjscc','ntscc','sqgan','adjscc','shared_semantic_codebook','diffcom','hybrid_semantic_complementary'],
            'B':['swinjscc','jcm','mdjcm','mambajscc','deepjscc_q','diffjscc','masked_vqvae','dit_jscc','hda_deepsc']}
    for account,names in queues.items():
        path=target/f'account_{account.lower()}.ipynb'
        path.write_text(json.dumps(notebook(account,names),indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
        for item in json.loads(path.read_text(encoding='utf-8'))['cells']:
            if item['cell_type']=='code':
                compile(''.join(item['source']),str(path),'exec')
        print(path)
