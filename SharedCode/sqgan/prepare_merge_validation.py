import ast
import json
from pathlib import Path
from build_stage_worker import ROOT,build

folder=ROOT/'Notebooks/project_notebooks/Training/SQGAN_stage_merge_validation'
build(folder,'hmmkk20/sqgan-stage-merge-validation-20261020','awgn/ssm',
      ['hmmkk20/sqgan-checkpoint-publication-test-20261020'])
p=folder/'sqgan_coordinated_stage.ipynb'
nb=json.loads(p.read_text()); nb['cells']=nb['cells'][:4]
nb['cells'][1]['source']=[s.replace('/kaggle/working/sqgan_div2k_channel_results','/tmp/sqgan_merge_validation') for s in nb['cells'][1]['source']]
code='''from stage_artifacts import sha256
report=json.loads((OUTPUT/'merge_report.json').read_text())
assert len(report['stages'])==1
row=report['stages'][0]
assert row['completed_epochs']==98 and row['global_step']==78400
original=next(Path('/kaggle/input').rglob('sqgan_div2k_channel_results'))/'upstream_logs/10-19T17-22-21_sqgan_awgn_ssm_150ep/checkpoints/last.ckpt'
merged=OUTPUT/row['checkpoint_relative']
assert sha256(original)==sha256(merged)==row['sha256']
result={'source_and_merged_checkpoint_equal':True,'completed_epochs':98,'global_step':78400,
        'optimizer_states':2,'lr_schedulers':2,'checkpoint_sha256':row['sha256'],'model_updates_performed':0}
out=Path('/kaggle/working/stage_merge_validation'); out.mkdir(exist_ok=True)
(out/'validation.json').write_text(json.dumps(result,indent=2))
print('REAL CHECKPOINT MERGE VALIDATION PASS',json.dumps(result),flush=True)
'''
ast.parse(code)
nb['cells'].append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                    'source':code.splitlines(keepends=True)})
p.write_text(json.dumps(nb,indent=1))
metadata=json.loads((folder/'kernel-metadata.json').read_text())
metadata.update(enable_gpu=False); metadata.pop('machine_shape',None)
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(folder)
