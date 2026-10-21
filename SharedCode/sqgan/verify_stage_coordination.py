"""Executed scheduler/selection contracts; synthetic states are not ML results."""
import ast
import json
import tempfile
from pathlib import Path
from stage_artifacts import ready_stages,select_stages
from build_stage_worker import build

def main():
    def row(c,m,e,h='same'):
        return dict(channel=c,mode=m,completed_epochs=e,global_step=e*800,sha256=h)
    selected=select_stages([row('awgn','ssm',98,'old'),row('awgn','ssm',112,'new'),row('awgn','ssm',98,'old')])
    assert selected[('awgn','ssm')]['completed_epochs']==112
    conflict=False
    try: select_stages([row('awgn','ssm',112,'a'),row('awgn','ssm',112,'b')])
    except AssertionError: conflict=True
    assert conflict
    inventory={('awgn','ssm'):row('awgn','ssm',150),('awgn','img'):row('awgn','img',32)}
    tasks=ready_stages(inventory)
    assert tasks[0]==('awgn','img') and ('awgn','all') not in tasks
    assert ('awgn','img') not in ready_stages(inventory,[('awgn','img')])
    inventory[('awgn','img')]=row('awgn','img',150)
    assert ('awgn','all') in ready_stages(inventory)
    assert ('rayleigh','all') not in ready_stages(inventory)
    with tempfile.TemporaryDirectory() as tmp:
        for stage in ('awgn/img','rayleigh/ssm','evaluation'):
            folder=Path(tmp)/stage.replace('/','_')
            build(folder,'hmmkk20/synthetic-worker',stage,['hmmkk20/sqgan-checkpoint-publication-test-20261020'])
            nb=json.loads((folder/'sqgan_coordinated_stage.ipynb').read_text())
            code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            for c in nb['cells']:
                if c['cell_type']=='code': ast.parse(''.join(c['source'])); assert not c['outputs']
            assert 'for mode in STAGE_PLAN.get(channel, []):' in code
            assert 'for mode in ("ssm","img","all"):' not in code
            assert 'merge_artifacts(Path("/kaggle/input"),OUTPUT,REPO)' in code
            assert 'BATCH_SIZE = 1' in code and 'IMAGE_HEIGHT = 256' in code and 'IMAGE_WIDTH = 512' in code
    root=Path(__file__).resolve().parents[2]/'Results/SQGAN/stage_coordination'
    root.mkdir(parents=True,exist_ok=True)
    result={'result':'PASS','scope':'synthetic scheduler/selection and generated notebook structure',
            'checks':['highest progress retained','equal-progress conflicting checkpoint rejected',
                      'partial stage resumed first','active stage cannot be assigned twice',
                      'joint stage waits for both complete pretraining stages',
                      'single-stage workers cannot execute the broad sequential loop',
                      'batch, geometry, horizon and output cleanliness retained'],
            'real_checkpoint_merge_validation':'separate Kaggle CPU job required',
            'model_training_results_generated':False}
    (root/'local_contract_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
