"""Build single-stage workers around the unchanged author training code."""
import ast
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TEMPLATE=ROOT/'Notebooks/project_notebooks/Training/sqgan_div2k_channel/sqgan_div2k_awgn_rayleigh_training.ipynb'

def build(folder,kernel,stage,parents,template=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    nb=json.loads(Path(template or TEMPLATE).read_text())
    source=''.join(nb['cells'][1]['source'])
    plan={} if stage=='evaluation' else {stage.split('/')[0]:[stage.split('/')[1]]}
    source+='\nSTAGE_PLAN = '+repr(plan)+'\n'
    source+='WORKER_STAGE = '+repr(stage)+'\n'
    source+='\n'+(Path(__file__).parent/'input_paths.py').read_text()+'\nDIV2K_TRAIN,DIV2K_VAL=resolve_div2k()\n'
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    source=''.join(nb['cells'][3]['source']).split('# Restore only the exported training state')[0]
    module=(Path(__file__).parent/'stage_artifacts.py').read_text()
    source+='\n(COMPAT_DIR/"stage_artifacts.py").write_text('+repr(module)+')\n'
    source+='sys.path.insert(0,str(COMPAT_DIR))\nfrom stage_artifacts import merge_artifacts\nmerge_artifacts(Path("/kaggle/input"),OUTPUT,REPO)\n'
    nb['cells'][3]['source']=source.splitlines(keepends=True)
    source=''.join(nb['cells'][8]['source'])
    assert 'for mode in ("ssm","img","all"):' in source
    source=source.replace('for mode in ("ssm","img","all"):', 'for mode in STAGE_PLAN.get(channel, []):')
    guard='''                if mode=="all":
                    for dependency in ("ssm","img"):
                        required=stage_state_dir/f"{channel}_{dependency}.json"
                        assert required.exists() and json.loads(required.read_text())["completed_epochs"]==150, "Joint stage prerequisite missing"
'''
    source=source.replace('                cfg=channel_config(channel,mode,parts if mode=="all" else None)',guard+'                cfg=channel_config(channel,mode,parts if mode=="all" else None)')
    source=source.replace('manifest={"upstream_commit":', 'manifest={"worker_stage":WORKER_STAGE,"stage_plan":STAGE_PLAN,"upstream_commit":')
    nb['cells'][8]['source']=source.splitlines(keepends=True)
    nb['cells'][0]['source']=['# SQ-GAN coordinated worker: '+stage+'\n\nOnly the assigned stage may train. Shared parent artifacts are validated and selected by full-epoch progress. Joint training requires both 150-epoch prerequisites. The author architecture, batch size, dataset exposure and precision remain unchanged.\n']
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]; cell['execution_count']=None
            ast.parse(''.join(cell['source']))
    name='sqgan_coordinated_stage.ipynb'
    (folder/name).write_text(json.dumps(nb,indent=1))
    metadata={'id':kernel,'title':kernel.split('/')[1].replace('-',' '),'code_file':name,
              'kernel_type':'notebook','language':'python','is_private':True,'enable_gpu':True,
              'enable_tpu':False,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
              'dataset_sources':['sherylmehta/kodak-dataset'],'competition_sources':[],'model_sources':[],
              'kernel_sources':['jagan028/div2k-dataset-generation-for-isr',*sorted(set(parents))]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    return folder

if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4:])
