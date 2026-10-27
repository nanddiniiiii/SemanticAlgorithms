"""Prepare the published team notebook with SwinJSCC as the sole selected model."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
notebook=json.loads((ROOT/'Notebooks/TeamSuite/you.ipynb').read_text(encoding='utf-8'))
setup=notebook['cells'][1]
text=''.join(setup['source'])
before='MODELS_TO_RUN = ASSIGNED[:]'
if text.count(before)!=1:raise ValueError('Unexpected team notebook configuration')
text=text.replace(before,"MODELS_TO_RUN = ['swinjscc']")
setup['source']=text.splitlines(keepends=True)
for cell in notebook['cells']:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
folder=ROOT/'Kaggle/team_first_swin_training';folder.mkdir(parents=True,exist_ok=True)
(folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata={'id':'satyapaladuguplus/team-suite-first-swin-training','title':'Team suite first Swin training',
          'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
          'dataset_sources':['satyapaladuguplus/semantic-lossless-private-swin-assets',
                             'soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24'],
          'kernel_sources':[],'competition_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(folder)
