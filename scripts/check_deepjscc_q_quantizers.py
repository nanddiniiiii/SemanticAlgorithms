"""Audit pinned native quantizer behavior without changing its algorithms."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from src.sources import source_path,fetch_source
from src.artifacts import write_json


def main():
    receipt=fetch_source('deepjscc_q')
    path=source_path('deepjscc_q')/'modules/quantizers.py'
    tree=ast.parse(path.read_text())
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in {'SoftHardQuantize','LearnedQuantize'}]
    namespace={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(path),'exec'),namespace)
    embed=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]).t()
    options=dict(n_embed=4,embed_dim=2,embed_init=embed,commitment=0.,anneal='linear',
                 sigma_start=5.,sigma_max=200.,sigma_period=1000,sigma_scale=5.)
    image=torch.tensor([[.2,.3,.7,-.2]])
    fixed=namespace['SoftHardQuantize'](**options).eval()
    fixed_output,_=fixed(image)
    learned=namespace['LearnedQuantize'](**options,normalize=True).eval()
    before=int(learned.steps.item())
    learned_output,_=learned(image)
    after=int(learned.steps.item())
    learned.train();training_before=int(learned.steps.item());learned(image)
    result={'source_commit':receipt['commit'],'scope':'native quantizer behavior on synthetic input; no architecture/training reproduction',
        'fixed_quantizer_eval_finite':bool(torch.isfinite(fixed_output).all()),
        'learned_quantizer_eval_finite':bool(torch.isfinite(learned_output).all()),
        'learned_eval_annealing_counter_change':after-before,
        'learned_train_annealing_counter_change':int(learned.steps.item())-training_before,
        'source_modified':False}
    write_json(ROOT/'Results/implementation_checks/deepjscc_q_quantizers.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
