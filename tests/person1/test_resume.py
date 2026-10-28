import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from src.person1.protocol import options,require_complete
from src.person1.torch_worker import train,load


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Conv2d(3,3,1)
    def forward(self,x):return {'reconstruction':self.encoder(x)+torch.randn_like(x)*.01}
    def loss(self,out,batch):return (out['reconstruction']-batch['image']).square().mean()
    def optimizer(self):return torch.optim.Adam(self.parameters(),lr=.001)


def manifest(tmp_path):
    rows=[]
    for i in range(6):
        name=str(i)+'.png';Image.fromarray(np.full((8,8,3),20+i*15,dtype=np.uint8)).save(tmp_path/name)
        rows.append(dict(source='fixture',relative_path=name,split='train' if i<4 else 'val'))
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(images=rows,identity=identity,roots={'fixture':str(tmp_path)})))
    return path


def test_interrupted_continuation_matches_uninterrupted(tmp_path):
    source=manifest(tmp_path);config=options('deepjscc',source);config.update(epochs=2,batch_size=2,crop=8,max_seconds=100,data_parallel=False)
    torch.manual_seed(71);initial=Small();full=copy.deepcopy(initial);part=copy.deepcopy(initial)
    complete=train(full,config,tmp_path/'full')
    partial=train(part,{**config,'stop_after_steps':1},tmp_path/'chunk1')
    assert partial['status']=='partial' and partial['completed_epochs']=={'codec':0}
    resumed=Small();result=train(resumed,{**config,'resume_checkpoint':str(tmp_path/'chunk1/checkpoints/last.pt')},tmp_path/'chunk2')
    assert result['completed_epochs']=={'codec':2} and result['images_seen']==8
    for key,value in full.state_dict().items():assert torch.equal(value,resumed.state_dict()[key]),key
    assert load(tmp_path/'full/checkpoints/last.pt')['progress']['optimizer_states']['codec']['param_groups']==load(tmp_path/'chunk2/checkpoints/last.pt')['progress']['optimizer_states']['codec']['param_groups']
    require_complete(complete,{'codec':2})


def test_changed_training_contract_cannot_resume(tmp_path):
    import pytest
    source=manifest(tmp_path);config=options('deepjscc',source);config.update(epochs=2,batch_size=2,crop=8,max_seconds=100,data_parallel=False,stop_after_steps=1)
    train(Small(),config,tmp_path/'one')
    with pytest.raises(ValueError,match='contract changed'):
        train(Small(),{**config,'learning_rate':.02,'resume_checkpoint':str(tmp_path/'one/checkpoints/last.pt')},tmp_path/'two')
