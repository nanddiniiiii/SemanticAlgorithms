"""Small real-author CPU diagnostics. No synthetic result is benchmark quality."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/"SharedCode/reproduction"


def test_deep_author_backward_and_measured_rate():
    # Subprocess isolates upstream absolute names (model/channel/net/utils).
    script='''
import sys, torch
sys.path.insert(0, "SharedCode/reproduction")
from model_runtime.sources import verified_source
from model_runtime.adapters import TorchAdapter, reduce_loss
torch.set_num_threads(2)
source,_=verified_source("DeepJSCC")
model=TorchAdapter("DeepJSCC",source,device="cpu")
x=torch.rand(2,3,128,128)
result=model(x)
assert result["reconstruction"].shape==x.shape
assert result["complex_uses"].tolist()==[8192,8192]
loss=reduce_loss(result)
expected=model.core.loss(result["reconstruction"]*255,x*255)
assert torch.equal(loss,expected)
loss.backward()
grads=[p.grad for p in model.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert abs(result["complex_uses"][0].item()/x[0].numel()-1/6)<1e-12
'''
    subprocess.run([sys.executable,"-c",script],cwd=ROOT,check=True,timeout=90)


def test_swin_author_base_c256_cpu_core():
    script='''
import sys, torch
sys.path.insert(0,"SharedCode/reproduction/.cpu_dependencies")
sys.path.insert(0,"SharedCode/reproduction")
from model_runtime.sources import verified_source
from model_runtime.adapters import TorchAdapter, reduce_loss
torch.set_num_threads(2)
source,_=verified_source("SwinJSCC")
model=TorchAdapter("SwinJSCC",source,device="cpu")
assert model.core.config.encoder_kwargs["depths"]==[2,2,6,2]
assert model.core.config.encoder_kwargs["C"]==256
x=torch.rand(1,3,256,256)
result=model(x)
assert result["reconstruction"].shape==x.shape
assert result["complex_uses"].tolist()==[32768]
loss=reduce_loss(result)
assert torch.isfinite(loss)
loss.backward()
grads=[p.grad for p in model.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert abs(result["complex_uses"][0].item()/x[0].numel()-1/6)<1e-12
'''
    subprocess.run([sys.executable,"-c",script],cwd=ROOT,check=True,timeout=180)

