"""Actual two-device autograd and atomic-resume preflight on author models."""
import copy
import hashlib
import torch
from .adapters import reduce_loss
from .engine import make_optimizers, applied_step
from .state import capture_rng, restore_rng, atomic_save


def torch_diagnostic(adapter,output):
    if torch.cuda.device_count()<2:
        raise RuntimeError("Two CUDA devices required; construction alone cannot admit training")
    rng=capture_rng()
    initial={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    adapter.to("cuda:0")
    parallel=torch.nn.DataParallel(adapter,device_ids=[0,1])
    optimizers,schedulers=make_optimizers(adapter)
    participating=set()
    def observed(module,inputs,result):
        participating.add(int(inputs[0].device.index))
    observer=adapter.register_forward_hook(observed)
    try:
        x=torch.rand(16,3,adapter.crop,adapter.crop,device="cuda:0")
        parallel.train()
        result,loss=applied_step(parallel,adapter,optimizers,x)
        torch.cuda.synchronize()
        if participating!={0,1}:
            raise RuntimeError("Both GPUs did not participate in the diagnostic forward")
        if not torch.isfinite(loss) or not torch.isfinite(result["reconstruction"]).all():
            raise RuntimeError("Nonfinite author diagnostic output/loss")
        saved={"synthetic_diagnostic_only":True,"production_epochs_completed":0,
               "model":adapter.state_dict(),"optimizers":[o.state_dict() for o in optimizers],
               "schedulers":[s.state_dict() for s in schedulers],"rng":capture_rng()}
        path=output/"diagnostic_roundtrip.pt"
        atomic_save(saved,path)
        restored=torch.load(path,map_location="cuda:0",weights_only=False)
        adapter.load_state_dict(restored["model"],strict=True)
        for optimizer,state in zip(optimizers,restored["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler,state in zip(schedulers,restored["schedulers"]):
            scheduler.load_state_dict(state)
        if not all(torch.equal(value,restored["model"][key]) for key,value in adapter.state_dict().items()):
            raise RuntimeError("Strict diagnostic checkpoint roundtrip differs")
        counts=result["complex_uses"].detach().cpu().tolist()
        return {"runtime_validation":"PASSED_TWO_GPU_FORWARD_BACKWARD_OPTIMIZER_ROUNDTRIP",
                "participating_cuda_devices":sorted(participating),"diagnostic_global_batch":16,
                "synthetic_diagnostic_only":True,"production_epochs_completed":0,
                "measured_complex_uses_per_image":counts,
                "measured_cbr_per_image":[n/(3*adapter.crop**2) for n in counts],
                "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "optimizer_state_count":len(restored["optimizers"]),
                "full_kodak_geometry_gpu_validated":False}
    finally:
        observer.remove()
        adapter.load_state_dict(initial,strict=True)
        adapter.zero_grad(set_to_none=True)
        restore_rng(rng)

