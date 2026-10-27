"""Frozen mixed-data ADJSCC adaptation, always separate from native baselines."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=7200)
    args=parser.parse_args()
    import torch
    import random
    import numpy as np
    import queue
    from src.config import Config
    from src.adjscc import build_adjscc
    from src.sources import fetch_source
    from src.training import train
    from src.events import Events
    from src.artifacts import write_json,sha256,export_manifest
    assert torch.cuda.is_available()
    assert torch.cuda.device_count()>=2,'Custom adaptation requests two actual GPUs'
    assert sha256(args.checkpoint)=='c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'
    manifest=json.loads(args.manifest.read_text());assert manifest['status']=='frozen'
    args.output.mkdir(parents=True,exist_ok=False)
    fetch_source('diffcom')
    torch.manual_seed(22);random.seed(22);np.random.seed(22)
    config=Config('adjscc',variant='modified',reproduction_status='adapted',dataset='frozen_images',
        device='cuda:0',checkpoint=str(args.checkpoint),crop_size=256,batch_size=8,workers=2,
        epochs=150,max_seconds=args.max_seconds,snr_db=10,seed=22,learning_rate=1e-4,
        multi_gpu=True,amp=False,model_options={'latent_channels':2,'dataset_manifest':str(args.manifest.resolve())})
    codec=build_adjscc(config)
    write_json(args.output/'config.json',config.as_dict())
    write_json(args.output/'dataset_identity.json',{'manifest_sha256':manifest['manifest_sha256'],
        'manifest_file_sha256':sha256(args.manifest),'counts':manifest['counts'],
        'test_used_for_training':False,'transforms':'random256traincrop/center256devcrop; no flips',
        'scope':'controlled custom-data adaptation; native baseline unchanged; no guaranteed improvement'})
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'execution_device':'DataParallel cuda:0 andcuda:1 fortraining; developmentCUDA0',
        'multi_gpu':True,'precision':'float32'})
    devices=queue.Queue()
    hook=codec.core.jscc_encoder.register_forward_hook(lambda module,inputs,output:devices.put(str(inputs[0].device)))
    train(config,codec,args.output,Events(args.output,config),pilot=args.pilot)
    hook.remove()
    observed=set()
    while not devices.empty():observed.add(devices.get())
    write_json(args.output/'gpu_work.json',{'observed_encoder_devices':sorted(observed)})
    assert {'cuda:0','cuda:1'}.issubset(observed),'Did not observe encoder work on both GPUs'
    saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
    if args.pilot and saved['step']<2:
        raise RuntimeError('Pilot did not complete both optimizer steps; training gate failed')
    export_manifest(args.output)
    print('Custom ADJSCC '+('pilot' if args.pilot else 'bounded training')+' finished: '+str(args.output),flush=True)


if __name__=='__main__':main()
