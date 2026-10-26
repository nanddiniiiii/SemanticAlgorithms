"""Approved spatial reconstruction-only JCM adaptation; native JCM is untouched."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--resolved-sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=7200)
    args=parser.parse_args()
    import numpy as np
    import torch
    import queue
    from semantic_lossless.config import Config
    from semantic_lossless.jcm_spatial import build_jcm_spatial
    from semantic_lossless.training import train
    from semantic_lossless.events import Events
    from semantic_lossless.artifacts import write_json,sha256,export_manifest
    assert torch.cuda.device_count()>=2
    random.seed(22);np.random.seed(22);torch.manual_seed(22)
    manifest=json.loads(args.manifest.read_text());assert manifest['status']=='frozen'
    args.output.mkdir(parents=True,exist_ok=False)
    # Config/events use the parent algorithm; an explicit variant separates this run.
    config=Config('jcm',variant='modified',reproduction_status='adapted',dataset='frozen_images',
        checkpoint=str(args.checkpoint),device='cuda:0',crop_size=256,batch_size=8,workers=2,
        epochs=150,max_seconds=args.max_seconds,snr_db=10,seed=22,learning_rate=1e-4,
        multi_gpu=True,amp=False,model_options={'dataset_manifest':str(args.manifest.resolve()),
            'resolved_sources':str(args.resolved_sources.resolve()),
            'approved_variant':'spatial_reconstruction_only','native_jcm_input_contract':False})
    codec=build_jcm_spatial(args.checkpoint,device='cuda:0',snr_db=10)
    write_json(args.output/'config.json',config.as_dict());write_json(args.output/'model.json',codec.metadata)
    write_json(args.output/'dataset_identity.json',{'manifest_sha256':manifest['manifest_sha256'],
        'manifest_file_sha256':sha256(args.manifest),'counts':manifest['counts'],
        'training':'random256RGBcrop; noflips; nativeJCMclassificationlabelsnotused',
        'scope':'user-approvedarchitecture/objectiveadaptation; notnativeJCMpaperreproduction'})
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'multi_gpu':True,'precision':'FP32'})
    devices=queue.Queue()
    hook=codec.encoder.register_forward_pre_hook(lambda module,inputs:devices.put(str(inputs[0].device)))
    events=Events(args.output,config);events.identity['variant']='modified_spatial_reconstruction_only'
    train(config,codec,args.output,events,pilot=args.pilot)
    hook.remove();observed=set()
    while not devices.empty():observed.add(devices.get())
    write_json(args.output/'gpu_work.json',{'observed_encoder_devices':sorted(observed)})
    assert {'cuda:0','cuda:1'}.issubset(observed)
    state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
    if args.pilot and state['step']<2:raise RuntimeError('Pilot failed required two-step gate')
    codec.load_state_dict(state['state_dict'],strict=True)
    export_manifest(args.output)
    print('Spatial JCM adaptation finished: '+str(args.output),flush=True)


if __name__=='__main__':main()
