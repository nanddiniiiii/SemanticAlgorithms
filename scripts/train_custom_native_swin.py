"""Bounded mixed-data fine-tuning of the strictly loaded native Swin baseline."""
import argparse
import json
from pathlib import Path
import queue
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--resolved-sources',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=7200)
    args=parser.parse_args()
    import numpy as np
    import torch
    from src.artifacts import write_json,sha256,export_manifest
    from src.config import Config
    from src.data import loader
    from src.swin_native import build_native_swin
    from src.metrics import quality
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('This run requires two actual CUDA devices')
    random.seed(22);np.random.seed(22);torch.manual_seed(22)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    config=Config('swinjscc',variant='native_fixed_c32_custom_data',
        reproduction_status='adapted',dataset='frozen_images',device='cuda:0',
        batch_size=4,workers=2,crop_size=256,epochs=150,snr_db=10,
        learning_rate=1e-4,multi_gpu=True,amp=False,seed=22,
        model_options={'dataset_manifest':str(args.manifest.resolve()),
                       'resolved_sources':str(args.resolved_sources.resolve())})
    native,metadata=build_native_swin(args.checkpoint,'fixed_c32','cuda:0')
    write_json(args.output/'protocol.json',{'config':config.as_dict(),'model':metadata,
        'manifest_sha256':manifest['manifest_sha256'],
        'manifest_file_sha256':sha256(args.manifest),
        'native_architecture_loss_channel_preserved':True,
        'adaptations':['custom frozen parent images','author-weight fine-tuning',
            'global batch4 with DataParallel','fixed center256 development crops',
            'bounded update budget; not paper convergence'],
        'train_transform':'native random256 RGB crop /255, no flip or resize',
        'channel':'native fixed-C32 AWGN 10dB, including native replica-batch power normalization',
        'optimizer':'native Adam lr1e-4; no added scheduler',
        'test_pixels_opened':False})
    write_json(args.output/'environment.json',{'interpreter':sys.executable,
        'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    print('Verifying frozen training and development source bytes',flush=True)
    train_loader=loader(config,'train');dev_loader=loader(config,'val')
    print(json.dumps({'stage':'data_ready','train':len(train_loader.dataset),
        'development':len(dev_loader.dataset),'test_opened':False}),flush=True)
    observed=queue.Queue()
    hook=native.encoder.register_forward_pre_hook(
        lambda module,inputs:observed.put(str(inputs[0].device)))

    class TrainView(torch.nn.Module):
        def __init__(self,core):super().__init__();self.core=core
        def forward(self,image):
            recon,_,_,_,loss=self.core(image,10,32)
            # Expand the native replica mean so gather handles uneven chunks.
            return recon,loss.expand(image.shape[0])

    parallel=torch.nn.DataParallel(TrainView(native),device_ids=[0,1])
    optimizer=torch.optim.Adam(native.parameters(),lr=1e-4)
    start=time.monotonic();last_save=start;steps=0;images=0
    events=args.output/'steps.jsonl'

    def save(name,epoch):
        path=args.output/name;temporary=path.with_suffix('.tmp')
        torch.save({'state_dict':native.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'optimizer_steps':steps,'training_images_seen':images,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),'config':config.as_dict(),
            'manifest_sha256':manifest['manifest_sha256'],
            'resume_supported':False,'resume_note':'Weights and optimizer saved; exact mid-epoch sampler resume not implemented'},temporary)
        temporary.replace(path)

    def evaluate(label):
        native.eval();rows=[]
        torch.manual_seed(222)
        with torch.inference_mode():
            for index,batch in enumerate(dev_loader):
                image=batch['image'].cuda()
                recon=native(image,10,32)[0]
                values=quality(image,recon,use_lpips=False)
                rows.append({'sample':batch['sample'][0],**values})
                if args.pilot and index==1:break
                if (index+1)%50==0:
                    print(json.dumps({'stage':label,'images':index+1}),flush=True)
        write_json(args.output/(label+'.json'),{'images':len(rows),
            'mean_psnr_db':sum(r['psnr_db'] for r in rows)/len(rows),
            'rows':rows,'test_opened':False,'scope':'development, fixed center256 crops'})

    if not args.pilot:evaluate('development_before')
    # Restore a fixed training seed after deterministic development evaluation.
    torch.manual_seed(22)
    try:
        stop=False
        for epoch in range(1,151):
            parallel.train()
            for batch in train_loader:
                image=batch['image'].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                recon,losses=parallel(image);loss=losses.mean()
                if recon.shape!=image.shape or not torch.isfinite(loss):
                    raise FloatingPointError('Shape/nonfinite native Swin pilot failure')
                loss.backward()
                finite=torch.stack([torch.isfinite(p.grad).all() for p in native.parameters() if p.grad is not None]).all()
                if not finite:
                    raise FloatingPointError('Nonfinite native Swin gradients')
                optimizer.step();steps+=1;images+=len(image)
                if steps<=2 or steps%25==0:
                    row={'stage':'optimizer','step':steps,'epoch':epoch,'loss':float(loss.detach()),
                        'images_seen':images,'elapsed_seconds':time.monotonic()-start,
                        'peak_bytes':[torch.cuda.max_memory_allocated(i) for i in (0,1)]}
                    with events.open('a') as stream:stream.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                if time.monotonic()-last_save>=300:
                    save('last.pt',epoch);last_save=time.monotonic()
                if (args.pilot and steps>=2) or time.monotonic()-start>=args.max_seconds:
                    stop=True;break
            save('last.pt',epoch)
            if stop:break
        devices=set()
        while not observed.empty():devices.add(observed.get())
        write_json(args.output/'gpu_work.json',{'observed_encoder_devices':sorted(devices)})
        if not {'cuda:0','cuda:1'}.issubset(devices):
            raise RuntimeError('Both GPUs did not perform encoder work')
        if args.pilot and steps<2:
            raise RuntimeError('Pilot budget ended before two optimizer steps')
        saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
        native.load_state_dict(saved['state_dict'],strict=True)
        evaluate('development_after')
        write_json(args.output/'status.json',{'status':'pilot_completed' if args.pilot else 'budget_stopped',
            'optimizer_steps':steps,'images_seen':images,'epochs_entered':epoch,
            'elapsed_seconds':time.monotonic()-start,'strict_reload':True,
            'test_opened':False,'paper_convergence_reproduced':False})
    except Exception as error:
        write_json(args.output/'status.json',{'status':'failed','optimizer_steps':steps,
            'error':str(error),'test_opened':False})
        raise
    finally:
        hook.remove();export_manifest(args.output)


if __name__=='__main__':main()
