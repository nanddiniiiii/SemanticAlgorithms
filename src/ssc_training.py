"""SSC distributed training preserving native per-depth EMA synchronization."""
import argparse
import json
from pathlib import Path
import time
import os


def update_parameter_ema(model, state, decay=.999):
    """Update every state-dict alias of tied parameters; buffers follow native behavior."""
    import torch
    with torch.no_grad():
        for name,parameter in model.named_parameters(remove_duplicate=False):
            state[name].mul_(decay).add_(parameter.detach(),alpha=1-decay)


def run(options, directory, pilot=False):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from .dataset import ParentImages
    from .models import build,load_training_state
    from .runtime import write_json
    if int(options.get('epochs',150))<1 or int(options.get('batch_size',2))<1 or not 0<float(options.get('max_seconds',14400))<=14400:
        raise ValueError('SSC requires positive epochs/batch and budget in (0,14400] seconds')
    if options.get('algorithm')!='ssc':raise ValueError('SSC worker requires algorithm=ssc')
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    use_cuda=options.get('device','cpu').startswith('cuda')
    if use_cuda:torch.cuda.set_device(rank)
    device=f'cuda:{rank}' if use_cuda else 'cpu'
    if world>1:dist.init_process_group(backend='nccl' if use_cuda else 'gloo')
    try:
        torch.manual_seed(int(options.get('seed',22)))
        model=build('ssc',options,device)
        parallel=DistributedDataParallel(model,device_ids=[rank] if use_cuda else None,
            broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
        torch.manual_seed(int(options.get('seed',22))+rank)
        optimizer=model.optimizer()
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[250000,400000,450000,475000],gamma=.5)
        data=ParentImages(options['manifest'],'train',256)
        if len(data)==0:raise ValueError('SSC training split is empty')
        sampler=torch.utils.data.DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=int(options.get('seed',22))) if world>1 else None
        global_batch=int(options.get('batch_size',2))
        if global_batch%world:raise ValueError('SSC global batch must divide the distributed world size')
        loader=torch.utils.data.DataLoader(data,batch_size=global_batch//world,sampler=sampler,shuffle=sampler is None,
            num_workers=int(options.get('workers',0)),drop_last=False)
        run=Path(directory);(run/'checkpoints').mkdir(parents=True,exist_ok=True)
        model.train();started=time.monotonic();step=0;images=0;last_save=started
        ema={key:value.detach().clone() for key,value in model.state_dict().items()}
        def save():
            if rank:return
            destination=run/'checkpoints/last.pt';temporary=destination.with_suffix('.tmp')
            torch.save({'state_dict':model.state_dict(),'ema_state_dict':ema,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'step':step,'images_seen':images,'epoch':epoch,
                'model_ema_decay':.999,'torch_rng':torch.get_rng_state(),'resume_supported':False},temporary)
            temporary.replace(destination)
        for epoch in range(int(options.get('epochs',150))):
            if sampler:sampler.set_epoch(epoch)
            for batch in loader:
                stop=torch.tensor(int((pilot and step>=2) or time.monotonic()-started>=min(float(options.get('max_seconds',14400)),14400)),device=device)
                if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():break
                image=batch['image'].to(device)
                optimizer.zero_grad();output=parallel(image);loss=model.loss(output,{'image':image})
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite SSC objective')
                loss.backward();optimizer.step();scheduler.step();step+=1;images+=len(image)*world
                update_parameter_ema(model,ema)
                if rank==0:
                    with (run/'training.jsonl').open('a') as stream:stream.write(json.dumps({'step':step,'epoch':epoch,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started})+'\n')
                    if step==1 or step%25==0:print('SSC step',step,'loss',float(loss.detach()),flush=True)
                if time.monotonic()-last_save>=300:save();last_save=time.monotonic()
            save()
            if stop.item():break
        if step< (2 if pilot else 1):raise RuntimeError('SSC did not complete required optimizer updates')
        if world>1:dist.barrier()
        saved=torch.load(run/'checkpoints/last.pt',map_location=device,weights_only=True)
        load_training_state(model,saved['state_dict'])
        if rank==0:
            write_json(run/'model.json',model.metadata)
            write_json(run/'training_summary.json',{'status':'pilot_completed' if pilot else 'budget_limited_training_completed',
                'steps':step,'images_seen':images,'distributed_world_size':world,'strict_reload':True,
                'elapsed_seconds':time.monotonic()-started,'objective':'native MSE+commitment+bitflip',
                'model_ema_saved':True,'paper_convergence_reproduced':False})
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run',required=True);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.run,args.pilot)

if __name__=='__main__':main()
