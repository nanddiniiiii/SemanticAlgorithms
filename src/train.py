"""Budgeted common-data training; model-native objectives, atomic checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

from .runtime import write_json


def train(model, manifest, run, options, pilot=False):
    import numpy as np
    import torch
    from .dataset import ParentImages
    run=Path(run); (run/'checkpoints').mkdir(exist_ok=True,parents=True)
    device=next(model.parameters()).device
    seed=int(options.get('seed',22))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    crop=int(options.get('crop',256));batch_size=int(options.get('batch_size',2))
    if batch_size<1:
        raise ValueError('batch_size must be positive')
    dataset=ParentImages(manifest,'train',crop)
    development=ParentImages(manifest,'val',crop)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,
        num_workers=0 if pilot else int(options.get('workers',2)),pin_memory=device.type=='cuda',
        generator=torch.Generator().manual_seed(seed))
    multi=bool(options.get('data_parallel',True)) and device.type=='cuda'
    if multi and (torch.cuda.device_count()<2 or batch_size<2):
        raise RuntimeError('Requested DataParallel needs two CUDA GPUs and a global batch >=2')
    parallel=torch.nn.DataParallel(model,device_ids=[0,1]) if multi and not getattr(model,'native_data_parallel',False) else model
    observed_devices=set();hooks=[]
    candidates={'encoder','core.encoder','core.jscc_encoder','core.ga',
                'core.ntc.g_a0','core.patch_downsample_1','networks.E.module','networks.E','encoder.encoder_2d'}
    if multi:
        def observe(module,inputs):
            if inputs and torch.is_tensor(inputs[0]):observed_devices.add(str(inputs[0].device))
        for name,module in model.named_modules():
            if name in candidates:hooks.append(module.register_forward_pre_hook(observe))
        if not hooks:raise RuntimeError('No encoder-work probe registered for requested DataParallel model')
    budget=float(options.get('max_seconds',14400))
    if not 0<budget<=14400:
        raise ValueError('Training budget must be in (0,14400] seconds')
    deadline=time.monotonic()+budget;started=time.monotonic();last_save=started
    steps=0;images_seen=0;best=float('inf');epoch=0
    # The four-hour envelope includes both ISEC prerequisites, not four hours each.
    stages=['jscc','denoiser'] if hasattr(model,'stage') else ['codec']
    completed_stages=[];stage_step_counts={}

    def save(name, epoch_complete=False):
        target=run/'checkpoints'/name;temporary=target.with_suffix('.tmp')
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'aux_optimizer':aux_optimizer.state_dict() if aux_optimizer else None,
                    'scheduler':scheduler.state_dict() if scheduler else None,
                    'step':steps,'images_seen':images_seen,'epoch':epoch,
                    'epoch_complete':epoch_complete,'stage':stage_name,'completed_stages':completed_stages[:],
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'python_rng':random.getstate(),'numpy_rng':{**dict(zip(
                        ('generator','keys','position','has_gauss','cached_gaussian'), np.random.get_state())),
                        'keys':np.random.get_state()[1].tolist()},
                    'options':options,'resume_supported':False,
                    'resume_note':'Optimizer/RNG preserved; exact mid-epoch sampler resume is not implemented'},temporary)
        temporary.replace(target)

    for stage_index,stage_name in enumerate(stages):
        if time.monotonic()>=deadline:
            break
        if hasattr(model,'stage'):
            model.stage=stage_name
            for part in (model.encoder,model.decoder):
                for parameter in part.parameters():
                    parameter.requires_grad_(stage_name=='jscc');parameter.grad=None
            for parameter in model.denoiser.parameters():parameter.requires_grad_(stage_name=='denoiser')
        optimizer=model.optimizer()
        if hasattr(model,'scheduler'):
            scheduler=model.scheduler(optimizer)
        elif options.get('algorithm')=='isec':
            interval=30 if stage_name=='jscc' else max(1,int(options.get('epochs',150))//2)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=interval,gamma=.1)
        elif options.get('algorithm')=='mdjcm':
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4000,4500],gamma=.1)
        else:scheduler=None
        quantiles=[p for n,p in model.named_parameters() if n.endswith('.quantiles') and p.requires_grad]
        aux_optimizer=torch.optim.Adam(quantiles,lr=1e-3) if quantiles else None
        if aux_optimizer and not hasattr(model,'aux_loss'):
            raise RuntimeError('Entropy quantiles found without an auxiliary-objective adapter')
        stage_deadline=deadline if stage_index==len(stages)-1 else started+budget/len(stages)
        stage_steps=0
        for epoch in range(1,int(options.get('epochs',150))+1):
            parallel.train()
            if hasattr(model,'stage') and stage_name=='denoiser':
                model.encoder.eval();model.decoder.eval()
            epoch_complete=True
            for batch in loader:
                if time.monotonic()>=stage_deadline and stage_steps:
                    epoch_complete=False;break
                image=batch['image'].to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if aux_optimizer:aux_optimizer.zero_grad(set_to_none=True)
                try:
                    output=parallel(image)
                    loss=model.loss(output,{'image':image})
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise FloatingPointError('Nonfinite gradient')
                    clip=float(options.get('clip_norm',1.0))
                    if clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                    optimizer.step()
                    if aux_optimizer:
                        aux=model.aux_loss();aux.backward();aux_optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    write_json(run/'oom.json',{'batch_size':batch_size,'crop':crop,'steps':steps,
                        'action':'Stop; reduce batch explicitly and create a new run; no hidden protocol change'})
                    if steps:save('last.pt')
                    raise
                steps+=1;stage_steps+=1;images_seen+=len(image)
                event={'stage':stage_name,'epoch':epoch,'step':steps,'loss':float(loss.detach()),
                       'images_seen':images_seen,'elapsed_seconds':time.monotonic()-started}
                with (run/'training.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(event)+'\n')
                if stage_steps<=2 or steps%25==0:print(json.dumps(event),flush=True)
                if time.monotonic()-last_save>=300:
                    save('last.pt');last_save=time.monotonic()
                if pilot and stage_steps>=2:
                    epoch_complete=False;break
            if scheduler and epoch_complete:scheduler.step()
            save('last.pt',epoch_complete)
            if (pilot and stage_steps>=2) or time.monotonic()>=stage_deadline:break
            if pilot:continue
            # Select only on development parents; Kodak is not opened here.
            parallel.eval();loss_sum=0.;count=0
            with torch.no_grad():
                for index in range(min(len(development),int(options.get('dev_images',16)))):
                    if time.monotonic()>=stage_deadline:break
                    sample=development[index]['image'].unsqueeze(0).to(device)
                    prediction=model(sample)
                    value=model.loss(prediction,{'image':sample})
                    loss_sum+=float(value);count+=1
            if count and loss_sum/count<best:
                best=loss_sum/count;save('best.pt',epoch_complete)
            print(f'Development {stage_name}: {count} images; mean objective={loss_sum/count if count else None}',flush=True)
        if pilot and stage_steps<2:
            raise RuntimeError(f'Pilot incomplete: {stage_name} performed {stage_steps}/2 required updates')
        stage_step_counts[stage_name]=stage_steps
        completed_stages.append(stage_name)
        save('last.pt',epoch_complete)
    if steps==0:
        raise RuntimeError('Budget expired before any optimizer update')
    if set(completed_stages)!=set(stages) or any(stage_step_counts.get(name,0)<1 for name in stages):
        write_json(run/'training_summary.json',{'status':'partial_prerequisites',
            'steps':steps,'stages_executed':completed_stages,'stage_optimizer_steps':stage_step_counts,
            'reason':'Not every required stage received updates; checkpoint is not an implemented ISEC result'})
        raise RuntimeError('Training budget expired before all prerequisite stages executed')
    for hook in hooks:hook.remove()
    if pilot and set(completed_stages)!=set(stages):
        raise RuntimeError('Pilot did not execute every prerequisite stage: '+str(completed_stages))
    if multi and not {'cuda:0','cuda:1'}.issubset(observed_devices):
        raise RuntimeError('DataParallel requested but encoder work was not observed on both GPUs: '+str(observed_devices))
    # Validate current geometry buffers against the saved checkpoint.
    saved=torch.load(run/'checkpoints'/'last.pt',map_location=device,weights_only=True)
    model.load_state_dict(saved['state_dict'],strict=True)
    result={'status':'pilot_completed' if pilot else 'budget_limited_training_completed',
            'steps':steps,'images_seen':images_seen,'stages_executed':completed_stages,
            'stage_optimizer_steps':stage_step_counts,'observed_encoder_devices':sorted(observed_devices),
            'elapsed_seconds':time.monotonic()-started,'strict_reload':True,
            'paper_convergence_reproduced':False,'data_parallel_requested':multi,
            'budget_seconds':budget,'deadline_scope':'soft boundary between steps; checkpoint and one in-flight step may exceed budget'}
    write_json(run/'training_summary.json',result)
    return result
