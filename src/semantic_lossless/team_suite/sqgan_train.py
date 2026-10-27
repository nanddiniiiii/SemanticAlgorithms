"""Author masked SQ-GAN all-mode fine-tuning with frozen pseudo-segmentation.

The author codec has no radio channel. Training remains source-codec training;
we do not rename it AWGN JSCC or inject Gaussian noise into discrete VQ indices.
"""
from pathlib import Path
import importlib
import time

from .runtime import write_json


def align_cityscapes_loss_weights(loss):
    """The released loss excludes null but its YAML supplies the null weight."""
    weights=loss.label_weight_tensor
    if weights is None or weights.numel()!=20:
        raise ValueError('Unexpected SQ-GAN Cityscapes loss-weight configuration')
    loss.label_weight_tensor=weights[:-1].clone()


def build(options):
    import torch
    import random
    import numpy as np
    import yaml
    from torch import nn
    from torch.nn import functional as F
    from transformers import SegformerForSemanticSegmentation
    from semantic_lossless.sources import source_path
    from semantic_lossless.codecs import upstream_import
    seed=int(options.get('seed',22));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    directory=source_path('sqgan')
    params=yaml.safe_load((directory/'config/sqgan_cityscapes.yml').read_text())['model']['params']
    params.update(mode='all',ckpt_path=None,ckpt_path_parts=None)
    with upstream_import(directory,('models','modules','utils')):
        core=importlib.import_module('models.sqgan').SQGAN(**params)
    align_cityscapes_loss_weights(core.loss)
    names=['encoder','decoder','masker','demasker','quantize','encoder_semantic','decoder_semantic',
           'masker_semantic','demasker_semantic','quantize_semantic']
    for name in names:
        state=torch.load(Path(options['checkpoint'])/(name+'.ckpt'),map_location='cpu',weights_only=True)
        getattr(core,name).load_state_dict(state,strict=True)
    original_masker_settings={name:(getattr(core,name).topk_ratio_prob,
        list(getattr(core,name).topk_ratio_range)) for name in ('masker','masker_semantic')}
    teacher=SegformerForSemanticSegmentation.from_pretrained(options['teacher_model'],
        revision=options['teacher_revision']).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)

    class View(nn.Module):
        def __init__(self):
            super().__init__();self.core=core;self.teacher=teacher
            self.metadata={'source_codec_only':True,'native_mode':'all','pseudo_labels':True,
                'teacher':options['teacher_model'],'teacher_revision':options['teacher_revision'],
                'teacher_preprocessing':'ImageNet normalization at actual256x512 crop; no1024 resize',
                'teacher_in_forward_cost':True,
                'training':'author adversarial + perceptual + VQ objective; pretrained components; random discriminator',
                'radio_channel':'not implemented; no claim of AWGN training',
                'adaptations':['frozen teacher on custom crops','bounded fine-tuning instead of native convergence',
                               'remove unused null-class weight to match author loss semantic[:, :-1]',
                               'no augmentation of traffic objects outside native training-step label enhancer',
                               'manual two-optimizer loop instead of Lightning Trainer']}

        def train(self,mode=True):
            super().train(mode);self.teacher.eval()
            for name in ('masker','masker_semantic'):
                masker=getattr(self.core,name)
                if mode:
                    masker.topk_ratio_prob,masker.topk_ratio_range=original_masker_settings[name]
                else:
                    fraction=float(options.get('semantic_fraction' if name.endswith('_semantic') else 'image_fraction',.25))
                    if not 0<fraction<=1:raise ValueError('SQ-GAN evaluation retention must be in (0,1]')
                    masker.topk_ratio_prob='custom';masker.topk_ratio_range=[fraction,fraction]
            return self

        def forward(self,image):
            mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
            std=image.new_tensor([.229,.224,.225])[None,:,None,None]
            normalized=(image-mean)/std
            with torch.no_grad():
                logits=self.teacher(pixel_values=normalized).logits
                ids=F.interpolate(logits,size=image.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
                semantic=F.one_hot(ids,num_classes=20).permute(0,3,1,2).float()
                label=((ids==6)|(ids==7)).float().unsqueeze(1)
            if self.training:
                label,normalized,semantic=self.core.labelenhancer.process_masks(label,normalized,semantic)
            result=self.core.forward_all(normalized,semantic)
            output={'reconstruction':result[0]*std+mean,'xrec':result[0],'inputs':normalized,
                    'semantic':semantic,'qloss':result[1].mean().expand(len(image)),
                    'complex_uses':image.new_full((len(image),),-1)}
            if not self.training:
                from semantic_lossless.bitstream import encode_packet
                image_mask,semantic_mask=result[6],result[7]
                bits=[]
                for index in range(len(image)):
                    packet=encode_packet(image_mask['sample_index'][index].tolist(),result[2][2][index].flatten().tolist(),
                        semantic_mask['sample_index'][index].tolist(),result[5][2][index].flatten().tolist())
                    bits.append(len(packet)*8)
                output['source_bits']=image.new_tensor(bits)
            return output

    return View().to(options.get('device','cpu'))


def execute(stage,options,run):
    from contextlib import nullcontext
    import torch
    from .dataset import ParentImages
    from .evaluate import evaluate
    model=build(options);device=next(model.parameters()).device
    write_json(run/'model.json',model.metadata)
    if stage=='evaluate':
        state=torch.load(run/'checkpoints'/'last.pt',map_location=device,weights_only=True)
        model.load_state_dict(state['state_dict'],strict=True)
        write_json(run/'evaluation_reload.json',{'checkpoint':str(run/'checkpoints'/'last.pt'),
            'selection_rule':'last checkpoint at training stop','development_best_checkpoint_used':False,
            'learned_parameters_strict':True})
        return evaluate(model,options['manifest'],run,{**options,'evaluation_tile':[256,512]})
    pilot=stage=='pilot'
    if not pilot and not (run/'pilot_passed.json').exists():
        raise RuntimeError('SQ-GAN requires successful adversarial pilot before training')
    target=run/'pilot' if pilot else run
    target.mkdir(exist_ok=True);(target/'checkpoints').mkdir(exist_ok=True)
    checkpoint=target/'checkpoints'/'last.pt'
    if checkpoint.exists():raise FileExistsError('Refusing to overwrite SQ-GAN run')
    dataset=ParentImages(options['manifest'],'train',[256,512])
    batch=int(options.get('batch_size',2))
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0)
    if options.get('data_parallel',True) and device.type=='cuda':
        if torch.cuda.device_count()<2 or batch<2:raise RuntimeError('SQ-GAN DataParallel requires two GPUs and batch>=2')
        parallel=torch.nn.DataParallel(model)
    else:parallel=model
    discriminator=list(model.core.loss.discriminator.parameters())
    lr=float(options.get('learning_rate',1.25e-5))
    model.core.learning_rate=lr
    model.core.min_learning_rate=0.
    model.core.steps_per_epoch=len(loader)
    model.core.training_steps=len(loader)*int(options.get('epochs',150))
    optimizers,scheduler_configs=model.core.configure_optimizers()
    opt_g,opt_d=optimizers
    schedulers=[config['scheduler'] if isinstance(config,dict) else config for config in scheduler_configs]
    from .ema import ReplicaEMA
    ema=ReplicaEMA({'image':model.core.quantize.codebook,'semantic':model.core.quantize_semantic.codebook})
    write_json(target/'vq_ema_protocol.json',{'mechanism':'collect quantizer observations from all replicas; apply one native EMA update per forward',
        'reason':'DataParallel otherwise drops non-primary replica codebook updates',
        'native_update_formulas_preserved':True,'single_gpu_also_supported':True})
    started=time.monotonic();last_save=started;steps=0;epoch=0
    def save():
        temporary=checkpoint.with_suffix('.tmp')
        torch.save({'state_dict':model.state_dict(),'generator_optimizer':opt_g.state_dict(),
            'discriminator_optimizer':opt_d.state_dict(),'step':steps,'epoch':epoch,
            'schedulers':[scheduler.state_dict() for scheduler in schedulers],
            'torch_rng':torch.get_rng_state(),'options':options,'resume_supported':False},temporary)
        temporary.replace(checkpoint)
    for epoch in range(1,int(options.get('epochs',150))+1):
        parallel.train()
        for item in loader:
            image=item['image'].to(device)
            for optimizer_index,optimizer in enumerate((opt_g,opt_d)):
                for parameter in discriminator:parameter.requires_grad_(optimizer_index==1)
                optimizer.zero_grad(set_to_none=True)
                # The author's discriminator loss detaches real/fake inputs.
                # Its generator graph is therefore unnecessary on this pass.
                with torch.no_grad() if optimizer_index==1 else nullcontext():
                    out=parallel(image)
                loss,_=model.core.loss(codebook_loss=out['qloss'].mean(),semantic=out['semantic'],
                    inputs=out['inputs'],reconstructions=out['xrec'],optimizer_idx=optimizer_index,
                    global_step=steps,last_layer=model.core.get_last_layer(),split='train')
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SQ-GAN objective')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite SQ-GAN gradient')
                optimizer.step()
                last_loss=float(loss.detach())
                optimizer.zero_grad(set_to_none=True)
                counts=ema.update()
                if not {'image','semantic'}.issubset(counts):
                    raise RuntimeError('Missing image/semantic codebook EMA observations')
                del out,loss
            steps+=1
            for scheduler in schedulers:scheduler.step()
            if steps<=2 or steps%10==0:print(f'SQ-GAN step {steps}, discriminator loss {last_loss:.6f}',flush=True)
            if time.monotonic()-last_save>=300:save();last_save=time.monotonic()
            if (pilot and steps>=2) or time.monotonic()-started>=float(options.get('max_seconds',14400)):break
        save()
        if (pilot and steps>=2) or time.monotonic()-started>=float(options.get('max_seconds',14400)):break
    if not steps:raise RuntimeError('No SQ-GAN updates')
    if pilot and steps<2:raise RuntimeError('SQ-GAN pilot did not complete two adversarial updates')
    ema.close()
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(state['state_dict'],strict=True)
    write_json(target/'training_summary.json',{'steps':steps,'epochs_entered':epoch,
        'elapsed_seconds':time.monotonic()-started,'strict_reload':True,'source_codec_only':True,
        'teacher_cost_included_in_elapsed_time':True,'paper_convergence_reproduced':False})
    if pilot:
        evaluate(model,options['manifest'],target,{**options,'crop':256,'evaluation_tile':[256,512]},pilot=True)
        write_json(run/'pilot_passed.json',{'scope':'two adversarial steps, strict reload and development inference',
            'source_codec_only':True,'AWGN_trained':False})
