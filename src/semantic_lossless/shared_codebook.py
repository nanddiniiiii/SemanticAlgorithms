"""Native shared semantic-aware codebook components and construction stages.

Integrates SAC without the proposed information-theoretic loss term; this does
not mean lossless reconstruction. Common transport, holdout
splits, and the corrected per-image construction search are explicit adaptations.
"""
import ast
import importlib
import math
import sys
import types
from .artifacts import sha256, write_json
from .paths import resolve_path
from .sources import source_path, fetch_source


def native_classifier():
    import torch
    from torch import nn
    path=source_path('shared_semantic_codebook')/'SC_construction.py'
    tree=ast.parse(path.read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in {'conv_relu','inception','GoogLeNet'}]
    if len(nodes)!=3:
        raise ValueError('Pinned classifier definition changed')
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    namespace={'torch':torch,'nn':nn}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['GoogLeNet'](3,10)


def build_shared(config):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .codecs import upstream_import
    directory=source_path('shared_semantic_codebook')
    if not (directory/'source_receipt.json').is_file():
        raise FileNotFoundError('Run prepare --fetch first for shared_semantic_codebook')
    receipt=fetch_source('shared_semantic_codebook')
    asset=config.model_options.get('codebook')
    if not asset:
        raise FileNotFoundError('Construct a semantic-aware codebook first and set model_options.codebook')
    path=resolve_path(asset)
    book=torch.load(path,map_location='cpu',weights_only=True)
    if book.get('construction')!='semantic_weighted_mse_ce':
        raise ValueError('A generic random/VQ codebook is not a semantic-aware source codebook')
    tensor=book['codebook']
    if tensor.ndim!=4 or tuple(tensor.shape[1:])!=(3,256,256) or not torch.isfinite(tensor).all():
        raise ValueError('Codebook must contain finite RGB 256x256 prototype images')
    channels=int(config.model_options.get('latent_channels',4))
    if channels not in {4,8,12,16,20}:
        raise ValueError('Choose native latent_channels 4,8,12,16,20')
    class Shared(nn.Module):
        trainable=True
        rate_complete=False
        def __init__(self):
            super().__init__()
            self.register_buffer('codebook',tensor)
            encoder_options=dict(img_size=(256,256),patch_size=2,in_chans=3,
                embed_dims=[128,192,256,320],depths=[2,2,6,2],num_heads=[4,6,8,10],C=channels,
                window_size=8,mlp_ratio=4.,qkv_bias=True,qk_scale=None,norm_layer=nn.LayerNorm,patch_norm=True)
            decoder_options=dict(img_size=(256,256),embed_dims=[320,256,192,128],depths=[2,6,2,2],
                num_heads=[10,8,6,4],C=channels,window_size=8,mlp_ratio=4.,qkv_bias=True,qk_scale=None,norm_layer=nn.LayerNorm,patch_norm=True)
            with upstream_import(directory,('net',)):
                # Device-only compatibility fix. No source file is overwritten.
                source=directory/'net/encoderCB.py'
                text=source.read_text()
                if text.count('device = x.get_device()')!=1:
                    raise ValueError('Pinned encoder device compatibility patch no longer applies')
                patched=text.replace('device = x.get_device()','device = x.device')
                module=types.ModuleType('net.encoderCB');module.__file__=str(source);module.__package__='net'
                sys.modules['net.encoderCB']=module
                exec(compile(patched,str(source),'exec'),module.__dict__)
                self.encoder=module.create_encoder(**encoder_options)
                self.decoder=importlib.import_module('net.decoderCB').create_decoder(**decoder_options)
            self.metadata={'source_commit':receipt['commit'],'paper_branch':'SAC without information-theoretic loss',
                'codebook_sha256':sha256(path),'codebook_entries':len(tensor),'codebook_construction':{k:v for k,v in book.items() if k!='codebook'},
                'channel':'common per-image complex AWGN with latent scale restored','snr_definition':'Es/N0_dB',
                'side_information':'codeword index assumed received correctly; bit cost counted but protection not implemented',
                'adaptations':['encoder x.get_device -> x.device for CPU compatibility','training-only holdout','common runner scheduler','no periodic classifier-only fine-tune stage'],
                'latent_channels':channels,'native_eval_sampling':'Gaussian encoder sampling retained'}
            if config.checkpoint:
                saved=torch.load(resolve_path(config.checkpoint),map_location='cpu',weights_only=True)
                self.load_state_dict(saved.get('state_dict',saved),strict=True)
                self.metadata.update(checkpoint=str(resolve_path(config.checkpoint)),checkpoint_sha256=sha256(resolve_path(config.checkpoint)),strict_load=True)
            elif not config.model_options.get('from_scratch',False):
                raise FileNotFoundError('Set a trained codec checkpoint or explicitly enable from_scratch')
        def forward(self,image):
            if image.shape[-2:]!=(256,256):
                raise ValueError('Native shared-codebook codec requires explicit 256x256 inputs')
            with torch.autocast(device_type=image.device.type,enabled=False):
                a=image.float().flatten(1); b=self.codebook.float().flatten(1)
                distances=(a.square().sum(1,keepdim=True)+b.square().sum(1)[None]-2*a@b.T)/a.shape[1]
                indices=distances.argmin(1)
            prior=self.codebook[indices]
            feature=self.encoder(image,prior,config.snr_db,'WITT_W/O')
            flat=feature.flatten(1)
            if flat.shape[1]%2:
                raise ValueError('Odd latent packing is unsupported')
            scale=(2*flat.square().mean(1,keepdim=True)).clamp_min(1e-12).sqrt()
            received=((flat/scale+torch.randn_like(flat)*(0.5*10**(-config.snr_db/10))**0.5)*scale).reshape_as(feature)
            reconstruction=self.decoder(received,prior,config.snr_db,'WITT_W/O')
            return {'reconstruction':reconstruction,'complex_uses':torch.full((len(image),),flat.shape[1]//2,device=image.device),
                    'codeword_index_bits':math.ceil(math.log2(len(self.codebook))),'codeword_indices':indices}
        def loss(self,output,batch):
            # Native MSE distortion uses 0..255 scaling and clamps reconstruction.
            return F.mse_loss(output['reconstruction'].clamp(0,1)*255,batch['image']*255)
        def optimizer(self):
            return torch.optim.Adam(self.parameters(),lr=config.learning_rate)
    return Shared().to(config.device)


def construct(config):
    """Train the native classifier, then fit an online semantic-aware codebook."""
    import time
    import random
    from dataclasses import replace
    from datetime import datetime,timezone
    import uuid
    import torch
    import numpy as np
    from torch.nn import functional as F
    from .data import loader
    from .events import Events
    from .artifacts import export_manifest
    if config.algorithm!='shared_semantic_codebook' or config.dataset!='stl10':
        raise ValueError('Construction requires shared_semantic_codebook and labeled STL-10')
    fetch_source('shared_semantic_codebook')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    run=resolve_path(config.output_root)/config.algorithm/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-construct-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',config.as_dict())
    events=Events(run,config)
    start=time.monotonic()
    classifier=native_classifier().to(config.device)
    completed=0
    images_seen=0
    try:
        supplied=config.model_options.get('classifier_checkpoint')
        if supplied:
            state=torch.load(resolve_path(supplied),map_location=config.device,weights_only=True)
            classifier.load_state_dict(state.get('state_dict',state),strict=True)
        else:
            classification=replace(config,model_options={**config.model_options,'native_image_size':96})
            train_loader=loader(classification,'train')
            val_loader=loader(classification,'val')
            epochs=int(config.model_options.get('classifier_epochs',150))
            parallel=classifier
            if config.multi_gpu:
                if torch.cuda.device_count()<2:
                    raise RuntimeError('Classifier dual-GPU training requires two GPUs')
                parallel=torch.nn.DataParallel(classifier)
            for epoch in range(epochs):
                lr=.02 if epoch<20 else .01 if epoch<40 else .005 if epoch<80 else .002 if epoch<120 else .001 if epoch<150 else .0005
                optimizer=torch.optim.SGD(classifier.parameters(),lr=lr)
                parallel.train()
                for batch in train_loader:
                    image=batch['image'].to(config.device);labels=batch['label'].to(config.device)
                    loss=F.cross_entropy(parallel(image),labels)
                    if not torch.isfinite(loss):raise FloatingPointError('Classifier loss is nonfinite')
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    images_seen+=len(image)
                    if config.max_seconds and time.monotonic()-start>=config.max_seconds:
                        raise TimeoutError('Classifier budget exhausted; codec/codebook construction has not completed')
                completed=epoch+1
                classifier.eval();correct=0;count=0
                with torch.inference_mode():
                    for batch in val_loader:
                        image=batch['image'].to(config.device);label=batch['label'].to(config.device)
                        correct+=int((classifier(image).argmax(1)==label).sum());count+=len(label)
                events.emit('classifier','epoch_completed',metrics={'validation_accuracy':correct/count,'training_images_seen':images_seen},epoch=completed)
                torch.save({'state_dict':classifier.state_dict(),'epochs_completed':completed,'training_images_seen':images_seen},run/'classifier.pt')
        classifier.eval()
        for parameter in classifier.parameters():parameter.requires_grad_(False)
        if supplied:
            classifier_hash=sha256(resolve_path(supplied))
        else:
            classifier_hash=sha256(run/'classifier.pt')
        construction=replace(config,model_options={**config.model_options,'native_image_size':256})
        data=loader(construction,'train')
        size=int(config.model_options.get('codebook_size',32))
        if not 2<=size<=128:raise ValueError('Choose codebook_size 2..128')
        prototypes=[]
        for batch in data:
            prototypes.extend(batch['image'])
            if len(prototypes)>=size:break
        if len(prototypes)<size:raise ValueError('Not enough training images to initialize the codebook')
        book=torch.stack(prototypes[:size]).to(config.device)
        counters=torch.ones(size,device=config.device)
        weight=float(config.model_options.get('semantic_weight',10.))
        iterations=0
        epochs=int(config.model_options.get('codebook_epochs',300))
        per_epoch=int(config.model_options.get('codebook_batches_per_epoch',10))
        with torch.inference_mode():
            for epoch in range(epochs):
                for index,batch in enumerate(data):
                    images=batch['image'].to(config.device);labels=batch['label'].to(config.device)
                    logits=classifier(F.interpolate(book,size=(96,96),mode='bilinear',align_corners=False))
                    semantic=-F.log_softmax(logits,dim=1)
                    for image,label in zip(images,labels):
                        # Per-image argmin fixes the upstream minimum carried
                        # across unrelated images in the same batch.
                        distances=(book-image).square().mean((1,2,3))+weight*semantic[:,label]
                        selected=int(distances.argmin())
                        counters[selected]+=1
                        book[selected]+=(image-book[selected])/counters[selected]
                    iterations+=1
                    events.emit('codebook',metrics={'iterations':iterations,'entries':size},epoch=epoch+1)
                    if config.max_seconds and time.monotonic()-start>=config.max_seconds:
                        raise TimeoutError('Construction budget exhausted before the requested codebook passes')
                    if index+1>=per_epoch:break
        artifact={'codebook':book.cpu(),'construction':'semantic_weighted_mse_ce','classifier_sha256':classifier_hash,
                  'classifier_epochs_completed':completed if not supplied else None,'classifier_training_images_seen':images_seen if not supplied else None,
                  'codebook_epochs':epochs,'codebook_batches_per_epoch':per_epoch,'iterations':iterations,'semantic_weight':weight,
                  'fit_split':'STL-10 training-only 90 percent subset','seed':config.seed,'source_commit':fetch_source('shared_semantic_codebook')['commit'],
                  'adaptations':['per-image minimum reset','training-only holdout','bilinear explicit resize','no use of held-out test labels']}
        torch.save(artifact,run/'codebook.pt')
        write_json(run/'status.json',{'status':'completed','codebook':str((run/'codebook.pt').resolve())})
        events.emit('construct','completed',metrics={'iterations':iterations,'classifier_epochs_completed':completed})
        export_manifest(run)
        print(str(run),flush=True)
        return run
    except BaseException as error:
        events.emit('construct','failed',error=str(error))
        write_json(run/'status.json',{'status':'failed','error':str(error),'classifier_epochs_completed':completed})
        raise
