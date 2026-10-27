"""Inference-only SQ-GAN integration using the ten strict-loaded author modules.

Forward ordering follows models/sqgan.py:forward_all at the pinned source commit.
No Lightning trainer or discriminator is constructed for inference.
"""
import importlib
import json
import math
from pathlib import Path
from .artifacts import sha256
from .paths import project_root, resolve_path
from .sources import fetch_source, source_path


def build_sqgan(config):
    import torch
    from torch import nn
    import yaml
    from .codecs import upstream_import
    if not (source_path('sqgan')/'source_receipt.json').is_file():
        raise FileNotFoundError('Run prepare --fetch first to retrieve SQ-GAN sources')
    source = fetch_source('sqgan')  # validates the existing pinned source files
    directory = source_path('sqgan')
    params = yaml.safe_load((directory/'config/sqgan_cityscapes.yml').read_text())['model']['params']
    checkpoints = resolve_path(config.checkpoint) if config.checkpoint else project_root()/'Models/pretrained_downloads/SQ-GAN/extracted/Final_ckpt_parts'
    names = ['encoder','decoder','masker','demasker','quantize','encoder_semantic','decoder_semantic','masker_semantic','demasker_semantic','quantize_semantic']
    config_names = {'quantize':'vqconfig','quantize_semantic':'vqconfig_semantic'}
    class SQGAN(nn.Module):
        trainable = False
        rate_complete = False
        requires_semantic = True
        def __init__(self):
            super().__init__()
            receipt = []
            with upstream_import(directory, ('modules',)):
                for name in names:
                    key = config_names.get(name,name+'_config')
                    entry = params[key]
                    module, cls = entry['target'].rsplit('.',1)
                    component = getattr(importlib.import_module(module),cls)(**entry['params'])
                    checkpoint = checkpoints/(name+'.ckpt')
                    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
                    component.load_state_dict(state,strict=True)
                    setattr(self,name,component)
                    receipt.append({'component':name,'sha256':sha256(checkpoint),'strict_load':True})
            self.image_fraction=float(config.model_options.get('image_fraction',0.25))
            self.semantic_fraction=float(config.model_options.get('semantic_fraction',0.25))
            for name, fraction in [('masker',self.image_fraction),('masker_semantic',self.semantic_fraction)]:
                if not 0 < fraction <= 1:
                    raise ValueError('SQ-GAN retained fractions must lie in (0,1]')
                masker=getattr(self,name)
                masker.topk_ratio_prob='custom'
                masker.topk_ratio_range=[fraction,fraction]
            self.metadata={'source_commit':source['commit'],'components':receipt,'training_adapter':False,
                'preprocessing':'RGB ImageNet normalization; paired 20-class trainIds; 256x512',
                'channel':'ideal source codec; no AWGN/FEC stage implemented',
                'rate_scope':'actual experimental SQ01 source packet bytes including bitmap/header/CRC; no radio/FEC cost',
                'packet_format':'our SQ01 adapter format; preagreed synchronized checkpoint; not author entropy coder',
                'image_fraction':self.image_fraction,'semantic_fraction':self.semantic_fraction}
        def encode(self,image,semantic):
            from .bitstream import encode_packet
            if len(image)!=1:
                raise ValueError('SQ packet encoder currently processes one image at a time')
            if image.shape[-2:]!=(256,512) or semantic.shape[1:]!=(20,256,512):
                raise ValueError('Native SQ-GAN requires RGB 256x512 and a paired 20-class map')
            mean=image.new_tensor([0.485,0.456,0.406])[None,:,None,None]
            std=image.new_tensor([0.229,0.224,0.225])[None,:,None,None]
            x=(image-mean)/std
            image_mask=self.masker(self.encoder(x,semantic[:,:-1]),semantic[:,:-1])
            semantic_mask=self.masker_semantic(self.encoder_semantic(semantic),semantic[:,:-1])
            _,_,image_info=self.quantize(image_mask['sample_features'])
            _,_,semantic_info=self.quantize_semantic(semantic_mask['sample_features'])
            packet=encode_packet(image_mask['sample_index'][0].tolist(),image_info[2].flatten().tolist(),
                                 semantic_mask['sample_index'][0].tolist(),semantic_info[2].flatten().tolist())
            return packet
        def decode(self,packet):
            from .bitstream import decode_packet
            image_section,semantic_section=decode_packet(packet)
            parameter=next(self.parameters())
            def restore(section,quantizer,demasker):
                positions,codes=section
                indices=torch.tensor(codes,device=parameter.device,dtype=torch.long)[None]
                vectors=quantizer.get_codebook_entry(indices).permute(0,2,1).contiguous()
                selected=torch.tensor(positions,device=parameter.device,dtype=torch.long)[None]
                mask=parameter.new_zeros(1,512)
                mask.scatter_(1,selected,1.)
                return demasker(vectors,vectors[:,:,:0],selected,None,mask)
            semantic_logits=self.decoder_semantic(restore(semantic_section,self.quantize_semantic,self.demasker_semantic))
            predicted=torch.zeros_like(semantic_logits).scatter_(1,semantic_logits.argmax(1,keepdim=True),1.)
            reconstruction=self.decoder(restore(image_section,self.quantize,self.demasker),[predicted[:,:-1] for _ in range(5)])
            mean=reconstruction.new_tensor([0.485,0.456,0.406])[None,:,None,None]
            std=reconstruction.new_tensor([0.229,0.224,0.225])[None,:,None,None]
            selected=len(image_section[0])+len(semantic_section[0])
            return {'reconstruction':reconstruction*std+mean,'complex_uses':torch.full((1,),-1,device=parameter.device),
                    'semantic_logits':semantic_logits,'source_bits':len(packet)*8,'source_packet':packet,
                    'vq_index_bits':selected*10,'mask_bitmap_bits':1024,'packet_header_crc_bits':128}
        def forward(self,image,semantic=None):
            if semantic is None:
                raise ValueError('SQ-GAN requires a paired semantic map at the encoder; never substitute zero masks')
            return self.decode(self.encode(image,semantic))
    return SQGAN().to(config.device)
