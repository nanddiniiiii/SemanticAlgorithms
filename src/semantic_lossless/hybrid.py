"""Our complementary-projection prototype, not a paper reproduction.

A frozen ADJSCC stream carries the base. Fixed DCT residual coefficients describe
the difference from the transmitter's LOCAL NOISELESS base reconstruction. The
transmitter never reads the receiver's noisy reconstruction (no feedback oracle).
"""
import math
import struct
from dataclasses import replace
from .paths import project_root


def build_hybrid(config):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .adjscc import build_adjscc
    checkpoint=config.model_options.get('base_checkpoint')
    if not checkpoint:
        candidate=project_root()/'Models/pretrained_downloads/ADJSCC/ADJSCC_C=2.pth.tar'
        if not candidate.is_file():
            raise FileNotFoundError('Set base_checkpoint to the DiffCom ADJSCC C=2 checkpoint')
        checkpoint=str(candidate)
    base_config=replace(config,algorithm='adjscc',checkpoint=checkpoint,
        model_options={'latent_channels':2,'from_scratch':False})
    base=build_adjscc(base_config)
    count=int(config.model_options.get('coefficients_per_block',4))
    if not 1<=count<=64:
        raise ValueError('coefficients_per_block must lie in 1..64')
    class Hybrid(nn.Module):
        trainable=False
        rate_complete=False
        def __init__(self):
            super().__init__()
            self.base=base
            for parameter in self.base.parameters():parameter.requires_grad_(False)
            rows=torch.arange(8).float()[:,None];columns=torch.arange(8).float()[None]
            dct=torch.cos(math.pi*(columns+.5)*rows/8)*math.sqrt(2/8)
            dct[0]/=math.sqrt(2)
            self.register_buffer('dct',dct)
            # Fixed zig-zag order: no image-dependent basis/index transmission.
            order=sorted([(u,v) for u in range(8) for v in range(8)],key=lambda p:(p[0]+p[1],p[0] if (p[0]+p[1])%2 else -p[0]))
            self.register_buffer('selected',torch.tensor([u*8+v for u,v in order[:count]],dtype=torch.long))
            self.metadata={'type':'our untrained complementary-projection reference; not author HDA-DeepSC or hybrid paper implementation',
                'base':base.metadata,'coefficients_per_block':count,'basis':'fixed orthonormal 8x8 DCT; zig-zag first q',
                'transmitter_reference':'local noiseless ADJSCC reconstruction; no receiver reconstruction is accessed',
                'channel':'two unit-energy complex AWGN streams at the configured Es/N0',
                'side_information':'112 assumed-correct bits for coefficient scale/dimensions/mode; no protection modeled',
                'rate_scope':'base complex symbols plus complementary complex symbols; side bits separate',
                'training':'frozen pretrained base; no improvement/convergence claim'}
        def forward(self,image):
            if len(image)!=1:
                raise ValueError('Hybrid reference currently evaluates one image at a time')
            original_h,original_w=image.shape[-2:]
            padded=F.pad(image,(0,(-original_w)%8,0,(-original_h)%8),mode='replicate')
            batch,channels,height,width=padded.shape
            context=padded.new_full((batch,1),config.snr_db)
            latent=self.base.core.jscc_encoder(padded,context)
            flat=latent.flatten(1)
            scale=(2*flat.square().mean(1,keepdim=True)).clamp_min(1e-12).sqrt()
            clean=flat/scale
            local=self.base.core.jscc_decoder(clean.reshape_as(latent),context)
            residual=padded-local
            blocks=residual.unfold(2,8,8).unfold(3,8,8).contiguous()
            coefficients=self.dct@blocks@self.dct.T
            selected=coefficients.flatten(-2)[...,self.selected].reshape(batch,-1)
            real_count=selected.shape[1]
            if real_count%2:
                selected=F.pad(selected,(0,1))
            residual_scale=(2*selected.square().mean(1,keepdim=True)).clamp_min(1e-12).sqrt()
            side_packet=struct.pack("!IIfBB",original_h,original_w,float(residual_scale.item()),count,1)
            decoded_h,decoded_w,decoded_scale,decoded_count,version=struct.unpack("!IIfBB",side_packet)
            if decoded_count!=count or version!=1:
                raise ValueError("Hybrid mode disagreement")
            receiver_scale=selected.new_tensor(decoded_scale)
            noise_std=(.5*10**(-config.snr_db/10))**.5
            received_base=clean+torch.randn_like(clean)*noise_std
            received_complement=(selected/residual_scale+torch.randn_like(selected)*noise_std)*receiver_scale
            base_reconstruction=self.base.core.jscc_decoder(received_base.reshape_as(latent),context)
            received_complement=received_complement[:,:real_count].reshape(batch,channels,height//8,width//8,count)
            grid=coefficients.new_zeros(batch,channels,height//8,width//8,64)
            grid[...,self.selected]=received_complement
            recovered=self.dct.T@grid.reshape(batch,channels,height//8,width//8,8,8)@self.dct
            recovered=recovered.permute(0,1,2,4,3,5).reshape(batch,channels,height,width)
            reconstruction=(base_reconstruction+recovered)[...,:decoded_h,:decoded_w]
            base_uses=flat.shape[1]//2
            complementary_uses=selected.shape[1]//2
            return {'reconstruction':reconstruction,'complex_uses':torch.tensor([base_uses+complementary_uses],device=image.device),
                    'base_complex_uses':base_uses,'complementary_complex_uses':complementary_uses,'side_information_bits':len(side_packet)*8,'side_packet':side_packet}
    return Hybrid().to(config.device).eval()
