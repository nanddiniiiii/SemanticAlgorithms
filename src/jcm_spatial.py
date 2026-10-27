"""User-approved larger-image reconstruction variant, NOT native JCM.

Retains native residual analysis/synthesis and stochastic 16-QAM amplitudes.
Replaces the global32x32 probability FC by a spatial1x1 probability head,
removes classification, and uses per-image normalization for DataParallel.
"""
from types import SimpleNamespace
import importlib
from pathlib import Path

from .codecs import upstream_import
from .sources import fetch_source,source_path
from .artifacts import sha256


def build_jcm_spatial(checkpoint,device='cpu',snr_db=10,learning_rate=1e-4):
    import torch
    from torch import nn
    from torch.nn import functional as F
    checkpoint=Path(checkpoint)
    fetch_source('jcm')
    if sha256(checkpoint)!='b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897':
        raise ValueError('Expected the verified author16QAMcheckpoint')
    with upstream_import(source_path('jcm'),('network','modules')):
        native_module=importlib.import_module('network')
        native=native_module.JCM(SimpleNamespace(channel_use=128,mod_method='16qam',
            mode='test',snr_train=12,snr_test=12),torch.device('cpu'))
    native.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)

    class SpatialJCM(nn.Module):
        trainable=True
        rate_complete=False
        def __init__(self):
            super().__init__()
            self.encoder=native.encoder
            self.decoder=native.decoder_recon
            self.probability_head=nn.Sequential(nn.Conv2d(256,16*4,1),nn.ReLU())
            self.register_buffer('amplitudes',torch.tensor([-3.,-1.,1.,3.]))
            self.metadata={'variant':'user_approved_spatial_reconstruction_JCM','native_reproduction':False,
                'initialization':'native16QAMencoder_and_reconstruction_decoder; new probability head',
                'checkpoint_sha256':sha256(checkpoint),'source_commit':fetch_source('jcm')['commit'],
                'changes':['globalFCprobability head replaced by1x1spatialhead',
                    'native decoder4x4reshape replaced by matching spatialreshape',
                    'classification head/objective removed; reconstructionMSEonly',
                    'per-image normalization replaces nativebatchwide normalization',
                    'RGBdimensions multiples8; RGB0to1 external range retained'],
                'modulation':'native hardGumbelsoftmax tau1.5 with16QAMrealamplitudes',
                'rate':'16real amplitudes perH/8xW/8;8complexuses perlatentposition; CBR1/24',
                'radio_scope':'finite-amplitude noisy-symbol simulation; no actual bit packing/demapping/LDPC/framing; raw label count is not a transmitted packet size',
                'digital_limitation':'sample-dependent normalization rescales constellation; no implemented receiver demapper or scale signalling; not a complete standard16QAMlink',
                'normalization':'per-image mean squared real amplitude1; equivalent complexEs2; realnoisevariance10^(-SNR/10)',
                'batchnorm':'nativeBatchNorm retained; DataParallel replica statistics declared, not singleGPUequivalence',
                'input_contract':'user-approved separatelargerimagevariant; nativeCIFAR10JCM unchanged'}
        def forward(self,image):
            batch,channels,height,width=image.shape
            if channels!=3 or height%8 or width%8:
                raise ValueError('SpatialJCM requires RGBdimensions divisible by8')
            feature=self.encoder(image*2-1)
            h,w=feature.shape[-2:]
            logits=self.probability_head(feature).reshape(batch,16,4,h,w).permute(0,3,4,1,2)
            choices=F.gumbel_softmax(logits,hard=True,tau=1.5,dim=-1)
            real_symbols=(choices*self.amplitudes).sum(-1).permute(0,3,1,2).contiguous()
            power=real_symbols.square().mean((1,2,3),keepdim=True).clamp_min(1e-12)
            normalized=real_symbols/power.sqrt()
            received=normalized+torch.randn_like(normalized)*10**(-snr_db/20)
            decoder=self.decoder
            value=decoder.conv1(received)
            value=decoder.layer1(value);value=decoder.layer2(value)
            value=decoder.DepthToSpace1(value);value=decoder.conv2(value)
            value=decoder.layer3(value);value=decoder.DepthToSpace2(value)
            reconstruction=(decoder.conv3(value)+1)/2
            assert reconstruction.shape==image.shape
            return {'reconstruction':reconstruction,
                'complex_uses':torch.full((batch,),8*h*w,device=image.device,dtype=torch.long),
                'raw_symbol_label_bits':torch.full((batch,),32*h*w,device=image.device,dtype=torch.long)}
        def loss(self,output,batch):return F.mse_loss(output['reconstruction'],batch['image'])
        def optimizer(self):return torch.optim.Adam(self.parameters(),lr=learning_rate)
    return SpatialJCM().to(device)
