"""DiffCom/HiFi-DiffCom AWGN inference with pinned author diffusion components.

The lightweight operator follows the author's AWGN equations. Optional plotting,
OFDM, NTSCC+, and perceptual-metric packages are not imported to build this codec.
"""
import ast
import importlib
import math
import sys
import types
from dataclasses import replace
from types import SimpleNamespace
from .artifacts import sha256
from .paths import project_root,resolve_path
from .sources import source_path,fetch_source


def build_diffcom(config):
    import numpy as np
    import torch
    from torch import nn
    from .adjscc import build_adjscc
    from .codecs import upstream_import
    directory=source_path('diffcom')
    if not (directory/'source_receipt.json').is_file():
        raise FileNotFoundError('Run prepare --fetch first for DiffCom')
    receipt=fetch_source('diffcom')
    checkpoint=resolve_path(config.checkpoint) if config.checkpoint else project_root()/'Models/pretrained_downloads/DiffCom/256x256_diffusion_uncond.pt'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    ad_checkpoint=config.model_options.get('base_checkpoint',str(project_root()/'Models/pretrained_downloads/ADJSCC/ADJSCC_C=2.pth.tar'))
    base=build_adjscc(replace(config,algorithm='adjscc',checkpoint=ad_checkpoint,model_options={'latent_channels':2}))
    steps=int(config.model_options.get('sampling_steps',1000))
    if not 1<=steps<=1000 or 1000%steps:
        raise ValueError('sampling_steps must be a positive divisor of 1000')
    method=config.model_options.get('conditioning_method','hifi_diffcom')
    if method not in {'diffcom','hifi_diffcom'}:
        raise ValueError('This integration supports AWGN DiffCom and HiFi-DiffCom only')
    with upstream_import(directory,('guided_diffusion','utils','conditioning_method')):
        factory=importlib.import_module('guided_diffusion.script_util')
        options=factory.model_and_diffusion_defaults()
        options.update(image_size=256,num_channels=256,num_res_blocks=2,num_head_channels=64,
            attention_resolutions='8,16,32',learn_sigma=True,noise_schedule='linear',resblock_updown=True,
            use_scale_shift_norm=True,dropout=.1,use_checkpoint=False)
        with torch.device('meta'):
            unet,diffusion=factory.create_model_and_diffusion(**options)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        unet.load_state_dict(state,strict=True,assign=True)
        del state
        # Execute the two unmodified author helper definitions, without importing
        # unused optional plotting/perceptual-metric dependencies from utils.util.
        helper=types.ModuleType('utils.utils_model')
        namespace=helper.__dict__;namespace.update(torch=torch,np=np)
        path=directory/'utils/utils_model.py'
        tree=ast.parse(path.read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in {'find_nearest','model_fn'}]
        if len(nodes)!=2:raise ValueError('Pinned diffusion helper definitions changed')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
        package=types.ModuleType('utils');package.__path__=[str(directory/'utils')]
        package.utils_model=helper
        sys.modules['utils']=package;sys.modules['utils.utils_model']=helper
        conditioning=importlib.import_module('conditioning_method.diffcom')
        condition=conditioning.get_conditioning_method(method)
        loss_class=conditioning.ConsistencyLoss
    class DiffCom(nn.Module):
        trainable=False
        rate_complete=False
        requires_gradient_inference=True
        def __init__(self):
            super().__init__()
            self.base=base
            self.unet=unet
            for parameter in self.base.parameters():parameter.requires_grad_(False)
            # The author's attention CheckpointFunction differentiates its
            # parameter inputs explicitly. Keep the native UNet flags enabled;
            # no optimizer or weight updates are performed during sampling.
            for parameter in self.unet.parameters():parameter.requires_grad_(True)
            self.metadata={'source_commit':receipt['commit'],'diffusion_checkpoint':str(checkpoint),
                'diffusion_sha256':sha256(checkpoint),'strict_load':True,'base':base.metadata,
                'method':method,'requested_sampling_steps':steps,'native_denoiser_size':256,
                'channel':'unit-energy complex AWGN; B=1 identity interleaver',
                'assumed_receiver_side_information':'transmit latent power scale, 32 bits recorded separately; protection not modeled',
                'native_components':'author UNet, diffusion factory, model_fn, consistency loss and conditioning step',
                'adaptations':['identity rather than shuffled AWGN interleaving','lightweight AWGN operator','schedule arrays ported from NoiseSchedule'],
                'compute_scope':'inference includes posterior gradient steps; no model training',
                'autograd_policy':'native UNet parameter gradient flags retained for custom checkpoint backward; no optimizer or weight updates',
                'sampling_reproducibility':'caller seeds each trial; no forced seed reset inside the image forward'}
        @torch.inference_mode(False)
        @torch.enable_grad()
        def forward(self,image):
            if tuple(image.shape)!=(1,3,256,256):
                raise ValueError('DiffCom integration requires one explicit RGB 256x256 crop; no hidden resizing')
            context=image.new_full((1,1),config.snr_db)
            with torch.no_grad():
                clean_feature=self.base.core.jscc_encoder(image,context)
                shape=clean_feature.shape
                raw=clean_feature.flatten(1)
                power_scale=(2*raw.square().mean()).clamp_min(1e-12).sqrt().detach()
                signal=raw/power_scale
                measurement_signal=signal+torch.randn_like(signal)*(.5*10**(-config.snr_db/10))**.5
                baseline=self.base.core.jscc_decoder(measurement_signal.reshape(shape),context)
            owner=self
            class Operator:
                def encode(self,x):
                    return owner.base.core.jscc_encoder(x,context).flatten(1)
                def forward(self,s,cof=None):return s/power_scale
                def transpose(self,s,cof=None):return s
                def decode(self,s):return owner.base.core.jscc_decoder(s.reshape(shape),context)
            operator=Operator()
            settings={
                'diffcom':{'lr_schedule':'constant','learning_rate':1.,'zeta':1.,'gamma':0.},
                'hifi_diffcom':{'lr_schedule':'constant','learning_rate':1.,'zeta':.25,'gamma':.25}}
            if 'learning_rate' in config.model_options:
                settings[method]['learning_rate']=float(config.model_options['learning_rate'])
            native=SimpleNamespace(conditioning_method=method,ddim_sample=False,diffcom_series=settings)
            losses=loss_class(native,image.device)
            betas=np.linspace(.0001,.02,1000,dtype=np.float32)
            alphas=torch.from_numpy(1-betas).to(image.device)
            alpha_bar=np.cumprod(alphas.cpu(),axis=0).to(image.device)
            sqrt_alpha=alpha_bar.sqrt();sqrt_noise=(1-alpha_bar).sqrt()
            if config.model_options.get('adaptive_start',True):
                log_snr=torch.log10(alpha_bar/(1-alpha_bar))
                dsnr=np.log10((1+10**(config.snr_db/10))**(1/48))*5
                start=int(1000-np.searchsorted(log_snr.cpu().numpy()[::-1],dsnr))
            else:start=999
            skip=1000//steps
            sequence=[int(i*skip) for i in np.arange(0,start//skip)][::-1]
            if not sequence:raise ValueError('Too few sampling steps for the adaptive start; increase sampling_steps')
            schedule=SimpleNamespace(seq=sequence,t_start=start,reduced_alpha_cumprod=sqrt_noise/sqrt_alpha)
            x=sqrt_alpha[start]*(2*baseline-1)+sqrt_noise[start]*torch.randn_like(image)
            measurement={'ofdm_sig':measurement_signal,'x_mse':baseline}
            for index,_ in enumerate(sequence):
                estimate,_,x,_,_=condition.conditioning(native,index,schedule,x.detach(),None,None,
                    measurement,self.unet,diffusion,operator,losses,last_timestep=index==len(sequence)-1)
            # Native main_diffcom.py reports the final posterior state x_t,
            # rather than the intermediate clean-image prediction x_0_hat.
            return {'reconstruction':(x.detach()+1)/2,
                    'complex_uses':torch.tensor([raw.numel()//2],device=image.device),
                    'side_information_bits':32,'denoiser_evaluations':len(sequence)}
    return DiffCom().to(config.device).eval()
