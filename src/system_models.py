"""Author architecture adapters for spatial VQ, OFDM and MIMO transmission."""
import importlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

from .codecs import upstream_import
from .native_runtime import runtime_source,_device


def build_mimo(options,device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    source=runtime_source('deepjscc_mimo')
    with upstream_import(source,('Models',)):
        module=importlib.import_module('Models.transformer')
        core=module.JSCCModel(patch_size=(4,4),in_channels=3,out_channels=3,hidden_size=256,
            num_hidden_layers=8,num_attention_heads=8,tcn=int(options.get('tcn',8)),MIMO_num=int(options.get('antennas',2)))
    class MIMO(nn.Module):
        trainable=True;rate_complete=False
        def __init__(self):
            super().__init__();self.encoder=core
            self.metadata={'source':'eedavidwu/DeepJSCC-MIMO','native_geometry':[3,32,32],
                'channel':'iid complex MIMO, SVD precoding and perfect CSI',
                'rate_scope':'aggregate antenna-symbols; excludes CSI acquisition',
                'compatibility':str(source/'compatibility.json'),'paper_convergence_reproduced':False}
        def forward(self,image):
            if image.shape[-2:]!=(32,32):raise ValueError('Released MIMO implementation requires 32x32 tiles and 64 tokens')
            token=_device.set(image.device)
            try:recon=self.encoder(image,float(options.get('snr_db',10)),int(options.get('antennas',2)))
            finally:_device.reset(token)
            uses=64*int(options.get('tcn',8))/2
            return {'reconstruction':recon,'complex_uses':image.new_full((len(image),),uses)}
        def loss(self,out,batch):return F.mse_loss(out['reconstruction'],batch['image'])
        def optimizer(self):return torch.optim.Adam(self.parameters(),lr=1e-4)
    return MIMO().to(device)


def build_ofdm(options,device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    source=runtime_source('ofdm_guided_jscc')
    with upstream_import(source,('models','configs','util')):
        cfg=copy.deepcopy(importlib.import_module('configs.config_train').cfg)
        cfg.gpu_ids=[0,1] if str(device).startswith('cuda') and options.get('data_parallel',True) and torch.cuda.device_count()>=2 else ([int(str(device).split(':')[-1])] if str(device).startswith('cuda') else [])
        cfg.SNR=float(options.get('snr_db',10));cfg.isTrain=True;cfg.gan_mode='none'
        cfg.checkpoints_dir=str(Path(options.get('models_root','Models'))/'native_ofdm');cfg.name='native'
        crop=int(options.get('crop',256));cfg.size_h=cfg.size_w=crop
        latent=(crop//2**cfg.n_downsample)**2*(cfg.C_channel//2)
        if latent%(cfg.P*cfg.M):raise ValueError('OFDM latent must divide exactly into P*M subcarriers')
        cfg.S=latent//(cfg.P*cfg.M)
        core=importlib.import_module('models.JSCCOFDM_model').JSCCOFDMModel(cfg)
        native_scheduler=importlib.import_module('models.networks').get_scheduler
    class OFDM(nn.Module):
        trainable=True;rate_complete=False
        native_data_parallel=bool(cfg.gpu_ids and len(cfg.gpu_ids)>1)
        def __init__(self):
            super().__init__();self.networks=nn.ModuleDict({name:(getattr(core,'net'+name).module if isinstance(getattr(core,'net'+name),nn.DataParallel) else getattr(core,'net'+name)) for name in core.model_names});self.channel=core.channel
            self.parallel_networks={name:nn.DataParallel(network,device_ids=cfg.gpu_ids) if len(cfg.gpu_ids)>1 else network for name,network in self.networks.items()}
            self.metadata={'source':'mingyuyng/OFDM-guided-JSCC','channel':'native frequency-selective multipath',
                'subcarriers':cfg.M,'cyclic_prefix':cfg.K,'paths':cfg.L,'pilot_symbols':cfg.N_pilot,
                'receiver':cfg.feedforward,'snr_convention':cfg.SNR_cal,'compatibility':str(source/'compatibility.json'),
                'rate_scope':'payload+pilots+CP complex time samples; training dataset adapted'}
        def forward(self,image):
            if image.shape[-2:]!=(crop,crop):raise ValueError('OFDM uses configured square tiles')
            # Native BaseModel is stateful and not an nn.Module; copy per call.
            active=copy.copy(core)
            for name,network in self.parallel_networks.items():setattr(active,'net'+name,network)
            active.channel=self.channel;active.set_input(image*2-1);active.forward()
            recon=(active.fake+1)/2
            objective=F.mse_loss(active.fake,active.real_B)*cfg.lam_L2
            if cfg.feedforward in {'OFDM-CE-sub-EQ','OFDM-CE-sub-EQ-sub'}:
                objective=objective+cfg.lam_h*(active.H_est-active.H_true).square().mean()*2
            uses=cfg.P*(cfg.S+cfg.N_pilot)*(cfg.M+cfg.K)
            return {'reconstruction':recon,'complex_uses':image.new_full((len(image),),uses),
                    'native_loss':objective.expand(len(image))}
        def loss(self,out,batch):return out['native_loss'].mean()
        def optimizer(self):return torch.optim.Adam(self.parameters(),lr=cfg.lr,betas=(cfg.beta1,.999))
        def scheduler(self,optimizer):return native_scheduler(optimizer,cfg)
    return OFDM().to(device)


def build_ssc(options,device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    import yaml
    from addict import Dict
    source=runtime_source('ssc')
    config=yaml.safe_load((source/'options/train_SSC_from_scratch.yml').read_text())
    args=Dict(config['network_g']['args']);crop=int(options.get('crop',256))
    if crop!=256:raise ValueError('SSC released recipe uses 256x256 crops')
    channel_number=int(args.C)
    encoder=dict(img_size=(crop,crop),patch_size=2,in_chans=3,embed_dims=[128,192,256,320],depths=[2,2,2,2],
        num_heads=[4,6,8,10],C=channel_number,window_size=8,mlp_ratio=4.,qkv_bias=True,qk_scale=None,norm_layer=nn.LayerNorm,patch_norm=True)
    decoder=dict(img_size=(crop,crop),embed_dims=[320,256,192,128],depths=[2,2,2,2],num_heads=[10,8,6,4],
        C=channel_number,window_size=8,mlp_ratio=4.,qkv_bias=True,qk_scale=None,norm_layer=nn.LayerNorm,patch_norm=True)
    with upstream_import(source,('ssc','basicsr','loss')):
        sys.path.insert(0,str(source/'ssc'))
        try:
            module=importlib.import_module('ssc.archs.SwinSSC_arch')
            core=module.SwinSSC(args,encoder,decoder,config['network_g']['rq_kwargs'])
            FA=importlib.import_module('ssc.faim').FA_IM_Channel
        finally:sys.path.remove(str(source/'ssc'))
    class SSC(nn.Module):
        trainable=True;rate_complete=False;requires_ddp=True
        def __init__(self):
            super().__init__();self.core=core;self.fa=None
            self.metadata={'source':'gxh1106/SSC','training_channel':'author differentiable index bit-flip surrogate',
                'evaluation_channel':'native FA-IM with adaptive stream splitting',
                'objective':'MSE + 0.25 commitment + native bit-flip regularizer','model_ema_decay':.999,
                'rate_scope':'index bits exact; physical FA-IM resource count not certified',
                'native_recipe':str(source/'options/train_SSC_from_scratch.yml')}
        def forward(self,image):
            token=_device.set(image.device)
            try:return self._forward(image)
            finally:_device.reset(token)
        def _forward(self,image):
            if self.training:
                recon,cbr,snr,commit,indices=self.core(image,float(options.get('snr_db',10)))
            else:
                if self.fa is None:
                    self.fa=FA(K=4,N=16,Nr=8,M=64,num_H=100,W=2,L_paths=10,device=str(image.device),codebook_size=16)
                recon,cbr,snr,commit,indices=self.core.forward_faim(image,float(options.get('snr_db',10)),self.fa,int(torch.randint(100,()).item()))
            objective=F.mse_loss(recon,image)+.25*commit+self.core.quantizer.bit_flip_regularization_loss()
            return {'reconstruction':recon,'native_loss':objective.expand(len(image)),
                    'source_bits':indices.numel()*4//len(image)}
        def loss(self,out,batch):return out['native_loss'].mean()
        def optimizer(self):return torch.optim.Adam(self.parameters(),lr=3e-5,betas=(.9,.99),weight_decay=0)
    return SSC().to(device)
