"""ADJSCC using DiffCom's released PyTorch attention/GDN architecture.

This is a separate implementation lineage from the original TensorFlow ADJSCC.
The common transport uses per-image unit-energy complex AWGN, Es/N0 in dB.
"""
import importlib
import random
from .artifacts import sha256
from .paths import resolve_path
from .sources import source_path, fetch_source


def build_adjscc(config):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .codecs import upstream_import
    directory=source_path('diffcom')
    if not (directory/'source_receipt.json').is_file():
        raise FileNotFoundError('Run prepare --fetch for adjscc to retrieve the pinned DiffCom source')
    receipt=fetch_source('diffcom')
    channels=int(config.model_options.get('latent_channels',2))
    if channels <= 0 or channels % 2:
        raise ValueError('Choose positive even latent_channels for unambiguous I/Q pairing')
    class ADJSCC(nn.Module):
        trainable=True
        rate_complete=False
        def __init__(self):
            super().__init__()
            with upstream_import(directory,('_djscc',)):
                module=importlib.import_module('_djscc.network')
                self.core=module.ADJSCC(channels,None,torch.device('cpu'))
            # Author GDN stores constants as plain tensors. Nonpersistent buffers
            # preserve state-dict compatibility while moving correctly across GPUs.
            for item in self.core.modules():
                if item.__class__.__name__=='GDN':
                    for name in ['reparam_offset','pedestal','beta_bound','gamma_bound']:
                        value=getattr(item,name)
                        delattr(item,name)
                        item.register_buffer(name,value,persistent=False)
            self.metadata={'source_commit':receipt['commit'],'architecture_lineage':'DiffCom released PyTorch ADJSCC, not original TF checkpoint',
                'channel':'per-image unit-complex-energy AWGN','snr_definition':'Es/N0_dB','rate_scope':'complex latent uses; excludes pilots/headers/CSI',
                'latent_channels':channels,'transport_adaptation':'per-image normalization rather than author batch-wide normalization',
                'initialization':'random_author_architecture','train_snr_range':config.model_options.get('train_snr_range'),
                'scheduler':'shared runner cosine restart schedule; not a claimed native training reproduction'}
            if config.checkpoint:
                if config.model_options.get('from_scratch',False):
                    raise ValueError('from_scratch and checkpoint are mutually exclusive')
                path=resolve_path(config.checkpoint)
                state=torch.load(path,map_location='cpu',weights_only=True)
                state=state.get('state_dict',state)
                mapped={}
                for key,value in state.items():
                    if key.startswith('distortion_loss'):
                        continue  # author DiffCom loader omits this obsolete loss state
                    if key.startswith('core.'):
                        key=key[5:]
                    key=key.replace('Encoder.','jscc_encoder.').replace('Decoder.','jscc_decoder.')
                    mapped[key]=value
                self.core.load_state_dict(mapped,strict=True)
                self.metadata.update(initialization='pretrained_strict_load',checkpoint=str(path),checkpoint_sha256=sha256(path),strict_load=True)
            elif not config.model_options.get('from_scratch',False):
                raise FileNotFoundError('Set an ADJSCC checkpoint or explicitly opt into from_scratch training')
        def forward(self,image):
            height,width=image.shape[-2:]
            if height%4 or width%4:
                raise ValueError('ADJSCC dimensions must be multiples of four; crop or pad explicitly')
            snr=config.snr_db
            limits=config.model_options.get('train_snr_range')
            if self.training and limits is not None:
                if len(limits)!=2 or limits[1]<limits[0]:
                    raise ValueError('train_snr_range must be [low,high]')
                snr=random.uniform(*limits)
            context=image.new_full((len(image),1),snr)
            latent=self.core.jscc_encoder(image,context)
            values=latent.flatten(1)
            if values.shape[1]%2:
                raise ValueError('An even number of real values is required')
            scale=(2*values.square().mean(1,keepdim=True)).clamp_min(1e-12).sqrt()
            transmitted=values/scale
            received=transmitted+torch.randn_like(transmitted)*(0.5*10**(-snr/10))**0.5
            reconstruction=self.core.jscc_decoder(received.reshape_as(latent),context)
            return {'reconstruction':reconstruction,'complex_uses':torch.full((len(image),),values.shape[1]//2,device=image.device),
                    'snr_used':context.flatten()}
        def loss(self,output,batch):
            return F.mse_loss(output['reconstruction'],batch['image'])
        def optimizer(self):
            return torch.optim.Adam(self.parameters(),lr=config.learning_rate)
    return ADJSCC().to(config.device)
