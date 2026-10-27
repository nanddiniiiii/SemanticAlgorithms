"""Pinned author architectures, with explicit common-data experiment variants.

This module never substitutes a smaller autoencoder for an unavailable method.
One model runs per subprocess because upstreams reuse names such as `net`.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

from src.sources import fetch_source, source_path
from src.codecs import upstream_import


def legacy_ste_round(value):
    """CompressAI1.2.0's original function, renamed in newer releases.

    Source: InterDigitalInc/CompressAI v1.2.0, compressai/ops/ops.py.
    Preserve rounding forward and the identity straight-through derivative.
    """
    import torch
    return torch.round(value)-value.detach()+value


def load_training_state(model,state):
    """Strict learned-weight reload; resolution masks are derived, not weights."""
    buffers=dict(model.named_buffers())
    allowed={'.'.join(part for part in (prefix,name) if part)
             for prefix,module in model.named_modules() for name in module._buffers
             if name=='attn_mask'}
    unexpected=[name for name in state if name.endswith('attn_mask') and name not in allowed]
    if unexpected:raise RuntimeError('Unexpected resolution-mask keys: '+str(unexpected))
    derived=allowed & buffers.keys()
    saved={name:value for name,value in state.items() if name not in allowed}
    # Only known resolution-mask buffers may be regenerated. Every learned
    # parameter and every other buffer remains subject to strict validation.
    saved.update({name:buffers[name] for name in derived})
    model.load_state_dict(saved,strict=True)
    return {'regenerated_resolution_masks':sorted(derived),
            'discarded_saved_resolution_masks':sorted(name for name in state if name in allowed),
            'learned_parameters_strict':True}


def build(name, options, device='cpu'):
    import torch
    import random
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    from src.config import Config

    seed=int(options.get('seed',22))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

    from Infrastructure.conditions import validate_model_conditions
    conditions=validate_model_conditions(name, options)
    snr = conditions.train.snr_db
    checkpoint = options.get('checkpoint')
    lr = float(options.get('learning_rate', 1e-4))
    common = dict(algorithm=name, device=device, snr_db=snr, learning_rate=lr,
                  checkpoint=checkpoint or '', model_options=options)

    if name == 'adjscc':
        from src.adjscc import build_adjscc
        fetch_source('diffcom')
        return build_adjscc(Config(**common))
    if name == 'jcm':
        from src.jcm_spatial import build_jcm_spatial
        if not checkpoint:
            raise FileNotFoundError('Spatial JCM requires the verified author 16-QAM initializer')
        return build_jcm_spatial(checkpoint, device, snr, lr)
    if name == 'shared_semantic_codebook':
        from src.shared_codebook import build_shared
        base=build_shared(Config(**common))
        class SharedTensorView(nn.Module):
            trainable=True
            def __init__(self):
                super().__init__();self.core=base;self.metadata=base.metadata
            def forward(self,image):
                result=self.core(image)
                # DataParallel cannot gather a Python integer dictionary leaf.
                result['codeword_index_bits']=image.new_full((len(image),),result['codeword_index_bits'])
                return result
            def loss(self,output,batch):return self.core.loss(output,batch)
            def optimizer(self):return self.core.optimizer()
        return SharedTensorView().to(device)
    if name == 'sqgan':
        raise NotImplementedError('SQ-GAN uses its native staged adversarial trainer, not an MSE-only loop')
    if name in {'gbsed', 'deepjscc_wz'}:
        raise ValueError('Use the native task runner for ' + name)

    class View(nn.Module):
        trainable = True
        rate_complete = False

        def loss(self, output, batch):
            return output.get('native_loss', F.mse_loss(output['reconstruction'], batch['image'])).mean()

        def optimizer(self):
            return torch.optim.Adam((p for n, p in self.named_parameters()
                                     if p.requires_grad and not n.endswith('.quantiles')), lr=lr)

    if name == 'swinjscc':
        from src.swin_native import build_native_swin
        if options.get('swin_variant','fixed_c32')=='fixed_c32' and int(options.get('rate',32))!=32:
            raise ValueError('Native fixed-C32 checkpoint cannot implement a different rate head')
        if not checkpoint:
            raise FileNotFoundError('Configure a verified native SwinJSCC checkpoint')
        core, metadata = build_native_swin(checkpoint, options.get('swin_variant', 'fixed_c32'), device)
        if options.get('channel_backend')=='infrastructure':
            class Transport(nn.Module):
                def forward(self, feature, chan_param, avg_pwr=False):
                    from Infrastructure.Artifacts import LatentArtifact
                    condition=conditions.for_stage('train' if self.training else 'test')
                    return condition.build().transmit(LatentArtifact(feature,None,'swinjscc'),condition.snr_db).received
            core.channel=Transport()
            metadata.update(native_math_unmodified=False,channel='Infrastructure per-image transport',
                train_channel=conditions.train.channel,test_channel=conditions.test.channel,
                adaptation='fixed-C32 channel replacement; per-image normalization, perfect-CSI Rayleigh optional')

        class Swin(View):
            def __init__(self):
                super().__init__(); self.core = core; self.metadata = metadata

            def forward(self, image):
                rate = int(options.get('rate', 32))
                condition=conditions.for_stage('train' if self.training else 'test')
                recon, cbr, _, _, loss = self.core(image, condition.snr_db, rate)
                return {'reconstruction': recon,
                        'complex_uses': image.new_full((len(image),), float(cbr) * image[0].numel()),
                        'native_loss': loss.expand(len(image))}
        return Swin().to(device)

    receipt = fetch_source(name)
    directory = source_path(name)
    if name == 'deepjscc':
        train_channel=conditions.train.build()
        test_channel=conditions.test.build()
        with upstream_import(directory, ('model', 'channel')):
            module = importlib.import_module('model')
            core = module.DeepJSCC(c=int(options.get('latent_channels', 32)))

        class Deep(View):
            def __init__(self):
                super().__init__(); self.core = core
                self.metadata = {'source_commit': receipt['commit'], 'lineage': 'third-party DeepJSCC reproduction',
                                 'paper_exact': False,
                                 'train_channel':conditions.train.channel, 'train_snr_db':conditions.train.snr_db,
                                 'test_channel':conditions.test.channel, 'test_snr_db':conditions.test.snr_db,
                                 'train_coherence':conditions.train.coherence,'test_coherence':conditions.test.coherence,
                                 'changes': ['restore batch dimension after upstream singleton squeeze',
                                             'Infrastructure per-image channel and use counting'],
                                 'known_architecture_difference': 'upstream transposed kernel6 versus paper kernel5'}

            def forward(self, image):
                latent = self.core.encoder(image)
                if latent.ndim == 3:
                    latent = latent.unsqueeze(0)
                from Infrastructure.Artifacts import LatentArtifact
                condition=conditions.for_stage('train' if self.training else 'test')
                channel=train_channel if self.training else test_channel
                received=channel.transmit(
                    LatentArtifact(tensor=latent,rate_mask=None,source_model='deepjscc'),condition.snr_db).received
                recon = self.core.decoder(received)
                if recon.shape != image.shape:
                    raise ValueError('DeepJSCC source input geometry does not reconstruct exactly')
                return {'reconstruction': recon,
                        'complex_uses': image.new_full((len(image),), latent[0].numel()/2)}
        model = Deep()
    elif name == 'ntscc_plus':
        from importlib.metadata import version
        if version('natten').split('+')[0] != '0.14.6':
            raise RuntimeError('Author NTSCC+ requires NATTEN 0.14.6; do not silently use newer kernels')
        if not device.startswith('cuda'):
            raise RuntimeError('Pinned native NTSCC+ channel is CUDA-only; CPU is not a certified path')
        config = SimpleNamespace(device=torch.device(device), logger=None, channel_type='awgn', SNR=snr,
            eta=float(options.get('eta', .2)), multiple_rate=[1,4,8,12,16,20,24,32,48,64,80,96,112,128,
                144,160,176,192,208,224,240,256,272,288,304,320])
        with upstream_import(directory, ('net', 'layer', 'channel')):
            core = importlib.import_module('net.ntscc').NTSCC_plus(config, N=128, M=320,
                input_resolution=(16,16), depths=[4], num_heads=[20,20], window_size=4)

        class NTSCC(View):
            def __init__(self):
                super().__init__(); self.core = core
                self.metadata = {'source_commit': receipt['commit'], 'rate_scope': 'main complex symbols only; side info not protected',
                                 'objective': 'author lambda255^2*(MSE_JSCC+MSE_NTC)+bpp_y+bpp_z'}

            def forward(self, image):
                result = self.core(image)
                bpp = sum(-p.log().sum()/math.log(2)/len(image)/image.shape[-2]/image.shape[-1]
                          for p in result['likelihoods'].values())
                loss = float(options.get('lambda', .72))*255**2*(
                    F.mse_loss(result['x_hat'], image)+F.mse_loss(result['x_hat_ntc'], image))+bpp
                return {'reconstruction': result['x_hat'], 'native_loss': loss.expand(len(image)),
                        'complex_uses': image.new_full((len(image),), float(result['k'])/len(image)),
                        'codeword_index_bits':image.new_full((len(image),),result['indexes'].numel()/len(image)*
                                                           math.ceil(math.log2(len(config.multiple_rate)))),
                        'estimated_source_bpp': bpp.detach().expand(len(image))}

            def aux_loss(self):
                return self.core.ntc.aux_loss()
        model = NTSCC()
    elif name == 'mdjcm':
        import copy
        import compressai.ops as operations
        missing_legacy_round=not hasattr(operations,'ste_round')
        if missing_legacy_round:operations.ste_round=legacy_ste_round
        try:
            with upstream_import(directory, ('net', 'layer', 'channel', 'config', 'utils', 'loss', 'cnn_utils', 'interleavers')):
                config = copy.deepcopy(importlib.import_module('config').config)
                # Copy nested dicts: toggling the class flag alone leaves modulation off
                # in the already-created native encoder/decoder kwargs.
                config = SimpleNamespace(**{k: copy.deepcopy(v) for k,v in vars(config).items()
                                            if not k.startswith('_')})
                config.modulation = True
                config.device = torch.device(device)
                config.fe_kwargs['modulation'] = True
                config.fd_kwargs['modulation'] = True
                config.channel = {'type': 'awgn', 'chan_param': snr}
                module = importlib.import_module('net.MDJCM')
                # Correct a demonstrated wrong-super call without editing source bytes.
                module.MDJCM_Model.forward_NTC = lambda self, image, **kw: module.NTC_Hyperprior.forward(self, image, **kw)
                core = module.MDJCM_Model(config)
        finally:
            if missing_legacy_round:delattr(operations,'ste_round')

        class MDJCM(View):
            def __init__(self):
                super().__init__(); self.core = core
                self.metadata = {'source_commit': receipt['commit'], 'modulation': True,
                    'modulation_order': int(options.get('modulation_order',64)),
                    'patch': 'forward_NTC explicitly dispatches NTC_Hyperprior.forward',
                    'legacy_round_symbol_patch':missing_legacy_round,
                    'legacy_round_source':'original CompressAI1.2.0 straight-through rounding function',
                    'rate_scope': 'native main-symbol count; headers/indexes and side protection excluded'}

            def forward(self, image):
                ntc, by, bz, digital, cbr, _, recon, _ = self.core(image,
                    int(options.get('modulation_order',64)), snr)
                loss = ntc + digital + config.train_lambda*(by+bz)
                # Upstream cbr divides total batch uses by one image's dimension.
                return {'reconstruction': recon, 'native_loss': loss.expand(len(image)),
                        'complex_uses': image.new_full((len(image),), float(cbr)*image[0].numel()/len(image)),
                        'estimated_source_bpp': (by+bz).detach().expand(len(image))}

            def aux_loss(self):
                return self.core.aux_loss()
        model = MDJCM()
    elif name == 'hjscc':
        if snr != 10 or not device.startswith('cuda'):
            raise ValueError('Pinned native HJSCC hardcodes AWGN10 and CUDA; preserve that contract')
        directory = directory/'HJSCC'
        with upstream_import(directory, ('net', 'layer', 'config', 'timm', 'utils', 'channel', 'loss')):
            config = importlib.import_module('config').config
            config.device = torch.device(device)
            native = importlib.import_module('net')
            core = native.Model(config)

        class HJSCC(View):
            def __init__(self):
                super().__init__(); self.core = core
                self.metadata = {'source_commit': receipt['commit'], 'native_lambda':512,
                    'channel':'native AWGN10', 'reconstruction':'last patch_upsample_5 output, native process_output',
                    'rate_scope':'native total CBR reported; explicit protected headers unavailable'}

            def forward(self, image):
                import threading
                captured = []
                thread_id=threading.get_ident()
                # DataParallel replicas can share the module hook registry; a
                # callback must only capture its own replica thread's output.
                def capture(m,a,out):
                    if threading.get_ident()==thread_id:captured.append(out)
                hook = self.core.patch_upsample_5.register_forward_hook(capture)
                try:
                    loss, _, native_psnr, _, cbr, bpp = self.core(image, train=int(self.training))
                finally:
                    hook.remove()
                if not captured:
                    raise RuntimeError('Native reconstruction hook did not execute')
                recon = native.process_output(captured[-1][...,:image.shape[-2],:image.shape[-1]])
                if recon.device != image.device:
                    raise RuntimeError('HJSCC reconstruction came from another replica')
                measured_psnr=float(-10*torch.log10(F.mse_loss(recon.detach(),image)))
                if abs(measured_psnr-float(native_psnr))>1e-3:
                    raise RuntimeError('HJSCC reconstruction hook disagrees with native reported PSNR')
                return {'reconstruction': recon, 'native_loss': loss.expand(len(image)),
                        'complex_uses': image.new_full((len(image),), float(cbr)*image[0].numel()/len(image)),
                        'estimated_source_bpp': image.new_full((len(image),),float(bpp))}
        model = HJSCC()
    elif name == 'isec':
        return build_isec(options, device)
    else:
        raise ValueError('No verified architecture factory: ' + name)

    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        if state and all(k.startswith('core.') for k in state):
            model.load_state_dict(state, strict=True)
        else:
            model.core.load_state_dict(state, strict=True)
        from .dataset import digest
        model.metadata.update(checkpoint_sha256=digest(checkpoint), strict_load=True)
    elif not options.get('from_scratch',False):
        raise FileNotFoundError('Provide checkpoint or explicitly select from_scratch for ' + name)
    return model.to(device)


def build_isec(options, device):
    """Author OpenImages architecture, denoiser and iterative MAP update."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    receipt = fetch_source('isec')
    width = int(options.get('latent_channels',16))
    with upstream_import(source_path('isec'), ('models',)):
        ae = importlib.import_module('models.autoencoders')
        enc = ae.Encoder(width,128,2,3)
        dec = ae.Decoder(width,128,2,3)
        denoiser = importlib.import_module('models.bfcnn').BF_CNN(1,64,3,20,width)
    sigma = 10**(-float(options.get('snr_db',10))/20)

    class ISEC(nn.Module):
        trainable = True
        stage = 'jscc'
        needs_grad_evaluation = True

        def __init__(self):
            super().__init__(); self.encoder=enc; self.decoder=dec; self.denoiser=denoiser
            self.metadata={'source_commit':receipt['commit'], 'native_input_range':'[-1,1]',
                'training_stages':['JSCC','frozen-encoder bias-free denoiser'],
                'rate_scope':'complex-equivalent real latent pairs; no framing',
                'refinement_iterations':int(options.get('iterations',50))}

        def forward(self, image):
            latent = self.encoder(image*2-1)
            received = latent + torch.randn_like(latent)*sigma
            if self.training and self.stage == 'denoiser':
                # Author bias-free network predicts positive additive noise;
                # evaluation uses its negative as the score direction.
                noise = torch.randn_like(latent)*sigma
                predicted = self.denoiser(latent.detach()+noise)
                loss = F.mse_loss(predicted, noise)
                recon = self.decoder(received)
            else:
                refined = received
                if not self.training and self.stage == 'denoiser':
                    # Autograd only for the codeword, never optimizer updates.
                    with torch.enable_grad():
                        for _ in range(int(options.get('iterations',50))):
                            z = refined.detach().requires_grad_(True)
                            prior = -self.denoiser(z).detach()
                            prediction = self.encoder(self.decoder(z))
                            objective = (prediction-received.detach()).square().sum()/(2*sigma**2)
                            grad, = torch.autograd.grad(objective,z)
                            refined = (z + float(options.get('refinement_lr',.001))*(-grad+
                                float(options.get('alpha',4))*prior)).detach()
                recon = self.decoder(refined)
                loss = F.mse_loss(recon,image*2-1)
            return {'reconstruction':(recon+1)/2, 'native_loss':loss.expand(len(image)),
                    'complex_uses': image.new_full((len(image),),latent[0].numel()/2)}

        def loss(self, output, batch):
            return output['native_loss'].mean()

        def optimizer(self):
            params = self.denoiser.parameters() if self.stage=='denoiser' else list(self.encoder.parameters())+list(self.decoder.parameters())
            rate=options.get('denoiser_learning_rate',2e-4) if self.stage=='denoiser' else options.get('learning_rate',1e-3)
            return torch.optim.Adam(params,lr=float(rate))

    model=ISEC().to(device)
    checkpoint=options.get('checkpoint')
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state.get('state_dict',state),strict=True)
    elif not options.get('from_scratch',False):
        raise FileNotFoundError('ISEC requires both native JSCC/denoiser weights or explicit two-stage scratch training')
    return model
