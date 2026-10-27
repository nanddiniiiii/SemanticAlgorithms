"""Full-image paired metrics, rate accounting and inference-only profiling."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from .runtime import write_json


def image_quality(original, reconstruction, perceptual=True):
    from semantic_lossless.metrics import quality
    result=quality(original,reconstruction,use_lpips=False)
    import torch
    from torch.nn import functional as F
    a=original.float();b=reconstruction.float().clamp(0,1)
    result['mae']=float((a-b).abs().mean())
    if min(a.shape[-2:])>160:
        try:
            from pytorch_msssim import ms_ssim
            result['ms_ssim']=float(ms_ssim(a,b,data_range=1,size_average=True))
        except ImportError:
            result['ms_ssim']=None
    else:
        result['ms_ssim']=None
    if perceptual:
        try:
            # Cache by device, not one global metric on the wrong GPU.
            import lpips
            if not hasattr(image_quality,'networks'):image_quality.networks={}
            key=str(a.device)
            if key not in image_quality.networks:
                image_quality.networks[key]=lpips.LPIPS(net='alex').to(a.device).eval()
            with torch.no_grad():
                result['lpips']=float(image_quality.networks[key](a*2-1,b*2-1).mean())
        except (ImportError, OSError, RuntimeError) as exc:
            result['lpips']=None;result['lpips_unavailable_reason']=str(exc)
    else:
        result['lpips']=None;result['lpips_unavailable_reason']='Disabled for structural CPU pilot'
    if perceptual:
        try:
            from torchvision.models import resnet18,ResNet18_Weights
            if not hasattr(image_quality,'semantic_networks'):image_quality.semantic_networks={}
            key=str(a.device)
            if key not in image_quality.semantic_networks:
                weights=ResNet18_Weights.IMAGENET1K_V1
                network=resnet18(weights=weights).to(a.device).eval()
                image_quality.semantic_networks[key]=(network,weights.transforms())
            network,transform=image_quality.semantic_networks[key]
            pair=transform(torch.cat([a,b],dim=0))
            captured=[]
            hook=network.avgpool.register_forward_hook(lambda m,args,out:captured.append(out.flatten(1)))
            try:
                with torch.no_grad():logits=network(pair)
            finally:hook.remove()
            first,second=captured[0].chunk(2)
            result['resnet18_feature_cosine']=float(F.cosine_similarity(first,second).mean())
            predicted_a,predicted_b=logits.chunk(2)
            result['resnet18_prediction_agreement']=float((predicted_a.argmax(1)==predicted_b.argmax(1)).float().mean())
            result['semantic_proxy_note']='Frozen ImageNet ResNet18; embedding/agreement proxy, not labelled semantic accuracy'
        except (ImportError,OSError,RuntimeError) as exc:
            result['resnet18_feature_cosine']=None
            result['semantic_proxy_unavailable_reason']=str(exc)
    return result


def predict(model,image,options):
    """Native-size constraints use declared tiling, with every tile charged."""
    import torch
    from torch.nn import functional as F
    tile=options.get('evaluation_tile')
    if not tile:return model(image)
    th,tw=tile
    h,w=image.shape[-2:];ph,pw=(-h)%th,(-w)%tw
    padded=F.pad(image,(0,pw,0,ph),mode='replicate') if ph or pw else image
    recon=torch.empty_like(padded);uses=0.;bits=0.;side_bits=0.;uses_known=True;bits_known=True;side_known=True;tiles=0
    for y in range(0,padded.shape[-2],th):
        for x in range(0,padded.shape[-1],tw):
            out=model(padded[...,y:y+th,x:x+tw])
            recon[...,y:y+th,x:x+tw]=out['reconstruction']
            tiles+=1
            value=out.get('complex_uses')
            if value is None or float(value.reshape(-1)[0])<0:uses_known=False
            else:uses+=float(value.sum())
            value=out.get('source_bits')
            if value is None:bits_known=False
            else:bits+=float(value)
            value=out.get('codeword_index_bits')
            if value is None:side_known=False
            else:side_bits+=float(value.sum()) if torch.is_tensor(value) else float(value)
    return {'reconstruction':recon[...,:h,:w],
            'complex_uses':image.new_tensor([uses if uses_known else -1]),
            'source_bits':bits if bits_known else None,'tiles':tiles,'padded_height':padded.shape[-2],
            'codeword_index_bits':image.new_tensor([side_bits]) if side_known else None,
            'padded_width':padded.shape[-1]}


def evaluate(model,manifest,run,options,pilot=False):
    import numpy as np
    import torch
    from PIL import Image
    from .dataset import ParentImages
    run=Path(run);folder=run/'metrics'
    if (folder/'per_image.json').exists():
        raise FileExistsError('Evaluation evidence already exists; use a new run to preserve prior results')
    folder.mkdir(parents=True,exist_ok=True)
    (run/'reconstructions').mkdir(exist_ok=True)
    # A pilot checks development pixels, never the final test set.
    dataset=ParentImages(manifest,'val' if pilot else 'test',int(options.get('crop',256)))
    device=next(model.parameters()).device;model.eval()
    rows=[];failures=[];trials=1 if pilot else int(options.get('trials',3))
    first_image=None
    for index in range(min(len(dataset),1 if pilot else len(dataset))):
        sample=dataset[index];image=sample['image'].unsqueeze(0).to(device)
        if first_image is None:first_image=image
        for trial in range(trials):
            seed=int(options.get('seed',22))+10000*index+trial
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.synchronize(device)
            started=time.perf_counter()
            try:
                # ISEC enables local codeword gradients internally; no_grad is
                # intentional here rather than inference_mode.
                with torch.no_grad():out=predict(model,image,options)
                if device.type=='cuda':torch.cuda.synchronize(device)
                latency=time.perf_counter()-started
                reconstruction=out['reconstruction']
                if reconstruction.shape!=image.shape:raise ValueError('Reconstruction geometry mismatch')
                if not torch.isfinite(reconstruction).all():raise FloatingPointError('Nonfinite reconstruction')
                measured=image_quality(image,reconstruction,perceptual=not pilot)
                rawuses=out.get('complex_uses')
                uses=float(rawuses.sum()) if rawuses is not None and (rawuses>=0).all() else None
                bits=out.get('source_bits')
                if torch.is_tensor(bits):bits=float(bits.sum())
                side=out.get('codeword_index_bits')
                if torch.is_tensor(side):side=float(side.sum())
                row={'algorithm':options['algorithm'],'sample':sample['sample'],'trial':trial,'seed':seed,
                    'height':image.shape[-2],'width':image.shape[-1],'rgb_samples':image.numel(),
                    'snr_db':options.get('snr_db',10),'channel':'AWGN' if uses is not None else 'not_simulated',
                    'complex_channel_uses':uses,'cbr_complex_per_rgb_sample':uses/image.numel() if uses is not None else None,
                    'complex_uses_per_spatial_pixel':uses/(image.shape[-2]*image.shape[-1]) if uses is not None else None,
                    'source_bits':bits,'source_bpp':bits/(image.shape[-2]*image.shape[-1]) if bits is not None else None,
                    'raw_rgb8_bits':image.numel()*8,
                    'raw_rgb8_to_source_ratio':image.numel()*8/bits if bits is not None and bits>0 else None,
                    'assumed_correct_side_bits':side,
                    'rate_includes_headers_fec_pilots':False,'one_pass_seconds':latency,**measured}
                rows.append(row)
                if trial==0:
                    array=(reconstruction[0].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                    Image.fromarray(array).save(run/'reconstructions'/f'{index:03d}.png')
                print(json.dumps({'evaluation':index+1,'trial':trial,'psnr_db':row['psnr_db'],
                                  'complex_uses':uses,'seconds':latency}),flush=True)
            except Exception as exc:
                failures.append({'sample':sample['sample'],'trial':trial,'error':str(exc)})
                print('Evaluation failure: '+json.dumps(failures[-1]),flush=True)
            # Persist per sample; a late failure cannot erase earlier evidence.
            write_json(folder/'per_image.json',{'rows':rows,'failures':failures})
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (folder/'per_image.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    summary={'successful_trials':len(rows),'failed_trials':len(failures),
             'requested_trials':min(len(dataset),1 if pilot else len(dataset))*trials,
             'split':'development' if pilot else 'Kodak24 test',
             'pilot':pilot,'mean_over_successful_trials_only':True}
    for metric in ('psnr_db','ssim','ms_ssim','lpips','mae','mse','cbr_complex_per_rgb_sample',
                   'source_bpp','exact_uint8_pixels','one_pass_seconds','resnet18_feature_cosine',
                   'resnet18_prediction_agreement'):
        values=[r[metric] for r in rows if r.get(metric) is not None]
        summary[metric]=float(np.mean(values)) if values else None
    write_json(folder/'summary.json',summary)
    coverage={
        'pixel_metrics':'MSE, MAE, PSNR, SSIM, MS-SSIM, exact pixel/sample fractions and maximum error',
        'perceptual':'LPIPS AlexNet; unavailable reason recorded if dependencies/weights fail',
        'semantic_task_accuracy':'No true Kodak task labels; frozen ResNet18 feature cosine and prediction agreement are proxies only',
        'semantic_miou':'Requires genuine labels or separately labelled pseudo-reference evaluation',
        'fid_kid':'Not reported on 24 images as a dependable distributional comparison',
        'ber_bler_packet_error':'Unavailable without actual discrete bits, protected blocks and failure decisions',
        'spectral_efficiency_airtime':'Requires a defined waveform, bandwidth, pilots, coding and resource allocation',
        'source_compression_ratio':'Only derivable where an actual source bitstream is available; CBR is separate',
        'flops':'Inference only; supported-operator coverage recorded',
        'runtime_reliability':'Every requested image/trial counted, including failures',
        'component_encoder_decoder_timing':'Measured separately where parent module hooks execute; omitted components explicitly not measured'}
    write_json(folder/'metric_coverage.json',coverage)
    if first_image is not None and rows:
        # Evaluators are not part of the communication model residency. Move
        # cached metric networks out before measuring the codec's CUDA memory.
        for network in getattr(image_quality,'networks',{}).values():network.cpu()
        for network,_ in getattr(image_quality,'semantic_networks',{}).values():network.cpu()
        image_quality.networks={};image_quality.semantic_networks={}
        if device.type=='cuda':torch.cuda.empty_cache()
        from .profile import profile
        measured=profile(model,first_image,lambda m,x:predict(m,x,options),
                         warmup=1 if pilot else 3,repeats=1 if pilot else 10)
        write_json(folder/'profile.json',measured)
    if not rows:raise RuntimeError('No successful evaluation trials; see per_image.json')
    if failures:raise RuntimeError('Incomplete evaluation; successful rows and failures are preserved in per_image.json')
    return summary
