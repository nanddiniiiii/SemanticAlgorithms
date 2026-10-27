"""Retained NTSCC q1 GPU evaluation with an explicit incomplete side-rate ledger."""
import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_msssim import ms_ssim
    from src.config import Config
    from src.codecs import build_codec
    from src.metrics import quality,profile_forward
    from src.compute import registered_kernel_flops
    from src.artifacts import write_json,sha256
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True,exist_ok=False)
    codec=build_codec(Config('ntscc',device='cuda:0',snr_db=10)).eval()
    assert codec.metadata['checkpoint_sha256']=='7a8d1990dbd678ea5d3daa02c3aeaaf4459b347a96d6aa81a5c02eae6bfc7323'
    assert not codec.adapter.config.use_side_info
    write_json(args.output/'model.json',{**codec.metadata,'side_refinement_enabled':False,
        'rate_warning':'rate indices/mask and receiver power scale not transported',
        'profile_scope':'actual retained adapter includes auxiliary NTC synthesis/loss before JSCC synthesis; not a minimal optimized transmitter'})
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'execution_device':'cuda:0','entropy_implementation':'declared CompressAI1.2 forward-compatible subset; no entropy coder'})
    paths=sorted(args.images.rglob('kodim*.png'));assert len(paths)==24
    rows=[];first=None
    for index,path in enumerate(paths):
        with Image.open(path) as handle:array=np.asarray(handle.convert('RGB'),dtype=np.float32).copy()/255
        image=torch.from_numpy(array).permute(2,0,1)[None].cuda()
        torch.manual_seed(22)
        with torch.inference_mode():
            output=codec(image);reconstruction=output['reconstruction']
            assert reconstruction.shape==image.shape
            values=quality(image,reconstruction,use_lpips=True)
            values['ms_ssim']=float(ms_ssim(image,reconstruction.clamp(0,1),data_range=1))
        uses=int(output['complex_uses'][0]);metadata_bits=(image.shape[-2]//16)*(image.shape[-1]//16)*4
        rows.append({'sample':path.name,'sha256':sha256(path),'height':image.shape[-2],'width':image.shape[-1],
            'main_stream_complex_uses':uses,'main_stream_cbr':uses/image.numel(),
            'fixed_width_rate_index_bits':metadata_bits,**values})
        with (args.output/'per_image.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        if first is None:
            first=image
            Image.fromarray(reconstruction[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()).save(args.output/'reconstruction.png')
        print({'stage':'quality','completed':index+1,'total':24,'psnr_db':values['psnr_db'],'main_uses':uses},flush=True)
    import src.metrics as metric_module
    if hasattr(metric_module,'_LPIPS'):del metric_module._LPIPS
    del image,reconstruction,output
    torch.cuda.empty_cache()
    class Full(torch.nn.Module):
        def __init__(self):super().__init__();self.codec=codec
        def forward(self,value):return self.codec(value)['reconstruction']
    profile=profile_forward(Full(),first,warmup=3,repeats=20)
    profile['registered_kernel_count']=registered_kernel_flops(Full(),first)
    profile['flops_scope']='full retained adapter including auxiliary NTC reconstruction/loss and final JSCC synthesis; not minimal encoder'
    write_json(args.output/'profile_full.json',profile)
    summary={'status':'completed','images':len(rows),'snr_db':10,'noise_seed':22,
        'scope':'retained q1 GPU evaluation with device and entropy compatibility adapter; not author training reproduction',
        'rate_complete':False,'fixed_width_rate_index_bits_per_image':6144,
        'rate_warning':'main symbols exclude metadata/protection/headers and receiver rescaling information; hyperprior refinement disabled',
        'semantic_metrics':None,'semantic_metric_reason':'Kodak has no independent task labels',
        'exact_rgb_images':sum(row['exact_uint8_pixels'] for row in rows)}
    for key in ['psnr_db','ssim','ms_ssim','lpips','main_stream_cbr']:
        summary['mean_'+key]=sum(row[key] for row in rows)/len(rows)
    write_json(args.output/'metrics.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
