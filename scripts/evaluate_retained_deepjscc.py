"""Historical DeepJSCC checkpoint diagnostic; never a matched complex-radio result."""
import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_msssim import ms_ssim
    from Archive.EncDec.Research.model_wrappers import DeepJSCCAdapter
    from src.artifacts import write_json
    from src.metrics import quality
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    adapter=DeepJSCCAdapter(ROOT/'Models/DeepJSCC/checkpoint_source',
        ROOT/'Models/trained_checkpoints/DeepJSCC/production/imagenet_10_0.33_200.00_32_19.pth',
        torch.device('cpu'),evaluation_snr_db=10)
    write_json(args.output/'model.json',adapter.validation)
    paths=sorted((ROOT/'Datasets/KODAK').glob('kodim*.png'));assert len(paths)==24
    rows=[]
    for index,path in enumerate(paths):
        with Image.open(path) as handle:array=np.asarray(handle.convert('RGB'),dtype=np.float32).copy()/255
        image=torch.from_numpy(array).permute(2,0,1)[None]
        result=adapter.run_once(image,seed=22,tile=128)
        values=quality(image,result.reconstruction)
        values['ms_ssim']=float(ms_ssim(image,result.reconstruction,data_range=1))
        rows.append({'sample':path.name,'height':image.shape[-2],'width':image.shape[-1],
            'tile_count':result.tile_count,'transmitted_real_values':result.latent_real_values,**values})
        if index==0:
            write_json(args.output/'tensor_shapes.json',{'input':list(image.shape),'tile_latent_shapes':result.latent_shapes,
                'reconstruction':list(result.reconstruction.shape),'tile_size':128})
            Image.fromarray(result.reconstruction[0].mul(255).round().byte().permute(1,2,0).numpy()).save(args.output/'reconstruction.png')
        with (args.output/'per_image.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print({'stage':'legacy_diagnostic','completed':index+1,'total':24,'psnr_db':values['psnr_db']},flush=True)
    summary={'status':'completed','images':len(rows),'device':'local CPU','torch':torch.__version__,
        'training_snr_db':200,'evaluation_real_channel_snr_db':10,'seed':22,
        'scope':'retained third-party historical checkpoint; tiled real-AWGN diagnostic, not original-paper reproduction or matched complex-radio comparison',
        'input_dimensions':'Kodak native resolution; independent128x128tiles; eachlatent19x29x29 real',
        'complex_uses':None,'rate_complete':False,
        'rate_warning':'odd15979realvalues per tile; no explicit complex packing/padding/radio protocol is implemented',
        'transmitted_real_values_per_kodak':rows[0]['transmitted_real_values'],
        'exact_rgb_images':sum(row['exact_uint8_pixels'] for row in rows),
        'unmeasured':['LPIPS','semantic fidelity','GPU latency','explicit complex radio cost']}
    for key in ['psnr_db','ssim','ms_ssim']:
        summary['mean_'+key]=sum(row[key] for row in rows)/len(rows)
    write_json(args.output/'metrics.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
