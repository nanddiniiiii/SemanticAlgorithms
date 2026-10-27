"""Identical frozen development protocol before/after custom-data adaptation."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--adapted',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from src.config import Config
    from src.adjscc import build_adjscc
    from src.sources import fetch_source
    from src.data import FrozenImages
    from src.metrics import quality
    from src.artifacts import write_json,sha256
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True,exist_ok=False)
    fetch_source('diffcom')
    config=Config('adjscc',device='cuda:0',snr_db=10,crop_size=256,checkpoint=str(args.checkpoint),
        model_options={'latent_channels':2,'dataset_manifest':str(args.manifest.resolve())})
    codec=build_adjscc(config).eval()
    if args.adapted:
        codec.load_state_dict(torch.load(args.adapted,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    dataset=FrozenImages(config,'val');rows=[]
    for index in range(len(dataset)):
        batch=dataset[index];image=batch['image'][None].cuda();torch.manual_seed(22+index)
        with torch.inference_mode():values=quality(image,codec(image)['reconstruction'])
        rows.append({'sample':batch['sample'],**values})
        if index%25==0:print({'stage':'development_evaluation','images':index+1,'total':len(dataset)},flush=True)
    write_json(args.output/'per_image.json',rows)
    write_json(args.output/'metrics.json',{'status':'completed','images':len(rows),
        'mean_psnr_db':sum(row['psnr_db'] for row in rows)/len(rows),
        'mean_ssim':sum(row['ssim'] for row in rows)/len(rows),
        'manifest_sha256':sha256(args.manifest),'adapted_checkpoint':sha256(args.adapted) if args.adapted else None,
        'scope':'development center256crops; pairedseeds22+index; not final heldout results',
        'test_accessed':False,'snr_db':10,'main_stream_cbr':1/48,'rate_complete':False})


if __name__=='__main__':main()
