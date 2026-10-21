"""Generate Cityscapes-ID pseudo masks for DIV2K/Kodak with a pinned-at-run SegFormer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image-root',required=True)
    ap.add_argument('--output-root',required=True)
    ap.add_argument('--model-id',default=MODEL_ID)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    image_root=Path(args.image_root); output_root=Path(args.output_root); output_root.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in image_root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if args.limit: paths=paths[:args.limit]
    processor=AutoImageProcessor.from_pretrained(args.model_id)
    model=SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(args.device).eval()
    num_labels=int(model.config.num_labels)
    if num_labels != 19: raise ValueError(f'Expected 19 Cityscapes classes, model has {num_labels}')
    with torch.inference_mode():
        for i,path in enumerate(paths,1):
            image=Image.open(path).convert('RGB'); width,height=image.size
            inputs=processor(images=image,return_tensors='pt').to(args.device)
            logits=model(**inputs).logits
            labels=F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).argmax(dim=1)[0]
            arr=labels.to('cpu',dtype=torch.uint8).numpy()
            if arr.min()<0 or arr.max()>18: raise ValueError(f'Invalid class ids in {path}')
            Image.fromarray(arr,mode='L').save(output_root/(path.stem+'.png'))
            if i%25==0 or i==len(paths): print(f'pseudo labels {i}/{len(paths)}')
    meta={'model_id':args.model_id,'resolved_revision':getattr(model.config,'_commit_hash',None),'num_classes':num_labels,'dataset_root':str(image_root),'images':len(paths),'mask_ids':'Cityscapes train IDs 0..18; null channel 19 unused'}
    (output_root/'pseudo_label_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
