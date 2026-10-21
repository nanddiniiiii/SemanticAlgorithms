"""Validate full benchmark image sets and hardware inside the model environment."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
import torch


def validate(train_root, val_root, kodak_root):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Exactly two CUDA GPUs required')
    if not all('T4' in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Select Kaggle T4 x2')
    sets=[]
    for root,expected in [(train_root,800),(val_root,100),(kodak_root,24)]:
        files=sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
        if len(files)!=expected: raise ValueError('Wrong image count: %s expected %s found %s' % (root,expected,len(files)))
        hashes=[]
        for path in files:
            with Image.open(path) as image: image.verify()
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if len(set(hashes))!=expected: raise ValueError('Duplicate source image content')
        sets.append(set(hashes))
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise ValueError('Dataset content overlaps')
    return {'event':'input_validation_passed','train':800,'validation':100,'kodak':24,'gpus':2}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['train-root','val-root','kodak-root']: parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.train_root,args.val_root,args.kodak_root)))
