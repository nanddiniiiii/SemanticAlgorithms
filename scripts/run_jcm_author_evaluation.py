"""Native JCM checkpoint evaluation, preserving author batching and metrics.

This is pretrained evaluation, not reproduction of training. Run standalone on
Kaggle or locally with JCM_DATA_ROOT and SEMANTIC_LOSSLESS_ROOT configured.
"""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from types import SimpleNamespace

ROOT = Path(os.environ.get('SEMANTIC_LOSSLESS_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import torchvision
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.artifacts import write_json, sha256
from src.codecs import upstream_import
from src.sources import fetch_source, source_path, SOURCES


def main():
    fetch_source('jcm')
    directory = source_path('jcm')
    checkpoint = ROOT / 'Models/pretrained_downloads/JCM/CIFAR_SNR12_Trans128_16qam.pth.tar'
    expected = 'b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897'
    mounted = list(Path('/kaggle/input').rglob(checkpoint.name)) if Path('/kaggle/input').is_dir() else []
    if len(mounted) == 1:
        checkpoint = mounted[0]
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        repo, commit = SOURCES['jcm']
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/checkpoints/16qam/CIFAR_SNR12.000_Trans128_16qam.pth.tar'
        partial = checkpoint.with_suffix('.partial')
        urllib.request.urlretrieve(url, partial)
        if sha256(partial) != expected:
            raise ValueError('Author checkpoint hash differs')
        partial.replace(checkpoint)
    assert sha256(checkpoint) == expected
    if not torch.cuda.is_available():
        raise RuntimeError('This scheduled evaluation requires CUDA')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = SimpleNamespace(channel_use=128, mod_method='16qam', mode='test', snr_train=12, snr_test=12)
    with upstream_import(directory, ('network', 'modules')):
        native = importlib.import_module('network').JCM(config, torch.device('cuda:0')).cuda().eval()
    native.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    # Execute exactly the three author metric function bodies; exclude plotting.
    tree = ast.parse((directory / 'utils.py').read_text())
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'PSNR','SSIM','MSE'}], type_ignores=[])
    namespace = {'np': np, 'comp_psnr': peak_signal_noise_ratio, 'comp_ssim': structural_similarity}
    exec(compile(selected, str(directory / 'utils.py'), 'exec'), namespace)
    data_root = os.environ.get('JCM_DATA_ROOT')
    if not data_root:
        roots = sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob('cifar-10-batches-py') if (p/'test_batch').is_file()})
        if len(roots) != 1:
            raise ValueError(f'Expected one native CIFAR cache: {roots}')
        data_root = str(roots[0])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)])
    dataset = torchvision.datasets.CIFAR10(data_root, train=False, transform=transform, download=False)
    # Match released main.py's test shuffle and batch size; no DataParallel.
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    output = ROOT / 'Results/author_evaluation/jcm_16qam_12db'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', {'scope':'author pretrained checkpoint evaluation; training not reproduced',
        'source_commit':SOURCES['jcm'][1], 'checkpoint_sha256':expected,
        'torch':torch.__version__, 'torchvision':torchvision.__version__, 'device':torch.cuda.get_device_name(0),
        'dataset':'official CIFAR-10 test, 10000 images', 'batch_size':32, 'shuffle':True,
        'precision':'float32', 'data_parallel':False, 'snr_db':12, 'channel_uses':128,
        'metric_note':'Native AST PSNR averages channel PSNR; native SSIM uses data_range=1 on [-1,1]. Standard RGB PSNR reported separately.',
        'remaining_differences':['modern Torch/torchvision/skimage environment; not author package versions', 'constellation plotting omitted; it uses Python random but not Torch forward RNG']})
    count=correct=0
    totals={'author_psnr_db':0., 'author_ssim':0., 'author_mse_01':0., 'standard_rgb_psnr_db':0.}
    start=time.monotonic()
    with torch.inference_mode():
        for step,(image,label) in enumerate(loader):
            image,label=image.cuda(),label.cuda()
            code,z,received,logits,reconstruction=native(image)
            assert reconstruction.shape == image.shape and torch.isfinite(reconstruction).all()
            if step == 0:
                write_json(output/'tensor_shapes.json', {k:list(v.shape) for k,v in {'input':image,'code':code,'z':z,'received':received,'logits':logits,'reconstruction':reconstruction}.items()})
            correct += int((logits.argmax(1)==label).sum())
            count += len(image)
            for key,function in [('author_psnr_db','PSNR'),('author_ssim','SSIM'),('author_mse_01','MSE')]:
                totals[key] += float(namespace[function](image,reconstruction))
            mse=((image-reconstruction)/2).square().mean((1,2,3))
            totals['standard_rgb_psnr_db'] += float((-10*torch.log10(mse)).sum())
            if step%25==0:
                print(json.dumps({'stage':'evaluation','images':count,'total':len(dataset),'elapsed_seconds':time.monotonic()-start}),flush=True)
    result={key:value/count for key,value in totals.items()}
    result.update(accuracy=correct/count, images=count, elapsed_seconds=time.monotonic()-start,
                  status='completed', scope='pretrained evaluation, not trained reproduction')
    assert count == 10000
    write_json(output/'metrics.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
