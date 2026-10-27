"""Stage-1 native Swin Kodak evaluation and explicit partial-FLOPs profiling."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--variant', choices=['fixed_c32', 'sa_ra'], required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=24)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_msssim import ms_ssim
    from src.artifacts import write_json, sha256
    from src.metrics import quality, profile_forward
    from src.swin_native import build_native_swin
    assert torch.cuda.is_available(), 'A real CUDA device is required by the unmodified native implementation'
    random.seed(22); np.random.seed(22); torch.manual_seed(22)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    model, metadata = build_native_swin(args.checkpoint, args.variant)
    write_json(args.output/'model.json', metadata)
    paths = sorted(args.images.rglob('kodim*.png'))[:args.max_images]
    assert paths, 'Kodak images not found'
    snr, rate = 10, 32
    first_image = None
    rows = []
    latent_shapes = {}
    def observe_encoder(module, inputs, output):
        feature = output[0] if isinstance(output, tuple) else output
        latent_shapes.update(feature_shape=list(feature.shape),
            mask_shape=list(output[1].shape) if isinstance(output, tuple) else None)
    shape_hook = model.encoder.register_forward_hook(observe_encoder)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = np.asarray(image.convert('RGB'), dtype=np.float32).copy()/255
        image = torch.from_numpy(image).permute(2, 0, 1)[None].cuda()
        assert image.shape[-2] % 128 == image.shape[-1] % 128 == 0
        with torch.inference_mode():
            torch.cuda.synchronize(); start = time.perf_counter()
            recon, cbr, _, native_mse, _ = model(image, snr, rate)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            assert recon.shape == image.shape and torch.isfinite(recon).all()
            values = quality(image, recon, use_lpips=True)
            values['ms_ssim'] = float(ms_ssim(image, recon.clamp(0, 1), data_range=1, size_average=True))
        uses = int(round(float(cbr)*image.numel()))
        assert uses == image.shape[-2]*image.shape[-1]*rate//512
        row = {'sample': path.name, 'sha256': sha256(path), 'height': image.shape[-2],
            'width': image.shape[-1], 'snr_db': snr, 'active_channels': rate,
            'complex_uses': uses, 'main_stream_cbr': float(cbr), 'rate_complete': False,
            'single_forward_seconds_including_native_loss': elapsed,
            'native_mse_255': float(native_mse), **values}
        rows.append(row)
        if first_image is None:
            first_image = image
            write_json(args.output/'tensor_shapes.json', {'input':list(image.shape),
                'reconstruction':list(recon.shape), **latent_shapes})
            Image.fromarray(recon[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(args.output/'reconstruction.png')
        with (args.output/'per_image.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({'stage': 'evaluation', 'variant': args.variant, 'images': index+1, 'psnr_db': values['psnr_db']}), flush=True)
    shape_hook.remove()
    del image, recon

    class Full(torch.nn.Module):
        def __init__(self): super().__init__(); self.native = model
        def forward(self, image): return self.native(image, snr, rate)[0]
    full = profile_forward(Full(), first_image, warmup=3, repeats=10)
    full['flops_scope'] = 'native codec forward including channel, clamping and native loss; evaluator LPIPS/MS-SSIM excluded'
    write_json(args.output/'profile_full.json', full)
    decoder_inputs = []
    hook = model.decoder.register_forward_pre_hook(lambda module, inputs: decoder_inputs.append(inputs[0].detach()))
    with torch.inference_mode(): model(first_image, snr, rate)
    hook.remove()
    class Encoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.core = model.encoder
        def forward(self, image): return self.core(image, snr, rate, model.model)
    class Decoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.core = model.decoder
        def forward(self, latent): return self.core(latent, snr, model.model)
    for name, module, tensor in [('encoder', Encoder(), first_image), ('decoder', Decoder(), decoder_inputs[0])]:
        result = profile_forward(module, tensor, warmup=3, repeats=10)
        result['flops_scope'] = 'native '+name+' only; partial supported-operator count'
        result['input_shape'] = list(tensor.shape)
        result['memory_note'] = 'Whole native model remains resident; not isolated encoder/decoder deployment memory'
        write_json(args.output/('profile_'+name+'.json'), result)
    summary = {'status': 'completed', 'variant': args.variant, 'images': len(rows), 'snr_db': snr,
        'mean_psnr_db': sum(r['psnr_db'] for r in rows)/len(rows),
        'mean_ssim': sum(r['ssim'] for r in rows)/len(rows),
        'mean_ms_ssim': sum(r['ms_ssim'] for r in rows)/len(rows),
        'mean_lpips': sum(r['lpips'] for r in rows)/len(rows),
        'scope': 'native pretrained Kodak evaluation; one channel trial; no training reproduction',
        'rate_complete': False, 'rate_limitation': 'native main-stream CBR; transport and shared side-information overhead not specified',
        'semantic_metrics': None, 'semantic_metric_reason': 'Kodak has no task labels; no invented semantic score',
        'successful_images': len(rows), 'noise_seed': 22,
        'profile_note': 'first image at native dimensions; profiled FLOPs are partial, not certified total FLOPs'}
    write_json(args.output/'metrics.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
