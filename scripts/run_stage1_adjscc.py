"""Pretrained DiffCom-lineage ADJSCC C2: Kodak quality and separate GPU profiles."""
import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_msssim import ms_ssim
    from src.artifacts import sha256, write_json
    from src.config import Config
    from src.adjscc import build_adjscc
    from src.sources import fetch_source
    from src.metrics import quality, profile_forward
    from src.compute import registered_kernel_flops
    assert torch.cuda.is_available()
    assert sha256(args.checkpoint) == 'c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(22)
    fetch_source('diffcom')
    config = Config('adjscc', device='cuda:0', snr_db=10, checkpoint=str(args.checkpoint),
                    model_options={'latent_channels': 2})
    model = build_adjscc(config).eval()
    write_json(args.output/'model.json', model.metadata)
    write_json(args.output/'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'execution_device': 'cuda:0', 'multi_gpu': False, 'python': sys.version})
    paths = sorted(args.images.rglob('kodim*.png'))
    assert len(paths) == 24, 'Full Kodak24 is required'
    rows, first, decoder_input = [], None, []
    hook = model.core.jscc_decoder.register_forward_pre_hook(
        lambda module, inputs: decoder_input.append(inputs[0].detach()) if not decoder_input else None)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            array = np.asarray(image.convert('RGB'), dtype=np.float32).copy()/255
        image = torch.from_numpy(array).permute(2, 0, 1)[None].cuda()
        with torch.inference_mode():
            output = model(image)
            reconstruction = output['reconstruction']
            assert reconstruction.shape == image.shape
            values = quality(image, reconstruction, use_lpips=True)
            values['ms_ssim'] = float(ms_ssim(image, reconstruction.clamp(0, 1), data_range=1))
        uses = int(output['complex_uses'][0])
        assert uses == image.shape[-2]*image.shape[-1]//16
        rows.append({'sample': path.name, 'sha256': sha256(path), 'height': image.shape[-2],
            'width': image.shape[-1], 'complex_uses': uses, 'main_stream_cbr': uses/image.numel(),
            'snr_db': 10, **values})
        with (args.output/'per_image.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if first is None:
            first = image
            write_json(args.output/'tensor_shapes.json', {'input': list(image.shape),
                'latent': list(decoder_input[0].shape), 'reconstruction': list(reconstruction.shape)})
            Image.fromarray(reconstruction[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(args.output/'reconstruction.png')
        print({'stage': 'quality', 'completed': index+1, 'total': 24, 'psnr_db': values['psnr_db']}, flush=True)
    hook.remove()
    # LPIPS is an evaluator, not part of codec residency during profiling.
    import src.metrics as metric_module
    if hasattr(metric_module, '_LPIPS'):
        del metric_module._LPIPS
    del image, reconstruction, output
    torch.cuda.empty_cache()
    context = first.new_full((1, 1), 10)

    class Component(torch.nn.Module):
        def __init__(self, core, kind):
            super().__init__(); self.core = core; self.kind = kind
        def forward(self, value):
            if self.kind == 'full': return self.core(value)['reconstruction']
            return self.core(value, context)

    for name, core, value in [('full', model, first), ('encoder', model.core.jscc_encoder, first),
                               ('decoder', model.core.jscc_decoder, decoder_input[0])]:
        module = Component(core, name)
        print({'stage': 'profile', 'component': name, 'input_shape': list(value.shape)}, flush=True)
        profile = profile_forward(module, value, warmup=3, repeats=20)
        profile['registered_kernel_count'] = registered_kernel_flops(module, value)
        profile['flops_scope'] = name+' forward only; inference; registered counts and profiler counts are separate estimates'
        profile['memory_note'] = 'Whole codec remains resident; evaluator removed; not isolated component deployment memory'
        write_json(args.output/('profile_'+name+'.json'), profile)
    summary = {'status': 'completed', 'images': len(rows), 'algorithm': 'adjscc',
        'variant': 'diffcom_pytorch_C2_per_image_awgn', 'snr_db': 10, 'noise_seed': 22,
        'scope': 'pretrained evaluation of declared PyTorch lineage and transport adaptation; not original TensorFlow reproduction',
        'rate_complete': False, 'rate_limitation': 'main stream only; headers, pilots and CSI excluded',
        'semantic_metrics': None, 'semantic_metric_reason': 'Kodak has no independent task labels',
        'main_stream_cbr': 1/48, 'exact_images': sum(r['exact_uint8_pixels'] for r in rows)}
    for key in ['psnr_db', 'ssim', 'ms_ssim', 'lpips']:
        summary['mean_'+key] = sum(row[key] for row in rows)/len(rows)
    write_json(args.output/'metrics.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
