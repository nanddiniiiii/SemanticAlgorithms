"""CPU Kodak source-codec baselines; no invented channel/protection comparison."""
import argparse
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT/'Datasets/KODAK')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import PIL
    from PIL import Image, features
    from src.metrics import quality
    from src.artifacts import write_json, sha256
    torch.set_num_threads(2)
    paths = sorted(args.images.glob('kodim*.png'))
    assert len(paths) == 24
    assert features.check_codec('jpg_2000'), 'Pillow needs OpenJPEG for this baseline'
    args.output.mkdir(parents=True, exist_ok=False)
    variants = [('RAW_RGB', None, {}), ('PNG', 'PNG', {'compress_level': 9, 'optimize': False}),
                ('JPEG2000_lossless', 'JPEG2000', {'irreversible': False})]
    variants += [('JPEG_q'+str(q), 'JPEG', {'quality': q, 'subsampling': 2, 'optimize': False})
                 for q in [10,30,50,70,90]]
    variants += [('JPEG2000_ratio'+str(r), 'JPEG2000',
                  {'irreversible': True, 'quality_mode': 'rates', 'quality_layers': [r]}) for r in [12,24,48]]
    all_rows, summaries = [], []
    for name, format_name, options in variants:
        rows = []
        for path in paths:
            with Image.open(path) as handle:
                image = handle.convert('RGB')
            original = np.asarray(image).copy()
            start = time.perf_counter()
            if format_name is None:
                encoded = original.tobytes()
            else:
                stream = io.BytesIO(); image.save(stream, format=format_name, **options); encoded = stream.getvalue()
            encode_seconds = time.perf_counter()-start
            start = time.perf_counter()
            if format_name is None:
                reconstructed = np.frombuffer(encoded, dtype=np.uint8).reshape(original.shape).copy()
            else:
                with Image.open(io.BytesIO(encoded)) as handle:
                    reconstructed = np.asarray(handle.convert('RGB')).copy()
            decode_seconds = time.perf_counter()-start
            reference = torch.from_numpy(original).permute(2,0,1)[None].float()/255
            recovered = torch.from_numpy(reconstructed).permute(2,0,1)[None].float()/255
            values = quality(reference, recovered)
            exact = np.array_equal(original, reconstructed)
            if name in {'RAW_RGB','PNG','JPEG2000_lossless'}:
                assert exact, name+' did not recover the original RGB samples'
            rows.append({'variant': name, 'sample': path.name, 'input_sha256': sha256(path),
                'shape_hwc': list(original.shape), 'encoded_bytes': len(encoded),
                'bpp': len(encoded)*8/(image.width*image.height),
                'raw_to_encoded_ratio': original.nbytes/len(encoded), 'exact_rgb': exact,
                'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds, **values})
        summary = {'variant': name, 'images': len(rows), 'format': format_name, 'options': options,
            'mean_bpp': sum(r['bpp'] for r in rows)/len(rows),
            'mean_encoded_bytes': sum(r['encoded_bytes'] for r in rows)/len(rows),
            'mean_psnr_db': sum(r['psnr_db'] for r in rows)/len(rows) if all(r['psnr_db'] is not None for r in rows) else None,
            'psnr_is_infinite': all(r['exact_rgb'] for r in rows),
            'mean_ssim': sum(r['ssim'] for r in rows)/len(rows),
            'exact_rgb_images': sum(r['exact_rgb'] for r in rows),
            'mean_encode_seconds': sum(r['encode_seconds'] for r in rows)/len(rows),
            'mean_decode_seconds': sum(r['decode_seconds'] for r in rows)/len(rows)}
        all_rows.extend(rows); summaries.append(summary)
        write_json(args.output/'per_image.json', all_rows)
        write_json(args.output/'metrics.json', {'status': 'running', 'variants': summaries})
        print(json.dumps({'stage': 'source_codec', **summary}), flush=True)
    result = {'status': 'completed', 'dataset': 'Kodak24 native RGB', 'variants': summaries,
        'python': sys.version, 'pillow': PIL.__version__, 'openjpeg': features.version_codec('jpg_2000'),
        'device': 'local CPU', 'torch_threads': 2, 'timing_repeats': 1,
        'runtime_scope': 'single CPU encode/decode observations; not a stable latency distribution or GPU comparison',
        'channel': None, 'snr_db': None, 'fec': None, 'radio_uses': None,
        'rate_scope': 'complete file bytes including codec headers, except headerless RAW_RGB; no radio/FEC overhead',
        'exactness_scope': 'decoded RGB samples, not byte-identical original PNG file',
        'unmeasured': ['LPIPS','semantic fidelity','radio errors','codec FLOPs']}
    write_json(args.output/'metrics.json', result)
    print('Traditional source-codec evaluation completed: '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
