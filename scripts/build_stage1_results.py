"""Build a Stage1 inventory from checked receipts; missing metrics stay null."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semantic_lossless.catalog import MODELS


def read_receipt(name):
    path = ROOT/'Docs/validation'/name
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    retained = {'deepjscc', 'ntscc', 'swinjscc'}
    downloaded = {'adjscc', 'jcm', 'sqgan', 'swinjscc', 'mambajscc', 'diffcom', 'diffjscc'}
    excluded = {'diffcom', 'diffjscc', 'dit_jscc', 'hybrid_semantic_complementary'}
    shapes = {'jcm': 'RGB 3x32x32', 'sqgan': 'RGB 3x256x512 + semantic 20x256x512',
        'shared_semantic_codebook': 'RGB 3x256x256; auxiliary classifier 3x96x96',
        'mambajscc': 'filename claims256; stored config not runtime-certified',
        'swinjscc': 'native config256; Kodak full-resolution evaluation submitted',
        'ntscc': 'Kodak native512x768 or768x512'}
    quality = {
        'jcm': ('jcm_pretrained_metrics.json', 'standard_rgb_psnr_db', 'author_ssim', 'CIFAR10 test10000'),
        'ntscc': ('ntscc_kodak24_metrics.json', 'mean_psnr_db', 'mean_ssim', 'Kodak24'),
        'sqgan': ('sqgan_cityscapes500_native_metrics.json', 'mean_psnr_native_saved_range1_db', None, 'Cityscapes val500'),
    }
    records = []
    for name, model in MODELS.items():
        receipt = None; metrics = {'psnr_db': None, 'ssim': None, 'ms_ssim': None, 'lpips': None}
        dataset = None
        if name in quality:
            filename, psnr_key, ssim_key, dataset = quality[name]
            receipt = read_receipt(filename)
            if receipt:
                metrics['psnr_db'] = receipt[psnr_key]
                metrics['ssim'] = receipt.get(ssim_key) if ssim_key else None
        group = 'reference_only_non_reconstruction' if name == 'sc_lmkb' else 'excluded_from_current_stage' if name in excluded else 'retain_existing' if name in retained else 'use_downloaded_author_weights' if name in downloaded else 'author_weights_inaccessible' if name == 'mdjcm' else 'code_only_candidate' if name in {'shared_semantic_codebook','deepjscc_q'} else 'unverified_executable_release'
        semantic = {'metric': 'classification_accuracy', 'value': receipt['accuracy']} if name == 'jcm' and receipt else {'metric': None, 'value': None, 'reason': 'Not measured on an independently labelled task in the current receipt'}
        rate = {'value': None, 'unit': None, 'complete': False}
        if name == 'jcm': rate.update(value=128, unit='complex_uses_per_image', note='native16QAM at12dB; unspecified transport overhead')
        if name == 'ntscc' and receipt: rate.update(value=receipt['mean_main_stream_cbr'], unit='complex_uses_per_RGB_sample', note='main stream only; side information omitted')
        if name == 'sqgan': rate.update(value=3584, unit='source_payload_bits_per_256x512_image', note='branch fractions.25/.25; SQ01 extension adds128bits; no radio chain')
        profile = read_receipt(name+'_cpu_profile.json')
        records.append({'algorithm': name, 'family': model.family, 'action_group': group,
            'retained_weights': name in retained, 'downloaded_author_weights': name in downloaded,
            'implementation_backend': model.backend, 'fidelity_note': model.note,
            'input_dimensions': shapes.get(name), 'dataset_evaluated': dataset,
            'pixel_and_perceptual': metrics, 'semantic': semantic, 'transmission': rate,
            'compute': {'encoder_flops': None, 'decoder_flops': None, 'full_flops': None,
                'counted_forward_flops': profile['forward_flops'] if profile else None,
                'counted_forward_coverage': profile['flops_coverage'] if profile else None,
                'parameters': profile['parameters'] if profile else None,
                'profile_input_shape': profile.get('input_shape') if profile else None,
                'reason': 'Partial supported-operator CPU forward count only; separate encoder/decoder and complete coverage pending' if profile else 'Validated Stage1 profile receipt not yet imported; no inferred FLOPs'},
            'runtime_reliability': {'latency': None, 'peak_memory': None,
                'reason': 'Do not substitute whole evaluation duration or optimizer pilot for isolated inference latency'},
            'evidence_receipt': 'Docs/validation/'+quality[name][0] if name in quality else None})
    records.append({'algorithm': 'moc_rvq', 'action_group': 'unverified_executable_release',
        'retained_weights': False, 'downloaded_author_weights': False,
        'fidelity_note': 'Checked author-associated repository has code/weights forthcoming; not executable'})
    result = {'stage': 1, 'stage2_started': False, 'snapshot_date': '2026-10-22',
        'scope': 'inventory and measured receipts; not a matched performance ranking',
        'missing_value_policy': 'null means unmeasured/unsupported, never zero', 'models': records}
    output = ROOT/'Docs/validation/stage1_results_inventory.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Stage 1 metric coverage', '', 'Generated from local receipts by scripts/build_stage1_results.py.',
        'Missing values are unmeasured, not zero. Different native datasets/protocols cannot be ranked directly.', '',
        '| Model | Action group | Evaluation data | RGB PSNR dB | SSIM | LPIPS | Profile input BxCxHxW | Counted forward GFLOPs (partial) |',
        '|---|---|---|---:|---:|---:|---|---:|']
    for row in records:
        metric = row.get('pixel_and_perceptual', {})
        def display(value): return '—' if value is None else f'{value:.6f}'
        ssim = display(metric.get('ssim'))
        if row['algorithm'] == 'jcm' and ssim != '—': ssim += ' (author convention)'
        counted = row.get('compute', {}).get('counted_forward_flops')
        shape = row.get('compute', {}).get('profile_input_shape')
        shape_text = 'x'.join(map(str, shape)) if shape else '—'
        lines.append(f"| {row['algorithm']} | {row['action_group']} | {row.get('dataset_evaluated') or '—'} | {display(metric.get('psnr_db'))} | {ssim} | {display(metric.get('lpips'))} | {shape_text} | {display(counted/1e9 if counted is not None else None)} |")
    lines += ['', 'JCM RGB PSNR is the standard RGB receipt value; its separately reported author PSNR is23.409674dB.',
        'SQ-GAN PSNR uses native saved-image postprocessing and has no simulated radio channel.',
        'See STAGE_ONE_BASELINES.md for fidelity, blockers, semantic/rate/runtime fields and the training queue.',
        'The JSON inventory preserves metric groups and reasons. GPU job submission does not establish completion.']
    (ROOT/'Docs/STAGE_ONE_METRIC_COVERAGE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert len(records) == len({r['algorithm'] for r in records}) == len(MODELS)+1
    print(json.dumps({'models': len(records), 'quality_receipts': len(quality), 'output': str(output)}))


if __name__ == '__main__':
    main()
