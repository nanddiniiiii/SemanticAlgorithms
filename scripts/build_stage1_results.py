"""Build a Stage1 inventory from checked receipts; missing metrics stay null."""
import json
import csv
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
        'swinjscc': 'RGB 3x512x768 or3x768x512; native Kodak24 evaluation',
        'ntscc': 'Kodak native512x768 or768x512'}
    quality = {
        'jcm': ('jcm_pretrained_metrics.json', 'standard_rgb_psnr_db', 'author_ssim', 'CIFAR10 test10000'),
        'ntscc': ('ntscc_kodak24_metrics.json', 'mean_psnr_db', 'mean_ssim', 'Kodak24'),
        'sqgan': ('sqgan_cityscapes500_native_metrics.json', 'mean_psnr_native_saved_range1_db', 'mean_ssim', 'Cityscapes val500'),
        'swinjscc': ('swin_fixed_c32_gpu_metrics.json', 'mean_psnr_db', 'mean_ssim', 'Kodak24; native fixed-C32'),
    }
    if read_receipt('adjscc_gpu_metrics.json'):
        quality['adjscc'] = ('adjscc_gpu_metrics.json', 'mean_psnr_db', 'mean_ssim', 'Kodak24; DiffCom PyTorch C2')
    if read_receipt('deepjscc_legacy_kodak24_metrics.json'):
        quality['deepjscc'] = ('deepjscc_legacy_kodak24_metrics.json', 'mean_psnr_db', 'mean_ssim', 'Kodak24; legacy real-channel diagnostic only')
    if read_receipt('ntscc_gpu_metrics.json'):
        quality['ntscc']=('ntscc_gpu_metrics.json','mean_psnr_db','mean_ssim','Kodak24; retained q1 GPU')
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
                metrics['ms_ssim'] = receipt.get('mean_ms_ssim')
                metrics['lpips'] = receipt.get('mean_lpips')
        group = 'reference_only_non_reconstruction' if name == 'sc_lmkb' else 'excluded_from_current_stage' if name in excluded else 'retain_existing' if name in retained else 'use_downloaded_author_weights' if name in downloaded else 'author_weights_inaccessible' if name == 'mdjcm' else 'code_only_candidate' if name in {'shared_semantic_codebook','deepjscc_q','amp_sc'} else 'unverified_executable_release'
        semantic = {'metric': 'classification_accuracy', 'value': receipt['accuracy']} if name == 'jcm' and receipt else {'metric': None, 'value': None, 'reason': 'Not measured on an independently labelled task in the current receipt'}
        rate = {'value': None, 'unit': None, 'complete': False}
        if name == 'jcm': rate.update(value=128, unit='complex_uses_per_image', note='native16QAM at12dB; unspecified transport overhead')
        if name == 'ntscc' and receipt: rate.update(value=receipt['mean_main_stream_cbr'], unit='complex_uses_per_RGB_sample', note='main stream only; rate-index/mask transport and power scale unmodeled; hyperprior refinement disabled', fixed_width_rate_index_bits_per_kodak=6144)
        if name == 'sqgan': rate.update(value=3584, unit='source_payload_bits_per_256x512_image', note='branch fractions.25/.25; SQ01 extension adds128bits; no radio chain')
        if name == 'swinjscc': rate.update(value=1/48, unit='complex_uses_per_RGB_sample', note='native fixed-C32,10dB; overhead unspecified')
        if name == 'adjscc': rate.update(value=1/48, unit='complex_uses_per_RGB_sample', note='DiffCom C2; per-image AWGN adaptation; overhead unspecified')
        profile = read_receipt(name+'_cpu_profile.json')
        gpu_profile = read_receipt('swin_fixed_c32_profile_full.json') if name == 'swinjscc' else read_receipt(name+'_profile_full.json') if name in {'adjscc','ntscc','jcm','sqgan'} else None
        if gpu_profile:
            profile = gpu_profile
        records.append({'algorithm': name, 'family': model.family, 'action_group': group,
            'retained_weights': name in retained, 'downloaded_author_weights': name in downloaded,
            'implementation_backend': model.backend, 'fidelity_note': model.note,
            'quality_scope': receipt.get('scope') if receipt else None,
            'matched_radio_comparison_eligible': False,
            'input_dimensions': shapes.get(name), 'dataset_evaluated': dataset,
            'pixel_and_perceptual': metrics, 'semantic': semantic, 'transmission': rate,
            'compute': {'encoder_flops': None, 'decoder_flops': None, 'full_flops': None,
                'counted_forward_flops': profile['forward_flops'] if profile else None,
                'counted_forward_coverage': profile['flops_coverage'] if profile else None,
                'parameters': profile['parameters'] if profile else None,
                'profile_input_shape': profile.get('input_shape') if profile else None,
                'reason': 'Partial GPU operator count; separate component receipts available; not total arithmetic' if gpu_profile else 'Partial supported-operator CPU forward count only; separate encoder/decoder and complete coverage pending' if profile else 'Validated Stage1 profile receipt not yet imported; no inferred FLOPs'},
            'runtime_reliability': {'latency': gpu_profile.get('latency_p50_seconds') if gpu_profile else None,
                'peak_memory': gpu_profile.get('peak_cuda_memory_bytes') if gpu_profile else None,
                'reason': f"T4 FP32 batch1 input{gpu_profile['input_shape']}; warmup{gpu_profile['warmup']}/repeats{gpu_profile['repeats']}; memory is process residency" if gpu_profile else 'Do not substitute whole evaluation duration or optimizer pilot for isolated inference latency'},
            'evidence_receipt': 'Docs/validation/'+quality[name][0] if name in quality else None})
    records.append({'algorithm': 'moc_rvq', 'action_group': 'unverified_executable_release',
        'retained_weights': False, 'downloaded_author_weights': False,
        'fidelity_note': 'Checked author-associated repository has code/weights forthcoming; not executable'})
    variants = {'swin_sa_ra': read_receipt('swin_sa_ra_gpu_metrics.json')}
    result = {'stage': 1, 'stage2_started': False, 'snapshot_date': '2026-10-25',
        'scope': 'inventory and measured receipts; not a matched performance ranking',
        'missing_value_policy': 'null means unmeasured/unsupported, never zero', 'models': records,
        'additional_checkpoint_variants': variants}
    output = ROOT/'Docs/validation/stage1_results_inventory.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'Docs/validation/stage1_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        headers=['algorithm','family','action_group','dataset','psnr_db','ssim','ms_ssim','lpips',
            'rate_value','rate_unit','rate_complete','parameters','counted_forward_flops_partial',
            'latency_p50_seconds','peak_cuda_memory_bytes','evidence_receipt','scope']
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader()
        for row in records:
            metric=row.get('pixel_and_perceptual',{});rate=row.get('transmission',{})
            compute=row.get('compute',{});runtime=row.get('runtime_reliability',{})
            writer.writerow({'algorithm':row['algorithm'],'family':row.get('family'),
                'action_group':row['action_group'],'dataset':row.get('dataset_evaluated'),**metric,
                'rate_value':rate.get('value'),'rate_unit':rate.get('unit'),'rate_complete':rate.get('complete'),
                'parameters':compute.get('parameters'),'counted_forward_flops_partial':compute.get('counted_forward_flops'),
                'latency_p50_seconds':runtime.get('latency'),'peak_cuda_memory_bytes':runtime.get('peak_memory'),
                'evidence_receipt':row.get('evidence_receipt'),'scope':row.get('quality_scope') or row.get('fidelity_note')})
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
