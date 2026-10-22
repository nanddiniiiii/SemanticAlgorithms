"""Small command line interface; native model adapters own architecture details."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
from .artifacts import write_json, export_manifest
from .config import load_config
from .paths import resolve_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'prepare', 'construct', 'pilot', 'train', 'evaluate', 'profile', 'export', 'collect'])
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--inference-only', action='store_true', help='Use a forward-only pilot, even for trainable models')
    parser.add_argument('--root', default='Results/experiments')
    parser.add_argument('--csv', default='Results/metrics.csv')
    args = parser.parse_args(argv)
    if args.command == 'list':
        from dataclasses import asdict
        from .catalog import MODELS
        print(json.dumps([asdict(m) for m in MODELS.values()], indent=2))
        return
    if args.command == 'collect':
        from .events import collect
        print(json.dumps({'appended_rows': collect(resolve_path(args.root), resolve_path(args.csv))}))
        return
    if args.command == 'export':
        if not args.run_dir:
            parser.error('export requires --run-dir')
        print(json.dumps(export_manifest(resolve_path(args.run_dir)), indent=2))
        return
    if not args.config:
        parser.error('--config is required')
    config = load_config(args.config)
    if args.command == 'construct':
        from .shared_codebook import construct
        construct(config)
        return
    if args.command == 'prepare':
        from .catalog import get_model
        from .data import check_separate_folders
        check_separate_folders(config)
        if args.fetch:
            from .sources import fetch_source
            fetch_source("diffcom" if config.algorithm in {"adjscc", "hybrid_semantic_complementary"} else config.algorithm)
        print(json.dumps({'config': config.as_dict(), 'model': get_model(config.algorithm).__dict__}, indent=2))
        return
    if args.run_dir:
        parser.error('Evaluation/profile create unique runs; --run-dir is for export only')
    from .events import Events
    from .codecs import build_codec
    from .data import loader, check_separate_folders
    from .metrics import quality, profile_forward
    import torch
    import random
    import numpy as np
    run = resolve_path(config.output_root) / config.algorithm / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'config.json', config.as_dict())
    events = Events(run, config)
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        check_separate_folders(config)
        codec = build_codec(config).eval()
        write_json(run / 'model.json', codec.metadata)
        write_json(run / 'environment.json', {'torch': torch.__version__, 'device': config.device, 'cuda': torch.version.cuda})
        if args.command == 'pilot' and (not codec.trainable or args.inference_only):
            split = 'val' if config.dataset == 'cityscapes' else 'test'
            batch = next(iter(loader(config, split)))
            image = batch['image'].to(config.device)
            with torch.inference_mode():
                output = codec(image, batch['semantic'].to(config.device)) if getattr(codec,'requires_semantic',False) else codec(image)
            if output['reconstruction'].shape != image.shape or not torch.isfinite(output['reconstruction']).all():
                raise RuntimeError('Inference pilot failed shape/finite checks')
            events.emit('pilot', 'completed', shape=list(image.shape), scope='one real input inference check; not a benchmark')
            write_json(run / 'status.json', {'status':'pilot_completed','scope':'inference_only'})
            export_manifest(run)
            print(str(run),flush=True)
            return
        if args.command in ('pilot', 'train'):
            from .training import train
            train(config, codec, run, events, pilot=args.command == 'pilot')
            export_manifest(run)
            print(str(run), flush=True)
            return
        dataset = loader(config, 'test')
        events.emit(args.command, 'started', test_samples=len(dataset.dataset))
        if args.command == 'profile':
            batch = next(iter(dataset))
            profiled = codec
            if getattr(codec,'requires_semantic',False):
                class Paired(torch.nn.Module):
                    def __init__(self, core, semantic):
                        super().__init__()
                        self.core, self.semantic = core, semantic
                    def forward(self, image):
                        return self.core(image, self.semantic)
                profiled = Paired(codec, batch['semantic'].to(config.device))
            gradient_sampling = getattr(codec,'requires_gradient_inference',False)
            result = profile_forward(profiled, batch['image'].to(config.device),
                warmup=0 if gradient_sampling else 3, repeats=1 if gradient_sampling else 10,
                count_flops=not gradient_sampling)
            if gradient_sampling:
                result['flops_limitation'] = 'Posterior sampling includes input-gradient computation; complete inference FLOPs profiling is pending. No denoiser-only count is substituted.'
            write_json(run / 'profile.json', result)
            events.emit('profile', 'completed', metrics={k: v for k, v in result.items() if isinstance(v, (int, float))}, coverage=result['flops_coverage'])
        else:
            with torch.inference_mode():
                for index, batch in enumerate(dataset):
                    image = batch['image'].to(config.device)
                    for trial in range(config.trials):
                        trial_seed = config.seed + index * config.trials + trial
                        torch.manual_seed(trial_seed)
                        random.seed(trial_seed)
                        np.random.seed(trial_seed % 2**32)
                        output = codec(image, batch['semantic'].to(config.device)) if getattr(codec,'requires_semantic',False) else codec(image)
                        metrics = quality(image, output['reconstruction'], config.lpips)
                        if 'logits' in output:
                            metrics['classification_correct'] = int(output['logits'].argmax(1).item() == batch['label'].item())
                        uses = int(output['complex_uses'].reshape(-1)[0])
                        for key in ['source_bits','source_bits_estimate','vq_index_bits','mask_bitmap_bits','packet_header_crc_bits','codeword_index_bits','base_complex_uses','complementary_complex_uses','side_information_bits','denoiser_evaluations']:
                            if key in output:
                                metrics[key] = output[key]
                        if 'source_bits' in output:
                            metrics['source_bpp'] = output['source_bits'] / (image.shape[-2] * image.shape[-1])
                        if 'source_bits_estimate' in output:
                            metrics['source_bpp_estimate'] = output['source_bits_estimate'] / (image.shape[-2] * image.shape[-1])
                        metrics['complex_channel_uses'] = uses if uses >= 0 else None
                        metrics['cbr_main_stream'] = uses / image.numel() if uses >= 0 else None
                        events.emit('evaluate', 'sample_completed', metrics=metrics, split='test', sample=batch['sample'][0], trial=trial, rate_complete=codec.rate_complete)
                        if config.save_reconstructions:
                            from PIL import Image
                            values = output['reconstruction'][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                            target = run / 'reconstructions'
                            target.mkdir(exist_ok=True)
                            Image.fromarray(np.rint(values * 255).astype('uint8')).save(target / f'{index:06d}-{trial}.png')
                            if 'source_packet' in output:
                                (target / f'{index:06d}-{trial}.sq01').write_bytes(output['source_packet'])
                            if 'side_packet' in output:
                                (target / f'{index:06d}-{trial}.side').write_bytes(output['side_packet'])
            events.emit('evaluate', 'completed', evaluated_samples=len(dataset.dataset), trials=config.trials)
        write_json(run / 'status.json', {'status': 'completed', 'command': args.command})
        export_manifest(run)
    except BaseException as exc:
        events.emit(args.command, 'failed', error=str(exc), error_type=type(exc).__name__)
        write_json(run / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(str(run), flush=True)
