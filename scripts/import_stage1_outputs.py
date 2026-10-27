"""Import downloaded native Stage1 events into the existing deduplicated CSV."""
import contextlib
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from src.artifacts import export_manifest, sha256, write_json
from src.config import Config
from src.events import Events, collect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'Results/stage1_downloads')
    args = parser.parse_args(argv)
    root = args.input.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    imported = []
    for metrics_path in sorted(root.rglob('adjscc/diffcom_C2/metrics.json')):
        source = metrics_path.parent
        summary = json.loads(metrics_path.read_text())
        if summary.get('status') != 'completed' or summary.get('images') != 24:
            continue
        shutil.copyfile(metrics_path, ROOT/'Docs/validation/adjscc_gpu_metrics.json')
        for path in source.glob('profile_*.json'):
            shutil.copyfile(path, ROOT/'Docs/validation'/('adjscc_'+path.name))
        identity = sha256(source/'per_image.csv')[:16]
        run = ROOT/'Results/experiments/adjscc'/('pretrained-C2-'+identity)
        if (run/'status.json').exists(): continue
        run.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file(): shutil.copyfile(path, run/path.name)
        events = Events(run, Config('adjscc', dataset='Kodak24', snr_db=10, seed=22))
        events.identity['variant'] = 'baseline_diffcom_C2_per_image_awgn'
        previous = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known = {record.get('sample') for record in previous if record.get('stage') == 'pretrained_evaluation'}
        with contextlib.redirect_stdout(io.StringIO()), (source/'per_image.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['sample'] in known: continue
                values = {key: float(value) for key, value in row.items()
                    if key not in {'sample', 'sha256'} and value not in {'', 'None'}}
                events.emit('pretrained_evaluation', 'sample_completed', values,
                    split='test', sample=row['sample'], trial=0, rate_complete=False, training_reproduced=False)
            for component in ['full', 'encoder', 'decoder']:
                if any(record.get('stage') == 'profile_'+component for record in previous): continue
                profile = json.loads((source/('profile_'+component+'.json')).read_text())
                values = {key: value for key, value in profile.items() if isinstance(value, (int, float))}
                values['registered_kernel_flops'] = profile['registered_kernel_count']['registered_kernel_flops']
                events.emit('profile_'+component, 'completed', values,
                    flops_coverage=profile['flops_coverage'], input_shape=profile['input_shape'])
        write_json(run/'status.json', {'status': 'completed', 'scope': summary['scope']})
        export_manifest(run)
        imported.append(str(run))
    for metrics_path in sorted(root.rglob('metrics.json')):
        source = metrics_path.parent
        if source.name not in {'fixed_c32', 'sa_ra'} or not (source/'per_image.csv').exists(): continue
        summary = json.loads(metrics_path.read_text())
        if summary.get('status') != 'completed': continue
        shutil.copyfile(metrics_path, ROOT/'Docs/validation'/('swin_'+source.name+'_gpu_metrics.json'))
        for path in source.glob('profile_*.json'):
            shutil.copyfile(path, ROOT/'Docs/validation'/('swin_'+source.name+'_'+path.name))
        identity = sha256(source/'per_image.csv')[:16]
        run = ROOT/'Results/experiments/swinjscc'/('native-'+source.name+'-'+identity)
        if (run/'status.json').exists(): continue
        run.mkdir(parents=True, exist_ok=True)
        for path in source.glob('*'):
            if path.is_file(): shutil.copyfile(path, run/path.name)
        events = Events(run, Config('swinjscc', dataset='Kodak24', snr_db=10, seed=22, rate=32))
        events.identity['variant'] = 'baseline_native_'+source.name
        previous = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known = {row.get('sample') for row in previous if row.get('stage') == 'native_evaluation'}
        with (source/'per_image.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == summary['images']
        with contextlib.redirect_stdout(io.StringIO()):
            for row in rows:
                if row['sample'] in known: continue
                values = {}
                for key, value in row.items():
                    if key in {'sample', 'sha256', 'rate_complete'} or value in {'', 'None'}: continue
                    try: values[key] = float(value)
                    except ValueError: pass
                events.emit('native_evaluation', 'sample_completed', values, split='test', sample=row['sample'], trial=0,
                    native_variant=source.name, rate_complete=False, training_reproduced=False)
            for component in ['full', 'encoder', 'decoder']:
                if any(record.get('stage') == 'profile_'+component for record in previous): continue
                path = source/('profile_'+component+'.json')
                if not path.exists(): continue
                profile = json.loads(path.read_text())
                events.emit('profile_'+component, 'completed',
                    {key: value for key, value in profile.items() if isinstance(value, (int, float))},
                    native_variant=source.name, flops_coverage=profile['flops_coverage'], input_shape=profile.get('input_shape'))
        write_json(run/'status.json', {'status': 'completed', 'scope': summary['scope']})
        export_manifest(run)
        imported.append(str(run))
    for status_path in sorted(root.rglob('classifier_training/status.json')):
        source = status_path.parent; summary = json.loads(status_path.read_text())
        if summary.get('status') not in {'completed', 'budget_stopped'}: continue
        epochs = [json.loads(line) for line in (source/'epochs.jsonl').read_text().splitlines()]
        write_json(ROOT/'Docs/validation/shared_classifier_training.json',
            {**summary, 'last_epoch': epochs[-1], 'checkpoint_sha256': sha256(source/'classifier.pt'),
             'scope': 'auxiliary STL10 classifier; training-set metrics only; no semantic codec trained'})
        identity = sha256(status_path)[:16]
        run = ROOT/'Results/experiments/shared_semantic_codebook'/('native-aux-classifier-'+identity)
        if (run/'status.json').exists(): continue
        run.mkdir(parents=True, exist_ok=True)
        events = Events(run, Config('shared_semantic_codebook', dataset='stl10', seed=1024))
        previous = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known_epochs = {record.get('epoch') for record in previous}
        with contextlib.redirect_stdout(io.StringIO()):
            for line in (source/'epochs.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['epoch'] in known_epochs: continue
                events.emit('auxiliary_classifier_training', 'epoch_record',
                    {key: value for key, value in row.items() if isinstance(value, (int, float))},
                    snr_db=None, split='train', epoch=row['epoch'], semantic_codec_trained=False)
        for path in source.iterdir():
            if path.is_file() and path.name != 'status.json': shutil.copyfile(path, run/path.name)
        checkpoint_folder = ROOT/'Models/trained_checkpoints/SharedSemanticCodebook'/('stage1_classifier_'+identity)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/'classifier.pt', checkpoint_folder/'classifier.pt')
        write_json(run/'status.json', summary)
        export_manifest(run)
        imported.append(str(run))
    count = collect(ROOT/'Results/experiments', ROOT/'Results/metrics.csv')
    print(json.dumps({'imported_runs': imported, 'appended_metric_rows': count}))


if __name__ == '__main__':
    main()
