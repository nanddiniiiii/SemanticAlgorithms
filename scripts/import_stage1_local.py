"""Import source-codec and legacy diagnostics with distinct, non-radio identities."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from src.artifacts import sha256,write_json,export_manifest
from src.events import Events,collect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['traditional','deepjscc_legacy'])
    parser.add_argument('input',type=Path)
    args=parser.parse_args();source=args.input.resolve()
    summary=json.loads((source/'metrics.json').read_text())
    assert summary['status']=='completed'
    if args.kind=='traditional':
        all_rows=json.loads((source/'per_image.json').read_text())
        groups={variant['variant']:[row for row in all_rows if row['variant']==variant['variant']]
                for variant in summary['variants']}
        write_json(ROOT/'Docs/validation/traditional_kodak24_metrics.json',summary)
        identity=sha256(source/'per_image.json')[:16]
    else:
        with (source/'per_image.csv').open(newline='') as stream:
            all_rows=list(csv.DictReader(stream))
        groups={'retained_training200dB_real_awgn_tiled':all_rows}
        write_json(ROOT/'Docs/validation/deepjscc_legacy_kodak24_metrics.json',summary)
        identity=sha256(source/'per_image.csv')[:16]
    for variant,rows in groups.items():
        assert len(rows)==len({row['sample'] for row in rows})==24
        algorithm='traditional_source_codec' if args.kind=='traditional' else 'deepjscc'
        run=ROOT/'Results/experiments'/algorithm/(variant+'-'+identity)
        if (run/'status.json').exists():continue
        run.mkdir(parents=True,exist_ok=True)
        write_json(run/'per_image.json',rows);write_json(run/'metrics.json',summary)
        identity_fields=SimpleNamespace(algorithm=algorithm,variant=variant,dataset='Kodak24',
            seed=22 if args.kind=='deepjscc_legacy' else None,snr_db=None)
        events=Events(run,identity_fields)
        previous=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known={row.get('sample') for row in previous}
        with contextlib.redirect_stdout(io.StringIO()):
            for row in rows:
                if row['sample'] in known:continue
                values={}
                for key,value in row.items():
                    if key in {'sample','variant','input_sha256','shape_hwc'} or value is None:continue
                    try:values[key]=float(value)
                    except (ValueError,TypeError):pass
                events.emit('source_codec' if args.kind=='traditional' else 'legacy_diagnostic','sample_completed',
                    values,sample=row['sample'],split='test',trial=0,
                    scope=summary.get('scope',summary.get('rate_scope')),radio_comparison_eligible=False,
                    legacy_real_channel_snr_db=10 if args.kind=='deepjscc_legacy' else None)
        if args.kind=='deepjscc_legacy':
            for name in ['model.json','tensor_shapes.json','reconstruction.png']:
                shutil.copyfile(source/name,run/name)
        write_json(run/'status.json',{'status':'completed','radio_comparison_eligible':False})
        export_manifest(run)
    print(json.dumps({'kind':args.kind,'groups':len(groups),
        'appended_rows':collect(ROOT/'Results/experiments',ROOT/'Results/metrics.csv')}))


if __name__=='__main__':main()
