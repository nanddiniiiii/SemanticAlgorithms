"""Import completed SQ-GAN source-codec results without inventing radio metrics."""
import argparse
import contextlib
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semantic_lossless.artifacts import sha256,write_json,export_manifest
from semantic_lossless.config import Config
from semantic_lossless.events import Events,collect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt_directory',type=Path)
    args=parser.parse_args();directory=args.receipt_directory.resolve()
    summary=json.loads((directory/'metrics.json').read_text())
    if summary.get('status')!='completed' or summary.get('images')!=500:
        raise ValueError('Expected completed official 500-image Cityscapes evaluation')
    with (directory/'per_image.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=500 or len({row['sample'] for row in rows})!=500:
        raise ValueError('Expected 500 unique sample records')
    write_json(ROOT/'Docs/validation/sqgan_cityscapes500_native_metrics.json',summary)
    for name in ['profile_full.json','semantic_branch.json']:
        if (directory/name).is_file():
            shutil.copyfile(directory/name,ROOT/'Docs/validation'/('sqgan_'+name))
    identity=sha256(directory/'per_image.csv')[:16]
    run=ROOT/'Results/experiments/sqgan'/('source-eval-'+identity)
    run.mkdir(parents=True,exist_ok=True)
    # Resume interrupted imports by sample identity; collect only after completion.
    if not (run/'status.json').is_file():
        for name in ['metrics.json','per_image.csv','config.json','model.json','native_equivalence.json','environment.json']:
            shutil.copyfile(directory/name,run/name)
        if (directory/'native_preprocessing.json').is_file():
            shutil.copyfile(directory/'native_preprocessing.json',run/'native_preprocessing.json')
        for name in ['profile_full.json','semantic_branch.json','reconstruction.png','source_packet.sq01']:
            if (directory/name).is_file():shutil.copyfile(directory/name,run/name)
        previous=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known={record.get('sample') for record in previous if record.get('status')=='sample_completed'}
        events=Events(run,Config('sqgan',dataset='cityscapes',seed=22))
        with (run/'import_stdout.log').open('a') as log, contextlib.redirect_stdout(log):
            for row in rows:
                if row['sample'] in known:continue
                values={key:float(value) for key,value in row.items() if key!='sample'}
                events.emit('source_codec_evaluation','sample_completed',values,split='val',sample=row['sample'],trial=0,
                    scope=summary['scope'],radio_channel_simulated=False,training_reproduced=False)
            if not any(record.get('status')=='completed' for record in previous):
                aggregates={key:value for key,value in summary.items() if key.startswith('mean_') and isinstance(value,(int,float))}
                events.emit('source_codec_evaluation','completed',{'images':500,**aggregates},split='val',
                    unmeasured=summary['unmeasured'],scope=summary['scope'])
        write_json(run/'status.json',{'status':'completed','scope':summary['scope']})
        export_manifest(run)
    print(json.dumps({'run':str(run),'appended_rows':collect(ROOT/'Results/experiments',ROOT/'Results/metrics.csv')}))


if __name__=='__main__':main()
