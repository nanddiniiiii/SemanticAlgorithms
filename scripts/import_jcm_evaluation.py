"""Import a completed native checkpoint evaluation into the unified ledger once."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from src.artifacts import sha256,write_json,export_manifest
from src.config import Config
from src.events import Events,collect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt_directory',type=Path)
    args=parser.parse_args()
    directory=args.receipt_directory.resolve()
    metrics=json.loads((directory/'metrics.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    if metrics.get('status')!='completed' or metrics.get('images')!=10000:
        raise ValueError('Expected completed full 10,000-image evaluation')
    if protocol.get('checkpoint_sha256')!='b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897':
        raise ValueError('This importer is specific to the verified JCM author checkpoint')
    identity=sha256(directory/'metrics.json')[:16]
    run=ROOT/'Results/experiments/jcm'/('author-eval-'+identity)
    if not (run/'status.json').is_file():
        run.mkdir(parents=True,exist_ok=True)
        for name in ['metrics.json','protocol.json','tensor_shapes.json']:
            shutil.copyfile(directory/name,run/name)
        config=Config('jcm',dataset='cifar10',crop_size=32,seed=42,snr_db=12)
        write_json(run/'config.json',config.as_dict())
        if not (run/'events.jsonl').exists():
            Events(run,config).emit('pretrained_evaluation','completed',
                metrics={key:value for key,value in metrics.items() if isinstance(value,(int,float)) and not isinstance(value,bool)},
                split='test',trial=0,scope=metrics['scope'],protocol_file='protocol.json',
                source_receipt_sha256=sha256(directory/'metrics.json'))
        write_json(run/'status.json',{'status':'completed','scope':'pretrained evaluation; training not reproduced'})
        export_manifest(run)
    print(json.dumps({'run':str(run),'appended_rows':collect(ROOT/'Results/experiments',ROOT/'Results/metrics.csv')}))


if __name__=='__main__':main()
