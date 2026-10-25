"""Idempotent import of recovered JCM profiles and NTSCC GPU quality/profile."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semantic_lossless.artifacts import write_json,sha256,export_manifest
from semantic_lossless.config import Config
from semantic_lossless.events import Events,collect


def main():
    receipts=json.loads((ROOT/'Results/stage1_operations/jcm_sqgan_ntscc_transition_receipt.json').read_text())
    final=json.loads((ROOT/'Results/stage1_operations/stage1_final_gpu_transition_receipt.json').read_text())
    for algorithm,source in [('jcm',ROOT/receipts['jcm_v1']['retrieved_path']),
                              ('ntscc',ROOT/final['ntscc_v1']['retrieved_path'])]:
        if algorithm=='jcm' and not (source/'status.json').exists():
            matches=list(source.rglob('native_profiles/status.json'))
            assert len(matches)==1
            source=matches[0].parent
        marker=source/('status.json' if algorithm=='jcm' else 'metrics.json')
        summary=json.loads(marker.read_text());assert summary['status']=='completed'
        run=ROOT/'Results/experiments'/algorithm/('gpu-extension-'+sha256(marker)[:16])
        if (run/'status.json').exists():continue
        run.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name!='status.json':shutil.copyfile(path,run/path.name)
        events=Events(run,Config(algorithm,dataset='CIFAR10-profiling-only' if algorithm=='jcm' else 'Kodak24',
            seed=42 if algorithm=='jcm' else 22,snr_db=12 if algorithm=='jcm' else 10))
        events.identity['variant']='baseline_native16QAM_profiles' if algorithm=='jcm' else 'baseline_retained_q1_gpu'
        previous=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
        known={record.get('sample') for record in previous if record.get('stage')=='gpu_evaluation'}
        with contextlib.redirect_stdout(io.StringIO()):
            if algorithm=='ntscc':
                with (source/'per_image.csv').open(newline='') as stream:
                    for row in csv.DictReader(stream):
                        if row['sample'] in known:continue
                        events.emit('gpu_evaluation','sample_completed',
                            {key:float(value) for key,value in row.items() if key not in {'sample','sha256'}},
                            sample=row['sample'],split='test',trial=0,rate_complete=False)
            profiles=sorted(source.glob('*batch*.json')) if algorithm=='jcm' else [source/'profile_full.json']
            for path in profiles:
                stage='profile_'+path.stem
                if any(record.get('stage')==stage for record in previous):continue
                profile=json.loads(path.read_text())
                values={key:value for key,value in profile.items() if isinstance(value,(int,float))}
                if profile.get('registered_kernel_count'):
                    values['registered_kernel_flops']=profile['registered_kernel_count']['registered_kernel_flops']
                events.emit(stage,'completed',values,input_shape=profile['input_shape'],
                    flops_coverage=profile['flops_coverage'],quality_result=algorithm!='jcm')
        write_json(run/'status.json',{'status':'completed','scope':summary['scope']})
        export_manifest(run)
    print({'appended_rows':collect(ROOT/'Results/experiments',ROOT/'Results/metrics.csv')})


if __name__=='__main__':main()
