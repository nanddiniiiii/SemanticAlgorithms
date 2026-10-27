"""Per-cell CLI for the three-team experiments. No implicit job submission."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from semantic_lossless.team_suite.runtime import stage,write_json,environment,export
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','prepare','pilot','train','evaluate','export'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    options=json.loads(args.config.read_text())
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if args.stage=='export':
        with stage(run,'export'):
            print('Finalizing stage logs before hashing artifacts',flush=True)
        export(run)
        return
    with stage(run,args.stage):
        write_json(run/'environment.json',environment(ROOT))
        if args.stage=='freeze':
            from semantic_lossless.team_suite.dataset import freeze
            freeze(options['dataset_roots'],options['manifest'],strict_counts=options.get('strict_counts',True))
            return
        name=options['algorithm']
        if args.stage=='prepare':
            from semantic_lossless.sources import fetch_source
            source='diffcom' if name=='adjscc' else name
            receipt=fetch_source(source)
            write_json(run/'source.json',receipt)
            if options.get('manifest'):
                frozen=json.loads(Path(options['manifest']).read_text())
                write_json(run/'dataset_identity.json',{'identity':frozen['identity'],'counts':frozen['counts'],
                    'protocol':frozen['protocol'],'test_used_for_training':False,
                    'common_rgb_applies':name not in {'gbsed','deepjscc_wz'}})
            if name in {'adjscc','swinjscc','jcm','sqgan'} and not options.get('checkpoint'):
                from semantic_lossless.team_suite.assets import obtain
                options['checkpoint']=str(obtain(name,ROOT/'Models/pretrained_downloads/team_suite',
                    [ROOT/'Models/pretrained_downloads',Path('/kaggle/input')]))
            if name in {'gbsed','deepjscc_wz','sqgan','shared_semantic_codebook'}:
                from semantic_lossless.team_suite.special import prepare
                options=prepare(name,options,run)
            write_json(run/'resolved_config.json',options)
            print('Initialization recorded. A successful training pilot is still required.',flush=True)
            return
        prepared=run/'resolved_config.json'
        if not prepared.is_file():raise FileNotFoundError('Run initialization first: '+str(prepared))
        options=json.loads(prepared.read_text())
        if name in {'gbsed','deepjscc_wz','sqgan'}:
            from semantic_lossless.team_suite.special import execute
            execute(name,args.stage,options,run)
            return
        from semantic_lossless.team_suite.models import build
        model=build(name,options,options.get('device','cpu'))
        if args.stage=='pilot':
            from semantic_lossless.team_suite.train import train
            from semantic_lossless.team_suite.evaluate import evaluate
            pilot=run/'pilot';pilot.mkdir(exist_ok=False)
            write_json(pilot/'model.json',model.metadata)
            train(model,options['manifest'],pilot,{**options,'max_seconds':min(options.get('max_seconds',14400),120)},pilot=True)
            evaluate(model,options['manifest'],pilot,options,pilot=True)
            write_json(run/'pilot_passed.json',{'scope':'two optimizer steps per stage, strict state reload, one development image',
                       'device':options.get('device','cpu'),'source':json.loads((run/'source.json').read_text())['commit']})
        elif args.stage=='train':
            if not (run/'pilot_passed.json').is_file():raise RuntimeError('Training blocked: pilot has not passed')
            from semantic_lossless.team_suite.train import train
            write_json(run/'model.json',model.metadata)
            if (run/'checkpoints'/'last.pt').exists():
                raise FileExistsError('Existing training checkpoint; create a new run rather than silently overwriting')
            train(model,options['manifest'],run,options)
        elif args.stage=='evaluate':
            import torch
            from semantic_lossless.team_suite.evaluate import evaluate
            path=run/'checkpoints'/'last.pt'
            if not path.is_file():raise FileNotFoundError('No trained checkpoint: '+str(path))
            state=torch.load(path,map_location=options.get('device','cpu'),weights_only=True)
            from semantic_lossless.team_suite.models import load_training_state
            reload_info=load_training_state(model,state['state_dict'])
            reload_info.update({'checkpoint':str(path),'selection_rule':'last checkpoint at training stop',
                                'development_best_checkpoint_used':False})
            write_json(run/'evaluation_reload.json',reload_info)
            if name=='isec':model.stage='denoiser'
            evaluate(model,options['manifest'],run,options)
        export(run)


if __name__=='__main__':main()
