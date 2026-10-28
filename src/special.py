"""Native-task subprocesses and prerequisites that are not generic RGB MSE."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.sources import source_path
from .runtime import write_json


def prepare(name,options,run):
    options=dict(options)
    if name=='deepjscc_wz':
        root=Path(options.get('paired_data_root',''))
        if not options.get('paired_data_root') or not root.is_dir():
            raise FileNotFoundError('Set paired_data_root to native KITTI/Cityscapes views and split files')
        if options.get('paired_dataset','KittiStereo') not in {'KittiStereo','Cityscape'}:
            raise ValueError('Unknown native paired dataset')
        counts=validate_pairs(root,options.get('paired_dataset','KittiStereo'))
        from .dataset import digest
        write_json(Path(run)/'native_data_identity.json',{'dataset':options.get('paired_dataset','KittiStereo'),
            'counts':counts,'split_manifest_sha256':{split:digest(root/('KITTI_stereo_'+split+'.txt')) for split in counts} if options.get('paired_dataset','KittiStereo')=='KittiStereo' else {},
            'validation_scope':'native file layout, split path disjointness; decoded pixel identity not checked',
            'common_rgb_manifest_applies':False,'side_information':'clean correlated right image, receiver only'})
        options['native_task_note']='Paired-view reconstruction; no Kodak common-RGB substitution; native DDP, not DataParallel'
    elif name=='gbsed':
        for key in ('training_extraction_config','training_learning_config',
                    'evaluation_extraction_config','evaluation_learning_config','receiver_weights'):
            if not options.get(key) or not Path(options[key]).is_file():
                raise FileNotFoundError('GBSED native graph lane requires '+key)
        options=prepare_graph_configs(options,run)
        options['native_task_note']='Scene-graph compression and risk classification; source encoder is not a learned RGB decoder'
    elif name=='sqgan':
        if not options.get('checkpoint') or not Path(options['checkpoint']).is_dir():
            raise FileNotFoundError('SQ-GAN needs the ten author components directory; see Docs/PRETRAINED_DOWNLOADS.md')
        required=['encoder','decoder','masker','demasker','quantize','encoder_semantic','decoder_semantic',
                  'masker_semantic','demasker_semantic','quantize_semantic']
        for component in required:
            if not (Path(options['checkpoint'])/(component+'.ckpt')).is_file():
                raise FileNotFoundError('Missing SQ-GAN component: '+component)
        from .assets import verified_sqgan_components
        options['checkpoint_matches_author_ledger']=verified_sqgan_components(options['checkpoint'])
        if not options['checkpoint_matches_author_ledger']:
            raise ValueError('SQ-GAN checkpoint components do not match the author download ledger')
        options['adaptation']='Frozen Cityscapes pseudo-labels; pretrained all-mode adversarial fine-tuning; ideal source training'
        options['teacher_model']=options.get('teacher_model','nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        options['teacher_revision']=options.get('teacher_revision','21b3847fae21ddee674abd31129307b6a1235bd9')
    elif name=='shared_semantic_codebook':
        if not options.get('classifier') or not Path(options['classifier']).is_file():
            raise FileNotFoundError('Provide the frozen native STL10 classifier for approved pseudo-label construction')
        from .shared import construct
        options['codebook']=str(construct(options,run))
        options.setdefault('evaluation_tile',[256,256])
    return options


def native_command(name,stage,options,run):
    root=source_path(name);pilot=stage=='pilot'
    if name=='deepjscc_wz':
        worker=Path(__file__).resolve().parents[1]/'scripts/wz_native.py'
        config=run/'wz_worker_config.json';write_json(config,options)
        return [sys.executable,str(worker),'--source',str(root),'--config',str(config),
                '--run',str(run),'--stage',stage],root
    if name=='gbsed':
        if stage=='train':
            return [sys.executable,'rs2vec_training.py','--train_config_filename',options['training_extraction_config'],
                    '--learning_config_filename',options['training_learning_config']],root/'learning'
        if stage=='evaluate':
            import yaml
            checkpoint=run/'native/checkpoints/risk.pt'
            if not checkpoint.is_file():raise FileNotFoundError('No newly trained native GBSED task checkpoint')
            path=Path(options['evaluation_learning_config'])
            config=yaml.safe_load(path.read_text())
            config['model_configuration']['model_load_path']=str(checkpoint.resolve())
            path.write_text(yaml.safe_dump(config,sort_keys=False))
        return [sys.executable,'pipeline.py','--extraction_filename',options['evaluation_extraction_config'],
            '--learning_filename',options['evaluation_learning_config'],'--com_model_endpoint',options['receiver_weights'],
            '--output_file',str(run/'graph_outputs.csv'),'--sem_fid_file',str(run/'semantic_fidelity.csv'),
            '--time_file',str(run/'transfer_times.csv')],root/'pipeline'
    raise ValueError(name)


def prepare_graph_configs(options,run):
    """Distinct native training/evaluation configs; every writable output isolated.

    The upstream trainer and pipeline use different schemas and objectives.
    Reusing one config for both could train on evaluation graphs or overwrite
    the author's shipped risk checkpoint.
    """
    import yaml
    from .dataset import digest
    options=dict(options);run=Path(run).resolve()
    folder=run/'native_configs';folder.mkdir(parents=True,exist_ok=True)
    data=run/'native/data';data.mkdir(parents=True,exist_ok=True)
    checkpoints=run/'native/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    records={}
    for key in ('training_extraction_config','training_learning_config',
                'evaluation_extraction_config','evaluation_learning_config'):
        source=Path(options[key]).resolve()
        config=yaml.safe_load(source.read_text())
        location=config.setdefault('location_data',{})
        for input_key in ('input_path','transfer_path'):
            value=location.get(input_key)
            if value is not None and 'extraction' in key:
                path=Path(value)
                if not path.is_absolute() or not path.exists():
                    raise ValueError('GBSED configs require absolute existing '+key+'.'+input_key)
        if key=='training_extraction_config':location['data_save_path']=str(data/'1043_train.pkl')
        elif key=='evaluation_extraction_config':location['data_save_path']=str(data/'1043_transfer.pkl')
        else:
            model=config['model_configuration']
            model['model_save_path']=str(checkpoints/'risk.pt')
            if key=='training_learning_config':
                location['input_path']=str(data/'1043_train.pkl')
                location['transfer_path']=str(data/'1043_valid.pkl')
            else:
                location['transfer_path']=str(data/'1043_transfer.pkl')
                supplied=Path(model.get('model_load_path',''))
                if not supplied.is_absolute() or not supplied.is_file():
                    raise ValueError('Evaluation config requires an absolute native risk classifier checkpoint for its pilot')
        target=folder/(key+'.yaml');target.write_text(yaml.safe_dump(config,sort_keys=False))
        records[key]={'original':str(source),'sha256':digest(source),'runtime':str(target)}
        options[key]=str(target)
    write_json(run/'native_config_provenance.json',records)
    return options


def execute(name,stage,options,run):
    if name=='deepjscc_wz':
        if stage=='train' and not (Path(run)/'pilot_passed.json').is_file():
            raise RuntimeError('Native-task training requires a successful real-data pilot')
        command,cwd=native_command(name,stage,options,run)
        from .runtime import run_process_group
        write_json(run/(stage+'_command.json'),{'command':command,'cwd':str(cwd),'paired_task':True})
        run_process_group(command,cwd, min(float(options.get('max_seconds',14400)),180) +600 if stage=='pilot' else float(options.get('max_seconds',14400))+600)
        return
    if name=='sqgan':
        from .sqgan_train import execute as sqgan_execute
        return sqgan_execute(stage,options,run)
    if stage=='train' and not (run/'pilot_passed.json').is_file():
        raise RuntimeError('Native-task training requires a successful real-data pilot')
    command,cwd=native_command(name,stage,options,run)
    (run/'native').mkdir(exist_ok=True)
    if name=='deepjscc_wz' and stage=='evaluate':(run/'native/evaluation').mkdir(exist_ok=True)
    write_json(run/(stage+'_command.json'),{'command':command,'cwd':str(cwd),
        'source':str(source_path(name)),'task':options.get('native_task_note'),
        'snr_adaptation':'WZ fixed AWGN10 instead of native [-5,5]' if name=='deepjscc_wz' else None,
        'data_parallel':'WZ uses native Lightning DDP; GBSED preserves mixed-framework native pipeline'})
    env=dict(os.environ, PYTHONPATH=str(source_path(name)),WANDB_MODE='offline',PYTHONUNBUFFERED='1')
    # Inherit a log file directly so output continues even when a native process
    # does not flush a Python-level stream. Notebook displays the logfile path.
    log=run/'logs'/(stage+'_native.log')
    print('Native command:',command,'log:',log,flush=True)
    budget=180 if stage=='pilot' else float(options.get('max_seconds',14400))
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        started=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-started>budget:
                process.terminate()
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise TimeoutError('Native process budget reached; inspect saved checkpoints; completion not certified')
            print(f'Native {name}/{stage} running, elapsed {time.monotonic()-started:.0f}s; {log}',flush=True)
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'Native {name} exited {process.returncode}; inspect {log}')
    if stage=='pilot':
        write_json(run/'pilot_passed.json',{'scope':'native data execution; consult native log',
            'rgb_metrics_applicable':name!='gbsed','checkpoint_reload_verified':False})
    write_json(run/(stage+'_native_result.json'),{'returncode':process.returncode,
        'elapsed_seconds':time.monotonic()-started,'native_files':[str(p.relative_to(run)) for p in (run/'native').rglob('*') if p.is_file()],
        'metric_integration':'Native output tables; common RGB metrics not inferred'})


def validate_pairs(root,dataset):
    root=Path(root).resolve();seen=set();counts={}
    for split in ('train','val','test'):
        if dataset=='KittiStereo':
            path=root/('KITTI_stereo_'+split+'.txt')
            if not path.is_file():raise FileNotFoundError('Missing native paired split: '+str(path))
            lines=path.read_text().splitlines()
            if any(not line.strip() or line!=line.strip() for line in lines):
                raise ValueError('Native KITTI split paths must not contain blank lines or surrounding whitespace')
            if not lines or len(lines)%2:raise ValueError('KITTI paired split must contain alternating left/right paths')
            pairs=[(root/lines[i],root/lines[i+1]) for i in range(0,len(lines),2)]
        else:
            base=root/'cityscape_dataset';left=base/'leftImg8bit'/split;right=base/'rightImg8bit'/split
            pairs=[(path,right/path.relative_to(left).parent/path.name.replace('_leftImg8bit','_rightImg8bit')) for path in left.glob('*/*_leftImg8bit.png')]
        if not pairs:raise ValueError('Native paired split is empty: '+split)
        current=set()
        for left,right in pairs:
            for path in (left,right):
                path=path.resolve();path.relative_to(root)
                if not path.is_file():raise FileNotFoundError('Missing stereo image: '+str(path))
                current.add(path)
            if left.resolve()==right.resolve():raise ValueError('Receiver side image must not be the target image')
        if seen & current:raise ValueError('Paired split path leakage detected')
        seen.update(current);counts[split]=len(pairs)
    return counts
