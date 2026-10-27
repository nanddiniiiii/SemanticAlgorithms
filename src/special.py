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
        if stage=='evaluate':
            checkpoint=options.get('native_trained_checkpoint')
            if not checkpoint:
                candidates=sorted((run/'native').rglob('*.ckpt'),key=lambda p:p.stat().st_mtime_ns)
                checkpoint=str(candidates[-1]) if candidates else None
            if not checkpoint:raise FileNotFoundError('No WZ native checkpoint to evaluate')
            return [sys.executable,'eval.py','trainer.devices=[0]','experiment=wz/wz_eval',
                    'ckpt_path='+str(Path(checkpoint).resolve()),
                    'paths.data_dir='+str(Path(options['paired_data_root']).resolve()),
                    'paths.output_dir='+str((run/'native/evaluation').resolve()),'logger=null'],root/'src'
        command=[sys.executable,'train.py','trainer.devices=[0,1]','+trainer.strategy=ddp',
            'seed='+str(int(options.get('seed',22))),
            'test=False',
            'model=base_deepjscc','model.N=256','model.snr=undefined',
            'model.csi.min=10.0','model.csi.max=10.0','model/loss=mse_lpips','model.loss.lpips_weight=0.001',
            'data.name='+options.get('paired_dataset','KittiStereo'),
            'model._target_=src.models.wz.deepjscc_wz_baseline2.DeepJSCCWZBaseline2','model.bw_factor=16',
            'paths.data_dir='+str(Path(options['paired_data_root']).resolve()),
            'paths.output_dir='+str((run/'native').resolve()),'+trainer.max_time=00:04:00:00',
            'logger=null']
        if pilot:command+=['+trainer.fast_dev_run=2']
        return command,root/'src'
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
