"""Run the author's paired-view WZ model without mixing project src namespaces."""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--stage',choices=['pilot','train','evaluate'],required=True)
    args=parser.parse_args();options=json.loads(Path(args.config).read_text());source=Path(args.source).resolve();run=Path(args.run).resolve()
    sys.path.insert(0,str(source))
    import torch
    import hydra
    import lightning as pl
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
    from datetime import timedelta
    pilot=args.stage=='pilot';evaluation=args.stage=='evaluate'
    target=run/'pilot' if pilot else run;target.mkdir(parents=True,exist_ok=True)
    native=target/'native';native.mkdir(exist_ok=True)
    snr=float(options.get('test_snr_db' if evaluation else 'train_snr_db',options.get('snr_db',10)))
    if int(options.get('batch_size',2))<1 or int(options.get('epochs',150))<1 or not 0<float(options.get('max_seconds',14400))<=14400:
        raise ValueError('Invalid WZ batch/epochs/training budget')
    if evaluation and (target/'metrics/summary.json').exists():raise FileExistsError('Preserve existing WZ evaluation; use a new run')
    overrides=['logger=null','paths.root_dir='+str(source),'paths.work_dir='+str(native),'paths.log_dir='+str(native/'logs'),'data.name='+options.get('paired_dataset','KittiStereo'),
        'paths.data_dir='+str(Path(options['paired_data_root']).resolve()),'paths.output_dir='+str(native),
        'model=base_deepjscc','model._target_=src.models.wz.deepjscc_wz_baseline2.DeepJSCCWZBaseline2',
        'model.N=256','model.bw_factor=16','model.snr='+str(snr),'model.csi.min='+str(snr),'model.csi.max='+str(snr),
        'model/loss=mse_lpips','model.loss.lpips_weight=0.001',
        'data.batch_size='+str(int(options.get('batch_size',2))),'data.num_workers='+str(int(options.get('workers',2)))]
    with initialize_config_dir(config_dir=str(source/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=overrides)
    pl.seed_everything(int(options.get('seed',22)),workers=True)
    data=hydra.utils.instantiate(cfg.data);data.setup()
    model=hydra.utils.instantiate(cfg.model,input_dims=data.input_dims,output_dir=str(native))
    model_dir=target/'checkpoints';model_dir.mkdir(exist_ok=True)
    save(target/'native_recipe.json',{'model':OmegaConf.to_container(cfg.model,resolve=True),
        'input_dims':list(data.input_dims),'paired_dataset':options.get('paired_dataset','KittiStereo'),
        'reconstruction':'left view; clean right view supplied only at receiver','snr_db':snr})
    use_cuda=options.get('device','cpu').startswith('cuda')
    devices=2 if use_cuda and options.get('data_parallel',True) and not evaluation else 1
    class Evidence(pl.Callback):
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.is_global_zero:
                loss=outputs['loss'] if isinstance(outputs,dict) else outputs
                if int(trainer.global_step)%25==0 or int(trainer.global_step)<=2:print('WZ step',trainer.global_step,'loss',float(loss.detach()),flush=True)
                with (target/'training.jsonl').open('a') as stream:stream.write(json.dumps({'step':trainer.global_step,'epoch':trainer.current_epoch,'loss':float(loss.detach())})+'\n')
        def on_test_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
            folder=target/'metrics';folder.mkdir(exist_ok=True)
            preds=outputs['preds'].clamp(0,1);truth=outputs['targets']
            from torchmetrics.functional.image import structural_similarity_index_measure
            path=folder/'per_image.csv'
            fields=['algorithm','sample','trial','height','width','psnr_db','mse','mae','ssim','pixel_exact_fraction','complex_channel_uses','cbr_complex_per_rgb_sample','channel','snr_db','receiver_side_information']
            exists=path.exists()
            with path.open('a',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields)
                if not exists:writer.writeheader()
                for index,(prediction,reference) in enumerate(zip(preds,truth)):
                    mse=float((prediction-reference).square().mean());h,w=reference.shape[-2:]
                    writer.writerow({'algorithm':'deepjscc_wz','sample':f'paired-test-{batch_idx}-{index}','trial':0,
                        'height':h,'width':w,'psnr_db':-10*math.log10(max(mse,1e-20)),'mse':mse,
                        'mae':float((prediction-reference).abs().mean()),
                        'ssim':float(structural_similarity_index_measure(prediction[None],reference[None],data_range=1.0)),
                        'pixel_exact_fraction':float((prediction.mul(255).round()==reference.mul(255).round()).all(dim=0).float().mean()),
                        'complex_channel_uses':3*h*w/16,'cbr_complex_per_rgb_sample':1/16,'channel':'native_AWGN_with_decoder_side_information',
                        'snr_db':snr,'receiver_side_information':'clean correlated right image; acquisition cost excluded'})
    checkpoints=ModelCheckpoint(dirpath=str(model_dir),filename='epoch-{epoch:03d}',save_last=True,
        monitor='val/psnr',mode='max',auto_insert_metric_name=False,save_on_train_epoch_end=False)
    callbacks=[Evidence(),checkpoints,EarlyStopping(monitor='val/psnr',patience=10,mode='max',min_delta=.001)]
    kwargs=dict(accelerator='gpu' if use_cuda else 'cpu',devices=devices,logger=False,
        default_root_dir=str(native),callbacks=callbacks,enable_progress_bar=True,max_epochs=int(options.get('epochs',150)),
        min_epochs=1,max_time=timedelta(seconds=min(float(options.get('max_seconds',14400)),14400)),
        strategy='ddp_find_unused_parameters_false' if devices>1 else 'auto',sync_batchnorm=devices>1)
    if pilot:kwargs.update(max_steps=2,limit_train_batches=2,limit_val_batches=1,num_sanity_val_steps=0)
    trainer=pl.Trainer(**kwargs);started=time.monotonic()
    checkpoint=model_dir/'last.ckpt'
    if evaluation:
        checkpoint=run/'checkpoints/last.ckpt'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False) # trusted local own Lightning checkpoint
        model.load_state_dict(state['state_dict'],strict=True)
        print('START WZ held-out native evaluation; checkpoint:',checkpoint,flush=True)
        result=trainer.test(model,datamodule=data,ckpt_path=None)
        finite=lambda values:{k:float(v) if math.isfinite(float(v)) else None for k,v in values.items()}
        save(target/'metrics/summary.json',{'algorithm':'deepjscc_wz','native_metrics':[finite(item) for item in result],
            'task':'paired-view reconstruction','selection':'last wall-clock checkpoint','strict_reload':True,
            'elapsed_seconds':time.monotonic()-started,'test_used_for_training':False})
        # Reuse the project profiler by file path, without importing its src package.
        path=Path(__file__).resolve().parents[1]/'src/profiling.py'
        spec=importlib.util.spec_from_file_location('project_wz_profiling',path);profile_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile_module)
        batch=next(iter(data.test_dataloader()));left,right=[value[:1].to(model.device) for value in batch[0]]
        class Paired(torch.nn.Module):
            def __init__(self):super().__init__();self.core=model
            def forward(self,image):return self.core(((image,right),self.core.generate_csi((image,right),stage='test')))
        wrapper=Paired().eval()
        report=profile_module.profile(wrapper,left,lambda m,x:m(x),warmup=1,repeats=3)
        report['scope']='encoder, channel, receiver side-information encoder, decoder; clean side image supplied'
        save(target/'metrics/profile.json',report)
    else:
        if checkpoint.exists():raise FileExistsError('Preserve existing WZ checkpoint; create a new run')
        print('START WZ native fit:',options.get('paired_dataset','KittiStereo'),'devices',devices,flush=True)
        trainer.fit(model,datamodule=data)
        trainer.save_checkpoint(str(checkpoint))
        if devices>1:trainer.strategy.barrier()
        if trainer.is_global_zero:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['state_dict'],strict=True)
            save(target/'training_summary.json',{'status':'pilot_completed' if pilot else 'budget_limited_training_completed',
                'steps':int(trainer.global_step),'strict_reload':True,'elapsed_seconds':time.monotonic()-started,
                'distributed_world_size':devices,'paper_convergence_reproduced':False})
            save(run/'model.json',{'algorithm':'deepjscc_wz','source_commit':'802394328838c478321413e5f72a771f86a7599d',
                'channel':'native AWGN with clean decoder-only side information','input_dims':list(data.input_dims),
                'native_recipe':'native_recipe.json','rate_scope':'transmitted left latent only; side image/CSI excluded'})
            if pilot and int(trainer.global_step)<2:raise RuntimeError('WZ pilot did not complete two steps')
            if pilot:save(run/'pilot_passed.json',{'scope':'native paired data, two updates and strict checkpoint reload','world_size':devices})

if __name__=='__main__':main()
