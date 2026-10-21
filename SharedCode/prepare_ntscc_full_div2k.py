"""Build a separate full-DIV2K training notebook; never submit or run training."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Notebooks/project_notebooks/NTSCC_plus_DIV2K_Kodak_150ep.ipynb'
DEST = ROOT / 'Notebooks/project_notebooks/Training/NTSCC_plus_Full_DIV2K_150ep'
PIN = '765c6f6c11cf3bbe249a26b624dbaf47a4f0676d'
nb = json.loads(SOURCE.read_text(encoding='utf-8'))
code = ''.join(nb['cells'][2]['source']).split('summaries=[]')[0]
code = code.replace('IMAGES_PER_EPOCH = 128', 'IMAGES_PER_EPOCH = 800')
code = code.replace("SRC = Path('/kaggle/working/NTSCC_plus')", "SRC = Path('/tmp/NTSCC_plus')")
code = code.replace("OUT = Path('/kaggle/working/jscc_results/NTSCC_plus')", "OUT = Path('/kaggle/working/ntscc_plus_full_div2k_results')")
code = code.replace('EPOCHS = 150', '''EPOCHS = 150
PROTOCOL = 'ntscc_plus_full_div2k_800x150_v1'
SESSION_DEADLINE = time.time() + 9.5 * 3600
SOURCE_PIN = '765c6f6c11cf3bbe249a26b624dbaf47a4f0676d'
''')
code = code.replace("REV=subprocess.check_output", "subprocess.run(['git','-C',str(SRC),'checkout','--detach',SOURCE_PIN],check=True)\nsubprocess.run(['git','-C',str(SRC),'diff','--exit-code',SOURCE_PIN],check=True)\nREV=subprocess.check_output")
start = code.index('# NATTEN >=0.21')
end = code.index('from net.ntscc import', start)
code = code[:start] + '''# Compatibility adapter only: the released attention dropout default is zero.
# Bind before importing the official source; no author source file is edited.
import natten
_ReleasedNA2D = natten.NeighborhoodAttention2D
class NeighborhoodAttention2D(_ReleasedNA2D):
    def __init__(self, *args, attn_drop=0.0, **kwargs):
        if attn_drop != 0.0:
            raise ValueError('Nonzero attention dropout is unsupported by this adapter')
        super().__init__(*args, **kwargs)
natten.NeighborhoodAttention2D = NeighborhoodAttention2D
assert REV == SOURCE_PIN
print('UPSTREAM SOURCE INTEGRITY PASS', REV, flush=True)
''' + code[end:]
code = code.replace("return torch.from_numpy(a.transpose(2,0,1).copy())", "return torch.from_numpy(a.transpose(2,0,1).copy()), idx")
code = code.replace('class RandomDIV2KSubsetSampler', 'class FullDIV2KEpochSampler')
code = code.replace('Sample a fresh subset without replacement from all 800 DIV2K images each epoch.', 'Visit every DIV2K image exactly once, in a shuffled order, every epoch.')
code = code.replace('assert 0 < samples_per_epoch <= len(dataset)', 'assert samples_per_epoch == len(dataset) == 800')
code = code.replace("def make_model(channel_name):", "AUTHOR_CHANNEL_FORWARD = Channel.channel_forward\n\ndef make_model(channel_name):\n    Channel.channel_forward = AUTHOR_CHANNEL_FORWARD")
code = code.replace("uses=float(out['k'].sum().item())", "uses=float(torch.as_tensor(out['k']).sum().item())")
code = code.replace("'model':'NTSCC_plus'", "'model':'NTSCC_plus_full_DIV2K'")

runtime = r'''
import shutil

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)

def write_rows(path, rows):
    if not rows: return
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)

# Only an explicitly attached output from THIS full-data protocol may be resumed.
previous = [p for p in Path('/kaggle/input').rglob('ntscc_plus_full_div2k_results')
            if p.is_dir() and (p/'protocol.json').is_file()]
assert len(previous) <= 1, 'Attach at most one previous full-data continuation output'
if previous:
    prior = json.loads((previous[0]/'protocol.json').read_text())
    assert prior['protocol'] == PROTOCOL and prior['source_commit'] == REV
    shutil.copytree(previous[0], OUT, dirs_exist_ok=True)
    print('RESTORED FULL-DATA OUTPUT', previous[0], flush=True)
protocol = {'protocol':PROTOCOL, 'source_commit':REV, 'epochs_per_channel':EPOCHS,
            'images_per_epoch':800, 'batch_size':BATCH_SIZE, 'crop':CROP,
            'snr_db':SNR_DB, 'lambda':LAMBDA, 'eta':ETA, 'channels':['AWGN','Rayleigh'],
            'amp_fp16':AMP_ENABLED, 'seed':SEED, 'session_hours':9.5,
            'rayleigh_extension':'quasi-static complex fading, perfect receiver CSI',
            'training_flops_precision':'FP32 profile; actual training uses FP16 autocast',
            'inference_input':'native Kodak image geometry',
            'rate_matching':'existing eta retained; not a matched-CBR comparison'}
atomic_json(OUT/'protocol.json', protocol)
assert len(TRAIN_PATHS) == IMAGES_PER_EPOCH == 800
assert IMAGES_PER_EPOCH % BATCH_SIZE == 0

summaries = []
session_stop = False
for channel_name in ('AWGN', 'Rayleigh'):
    run = OUT/channel_name; run.mkdir(exist_ok=True)
    if (run/'completed.json').exists():
        done = json.loads((run/'completed.json').read_text())
        assert done['protocol'] == PROTOCOL and done['epochs'] == 150
        summaries.append(done['summary'])
        print('SKIP COMPLETED CHANNEL', channel_name, flush=True)
        continue
    if time.time() + 900 >= SESSION_DEADLINE:
        session_stop=True; break

    # Stable parameter ordering is necessary to reload optimizer states reliably.
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED); random.seed(SEED)
    model=make_model(channel_name)
    model=torch.nn.DataParallel(model) if torch.cuda.device_count()>1 else model
    base=model.module if isinstance(model,torch.nn.DataParallel) else model
    g_params=[p for n,p in base.named_parameters() if not n.endswith('.quantiles')]
    aux_params=[p for n,p in base.named_parameters() if n.endswith('.quantiles')]
    optimizer=torch.optim.Adam(g_params,lr=1e-4)
    aux_optimizer=torch.optim.Adam(aux_params,lr=1e-3)
    del g_params, aux_params
    scaler=torch.amp.GradScaler('cuda',enabled=AMP_ENABLED)
    start_epoch=0; global_step=0; rows=[]
    checkpoint=run/'latest.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['protocol'] == PROTOCOL and saved['source_commit'] == REV
        assert saved['images_per_epoch'] == 800 and saved['batch_size'] == BATCH_SIZE
        base.load_state_dict(saved['state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        aux_optimizer.load_state_dict(saved['aux_optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch']; global_step=saved['global_step']; rows=saved['history']
        assert len(rows) == start_epoch and global_step == start_epoch*100
        assert all(r['images_in_epoch']==r['unique_images_in_epoch']==800 for r in rows)
        flops=saved['flops']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        del saved
        print('RESUME VERIFIED',channel_name,'completed_epochs',start_epoch,
              'next_epoch',start_epoch+1,'global_step',global_step,flush=True)
    else:
        flops=count_flops(base,channel_name)
        # Profiling must not initialize or alter the fresh training model's state.
        del model,base,optimizer,aux_optimizer; torch.cuda.empty_cache()
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED); random.seed(SEED)
        model=make_model(channel_name)
        model=torch.nn.DataParallel(model) if torch.cuda.device_count()>1 else model
        base=model.module if isinstance(model,torch.nn.DataParallel) else model
        optimizer=torch.optim.Adam([p for n,p in base.named_parameters() if not n.endswith('.quantiles')],lr=1e-4)
        aux_optimizer=torch.optim.Adam([p for n,p in base.named_parameters() if n.endswith('.quantiles')],lr=1e-3)
        print('FRESH FULL-DATA TRAINING',channel_name,'NO SUBSET CHECKPOINT LOADED',flush=True)
    dataset=RandomCropDIV2K()
    sampler=FullDIV2KEpochSampler(dataset,IMAGES_PER_EPOCH,SEED)
    worker_generator=torch.Generator()
    loader=DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler,num_workers=4,
                      pin_memory=True,drop_last=False,generator=worker_generator)
    assert len(loader)==100, len(loader)
    for epoch in range(start_epoch+1,EPOCHS+1):
        sampler.epoch=epoch-1
        worker_generator.manual_seed(SEED+epoch)
        model.train(); t0=time.time(); sums=np.zeros(3); samples=0; seen=[]
        successful_main_updates=0
        for batch_index,(x,indices) in enumerate(loader):
            seen.extend(indices.tolist()); batch_count=x.shape[0]
            x=x.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True); aux_optimizer.zero_grad(set_to_none=True)
            # Quantiles belong only to the entropy auxiliary optimizer. Keep
            # scaled RD gradients out of this unscaled FP32 optimizer path.
            for group in aux_optimizer.param_groups:
                for parameter in group['params']: parameter.requires_grad_(False)
            with torch.amp.autocast('cuda',dtype=torch.float16,enabled=AMP_ENABLED):
                out=model(x); loss,mse,bpp=criterion(out,x)
            assert torch.isfinite(loss).item(), 'Non-finite main training loss'
            scaler.scale(loss.float()).backward(); scaler.unscale_(optimizer)
            main_parameters=[p for group in optimizer.param_groups for p in group['params']]
            grad_norm=torch.nn.utils.clip_grad_norm_(main_parameters,1.0)
            scale_before=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            successful_main_updates += int(scaler.get_scale()>=scale_before)
            for group in aux_optimizer.param_groups:
                for parameter in group['params']: parameter.requires_grad_(True)
            aux_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=False): aux=base.ntc.aux_loss()
            assert torch.isfinite(aux).item(), 'Non-finite entropy auxiliary loss'
            aux.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all().item()
                       for group in aux_optimizer.param_groups for p in group['params']), 'Non-finite auxiliary gradient'
            aux_optimizer.step()
            sums += np.asarray([loss.detach().item(),mse.detach().item(),bpp.detach().item()])*batch_count
            samples+=batch_count; global_step+=1
            if batch_index%25==0:
                print('TRAIN PROGRESS',json.dumps({'channel':channel_name,'epoch':epoch,
                    'batch':batch_index+1,'batches':100,'images_seen':samples,'global_step':global_step}),flush=True)
        assert samples==len(seen)==len(set(seen))==800
        assert set(seen)==set(range(800)), 'Every DIV2K index must occur exactly once'
        row={'model':'NTSCC_plus_full_DIV2K','channel':channel_name,'epoch':epoch,
             'epochs_target':150,'snr_db':SNR_DB,'train_loss':sums[0]/samples,
             'train_psnr_db':-10*math.log10(max(sums[1]/samples,1e-12)),
             'estimated_bpp':sums[2]/samples,'epoch_seconds':time.time()-t0,
             'train_flops_per_sample_fp32_profile':flops[0],
             'inference_flops_per_image':flops[1],'batch_size':BATCH_SIZE,'crop':CROP,
             'images_in_epoch':samples,'unique_images_in_epoch':len(set(seen)),
             'batches_in_epoch':len(loader),'successful_main_optimizer_updates':successful_main_updates,
             'global_step':global_step,'amp_fp16':AMP_ENABLED,'source_commit':REV}
        rows.append(row)
        saved={'protocol':PROTOCOL,'epoch':epoch,'global_step':global_step,'source_commit':REV,
               'images_per_epoch':800,'batch_size':BATCH_SIZE,'state_dict':base.state_dict(),
               'optimizer':optimizer.state_dict(),'aux_optimizer':aux_optimizer.state_dict(),
               'scaler':scaler.state_dict(),'history':rows,'flops':flops,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        atomic_torch_save(saved,checkpoint)
        del saved
        # Drop the final batch's graph references before saving/evaluating or
        # constructing the next channel's model on the same GPU.
        del out, loss, mse, bpp, aux, x, grad_norm, main_parameters, parameter, group
        write_rows(run/'epoch_metrics.csv',rows)
        atomic_json(run/'progress.json',dict(protocol=PROTOCOL,epoch=epoch,global_step=global_step,
                    images_in_epoch=800,unique_images_in_epoch=800,checkpoint_bytes=checkpoint.stat().st_size))
        print('EPOCH SAVED',json.dumps(row),flush=True)
        if time.time()+max(900,2*row['epoch_seconds'])>=SESSION_DEADLINE and epoch<150:
            session_stop=True; break
    if rows[-1]['epoch']<150:
        del model,base,optimizer,aux_optimizer; torch.cuda.empty_cache(); break
    if time.time()+900>=SESSION_DEADLINE:
        session_stop=True
        del model,base,optimizer,aux_optimizer; torch.cuda.empty_cache(); break
    torch.manual_seed(SEED+1000); torch.cuda.manual_seed_all(SEED+1000)
    kodak=evaluate(base,channel_name)
    assert len(kodak)==24
    summary={'model':'NTSCC_plus_full_DIV2K','channel':channel_name,'epochs':150,
        'images_per_epoch':800,'image_exposures':120000,'snr_db':SNR_DB,
        'kodak_psnr_db':float(np.mean([r['psnr_db'] for r in kodak])),
        'kodak_ssim':float(np.mean([r['ssim'] for r in kodak])),
        'kodak_msssim':float(np.mean([r['msssim'] for r in kodak])),
        'channel_uses_per_image_mean':float(np.mean([r['complex_channel_uses'] for r in kodak])),
        'cbr':float(np.mean([r['cbr'] for r in kodak])),
        'train_flops_per_sample_fp32_profile':flops[0],
        'inference_flops_per_image':flops[1],'source_commit':REV}
    atomic_torch_save({'protocol':PROTOCOL,'epoch':150,'source_commit':REV,
                       'state_dict':base.state_dict()},run/'model.pth')
    atomic_json(run/'completed.json',{'protocol':PROTOCOL,'epochs':150,'summary':summary})
    summaries.append(summary); write_rows(OUT/'summary.csv',summaries)
    print('CHANNEL COMPLETE',json.dumps(summary),flush=True)
    del model,base,optimizer,aux_optimizer; torch.cuda.empty_cache()
write_rows(OUT/'summary.csv',summaries)
complete=len(summaries)==2
atomic_json(OUT/'run_manifest.json',{'protocol':protocol,'training_and_evaluation_complete':complete,
            'session_stopped_normally':session_stop,'completed_channels':[s['channel'] for s in summaries],
            'channel_progress':{c:json.loads((OUT/c/'progress.json').read_text())
                for c in ('AWGN','Rayleigh') if (OUT/c/'progress.json').exists()}})
print('FULL_DATA_TRAINING_COMPLETE' if complete else 'CHUNK SAVED; ATTACH OUTPUT TO NEXT RUN',flush=True)
'''
nb['cells'][0]['source'] = ['''# NTSCC_plus: 150 full DIV2K epochs per channel

READY TO RUN; no training or results are claimed for this corrected experiment yet.

Each epoch visits all 800 DIV2K training images exactly once in 100 batches of 8,
with random 256x256 crops. AWGN and Rayleigh each receive 150 full epochs (120,000
image exposures per channel). The author model is pinned to the previous run's
commit, with no source-file edits. A zero-dropout NATTEN API adapter is applied at
import time. Rayleigh remains the explicitly declared perfect-CSI extension.

This is a fresh experiment; the old 128-image-per-epoch weights are never loaded.
Native-resolution Kodak evaluation reports PSNR, SSIM, MS-SSIM, complex channel
uses and CBR. FLOPs are profiler estimates; fused NATTEN kernels may be omitted.
The FP32 training profile is labelled separately from FP16 actual training.
CBR/eta and preprocessing are unchanged from the previous NTSCC_plus experiment;
full training exposure does not by itself establish a matched-rate comparison.

Normal session stops occur at epoch boundaries before a 9.5-hour budget expires.
Attach the preceding successful full-data output to continue; both optimizers,
GradScaler and random states are saved every epoch. No prior subset output is used.
''']
nb['cells'][2]['source']=(code+runtime).splitlines(keepends=True)
nb['cells'][3]['source']=['''## Outputs and continuation

`ntscc_plus_full_div2k_results/` contains protocol.json, run_manifest.json,
AWGN/Rayleigh epoch_metrics.csv, progress.json and atomic latest.pt checkpoints.
After each channel reaches 150 full epochs, all 24 Kodak images are evaluated,
and model.pth, completed.json and the aggregate summary are saved.

If a chunk stops normally, save its output and attach that successful notebook
output to the next execution. Keep the source and protocol unchanged. A channel
is complete only after training and evaluation artifacts have been written.
''']
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['outputs']=[]; cell['execution_count']=None
        ast.parse(''.join(cell['source']))
DEST.mkdir(parents=True,exist_ok=True)
name='ntscc_plus_full_div2k_150ep.ipynb'
(DEST/name).write_text(json.dumps(nb,indent=1),encoding='utf-8')
metadata={'id':'hmmkk20/ntscc-plus-full-div2k-150ep-20261020',
          'title':'NTSCC Plus Full DIV2K 150ep 20261020','code_file':name,
          'language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
          'machine_shape':'NvidiaTeslaT4','dataset_sources':['sherylmehta/kodak-dataset'],
          'kernel_sources':['jagan028/div2k-dataset-generation-for-isr'],
          'competition_sources':[],'model_sources':[]}
(DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Prepared, not submitted:',DEST/name)
