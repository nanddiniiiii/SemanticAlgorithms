"""Validate and combine independent stage artifacts, never average checkpoints."""
from pathlib import Path
import hashlib
import json
import shutil

PIN='0e5cdd7c0240b89749dad684645c53f783ac24cc'
CHANNELS=('awgn','rayleigh')
MODES=('ssm','img','all')
STAGES=tuple((c,m) for c in CHANNELS for m in MODES)
PARTS={'ssm':('encoder_semantic','decoder_semantic','masker_semantic','demasker_semantic','quantize_semantic'),
       'img':('encoder','decoder','masker','demasker','quantize')}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def select_stages(records):
    """Strictly pick greatest progress; equal-progress conflicts require review."""
    selected={}
    for row in records:
        key=(row['channel'],row['mode'])
        assert key in STAGES
        old=selected.get(key)
        if old is None or row['global_step']>old['global_step']:
            selected[key]=row
        elif row['global_step']==old['global_step']:
            assert row['sha256']==old['sha256'], f'Conflicting checkpoints at equal progress: {key}'
    return selected

def ready_stages(inventory,leased=()):
    leased=set(leased)
    result=[]
    for key in STAGES:
        if key in leased or inventory.get(key,{}).get('completed_epochs',0)>=150: continue
        c,m=key
        if m=='all' and not all(inventory.get((c,p),{}).get('completed_epochs',0)>=150 for p in ('ssm','img')):
            continue
        result.append(key)
    # Preserve partial-stage work before starting untouched stages.
    return sorted(result,key=lambda k:(-inventory.get(k,{}).get('completed_epochs',0),STAGES.index(k)))

def validate_checkpoint(path,channel,mode):
    import torch
    from omegaconf import OmegaConf
    raw=torch.load(path,map_location='cpu',weights_only=False)
    step=int(raw['global_step'])
    assert step>0 and step%800==0, f'Checkpoint is not at a full epoch boundary: {step}'
    epochs=step//800  # 800 images / (1 image x 2 ranks), two optimizers/batch
    assert 0<epochs<=150
    assert int(raw['epoch']) in {epochs-1,epochs}
    assert len(raw['optimizer_states'])==len(raw['lr_schedulers'])==2
    assert all(o['state'] for o in raw['optimizer_states']), 'Optimizer history is missing'
    state=raw['state_dict']
    required=PARTS['ssm']+PARTS['img'] if mode=='all' else PARTS['ssm' if mode=='ssm' else 'img']
    for prefix in required:
        assert any(k.startswith(prefix+'.') for k in state), (mode,prefix)
    # Matching source configuration is mandatory for exposure and geometry.
    return raw,{'channel':channel,'mode':mode,'saved_epoch_index':int(raw['epoch']),
                'completed_epochs':epochs,'global_step':step,'target_epochs':150,
                'optimizer_states':2,'lr_schedulers':2,'sha256':sha256(path),
                'status':'complete' if epochs==150 else 'partial'}

def merge_artifacts(input_root,output,repo):
    import torch
    from omegaconf import OmegaConf
    output,repo=Path(output),Path(repo)
    roots=sorted(p for p in Path(input_root).rglob('sqgan_div2k_channel_results')
                 if (p/'run_manifest.json').is_file())
    assert roots,'No successful SQ-GAN artifact input supplied'
    candidates=[]
    label_protocol=None
    for root in roots:
        manifest=json.loads((root/'run_manifest.json').read_text())
        assert manifest['upstream_commit']==PIN and manifest['epochs_per_stage']==150
        assert float(manifest['snr_db'])==10.0
        assert set(manifest['channels'])==set(CHANNELS)
        metadata=json.loads((root/'pseudo_label_metadata.json').read_text())
        if label_protocol is None: label_protocol=metadata
        assert metadata==label_protocol,'Pseudo-label protocol differs across workers'
        for stage in manifest['stages']:
            c,m=stage['channel'],stage['mode']
            assert (c,m) in STAGES
            path=root/stage['checkpoint_relative']
            assert path.resolve().is_relative_to(root.resolve()) and path.is_file()
            cfg=root/f'config_{c}_{m}.yaml'
            config=OmegaConf.load(cfg)
            assert config.data.params.batch_size==1
            assert list(config.data.params.train.params.size)==[256,512]
            for k in ('vqconfig','vqconfig_semantic'):
                params=config.model.params[k].params
                assert params.channel==c and float(params.snr_db)==10.0 and params.perfect_csi
            raw,info=validate_checkpoint(path,c,m)
            del raw
            info.update(source_root=str(root),source_path=str(path),config=str(cfg))
            candidates.append(info)
    chosen=select_stages(candidates)
    output.mkdir(parents=True,exist_ok=True)
    # Shared data are copied once; mutable stage files are selected individually.
    first=roots[0]
    for name in ('pseudo_labels','pseudo_label_metadata.json'):
        src=first/name
        if src.is_dir(): shutil.copytree(src,output/name,dirs_exist_ok=True)
        else: shutil.copy2(src,output/name)
    state_dir=output/'stage_state'; state_dir.mkdir(exist_ok=True)
    merge_records=[]
    for (c,m),row in chosen.items():
        folder=repo/'logs'/f'coordinated_sqgan_{c}_{m}_150ep'
        (folder/'checkpoints').mkdir(parents=True,exist_ok=True)
        (folder/'configs').mkdir(exist_ok=True)
        shutil.copy2(row['source_path'],folder/'checkpoints/last.ckpt')
        shutil.copy2(row['config'],folder/'configs/project.yaml')
        shutil.copy2(row['config'],output/f'config_{c}_{m}.yaml')
        exported=output/'upstream_logs'/folder.name
        shutil.copytree(folder,exported,dirs_exist_ok=True)
        marker={k:v for k,v in row.items() if k not in {'source_root','source_path','config'}}
        marker['checkpoint_relative']=str((exported/'checkpoints/last.ckpt').relative_to(output))
        marker['bytes']=Path(row['source_path']).stat().st_size
        (state_dir/f'{c}_{m}.json').write_text(json.dumps(marker,indent=2))
        audit=Path(row['source_root'])/'audit'/f'{c}_{m}'
        if audit.exists(): shutil.copytree(audit,output/'audit'/f'{c}_{m}',dirs_exist_ok=True)
        if row['completed_epochs']==150:
            if m in PARTS:
                raw=torch.load(row['source_path'],map_location='cpu',weights_only=False)
                parts=output/c/'components'; parts.mkdir(parents=True,exist_ok=True)
                for name in PARTS[m]:
                    prefix=name+'.'
                    tensors={k[len(prefix):]:v for k,v in raw['state_dict'].items() if k.startswith(prefix)}
                    assert tensors
                    torch.save(tensors,parts/(name+'.ckpt'))
                del raw
            else:
                (output/c).mkdir(exist_ok=True)
                (output/c/'final_checkpoint.json').write_text(json.dumps(marker,indent=2))
        merge_records.append(marker)
    report={'upstream_commit':PIN,'selection':'highest validated full-epoch global_step per stage; no weight averaging',
            'stages':merge_records,'input_roots':[str(p) for p in roots]}
    (output/'merge_report.json').write_text(json.dumps(report,indent=2))
    print('STAGE ARTIFACT MERGE VERIFIED',json.dumps(report),flush=True)
    return report
