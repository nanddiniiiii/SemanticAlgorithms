"""Exclusive stage ownership and two-slot scheduling after a safe legacy handoff."""
import json
import os
import hashlib
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from continue_training import get_outputs
from stage_artifacts import PIN,STAGES,ready_stages
from build_stage_worker import build,TEMPLATE

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Results/SQGAN/stage_coordination'
LEGACY='hmmkk20/sqgan-div2k-150ep-continuation-02'
PROBE='hmmkk20/sqgan-speed-benchmark-20261020'
VALIDATOR='hmmkk20/sqgan-stage-merge-validation-20261020'

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    for retry in range(8):
        try:
            temp=path.with_suffix('.partial')
            temp.write_text(json.dumps(value,indent=2)); os.replace(temp,path); return
        except PermissionError:
            time.sleep(1)
    raise RuntimeError('Could not persist coordinator state')

def status(api,kernel): return api.kernels_status(kernel).to_dict()['status']

def version(api,kernel):
    owner,slug=kernel.split('/',1)
    request=ApiGetKernelRequest(); request.user_name=owner; request.kernel_slug=slug
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request).metadata.to_dict()['currentVersionNumber']

def fetch_manifest(api,kernel,folder):
    files=get_outputs(api,kernel)
    item=next(f for f in files if f.file_name.endswith('/run_manifest.json'))
    response=requests.get(item.url,timeout=(20,90)); response.raise_for_status()
    manifest=response.json()
    assert manifest['upstream_commit']==PIN and manifest['epochs_per_stage']==150
    assert set(manifest['channels'])=={'awgn','rayleigh'}
    assert float(manifest['snr_db'])==10.0
    file_names={f.file_name for f in files}
    for s in manifest['stages']:
        assert (s['channel'],s['mode']) in STAGES
        step=int(s['global_step']); assert step>0 and step%800==0 and step<=120000
        assert s['optimizer_states']==s['lr_schedulers']==2
        assert any(f.endswith('/'+s['checkpoint_relative'].replace('\\','/')) for f in file_names)
    folder.mkdir(parents=True,exist_ok=True); save(folder/'run_manifest.json',manifest)
    return manifest

def add_inventory(state,manifest,kernel):
    for row in manifest['stages']:
        key=row['channel']+'/'+row['mode']
        current=state['inventory'].get(key)
        epochs=int(row['global_step'])//800
        if current is None or row['global_step']>current['global_step']:
            state['inventory'][key]={'completed_epochs':epochs,'global_step':row['global_step'],
                                     'kernel':kernel,'status':'complete' if epochs==150 else 'partial'}
    if manifest.get('comparison_ready'):
        state['comparison_ready']=True; state['evaluation_kernel']=kernel

def dispatch(api,state,stage):
    # The reservation is persisted BEFORE submission; interrupted pushes cannot
    # cause a second owner to be assigned to the same stage.
    sequence=state['sequence']+1; state['sequence']=sequence
    slug=f'sqgan-stage-{stage.replace("/","-")}-{sequence:02d}-20261020'
    kernel='hmmkk20/'+slug
    parents=sorted({v['kernel'] for v in state['inventory'].values()})
    folder=ROOT/'Notebooks/project_notebooks/Training/sqgan_coordinated'/slug
    snapshot=OUT/'worker_template.ipynb'
    assert digest(snapshot)==state['worker_template_sha256'],'Frozen worker template changed'
    assert digest(Path(__file__).with_name('stage_artifacts.py'))==state['merge_module_sha256'],'Merge code changed; review before dispatch'
    build(folder,kernel,stage,parents,template=snapshot)
    state['leases'][stage]={'kernel':kernel,'status':'SUBMITTING','version':1,
                            'parents':parents,'created_at':now(),'folder':str(folder),
                            'notebook_sha256':digest(folder/'sqgan_coordinated_stage.ipynb')}
    save(OUT/'registry.json',state)
    try:
        reply=api.kernels_push(str(folder),acc='NvidiaTeslaT4').to_dict()
        if reply.get('error'): raise RuntimeError(reply['error'])
        if reply.get('url'):
            from urllib.parse import urlparse
            parts=urlparse(reply['url']).path.strip('/').split('/')
            assert parts[:1]==['code']
            state['leases'][stage]['kernel']='/'.join(parts[1:3])
        state['leases'][stage]['status']='SUBMITTED'
        save(OUT/'registry.json',state)
        print('EXCLUSIVE STAGE SUBMITTED',stage,state['leases'][stage]['kernel'],flush=True)
    except Exception as exc:
        if 'Maximum batch GPU session count' in str(exc) or 'Maximum weekly GPU quota' in str(exc):
            # Explicit resource rejection did not create a kernel: release only
            # this reservation and retry allocation later.
            del state['leases'][stage]
            state['resource_blocker']=str(exc)
            state['resource_retry_after_unix']=time.time()+(1800 if 'weekly GPU quota' in str(exc) else 60)
            save(OUT/'registry.json',state)
            print('GPU CAPACITY BUSY; STAGE RESERVATION RELEASED',stage,flush=True)
            return
        # Keep SUBMITTING ownership until the kernel's existence is checked.
        state['leases'][stage]['submission_exception']=repr(exc)
        save(OUT/'registry.json',state)
        raise

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'coordinator.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    try:
        original_send=requests.Session.send
        def send(session,request,**kwargs):
            kwargs.setdefault('timeout',(20,60)); return original_send(session,request,**kwargs)
        requests.Session.send=send
        api=KaggleApi(); api.authenticate()
        state_path=OUT/'registry.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {
            'schema':1,'phase':'WAITING_FOR_LEGACY_PUBLICATION','legacy_kernel':LEGACY,
            'inventory':{},'leases':{},'history':[],'sequence':0,'comparison_ready':False,
            'contract':{'source_commit':PIN,'images_per_epoch':800,'batch_per_gpu':1,'gpus':2,
                        'geometry':[256,512],'epochs_per_stage':150,'precision':32}}
        state['coordinator_pid']=os.getpid()
        snapshot=OUT/'worker_template.ipynb'
        if not snapshot.exists(): shutil.copy2(TEMPLATE,snapshot)
        state.setdefault('worker_template_sha256',digest(snapshot))
        state.setdefault('merge_module_sha256',digest(Path(__file__).with_name('stage_artifacts.py')))
        while True:
            if (OUT/'STOP').exists():
                state['phase']='COORDINATOR_STOPPED'; save(state_path,state); return
            try:
                assert version(api,LEGACY)==1,'Legacy notebook was edited; inspect before taking ownership'
                legacy_status=status(api,LEGACY)
                state['legacy_status']=legacy_status
                if state['phase']=='WAITING_FOR_LEGACY_PUBLICATION':
                    if legacy_status=='COMPLETE':
                        # Real checkpoint merge smoke validation must also have passed.
                        if status(api,VALIDATOR)!='COMPLETE':
                            raise RuntimeError('Real merge validation has not completed successfully')
                        proof_files=get_outputs(api,VALIDATOR)
                        proof=next(f for f in proof_files if f.file_name.endswith('/validation.json'))
                        response=requests.get(proof.url,timeout=(20,60)); response.raise_for_status()
                        assert response.json()['source_and_merged_checkpoint_equal']
                        manifest=fetch_manifest(api,LEGACY,OUT/'legacy_handoff')
                        add_inventory(state,manifest,LEGACY)
                        state['phase']='COORDINATING'; state['handoff_at']=now()
                        print('LEGACY OUTPUT VERIFIED; EXCLUSIVE STAGE SCHEDULING ENABLED',flush=True)
                    elif legacy_status not in {'RUNNING','QUEUED','PENDING','INITIALIZING'}:
                        raise RuntimeError('Legacy run did not publish successfully: '+legacy_status)
                if state['phase']=='COORDINATING':
                    for stage,lease in list(state['leases'].items()):
                        actual=status(api,lease['kernel'])
                        assert version(api,lease['kernel'])==lease['version']
                        lease['status']=actual
                        if actual=='COMPLETE':
                            folder=OUT/lease['kernel'].split('/')[1]
                            manifest=fetch_manifest(api,lease['kernel'],folder)
                            assert manifest['worker_stage']==stage
                            # A partial worker must advance its own assigned stage.
                            if stage!='evaluation':
                                row=next(s for s in manifest['stages'] if s['channel']+'/'+s['mode']==stage)
                                before=state['inventory'].get(stage,{}).get('global_step',0)
                                assert row['global_step']>before,'Worker did not advance its owned stage'
                            add_inventory(state,manifest,lease['kernel'])
                            state['history'].append({**lease,'stage':stage,'verified_at':now()})
                            del state['leases'][stage]
                        elif actual not in {'RUNNING','QUEUED','PENDING','INITIALIZING'}:
                            raise RuntimeError(f'Owned stage {stage} ended with {actual}; no blind restart')
                    if state['comparison_ready']:
                        state['phase']='TRAINING_AND_EVALUATION_COMPLETE'; save(state_path,state); return
                    # A benchmark session consumes a slot but is never cancelled here.
                    probe_busy=status(api,PROBE) in {'RUNNING','QUEUED','PENDING','INITIALIZING'}
                    free=2-int(probe_busy)-len(state['leases'])
                    if time.time()<state.get('resource_retry_after_unix',0): free=0
                    inventory={tuple(k.split('/')):v for k,v in state['inventory'].items()}
                    leased=[tuple(k.split('/')) for k in state['leases'] if k!='evaluation']
                    choices=ready_stages(inventory,leased)
                    for key in choices[:max(0,free)]:
                        dispatch(api,state,'/'.join(key))
                    if not choices and not state['leases'] and not probe_busy:
                        assert len(inventory)==6 and all(v['completed_epochs']==150 for v in inventory.values())
                        dispatch(api,state,'evaluation')
                state['checked_at']=now(); state.pop('last_error',None)
                save(state_path,state)
            except Exception as exc:
                state['last_error']=repr(exc); state['error_at']=now()
                save(state_path,state); print('COORDINATOR CHECK REQUIRES REVIEW',repr(exc),flush=True)
                # Fail closed on invariants; active Kaggle jobs continue untouched.
                if isinstance(exc,(AssertionError,StopIteration)):
                    state['phase']='NEEDS_REVIEW'; save(state_path,state); return
            time.sleep(60)
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__': main()
