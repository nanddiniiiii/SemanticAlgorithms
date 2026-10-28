"""Preserve partial weights and create immutable private continuation inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from src.dataset import digest
from src.runtime import write_json
from src.sharing import MODEL_NAMES


def preserve_chunks(root,destination,env):
    root=Path(root);destination=Path(destination)
    submitted=json.loads((destination/'submitted_version.json').read_text())
    username=submitted['kernel'].split('/')[0]
    raw=destination/'published_output'
    for marker in raw.rglob('chunk_checkpoint.json'):
        run=marker.parent;spec=json.loads(marker.read_text());name=spec['algorithm']
        checkpoint=(run/spec['checkpoint']).resolve();checkpoint.relative_to(run.resolve())
        if digest(checkpoint)!=spec['sha256']:raise ValueError('Partial checkpoint changed after retrieval')
        identity=json.loads((run/'person1_run.json').read_text());run_id=identity['run_id']
        target=root/'Models/teammate_1_models'/MODEL_NAMES[name]/run_id/'partial'
        if not target.exists():
            target.mkdir(parents=True)
            shutil.copy2(checkpoint,target/checkpoint.name)
            for filename in ('training_summary.json','resolved_config.json','source.json','model.json','person1_run.json','chunk_checkpoint.json'):
                if (run/filename).is_file():shutil.copy2(run/filename,target/filename)
        if digest(target/checkpoint.name)!=spec['sha256']:raise ValueError('Local preserved checkpoint failed hash')
        receipt=destination/('checkpoint_dataset-'+name+'.json')
        if receipt.is_file() and json.loads(receipt.read_text()).get('status')=='ready':continue
        slug='p1-'+name.replace('_','-')+'-'+hashlib.sha256(run_id.encode()).hexdigest()[:12]
        ref=username+'/'+slug
        upload=root/'Results/person1_checkpoints'/slug;upload.mkdir(parents=True,exist_ok=True)
        shutil.copy2(checkpoint,upload/checkpoint.name)
        write_json(upload/'resume_receipt.json',dict(algorithm=name,checkpoint=checkpoint.name,sha256=spec['sha256'],
                   contract=json.loads((run/'training_summary.json').read_text())['contract'],run_id=run_id))
        write_json(upload/'dataset-metadata.json',dict(id=ref,title='Person1 private '+name+' checkpoint',
                   licenses=[{'name':'other'}]))
        # No --public flag: checkpoint inputs must stay private. Unique immutable
        # identities prevent replacing an older chunk used in another version.
        if not receipt.is_file():
            write_json(receipt,dict(dataset=ref,status='upload_started',sha256=spec['sha256']))
            result=subprocess.run(['kaggle','datasets','create','-p',str(upload),'-t'],env=env,capture_output=True,text=True,timeout=7200)
            (destination/('checkpoint_upload-'+name+'.log')).write_text(result.stdout+'\n'+result.stderr)
            if result.returncode:raise RuntimeError('Private checkpoint upload failed; inspect saved log before retry')
        status=subprocess.run(['kaggle','datasets','status',ref],env=env,capture_output=True,text=True,timeout=90)
        if status.returncode or 'ready' not in status.stdout.lower():
            raise RuntimeError('Private checkpoint Dataset not ready yet; collector will retry without duplicate creation')
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
        api=KaggleApi();api.authenticate();request=ApiGetDatasetMetadataRequest()
        request.owner_slug=username;request.dataset_slug=slug
        with api.build_kaggle_client() as client:
            view=client.datasets.dataset_api_client.get_dataset_metadata(request)
        private=view.info.is_private if view.info else None
        if private is not True:raise ValueError('Continuation Dataset privacy could not be verified')
        write_json(receipt,dict(dataset=ref,status='ready',sha256=spec['sha256'],is_private=True))
