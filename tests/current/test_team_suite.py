"""Behavioral checks for splits, notebook assignments and safe artifact retrieval."""
import ast
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset import freeze,ParentImages,discover
from src.runtime import export,stage
from src.monitor import verify_download,collect


def test_discovery_nested_aliases_and_explicit_overrides(tmp_path):
    nested=tmp_path/'dataset'/'Flickr2K'/'Flickr2K_HR'
    nested.mkdir(parents=True)
    for index in range(2650):
        (nested/f'{index}.png').touch()
    assert discover(tmp_path,keys=['flickr2k_train'])['flickr2k_train']==str(nested.resolve())
    other=tmp_path/'another'/'Flickr2K_HR'
    other.mkdir(parents=True)
    for index in range(2650):
        (other/f'{index}.png').touch()
    with pytest.raises(ValueError,match='Multiple candidate'):
        discover(tmp_path,keys=['flickr2k_train'])
    assert discover(tmp_path,keys=[])=={}


def test_local_launcher_submits_and_collects_without_real_kaggle(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from src import launch
    from src.runtime import export
    notebook=tmp_path/'Notebooks/1.ipynb'
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({'cells':[{'cell_type':'code','source':["REVISION = '"+'a'*40+"'\n"]}]}))
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        if command[2]=='push':
            metadata=json.loads((Path(command[command.index('-p')+1])/'kernel-metadata.json').read_text())
            assert metadata['dataset_sources']==launch.DATASETS and metadata['enable_gpu']
            assert metadata['id']=='testuser/semantic-notebook-1'
        if command[2]=='status':return SimpleNamespace(returncode=0,stdout='COMPLETE',stderr='')
        if command[2]=='output':
            raw=Path(command[command.index('-p')+1]);run=raw/'team_results/you/run/deepjscc'
            (run/'metrics').mkdir(parents=True)
            (run/'metrics/per_image.csv').write_text('algorithm,psnr_db\ndeepjscc,10\n')
            (run/'checkpoints').mkdir()
            (run/'checkpoints/last.pt').write_bytes(b'weights')
            (run/'resolved_config.json').write_text(json.dumps({'algorithm':'deepjscc'}))
            for stage in ('train','evaluate'):
                (run/f'{stage}_status.json').write_text(json.dumps({'status':'completed'}))
            submission=next((tmp_path/'Results/team_downloads').rglob('submission.json'))
            write=json.loads(submission.read_text())
            (run/'delivery_identity.json').write_text(json.dumps({'delivery_run_id':write['run_id'],'team':'1'}))
            from src.sharing import package_run
            package_run(run,raw/'models','1','run-fixture')
            export(run)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    versions=iter([None,1,1,1])
    monkeypatch.setattr(launch,'current_version',lambda username,slug:next(versions))
    monkeypatch.setattr(launch.shutil,'which',lambda name:'fake-kaggle')
    monkeypatch.setattr(launch.subprocess,'run',fake_run)
    monkeypatch.setattr(sys,'argv',['1.py','--username','testuser','--no-publish'])
    launch.main('1')
    assert [cmd[2] for cmd in calls]==['push','status','output']
    assert (tmp_path/'Models/teammate_1_models/DeepJSCC/run-fixture/last.pt').is_file()
    assert 'deepjscc' in (tmp_path/'Results/comparison.csv').read_text()


def test_launcher_refuses_unpinned_notebook_before_submission(tmp_path,monkeypatch):
    import sys
    from src import launch
    notebook=tmp_path/'Notebooks/1.ipynb'
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({'cells':[{'source':["REVISION = ''\n"]}]}))
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch.shutil,'which',lambda name:'fake-kaggle')
    monkeypatch.setattr(sys,'argv',['1.py','--username','testuser','--no-publish'])
    with pytest.raises(RuntimeError,match='not published and pinned'):
        launch.main('1')
    assert not list((tmp_path/'Results').rglob('submission.json'))


def test_launcher_does_not_replace_running_notebook(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from src import launch
    notebook=tmp_path/'Notebooks/1.ipynb'
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({'cells':[{'cell_type':'code','source':["REVISION = '"+'a'*40+"'\n"]}]}))
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch.shutil,'which',lambda name:'fake-kaggle')
    monkeypatch.setattr(launch,'current_version',lambda username,slug:3)
    calls=[]
    def running(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='RUNNING',stderr='')
    monkeypatch.setattr(launch.subprocess,'run',running)
    monkeypatch.setattr(sys,'argv',['1.py','--username','testuser','--no-publish'])
    with pytest.raises(RuntimeError,match='Existing notebook is active'):
        launch.main('1')
    assert len(calls)==1 and calls[0][2]=='status'


def test_sqgan_components_require_expected_digests(tmp_path,monkeypatch):
    import hashlib
    from src import assets
    (tmp_path/'encoder.ckpt').write_bytes(b'expected')
    monkeypatch.setattr(assets,'SQGAN_DIGESTS',{'encoder.ckpt':hashlib.sha256(b'expected').hexdigest()})
    assert assets.verified_sqgan_components(tmp_path)
    (tmp_path/'encoder.ckpt').write_bytes(b'changed')
    assert not assets.verified_sqgan_components(tmp_path)


def test_sqgan_null_weight_exclusion_matches_loss_channels():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from src.sqgan_train import align_cityscapes_loss_weights
    loss=SimpleNamespace(label_weight_tensor=torch.arange(20))
    align_cityscapes_loss_weights(loss)
    assert loss.label_weight_tensor.tolist()==list(range(19))
    with pytest.raises(ValueError,match='Unexpected'):
        align_cityscapes_loss_weights(loss)

ROOT=Path(__file__).resolve().parents[2]


def sample_roots(tmp_path):
    roots={}
    for index,name in enumerate(('div2k_train','flickr2k_train','div2k_val','kodak_test')):
        path=tmp_path/name;path.mkdir()
        Image.new('RGB',(64,48),(index*50,20,3)).save(path/'image.png')
        roots[name]=str(path)
    return roots


def test_content_identity_is_portable_and_duplicate_rgb_rejected(tmp_path):
    roots=sample_roots(tmp_path)
    value=freeze(roots,tmp_path/'manifest.json',strict_counts=False)
    assert value['counts']=={'train':2,'val':1,'test':1}
    assert freeze(roots,tmp_path/'manifest.json',strict_counts=False)['identity']==value['identity']
    Image.open(Path(roots['div2k_train'])/'image.png').save(Path(roots['kodak_test'])/'different.bmp')
    with pytest.raises(ValueError,match='Duplicate RGB'):
        freeze(roots,tmp_path/'other.json',strict_counts=False)


def test_training_duplicate_is_retained_and_recorded(tmp_path):
    roots=sample_roots(tmp_path)
    Image.open(Path(roots['div2k_train'])/'image.png').save(Path(roots['flickr2k_train'])/'duplicate.png')
    value=freeze(roots,tmp_path/'manifest.json',strict_counts=False)
    assert value['counts']['train']==3
    assert len(value['training_duplicates_retained'])==1
    assert 'retain and record' in value['duplicate_policy']


def test_native_crop_and_full_test_are_different_contracts(tmp_path):
    pytest.importorskip('torch')
    roots=sample_roots(tmp_path);path=tmp_path/'manifest.json'
    freeze(roots,path,strict_counts=False)
    assert tuple(ParentImages(path,'train',[32,48])[0]['image'].shape)==(3,32,48)
    assert tuple(ParentImages(path,'test',32)[0]['image'].shape)==(3,48,64)
    with pytest.raises(ValueError,match='smaller'):
        ParentImages(path,'train',128)[0]


def test_tampered_manifest_and_changed_pixels_are_rejected(tmp_path):
    roots=sample_roots(tmp_path);path=tmp_path/'manifest.json'
    freeze(roots,path,strict_counts=False)
    Image.new('RGB',(64,48),'red').save(Path(roots['div2k_train'])/'image.png')
    with pytest.raises(ValueError,match='Changed dataset'):
        ParentImages(path,'train',32)
    value=json.loads(path.read_text());value['images'][0]['split']='test'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='identity'):
        ParentImages(path,'train')


def test_partial_evaluation_preserves_results_but_fails_stage(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from src import evaluate as evaluator, profiling
    roots=sample_roots(tmp_path);manifest=tmp_path/'manifest.json'
    freeze(roots,manifest,strict_counts=False)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()));self.calls=0
        def forward(self,image):
            self.calls+=1
            if self.calls==2:raise RuntimeError('simulated decode failure')
            return {'reconstruction':image*self.weight,'complex_uses':image.new_tensor([1])}
    monkeypatch.setattr(evaluator,'image_quality',lambda *a,**k:{'psnr_db':100.0})
    monkeypatch.setattr(profiling,'profile',lambda *a,**k:{'scope':'test stub'})
    with pytest.raises(RuntimeError,match='Incomplete evaluation'):
        evaluator.evaluate(Model(),manifest,tmp_path/'run',{'algorithm':'fixture','trials':2})
    evidence=json.loads((tmp_path/'run/metrics/per_image.json').read_text())
    assert len(evidence['rows'])==1 and len(evidence['failures'])==1
    with pytest.raises(FileExistsError,match='Evaluation evidence already exists'):
        evaluator.evaluate(Model(),manifest,tmp_path/'run',{'algorithm':'fixture','trials':2})
    assert json.loads((tmp_path/'run/metrics/per_image.json').read_text())==evidence


def test_export_after_stage_is_verifiable_and_detects_corruption(tmp_path):
    with stage(tmp_path,'pilot'):
        (tmp_path/'weights.bin').write_bytes(b'checkpoint bytes')
    export(tmp_path)
    assert verify_download(tmp_path)['verified_files']>=3
    (tmp_path/'weights.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='verification'):
        verify_download(tmp_path)


def test_three_notebooks_assign_tested_models_and_compile():
    roster=json.loads((ROOT/'configs/models.json').read_text())['teams']
    assigned=[]
    for team,names in roster.items():
        notebook=json.loads((ROOT/'Notebooks'/(team+'.ipynb')).read_text())
        assert notebook['metadata']['experiment']['models']==names
        assert len([c for c in notebook['cells'] if c['cell_type']=='code'])==14
        for item in notebook['cells']:
            if item['cell_type']=='code':ast.parse(''.join(item['source']))
        assigned+=names
    assert len(assigned)==12 and len(set(assigned))==9
    assert all(len(names)==4 and len(names)==len(set(names)) for names in roster.values())


def test_notebook_slots_follow_user_model_order_and_package_models(tmp_path):
    from scripts.build_notebooks import build
    notebook=build('1',['deepjscc','swinjscc','adjscc','isec'])
    calls=[];runs={}
    def initialize(name):
        run=tmp_path/name;run.mkdir();runs[name]=run
    def invoke(name,stage):
        calls.append((name,stage))
        if stage=='package':
            (runs[name]/'package_result.json').write_text('{"produced": true}')
    namespace={'SESSION':tmp_path,'TEAM':'1','MODELS_TO_RUN':['swinjscc','deepjscc'],
        'READY':{},'TRAINED':{},'EVALUATED':{},'PACKAGED':{},'EXPORTED':{},'RUNS':runs,
        'REPO':ROOT,'json':json,'initialize':initialize,'invoke':invoke,
        'write_json':lambda path,value:None,'environment':lambda root:{},'export':lambda path:None}
    for cell in notebook['cells'][2:]:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'notebook-stage','exec'),namespace)
    expected=['prepare','pilot','train','evaluate','package','export']
    assert calls==[(name,stage) for name in ['swinjscc','deepjscc'] for stage in expected]
    assert namespace['PACKAGED']=={'swinjscc':True,'deepjscc':True}


def test_collector_keeps_distinct_runs_excludes_pilots_and_is_repeatable(tmp_path):
    for folder in ('you/run1/published_output/model/metrics','teammate/run2/published_output/model/metrics',
                   'you/run1/published_output/model/pilot/metrics','unverified/model/metrics'):
        target=tmp_path/folder;target.mkdir(parents=True)
        with (target/'per_image.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['algorithm','sample','psnr_db'])
            writer.writeheader();writer.writerow({'algorithm':'a','sample':'kodak01','psnr_db':20})
    for team,run in [('you','run1'),('teammate','run2')]:
        export(tmp_path/team/run/'published_output/model')
        (tmp_path/team/run/'retrieval_verified.json').write_text('{}')
    assert collect(tmp_path)==2
    assert collect(tmp_path)==2
    assert len(list(csv.DictReader((tmp_path/'comparison.csv').open())))==2


def test_native_task_gates_do_not_substitute_rgb(tmp_path):
    from src.special import prepare
    with pytest.raises(FileNotFoundError,match='paired_data_root'):
        prepare('deepjscc_wz',{},tmp_path)
    with pytest.raises(FileNotFoundError,match='training_extraction_config'):
        prepare('gbsed',{},tmp_path)


def test_real_deepjscc_two_step_pilot_crosses_small_dataset_epoch(tmp_path):
    torch=pytest.importorskip('torch')
    from src.sources import source_path
    if not (source_path('deepjscc')/'source_receipt.json').is_file():
        pytest.skip('Pinned source must be fetched explicitly before numerical integration check')
    from src.models import build
    from src.train import train
    from src.evaluate import evaluate
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        roots=sample_roots(tmp_path);manifest=tmp_path/'manifest.json'
        freeze(roots,manifest,strict_counts=False)
        options={'algorithm':'deepjscc','from_scratch':True,'latent_channels':4,'crop':32,
                 'batch_size':2,'data_parallel':False,'epochs':3,'max_seconds':30}
        model=build('deepjscc',options)
        run=tmp_path/'run';run.mkdir()
        result=train(model,manifest,run,options,pilot=True)
        assert result['stage_optimizer_steps']=={'codec':2}
        assert result['strict_reload'] is True
        metrics=evaluate(model,manifest,run,options,pilot=True)
        assert metrics['successful_trials']==1
        assert (run/'metrics/profile.json').is_file()
        # A real saved model must survive the same folder-sharing path.
        from src.runtime import write_json
        from src.sharing import package_run,load_bundle
        write_json(run/'resolved_config.json',options)
        for stage in ('train','evaluate'):write_json(run/f'{stage}_status.json',{'status':'completed'})
        bundle=package_run(run,tmp_path/'Models','1','real-deepjscc')
        restored,_=load_bundle(bundle)
        image=torch.rand(1,3,32,32)
        torch.manual_seed(98);expected=model(image)['reconstruction']
        torch.manual_seed(98);actual=restored(image)['reconstruction']
        torch.testing.assert_close(actual,expected)
    finally:
        torch.set_num_threads(previous)


def test_graph_runtime_configs_preserve_originals_and_isolate_outputs(tmp_path):
    yaml=pytest.importorskip('yaml')
    from src.special import prepare_graph_configs
    train=tmp_path/'training.pkl';train.write_bytes(b'no deserialization in config test')
    evaluation=tmp_path/'evaluation.pkl';evaluation.write_bytes(b'other split')
    weights=tmp_path/'author.pt';weights.write_bytes(b'immutable original')
    options={};originals={}
    for key in ('training_extraction_config','training_learning_config',
                'evaluation_extraction_config','evaluation_learning_config'):
        value={'location_data':{'input_path':str(train if key.startswith('training') else evaluation),
                                'data_save_path':'unsafe/old/output.pkl'},
               'model_configuration':{'model_load_path':str(weights),'model_save_path':str(weights)}}
        path=tmp_path/(key+'.yaml');path.write_text(yaml.safe_dump(value))
        options[key]=str(path);originals[path]=path.read_bytes()
    run=tmp_path/'run'
    resolved=prepare_graph_configs(options,run)
    for key,path in resolved.items():
        config=yaml.safe_load(Path(path).read_text())
        output=config['location_data']['data_save_path'] if 'extraction' in key else config['model_configuration']['model_save_path']
        Path(output).relative_to(run.resolve())
    assert all(path.read_bytes()==before for path,before in originals.items())
    assert weights.read_bytes()==b'immutable original'


def test_notebook_does_not_report_success_after_evaluation_failure(tmp_path):
    from scripts.build_notebooks import build
    notebook=build('you',['deepjscc'])
    source=''.join(notebook['cells'][-1]['source'])
    namespace={'SESSION':tmp_path,'TEAM':'you','READY':{'deepjscc':True},
               'TRAINED':{'deepjscc':True},'EVALUATED':{'deepjscc':False},
               'PACKAGED':{'deepjscc':False},'EXPORTED':{'deepjscc':False},'export':lambda path:None,
               'MODELS_TO_RUN':['deepjscc'],'REPO':ROOT,'environment':lambda root:{},
               'write_json':lambda path,value:None,'json':json}
    with pytest.raises(RuntimeError,match='did not complete evaluation'):
        exec(compile(source,'summary','exec'),namespace)


def test_legacy_quantizer_preserves_rounded_values_and_identity_gradient():
    torch=pytest.importorskip('torch')
    from src.models import legacy_ste_round
    value=torch.tensor([-1.75,-.5,-.01,0.,.5,2.4],requires_grad=True)
    result=legacy_ste_round(value)
    torch.testing.assert_close(result,torch.round(value),atol=0,rtol=0)
    result.sum().backward()
    torch.testing.assert_close(value.grad,torch.ones_like(value),atol=0,rtol=0)


def test_checkpoint_regenerates_only_resolution_masks_and_rejects_weights():
    torch=pytest.importorskip('torch')
    from src.models import load_training_state
    model=torch.nn.Linear(2,2)
    model.register_buffer('attn_mask',torch.zeros(2,2))
    state={**model.state_dict(),'attn_mask':torch.ones(7,7)}
    result=load_training_state(model,state)
    assert result['learned_parameters_strict']
    assert tuple(model.attn_mask.shape)==(2,2)
    with pytest.raises(RuntimeError,match='Unexpected resolution-mask'):
        load_training_state(model,{**state,'unexpected.attn_mask':torch.zeros(7)})
    del state['weight']
    with pytest.raises(RuntimeError,match='Missing key'):
        load_training_state(model,state)


def test_replica_ema_matches_native_combined_batch_update():
    torch=pytest.importorskip('torch')
    pytest.importorskip('einops')
    import copy
    import importlib.util
    from src.sources import source_path
    path=source_path('sqgan')/'modules/vector_quantization/quantize2.py'
    if not path.is_file():pytest.skip('Pinned SQ-GAN source not fetched')
    spec=importlib.util.spec_from_file_location('test_sq_native_quantize',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from src.ema import ReplicaEMA
    torch.manual_seed(22)
    native=module.VQEmbedding(4,3,restart_unused_codes=False).train()
    aggregated=copy.deepcopy(native)
    chunks=[torch.randn(1,5,3),torch.randn(1,5,3)]
    native(torch.cat(chunks,dim=0))
    sync=ReplicaEMA({'vq':aggregated})
    for values in chunks:aggregated(values)
    assert sync.update()=={'vq':10}
    for key,value in native.state_dict().items():
        torch.testing.assert_close(value,aggregated.state_dict()[key])
    sync.close()
    assert aggregated.ema is True
