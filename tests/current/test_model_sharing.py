import json

import pytest

from src.runtime import export, write_json
from src.sharing import import_bundles, package_run, verify_bundle
from src.monitor import collect


def trained_run(tmp_path, algorithm='swinjscc'):
    run=tmp_path/'run'
    (run/'checkpoints').mkdir(parents=True)
    (run/'checkpoints/last.pt').write_bytes(b'final checkpoint')
    (run/'checkpoints/best.pt').write_bytes(b'development checkpoint')
    (run/'pilot/checkpoints').mkdir(parents=True)
    (run/'pilot/checkpoints/last.pt').write_bytes(b'pilot must not be shared')
    initializer=tmp_path/'author.model';initializer.write_bytes(b'author constructor weights')
    write_json(run/'resolved_config.json',{'algorithm':algorithm,'checkpoint':str(initializer),'manifest':'/remote/data.json'})
    write_json(run/'training_summary.json',{'steps':3,'status':'budget_limited_training_completed'})
    for stage in ('train','evaluate'):write_json(run/f'{stage}_status.json',{'status':'completed'})
    return run


def test_shareable_bundle_layout_integrity_and_assets(tmp_path):
    run=trained_run(tmp_path)
    target=package_run(run,tmp_path/'Models','1','run-01')
    assert target==tmp_path/'Models/teammate_1_models/SwinJSCC/run-01'
    assert (target/'last.pt').read_bytes()==b'final checkpoint'
    assert not (target/'pilot').exists()
    assert (target/'assets/checkpoint/author.model').is_file()
    portable=json.loads((target/'portable_config.json').read_text())
    assert portable['checkpoint']=='assets/checkpoint/author.model' and portable['manifest'] is None
    assert verify_bundle(target)['algorithm']=='swinjscc'
    with pytest.raises(FileExistsError):package_run(run,tmp_path/'Models','1','run-01')
    (target/'last.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='verification failed'):verify_bundle(target)


def test_bundle_import_is_idempotent_and_preserves_run_identity(tmp_path):
    run=trained_run(tmp_path)
    source=package_run(run,tmp_path/'download/models','2','run-02')
    targets=import_bundles(tmp_path/'download',tmp_path/'Models','2')
    assert len(targets)==1
    assert (tmp_path/'Models/teammate_2_models/SwinJSCC/run-02/last.pt').is_file()
    assert import_bundles(tmp_path/'download',tmp_path/'Models','2')==targets
    with pytest.raises(ValueError,match='different teammate'):
        import_bundles(tmp_path/'download',tmp_path/'Models','1')
    metadata=json.loads((source/'bundle.json').read_text())
    metadata['run_id']='../escape'
    write_json(source/'bundle.json',metadata)
    with pytest.raises(ValueError,match='folder identifier'):
        import_bundles(tmp_path/'download',tmp_path/'Models','2')


def test_no_training_checkpoint_means_no_model_package(tmp_path):
    run=tmp_path/'run';run.mkdir()
    write_json(run/'resolved_config.json',{'algorithm':'deepjscc'})
    assert package_run(run,tmp_path/'Models','1','untrained') is None
    assert not (tmp_path/'Models').exists()


def test_csv_contains_training_failures_and_evaluation_without_duplicates(tmp_path):
    root=tmp_path/'Results/team_downloads'
    retrieval=root/'1/run-a';run=retrieval/'published_output/session/deepjscc'
    (run/'metrics').mkdir(parents=True)
    write_json(run/'resolved_config.json',{'algorithm':'deepjscc','teammate':'1','run_id':'run-a'})
    write_json(run/'training_summary.json',{'steps':1})
    (run/'training.jsonl').write_text('{"step":1,"loss":0.5}\n')
    (run/'metrics/per_image.csv').write_text('algorithm,sample,psnr_db\ndeepjscc,kodak01,20\n')
    write_json(run/'metrics/per_image.json',{'rows':[],'failures':[{'sample':'kodak02','error':'decode failed'}]})
    write_json(run/'train_status.json',{'status':'completed'})
    export(run)
    export(run.parent)  # Nested receipts must not duplicate rows.
    write_json(retrieval/'retrieval_verified.json',{})
    output=tmp_path/'Results/comparison.csv'
    assert collect(root,output)==5
    original=output.read_bytes()
    assert collect(root,output)==5 and output.read_bytes()==original
    import csv
    rows=list(csv.DictReader(output.open()))
    assert {row['record_type'] for row in rows}=={'training_step','training_summary','stage_status','evaluation_trial','evaluation_failure'}
    assert all(row['teammate']=='1' and row['run_id']=='run-a' for row in rows)
    (run/'training.jsonl').write_text('{"step":1,"loss":100}\n')
    with pytest.raises(ValueError,match='training log changed'):collect(root,output)


def test_partial_training_is_not_a_shareable_model(tmp_path):
    run=trained_run(tmp_path)
    write_json(run/'train_status.json',{'status':'failed'})
    with pytest.raises(ValueError,match='Only completed'):
        package_run(run,tmp_path/'Models','1','partial')
    assert not (tmp_path/'Models').exists()

def test_interrupted_import_does_not_leave_final_bundle(tmp_path,monkeypatch):
    import src.sharing as sharing
    run=trained_run(tmp_path)
    package_run(run,tmp_path/'download/models','1','run-interrupted')
    original=sharing.shutil.copytree
    def interrupted(source,target,*args,**kwargs):
        target.mkdir()
        (target/'last.pt').write_bytes(b'partial')
        raise OSError('simulated interrupted transfer')
    monkeypatch.setattr(sharing.shutil,'copytree',interrupted)
    with pytest.raises(OSError,match='interrupted'):
        import_bundles(tmp_path/'download',tmp_path/'Models','1')
    assert not (tmp_path/'Models/teammate_1_models/SwinJSCC/run-interrupted').exists()
    monkeypatch.setattr(sharing.shutil,'copytree',original)
    assert len(import_bundles(tmp_path/'download',tmp_path/'Models','1'))==1
