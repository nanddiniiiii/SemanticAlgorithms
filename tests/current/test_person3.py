from pathlib import Path
import json
import pytest
from src.special import validate_pairs,native_command


def paired_fixture(root):
    for split in ('train','val','test'):
        paths=[]
        for side in ('left','right'):
            path=root/(split+'-'+side+'.png');path.write_bytes(b'image-fixture');paths.append(path.name)
        (root/('KITTI_stereo_'+split+'.txt')).write_text('\n'.join(paths)+'\n')


def test_wz_native_split_validation_and_no_target_sideinfo(tmp_path):
    paired_fixture(tmp_path)
    assert validate_pairs(tmp_path,'KittiStereo')=={'train':1,'val':1,'test':1}
    (tmp_path/'KITTI_stereo_val.txt').write_text('val-left.png\nval-left.png\n')
    with pytest.raises(ValueError,match='must not be the target'):validate_pairs(tmp_path,'KittiStereo')


def test_wz_worker_command_avoids_native_multirun_and_records_budget(tmp_path):
    options={'paired_data_root':str(tmp_path),'paired_dataset':'KittiStereo','max_seconds':7200}
    command,cwd=native_command('deepjscc_wz','evaluate',options,tmp_path)
    assert any(str(value).endswith('wz_native.py') for value in command)
    assert '--stage' in command and command[-1]=='evaluate'
    assert not any('wz_eval' in str(value) for value in command)
    saved=json.loads((tmp_path/'wz_worker_config.json').read_text())
    assert saved['max_seconds']==7200


def test_wz_worker_compiles_and_preserves_native_pair_contract():
    import ast
    path=Path(__file__).parents[2]/'scripts/wz_native.py'
    source=path.read_text(encoding='utf-8-sig');ast.parse(source)
    assert "model.load_state_dict(state['state_dict'],strict=True)" in source
    assert "trainer.test(model,datamodule=data,ckpt_path=None)" in source
