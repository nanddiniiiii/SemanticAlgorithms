import json
from pathlib import Path
import pytest
from scripts.build_notebooks import build
from src.notebook_support import require_integrated

ROOT=Path(__file__).parents[2]

def test_revised_roster_is_unique_and_notebooks_match():
    roster=json.loads((ROOT/'configs/models.json').read_text())
    all_names=[name for names in roster['teams'].values() for name in names]
    assert len(all_names)==len(set(all_names))==12
    assert 'swinjscc' not in all_names
    for team,names in roster['teams'].items():
        notebook=build(team,names)
        assert notebook['metadata']['experiment']['models']==names
        code=''.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
        assert 'ASSIGNED='+repr(names) in code

@pytest.mark.parametrize('name',['ssc','ofdm_guided_jscc','deepjscc_mimo'])
def test_integrated_system_slots_have_training_dependencies_and_sharing_names(name):
    from src.notebook_support import EXTRA,default_options
    from src.sharing import MODEL_NAMES
    require_integrated(name)
    assert name in EXTRA and name in MODEL_NAMES
    options=default_options(name,'manifest')
    assert options['max_seconds']==14400
