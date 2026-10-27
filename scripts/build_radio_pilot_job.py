"""Package B-only Sionna/ns-3 smoke without changing codec training environments."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']",
    "'train_shared_native_classifier.py', 'run_radio_environment_pilot.py']")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('radio_environment_pilot','satyapaladuguplus','semantic-radio-environment-pilot',
    'Semantic Radio Environment Pilot',[],'''
run_logged('sionna_dependencies',[sys.executable,'-m','pip','install','sionna-no-rt==2.2.0'],600)
run_logged('radio_correctness',[sys.executable,'scripts/run_radio_environment_pilot.py',
    '--output',str(root/'Results/radio_environment/pilot_v1')],2400)
''')
print('Built private B radio pilot: Sionna PHY2.2 + independent ns-3.47; no coupled RT claim')
