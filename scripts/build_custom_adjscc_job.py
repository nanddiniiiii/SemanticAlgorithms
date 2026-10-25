"""Account B: freeze actual mixture, validate two steps, then bounded adaptation."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'freeze_custom_dataset.py', 'prepare_custom_dataset_spec.py', 'train_custom_adjscc.py', 'evaluate_custom_adjscc_dev.py']")
prefix=prefix.replace("['swinjscc', 'shared_semantic_codebook']:", "['swinjscc', 'shared_semantic_codebook', 'diffcom']:")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('custom_adjscc','satyapaladuguplus','custom-mixture-adjscc-adaptation',
    'Custom Mixture ADJSCC Adaptation',[
        'soumikrakshit/div2k-high-resolution-images','awsaf49/coco-2017-dataset',
        'deeplumiere/textocr','sherylmehta/kodak-dataset',
        'satyapaladuguplus/semantic-lossless-private-adjscc-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','pytorch-msssim==1.0.0'],300)
sys.path.insert(0,str(root/'src'))
from semantic_lossless.artifacts import sha256
expected='c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'
weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob('*.tar') if p.is_file() and sha256(p)==expected})
assert len(weights)==1
output=root/'Results/custom_adaptation/adjscc/mixture_v1'
output.mkdir(parents=True)
spec=output/'resolved_sources.json';manifest=output/'frozen_manifest.json'
run_logged('resolve_official_partitions',[sys.executable,'scripts/prepare_custom_dataset_spec.py','--output',str(spec)],300)
run_logged('freeze_membership',[sys.executable,'scripts/freeze_custom_dataset.py','--spec',str(spec),'--output',str(manifest)],1800)
base=[sys.executable,'scripts/train_custom_adjscc.py','--manifest',str(manifest),'--checkpoint',str(weights[0])]
run_logged('custom_two_step_pilot',base+['--output',str(output/'pilot'),'--pilot','--max-seconds','600'],900)
pilot=json.loads((output/'pilot/status.json').read_text())
assert pilot['status']=='pilot_completed' and pilot['optimizer_steps']>=2
evaluation=[sys.executable,'scripts/evaluate_custom_adjscc_dev.py','--manifest',str(manifest),'--checkpoint',str(weights[0])]
run_logged('before_adaptation_development',evaluation+['--output',str(output/'before_dev')],1200)
# Start from pristine author weights, not pilot-updated parameters.
run_logged('custom_bounded_finetune',base+['--output',str(output/'finetune'),'--max-seconds','7200'],7800)
run_logged('after_adaptation_development',evaluation+['--adapted',str(output/'finetune/last.pt'),
    '--output',str(output/'after_dev')],1200)
''')
print('Built custom dataset ADJSCC job; native baseline jobs unchanged',flush=True)
