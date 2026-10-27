"""Reuse the tested Stage1 transport without regenerating running jobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
builder = ROOT/'scripts/build_stage1_jobs.py'
prefix = builder.read_text().split("write_job('stage1_swin'")[0]
prefix = prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'run_stage1_adjscc.py']")
prefix = prefix.replace("['swinjscc', 'shared_semantic_codebook']:", "['swinjscc', 'shared_semantic_codebook', 'diffcom']:")
namespace = {'__file__': str(builder)}
exec(compile(prefix, str(builder), 'exec'), namespace)
namespace['write_job']('stage1_adjscc', 'satyapaladuguplus', 'stage1-adjscc-kodak-profile',
    'Stage1 ADJSCC Kodak Profile',
    ['sherylmehta/kodak-dataset', 'satyapaladuguplus/semantic-lossless-private-adjscc-assets'], '''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','pytorch-msssim==1.0.0','lpips==0.1.4'],300)
sys.path.insert(0,str(root/'src'))
from src.artifacts import sha256
expected='c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'
weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob('*.tar') if p.is_file() and sha256(p)==expected})
pictures=sorted({p.resolve().parent for p in Path('/kaggle/input').rglob('kodim01.png')})
assert len(weights)==len(pictures)==1,(weights,pictures)
run_logged('adjscc_kodak_quality_and_profiles',[sys.executable,'scripts/run_stage1_adjscc.py',
    '--checkpoint',str(weights[0]),'--images',str(pictures[0]),
    '--output',str(root/'Results/stage1/adjscc/diffcom_C2')],3600)
''')
print('Built Stage1 ADJSCC job; existing A/B job folders unchanged', flush=True)
