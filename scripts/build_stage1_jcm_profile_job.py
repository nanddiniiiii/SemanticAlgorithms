"""Build a separate account-A profiling job without replacing running sources."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'run_stage1_jcm_profile.py']")
prefix=prefix.replace("['swinjscc', 'shared_semantic_codebook']:", "['swinjscc', 'shared_semantic_codebook', 'jcm']:")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('stage1_jcm_profile','satyapaladugu','stage1-jcm-native-profile',
    'Stage1 JCM Native Profile',['pankrzysiu/cifar10-python','satyapaladugu/semantic-lossless-private-jcm-assets'],'''
sys.path.insert(0,str(root/'src'))
from src.artifacts import sha256
expected='b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897'
weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob('*.tar') if p.is_file() and sha256(p)==expected})
data=sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob('cifar-10-batches-py') if (p/'test_batch').is_file()})
assert len(weights)==len(data)==1,(weights,data)
run_logged('jcm_native_profiles',[sys.executable,'scripts/run_stage1_jcm_profile.py','--checkpoint',str(weights[0]),
    '--data-root',str(data[0]),'--output',str(root/'Results/stage1/jcm/native_profiles')],1800)
''')
print('Built account-A JCM profiling job',flush=True)
