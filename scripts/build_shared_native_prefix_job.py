"""A bounded native-release construction/pilot/epoch check, not convergence."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'run_shared_native_prefix.py']")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('stage1_shared_native_prefix','satyapaladugu','stage1-shared-native-prefix',
    'Stage1 Shared Native Prefix',['pratt3000/stl10-binary-files',
    'satyapaladugu/semantic-lossless-private-shared-classifier-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','timm==0.4.12'],300)
files=sorted({p.resolve() for p in Path('/kaggle/input').rglob('train_X.bin')})
classifiers=sorted({p.resolve() for p in Path('/kaggle/input').rglob('classifier.pt')})
assert len(files)==len(classifiers)==1
data=root/'Datasets/STL10';data.mkdir(parents=True)
(data/'stl10_binary').symlink_to(files[0].parent,target_is_directory=True)
base=[sys.executable,'scripts/run_shared_native_prefix.py']
common=['--classifier',str(classifiers[0]),'--data-root',str(data)]
output=root/'Results/stage1/shared_semantic_codebook/native_release_prefix'
run_logged('codebook_construction',base+['construct']+common+['--output',str(output/'construction'),'--max-seconds','1800'],1900)
book=['--book',str(output/'construction/book_native_e290.pt')]
run_logged('codec_pilot',base+['train']+common+book+['--output',str(output/'pilot'),'--steps','2','--max-seconds','600'],700)
run_logged('codec_epoch_prefix',base+['train']+common+book+['--output',str(output/'epoch0'),'--max-seconds','3600'],3700)
''')
print('Built shared native prefix; launch only after review/pilot gates',flush=True)
