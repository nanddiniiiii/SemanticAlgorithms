"""Transport retained NTSCC sources separately from immutable checkpoint assets."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'run_stage1_ntscc_gpu.py']")
needle="payload = buffer.getvalue()"
extra="""
with zipfile.ZipFile(buffer,'a',zipfile.ZIP_DEFLATED) as archive:
    for name in ['EncDec/Research/model_wrappers.py','EncDec/Research/compressai_1_2_compat.py']:
        archive.writestr(name,(ROOT/name).read_bytes())
    for path in sorted((ROOT/'Models/NTSCC/official').rglob('*.py')):
        archive.writestr(path.relative_to(ROOT).as_posix(),path.read_bytes())
"""
prefix=prefix.replace(needle,extra+'\n'+needle)
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('stage1_ntscc','satyapaladugu','stage1-ntscc-kodak-profile',
    'Stage1 NTSCC Kodak Profile',['sherylmehta/kodak-dataset','satyapaladugu/semantic-lossless-private-ntscc-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','timm==0.4.12','pytorch-msssim==1.0.0','lpips==0.1.4'],300)
import shutil
weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob('ntscc_hyperprior_quality_1_psnr.pth')})
pictures=sorted({p.resolve().parent for p in Path('/kaggle/input').rglob('kodim01.png')})
assert len(weights)==len(pictures)==1
target=root/'Models/trained_checkpoints/NTSCC/production/ntscc_hyperprior_quality_1_psnr.pth'
target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(weights[0],target)
run_logged('ntscc_gpu_quality_and_profile',[sys.executable,'scripts/run_stage1_ntscc_gpu.py',
    '--images',str(pictures[0]),'--output',str(root/'Results/stage1/ntscc/q1_gpu')],3600)
''')
print('Built NTSCC account-A job',flush=True)
