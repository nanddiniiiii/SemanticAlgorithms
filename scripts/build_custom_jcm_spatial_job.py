"""WSL-account-B pilot and bounded approved JCM variant; no native baseline overwrite."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'prepare_custom_dataset_spec.py', 'train_custom_jcm_spatial.py']")
prefix=prefix.replace("['swinjscc', 'shared_semantic_codebook']:", "['swinjscc', 'shared_semantic_codebook', 'jcm']:")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('custom_jcm_spatial','satyapaladuguplus','custom-mixture-spatial-jcm',
    'Custom Mixture Spatial JCM',[
        'soumikrakshit/div2k-high-resolution-images','awsaf49/coco-2017-dataset',
        'deeplumiere/textocr','sherylmehta/kodak-dataset',
        'satyapaladuguplus/semantic-lossless-private-jcm-assets',
        'satyapaladuguplus/semantic-lossless-pseudo-label-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','pytorch-msssim==1.0.0'],300)
sys.path.insert(0,str(root/'src'))
from semantic_lossless.artifacts import sha256
weights=sorted({p.resolve() for p in Path('/kaggle/input').rglob('*.tar')
    if p.is_file() and sha256(p)=='b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897'})
manifests=sorted({p.resolve() for p in Path('/kaggle/input').rglob('frozen_manifest.json')})
assert len(weights)==len(manifests)==1
assert sha256(manifests[0])=='6e921827d329de0ba9593904f77ddac50d91ba941bb0ab93a701c4af3c339d9e'
output=root/'Results/custom_adaptation/jcm/spatial_reconstruction_v1';output.mkdir(parents=True)
spec=output/'resolved_sources.json'
run_logged('resolve_original_membership_paths',[sys.executable,'scripts/prepare_custom_dataset_spec.py','--output',str(spec)],300)
base=[sys.executable,'scripts/train_custom_jcm_spatial.py','--manifest',str(manifests[0]),
    '--checkpoint',str(weights[0]),'--resolved-sources',str(spec)]
run_logged('spatial_jcm_dual_gpu_pilot',base+['--pilot','--output',str(output/'pilot'),'--max-seconds','600'],900)
pilot=json.loads((output/'pilot/status.json').read_text())
assert pilot['status']=='pilot_completed' and pilot['optimizer_steps']>=2
run_logged('spatial_jcm_bounded_adaptation',base+['--output',str(output/'finetune'),'--max-seconds','7200'],7800)
''')
print('Built B-only spatial JCM candidate; requires assets and qualified pilot',flush=True)
