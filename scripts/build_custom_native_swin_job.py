"""Account-B native Swin custom-data pilot followed by bounded fine-tuning."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']",
    "'train_shared_native_classifier.py', 'prepare_custom_dataset_spec.py', 'train_custom_native_swin.py']")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('custom_native_swin','satyapaladuguplus','custom-mixture-native-swin',
    'Custom Mixture Native Swin',[
        'soumikrakshit/div2k-high-resolution-images','awsaf49/coco-2017-dataset',
        'deeplumiere/textocr','sherylmehta/kodak-dataset',
        'satyapaladuguplus/semantic-lossless-private-swin-assets',
        'satyapaladuguplus/semantic-lossless-pseudo-label-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','--no-deps','timm==0.4.12','pytorch-msssim==1.0.0'],300)
sys.path.insert(0,str(root/'src'))
def asset_root(slug):
    candidates=[Path('/kaggle/input/datasets/satyapaladuguplus')/slug,Path('/kaggle/input')/slug]
    found={p.resolve() for p in candidates if p.is_dir()}
    if len(found)!=1:raise RuntimeError('Ambiguous/missing private asset mount: '+slug)
    return found.pop()
weights=list(asset_root('semantic-lossless-private-swin-assets').rglob('SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C32.model'))
manifests=list(asset_root('semantic-lossless-pseudo-label-assets').rglob('frozen_manifest.json'))
assert len(weights)==len(manifests)==1
from src.artifacts import sha256
assert sha256(manifests[0])=='6e921827d329de0ba9593904f77ddac50d91ba941bb0ab93a701c4af3c339d9e'
output=root/'Results/custom_adaptation/swin_native/fixed_c32_mixture_v1';output.mkdir(parents=True)
spec=output/'resolved_sources.json'
run_logged('resolve_frozen_paths',[sys.executable,'scripts/prepare_custom_dataset_spec.py',
    '--frozen-manifest',str(manifests[0]),'--output',str(spec)],300)
base=[sys.executable,'scripts/train_custom_native_swin.py','--manifest',str(manifests[0]),
    '--resolved-sources',str(spec),'--checkpoint',str(weights[0])]
run_logged('native_swin_dual_gpu_pilot',base+['--pilot','--output',str(output/'pilot'),'--max-seconds','600'],1200)
pilot=json.loads((output/'pilot/status.json').read_text())
assert pilot['status']=='pilot_completed' and pilot['optimizer_steps']>=2 and pilot['strict_reload']
run_logged('native_swin_bounded_finetune',base+['--output',str(output/'finetune'),'--max-seconds','7200'],8400)
''')
print('Built private B-only native Swin job with frozen-v1 data and two-GPU pilot gate',flush=True)
