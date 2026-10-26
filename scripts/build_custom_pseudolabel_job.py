"""WSL account B: verified frozen teachers and native-input adaptation assets."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'scripts/build_stage1_jobs.py'
prefix=builder.read_text().split("write_job('stage1_swin'")[0]
prefix=prefix.replace("'train_shared_native_classifier.py']", "'train_shared_native_classifier.py', 'prepare_custom_dataset_spec.py', 'generate_custom_pseudolabels.py']")
namespace={'__file__':str(builder)}
exec(compile(prefix,str(builder),'exec'),namespace)
namespace['write_job']('custom_pseudolabels','satyapaladuguplus','custom-mixture-frozen-pseudolabels',
    'Custom Mixture Frozen Pseudo Labels',[
        'soumikrakshit/div2k-high-resolution-images','awsaf49/coco-2017-dataset',
        'deeplumiere/textocr','sherylmehta/kodak-dataset',
        'satyapaladuguplus/semantic-lossless-pseudo-label-assets'],'''
run_logged('dependencies',[sys.executable,'-m','pip','install','transformers==4.57.1'],600)
sys.path.insert(0,str(root/'src'))
from semantic_lossless.artifacts import sha256
manifests=sorted({p.resolve() for p in Path('/kaggle/input').rglob('frozen_manifest.json')})
classifiers=sorted({p.resolve() for p in Path('/kaggle/input').rglob('classifier.pt')})
segmenters=sorted({p.resolve().parent for p in Path('/kaggle/input').rglob('pytorch_model.bin')})
assert len(manifests)==len(classifiers)==len(segmenters)==1
assert sha256(manifests[0])=='6e921827d329de0ba9593904f77ddac50d91ba941bb0ab93a701c4af3c339d9e'
output=root/'Results/custom_adaptation/pseudo_labels/mixture_v1'
output.mkdir(parents=True)
spec=output/'resolved_sources.json'
run_logged('resolve_mounted_paths',[sys.executable,'scripts/prepare_custom_dataset_spec.py','--output',str(spec)],300)
base=[sys.executable,'scripts/generate_custom_pseudolabels.py','--manifest',str(manifests[0]),
    '--resolved-sources',str(spec),'--segmenter-root',str(segmenters[0]),'--classifier',str(classifiers[0])]
run_logged('largest_image_pilot',base+['--max-images','8','--output',str(output/'pilot')],1200)
pilot=json.loads((output/'pilot/status.json').read_text())
assert pilot['status']=='completed' and pilot['images']==8
run_logged('frozen_teacher_labels',base+['--output',str(output/'full')],5400)
''')
print('Built B-only frozen pseudo-label job; no codec training in this stage',flush=True)
