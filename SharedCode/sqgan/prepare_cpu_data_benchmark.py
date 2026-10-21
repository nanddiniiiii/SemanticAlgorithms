"""Prepare a CPU-only data-loading benchmark without consuming GPU sessions."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
parent=ROOT/'Notebooks/project_notebooks/Training'
nb=json.loads((parent/'sqgan_div2k_channel/sqgan_div2k_awgn_rayleigh_training.ipynb').read_text())
assign=next(n for n in ast.parse(''.join(nb['cells'][3]['source'])).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='DATASET_SOURCE')
dataset_source=ast.literal_eval(assign.value)
code='DATASET_SOURCE='+repr(dataset_source)+'\nexec(DATASET_SOURCE,globals())\n'
code+=r'''
import json, time, hashlib
from torch.utils.data import DataLoader
from PIL import Image
torch.set_num_threads(1)
OUT=Path('/kaggle/working/sqgan_cpu_data_benchmark'); OUT.mkdir(exist_ok=True)
prior=next(Path('/kaggle/input').rglob('sqgan_div2k_channel_results'))
images=Path('/kaggle/input/notebooks/jagan028/div2k-dataset-generation-for-isr/Training/HR/hr_images')
labels=prior/'pseudo_labels/div2k_train'
original=DIV2KWithPseudoLabels(str(images),str(labels),size=(256,512),train=True)
assert len(original)==800
cache=Path('/tmp/sqgan_cpu_cache'); cache.mkdir(exist_ok=True)
mapping={}; equality=[]
t0=time.perf_counter()
for i,(image_path,mask_path) in enumerate(original.pairs):
    with Image.open(image_path) as im, Image.open(mask_path) as ma:
        image,mask=original._resize_cover(im.convert('RGB'),ma.convert('L'))
    cp,mp=cache/f'{i}_rgb.png',cache/f'{i}_label.png'
    image.save(cp,compress_level=1); mask.save(mp,compress_level=1)
    mapping[str(image_path)]=(cp,mp)
cache_build_seconds=time.perf_counter()-t0
cached=DIV2KWithPseudoLabels(str(images),str(labels),size=(256,512),train=True)
cached.pairs=[mapping[str(p[0])] for p in original.pairs]
for idx in range(0,800,40):
    np.random.seed(20261020+idx); a=original[idx]
    np.random.seed(20261020+idx); b=cached[idx]
    same=all(torch.equal(a[k],b[k]) for k in ('image','label','semantic'))
    assert same,idx
    equality.append({'index':idx,'image_label_semantic_exactly_equal':same})
(OUT/'cache_equivalence.json').write_text(json.dumps(equality,indent=2))
reports=[]
for name,dataset,workers,pin in [('original_workers0',original,0,False),
                                ('cached_workers0',cached,0,False),
                                ('cached_workers2',cached,2,False)]:
    gen=torch.Generator().manual_seed(20261020)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=workers,
                      pin_memory=pin,generator=gen,persistent_workers=workers>0)
    seen=[]; t0=time.perf_counter(); first100=None
    for batch in loader:
        seen.extend(batch['filename'])
        assert batch['image'].shape==(1,3,256,512)
        if len(seen)==100: first100=time.perf_counter()-t0
    seconds=time.perf_counter()-t0
    assert len(seen)==len(set(seen))==800
    report={'variant':name,'cpu_only':True,'images_processed':800,'unique_images':800,
            'batch_size':1,'workers':workers,'full_pass_seconds':seconds,
            'first100_seconds':first100,'remaining700_seconds':seconds-first100,
            'cache_build_seconds':cache_build_seconds if 'cached' in name else 0,
            'gpu_training_speedup_measured':False,'production_changed':False}
    reports.append(report)
    (OUT/'data_loading_report.json').write_text(json.dumps(reports,indent=2))
    print('CPU DATA RESULT',json.dumps(report),flush=True)
print('CPU DATA BENCHMARK COMPLETE',flush=True)
'''
ast.parse(code)
nb['cells']=[{'cell_type':'markdown','metadata':{},'source':['# SQ-GAN CPU data benchmark\nNo training, no GPU use, no production changes. 800 images per measured pass.']},
             {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]
dest=parent/'SQGAN_cpu_data_benchmark'; dest.mkdir(exist_ok=True)
(dest/'sqgan_cpu_data_benchmark.ipynb').write_text(json.dumps(nb,indent=1))
meta={'id':'hmmkk20/sqgan-cpu-data-benchmark-20261020','title':'SQGAN CPU Data Benchmark 20261020',
      'code_file':'sqgan_cpu_data_benchmark.ipynb','kernel_type':'notebook','language':'python',
      'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
      'dataset_sources':[],'competition_sources':[],'model_sources':[],
      'kernel_sources':['jagan028/div2k-dataset-generation-for-isr','hmmkk20/sqgan-checkpoint-publication-test-20261020']}
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(dest)
