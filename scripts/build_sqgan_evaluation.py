"""Package native-component Cityscapes evaluation for account B."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root/'src').rglob('*.py')):
        archive.writestr(path.relative_to(root).as_posix(),path.read_bytes())
    archive.writestr('sources.lock.json',(root/'sources.lock.json').read_bytes())
assets=root/'Kaggle/assets/sqgan_source';assets.mkdir(parents=True,exist_ok=True)
snapshot=assets/'sqgan-pinned-source.zip'
upstream=root/'.upstream/sqgan'
receipt=json.loads((upstream/'source_receipt.json').read_text())
with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as archive:
    for item in receipt['files']:
        path=upstream/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        archive.writestr(path.relative_to(root).as_posix(),path.read_bytes())
    archive.writestr('.upstream/sqgan/source_receipt.json',(upstream/'source_receipt.json').read_bytes())
(assets/'dataset-metadata.json').write_text(json.dumps({'title':'Semantic Lossless Private SQGAN Source',
    'id':'satyapaladuguplus/semantic-lossless-private-sqgan-source','licenses':[{'name':'other'}]},indent=2))
payload=buffer.getvalue()
bootstrap=f'''import base64, hashlib, io, os, sys, zipfile
from pathlib import Path
blob=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
root=Path('/kaggle/working/Semantic_Lossless');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():(root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root);os.chdir(root);sys.path.insert(0,str(root/'src'))
snapshots=sorted({{p.resolve() for p in Path('/kaggle/input').rglob('sqgan-pinned-source.zip') if p.is_file()}})
if len(snapshots)==1:
    assert hashlib.sha256(snapshots[0].read_bytes()).hexdigest()=={hashlib.sha256(snapshot.read_bytes()).hexdigest()!r}
    with zipfile.ZipFile(snapshots[0]) as archive:
        for name in archive.namelist():(root/name).resolve().relative_to(root.resolve())
        archive.extractall(root)
else:
    # Kaggle automatically extracts uploaded ZIP datasets on some runtimes.
    import shutil
    sources=sorted({{p.parent.resolve() for p in Path('/kaggle/input').rglob('source_receipt.json') if (p.parent/'models/sqgan.py').is_file()}})
    assert len(sources)==1, {{'archives':list(map(str,snapshots)),'sources':list(map(str,sources)),'mounts':[str(p) for p in Path('/kaggle/input').iterdir()]}}
    shutil.copytree(sources[0],root/'.upstream/sqgan')
'''
body='''import ast, csv, json, random, time, types, subprocess
import numpy as np
import torch
from PIL import Image
from semantic_lossless.config import Config
from semantic_lossless.data import Cityscapes
from semantic_lossless.sources import fetch_source,source_path
from semantic_lossless.sqgan import build_sqgan
from semantic_lossless.artifacts import write_json
run=root/'Results/author_evaluation/sqgan_cityscapes_quarter'
run.mkdir(parents=True,exist_ok=False)
try:
    hardware=subprocess.run(['bash','-lc','command -v nvidia-smi && nvidia-smi -L; ls /dev/nvidia* 2>/dev/null'],capture_output=True,text=True)
    print(json.dumps({'stage':'environment','torch':torch.__version__,'torch_cuda':torch.version.cuda,'hardware_probe':hardware.stdout,'probe_stderr':hardware.stderr}),flush=True)
    assert torch.cuda.is_available(), 'Kaggle did not allocate a CUDA runtime; do not proceed as a GPU evaluation'
    write_json(run/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0)})
    fetch_source('sqgan')
    checkpoint_dirs=sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob('encoder_semantic.ckpt')})
    image_roots=sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob('leftImg8bit') if (p/'val').is_dir()})
    assert len(checkpoint_dirs)==len(image_roots)==1, {'weights':[str(p) for p in checkpoint_dirs],'data':[str(p) for p in image_roots]}
    config=Config('sqgan',dataset='cityscapes',device='cuda:0',data_root=str(image_roots[0]),checkpoint=str(checkpoint_dirs[0]),
        model_options={'label_format':'label_ids','image_fraction':.25,'semantic_fraction':.25})
    dataset=Cityscapes(config,'test')
    assert len(dataset)==500, f'Expected official 500-image val, got {len(dataset)}'
    codec=build_sqgan(config).eval()
    write_json(run/'config.json',config.as_dict());write_json(run/'model.json',codec.metadata)
    tree=ast.parse((source_path('sqgan')/'models/sqgan.py').read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward_all')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'native_forward_all','exec'),namespace)
    native_forward=types.MethodType(namespace['forward_all'],codec)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    torch.manual_seed(22);np.random.seed(22);random.seed(22)
    start=time.monotonic();total_psnr=0.
    with (run/'per_image.csv').open('w',newline='') as stream, torch.inference_mode():
        writer=csv.DictWriter(stream,fieldnames=['sample','psnr_png_db','source_bits','paper_payload_bits','packet_bpp','paper_bpp'])
        writer.writeheader()
        for index in range(len(dataset)):
            batch=dataset[index];image=batch['image'][None].cuda();semantic=batch['semantic'][None].cuda()
            if index==0:
                rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state();np_rng=np.random.get_state();py_rng=random.getstate()
                expected=native_forward((image-mean)/std,semantic)[0]*std+mean
                torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu_rng);np.random.set_state(np_rng);random.setstate(py_rng)
            result=codec(image,semantic)
            if index==0:
                error=float((result['reconstruction']-expected).abs().max())
                torch.testing.assert_close(result['reconstruction'],expected,rtol=1e-4,atol=2e-5)
                write_json(run/'native_equivalence.json',{'max_absolute_error':error,'input_shape':list(image.shape),'semantic_shape':list(semantic.shape),'scope':'one real image, native forward vs independent packet decoder'})
            reconstruction=result['reconstruction'].clamp(0,1).mul(255).round()/255
            original=image.clamp(0,1).mul(255).round()/255
            mse=(reconstruction-original).square().mean()
            assert torch.isfinite(reconstruction).all() and mse>0
            psnr=float(-10*torch.log10(mse));total_psnr+=psnr
            paper_bits=result['vq_index_bits']+result['mask_bitmap_bits']
            writer.writerow({'sample':batch['sample'],'psnr_png_db':psnr,'source_bits':result['source_bits'],
                'paper_payload_bits':paper_bits,'packet_bpp':result['source_bits']/(256*512),'paper_bpp':paper_bits/(256*512)})
            stream.flush()
            if index==0:
                Image.fromarray(reconstruction[0].mul(255).byte().permute(1,2,0).cpu().numpy()).save(run/'reconstruction.png')
                (run/'source_packet.sq01').write_bytes(result['source_packet'])
            if index%25==0: print(json.dumps({'stage':'evaluation','images':index+1,'total':len(dataset)}),flush=True)
    write_json(run/'metrics.json',{'status':'completed','images':len(dataset),'mean_psnr_png_db':total_psnr/len(dataset),
        'elapsed_seconds':time.monotonic()-start,'scope':'partial native-component evaluation at one source-codec rate; no channel model',
        'unmeasured':['LPIPS','MS-SSIM','InternImage mIoU'],'training_reproduced':False,
        'metric_protocol':'[0,1] RGB, clipped and rounded to 8-bit as saved sampling images; per-image RGB PSNR mean'})
    print((run/'metrics.json').read_text(),flush=True)
except Exception as error:
    write_json(run/'status.json',{'status':'failed','error':str(error)})
    raise
'''
script=bootstrap+body;compile(script,'sqgan_evaluation.py','exec')
folder=root/'Kaggle/sqgan_evaluation';folder.mkdir(parents=True,exist_ok=True)
(folder/'kernel.py').write_text(script)
metadata={'id':'satyapaladuguplus/sqgan-native-components-evaluation','title':'SQGAN Native Components Evaluation',
    'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,
    'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
    'docker_image_pinning_type':'original',
    'docker_image':'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
    'dataset_sources':['kavithak1388/cityscapes','satyapaladuguplus/semantic-lossless-private-sqgan-assets','satyapaladuguplus/semantic-lossless-private-sqgan-source'],
    'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built SQGAN native-component evaluation')
