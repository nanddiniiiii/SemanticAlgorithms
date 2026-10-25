"""Package native-component Cityscapes evaluation for account B."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
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
bootstrap=f'''import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
blob=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
root=Path('/kaggle/working/Semantic_Lossless');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():(root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root);os.chdir(root);sys.path.insert(0,str(root/'src'))
provenance={{'source_bundle_sha256':hashlib.sha256(blob).hexdigest(),
    'git_revision':{subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!r},
    'working_tree_dirty':{bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())!r},
    'python':sys.version,'interpreter':sys.executable,'scope':'native SQ-GAN evaluation with explicitly separate added metrics'}}
(root/'source_bundle.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({{'stage':'source_preflight',**provenance}}),flush=True)
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
body='''import ast, csv, json, random, time, types, subprocess, importlib
import numpy as np
import torch
from PIL import Image
from semantic_lossless.config import Config
from semantic_lossless.data import Cityscapes
from semantic_lossless.sources import fetch_source,source_path
from semantic_lossless.sqgan import build_sqgan
from semantic_lossless.codecs import upstream_import
from semantic_lossless.artifacts import write_json
from semantic_lossless.metrics import quality, profile_forward
from semantic_lossless.compute import registered_kernel_flops
from torchvision import transforms
run=root/'Results/author_evaluation/sqgan_cityscapes_quarter_native_save'
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
    # Preserve sample.py's normalized-space clamp and uint8 truncation exactly.
    # This differs from conventional [0,1] clipping and rounding in version 6.
    inverse=transforms.Compose([transforms.Normalize(mean=[-.485/.229,-.456/.224,-.406/.225],std=[1/.229,1/.224,1/.225])])
    upper=mean.new_tensor([2.24,2.42,2.64]).view(1,3,1,1)
    lower=mean.new_tensor([-2.11,-2.03,-1.80]).view(1,3,1,1)
    def native_pixels(value):
        normalized=torch.max(torch.min(value,upper),lower)
        array=inverse(normalized).squeeze(0).cpu().numpy().transpose(1,2,0)
        return (array*255).astype(np.uint8)
    # Independently validate the raw-label conversion and RGB transforms against
    # the pinned author validation dataset on the first real sample.
    first_path=dataset.paths[0]
    relative=first_path.relative_to(dataset.image_root)
    raw_path=dataset.label_root/relative.parent/first_path.name.replace('_leftImg8bit.png','_gtFine_labelIds.png')
    from semantic_lossless.data import cityscapes_label_lut
    with Image.open(raw_path) as handle:
        raw=np.asarray(handle,dtype=np.int64)
        mapped=np.asarray(cityscapes_label_lut(),dtype=np.uint8)[raw]
        mapped[mapped==19]=255
        label_copy=run/'first_native_trainIds.png';Image.fromarray(mapped).save(label_copy)
    with upstream_import(source_path('sqgan'),('data',)):
        native_images=importlib.import_module('data.cityscapes_base').ImagePaths(
            split='val',is_val=True,paths=[str(first_path)],elements_input={'file_path_':[str(first_path)],'ssm_path':[str(label_copy)]},
            size=256,ratio_dim=2,relevant_obj=[0]*19)
        reference_input=native_images[0]
    first=dataset[0]
    torch.testing.assert_close((first['image']-mean[0].cpu())/std[0].cpu(),reference_input['image'],rtol=0,atol=1e-6)
    torch.testing.assert_close(first['semantic'],reference_input['semantic'],rtol=0,atol=0)
    write_json(run/'native_preprocessing.json',{'status':'passed','scope':'one real image: native RGB validation transform and one-hot labels; official raw-ID conversion'})
    torch.manual_seed(22);np.random.seed(22);random.seed(22)
    start=time.monotonic();total_psnr=0.;total_native_range1=0.;total_native_auto=0.
    fidelity_totals={key:0. for key in ['ssim','ms_ssim','lpips','exact_uint8_sample_fraction','exact_uint8_spatial_pixel_fraction']}
    confusion=torch.zeros(20,20,dtype=torch.int64,device='cuda')
    from pytorch_msssim import ms_ssim
    with (run/'per_image.csv').open('w',newline='') as stream, torch.inference_mode():
        writer=csv.DictWriter(stream,fieldnames=['sample','psnr_png_db','psnr_native_saved_range1_db','psnr_native_saved_auto_range_db','source_bits','paper_payload_bits','packet_bpp','paper_bpp',*fidelity_totals])
        writer.writeheader()
        for index in range(len(dataset)):
            batch=dataset[index];image=batch['image'][None].cuda();semantic=batch['semantic'][None].cuda()
            rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state();np_rng=np.random.get_state();py_rng=random.getstate()
            native_result=native_forward((image-mean)/std,semantic)
            expected=native_result[0]*std+mean
            if index==0:
                torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu_rng);np.random.set_state(np_rng);random.setstate(py_rng)
                result=codec(image,semantic)
            else:
                # Count the actual selected positions from both native branches.
                # Packet length uses separate byte padding for the two ID arrays.
                ni=native_result[6]['sample_index'].numel();ns=native_result[7]['sample_index'].numel()
                packet_bits=128+1024+8*((ni*10+7)//8+(ns*10+7)//8)
                result={'source_bits':packet_bits,'vq_index_bits':(ni+ns)*10,'mask_bitmap_bits':1024,'reconstruction':expected}
            if index==0:
                error=float((result['reconstruction']-expected).abs().max())
                torch.testing.assert_close(result['reconstruction'],expected,rtol=1e-4,atol=2e-5)
                write_json(run/'native_equivalence.json',{'max_absolute_error':error,'input_shape':list(image.shape),'semantic_shape':list(semantic.shape),'scope':'one real image, native forward vs independent packet decoder'})
            reconstruction=expected.clamp(0,1).mul(255).round()/255
            original=image.clamp(0,1).mul(255).round()/255
            mse=(reconstruction-original).square().mean()
            assert torch.isfinite(reconstruction).all() and mse>0
            psnr=float(-10*torch.log10(mse));total_psnr+=psnr
            fidelity=quality(original,reconstruction,use_lpips=True)
            fidelity['ms_ssim']=float(ms_ssim(original,reconstruction,data_range=1))
            for key in fidelity_totals:fidelity_totals[key]+=fidelity[key]
            ground_truth=semantic.argmax(1);predicted=native_result[3].argmax(1)
            valid=ground_truth<19
            confusion+=torch.bincount((ground_truth[valid]*20+predicted[valid]).reshape(-1),minlength=400).reshape(20,20)
            native_reconstructed=native_pixels(native_result[0])
            native_original=native_pixels((image-mean)/std)
            native_a=native_original.astype(np.float64)/255
            native_b=native_reconstructed.astype(np.float64)/255
            native_mse=float(np.mean((native_a-native_b)**2))
            native_range1=float(-10*np.log10(native_mse))
            native_auto=float(10*np.log10((native_a.max()-native_a.min())**2/native_mse))
            total_native_range1+=native_range1;total_native_auto+=native_auto
            paper_bits=result['vq_index_bits']+result['mask_bitmap_bits']
            writer.writerow({'sample':batch['sample'],'psnr_png_db':psnr,'psnr_native_saved_range1_db':native_range1,'psnr_native_saved_auto_range_db':native_auto,'source_bits':result['source_bits'],
                'paper_payload_bits':paper_bits,'packet_bpp':result['source_bits']/(256*512),'paper_bpp':paper_bits/(256*512),
                **{key:fidelity[key] for key in fidelity_totals}})
            stream.flush()
            if index==0:
                Image.fromarray(native_reconstructed).save(run/'reconstruction.png')
                (run/'source_packet.sq01').write_bytes(result['source_packet'])
            if index%25==0: print(json.dumps({'stage':'evaluation','images':index+1,'total':len(dataset)}),flush=True)
    valid_confusion=confusion[:19,:].double()
    intersections=valid_confusion[:,:19].diag()
    unions=valid_confusion.sum(1)+valid_confusion[:,:19].sum(0)-intersections
    class_ious=torch.where(unions>0,intersections/unions,torch.nan)
    write_json(run/'semantic_branch.json',{'semantic_branch_miou':float(class_ious.nanmean()),
        'class_ious':class_ious.tolist(),'confusion_gt19_pred20':valid_confusion.cpu().tolist(),
        'scope':'decoded semantic branch vs supplied paired labels; NOT segmentation of reconstructed RGB or paper InternImage mIoU',
        'ignore_gt_class':19,'absent_class_policy':'exclude zero-union classes','paired_labels_available_at_encoder':True})
    import semantic_lossless.metrics as metric_module
    if hasattr(metric_module,'_LPIPS'):del metric_module._LPIPS
    del native_result,result,expected,reconstruction,original,image,semantic
    torch.cuda.empty_cache()
    first=dataset[0];profile_image=first['image'][None].cuda();profile_semantic=first['semantic'][None].cuda()
    class NativeFull(torch.nn.Module):
        def __init__(self):super().__init__();self.core=codec
        def forward(self,value):return native_forward((value-mean)/std,profile_semantic)[0]
    profile=profile_forward(NativeFull(),profile_image,warmup=3,repeats=20)
    profile['registered_kernel_count']=registered_kernel_flops(NativeFull(),profile_image)
    profile['flops_scope']='ten-component native forward_all at fixed .25/.25; includes both semantic and RGB branches, not packet coding or GAN training'
    profile['semantic_input_shape']=list(profile_semantic.shape)
    write_json(run/'profile_full.json',profile)
    write_json(run/'metrics.json',{'status':'completed','images':len(dataset),'mean_psnr_png_db':total_psnr/len(dataset),
        'mean_psnr_native_saved_range1_db':total_native_range1/len(dataset),'mean_psnr_native_saved_auto_range_db':total_native_auto/len(dataset),
        'elapsed_seconds':time.monotonic()-start,'scope':'native-component evaluation at one source-codec rate with added conventional RGB fidelity and semantic-branch metrics; no channel model',
        'unmeasured':['InternImage mIoU','separate encoder and decoder GPU profiles'],'training_reproduced':False,
        **{'mean_'+key:value/len(dataset) for key,value in fidelity_totals.items()},
        'metric_protocol':'native forward; exact native normalized-space clamp and NumPy uint8 truncation; fixed-range and auto-range RGB PSNR formulas; conventional rounded PSNR also recorded'})
    print((run/'metrics.json').read_text(),flush=True)
except Exception as error:
    write_json(run/'status.json',{'status':'failed','error':str(error)})
    raise
'''
body=body.replace("run.mkdir(parents=True,exist_ok=False)","run.mkdir(parents=True,exist_ok=False)\nsubprocess.run([sys.executable,'-m','pip','install','--no-deps','lpips==0.1.4','pytorch-msssim==1.0.0'],check=True)")
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
