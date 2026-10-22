"""Package a complete posterior reconstruction pilot, not a paper benchmark."""
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
payload=buffer.getvalue()
bootstrap=f'''import base64, hashlib, io, os, sys, zipfile
from pathlib import Path
blob=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
root=Path('/kaggle/working/Semantic_Lossless')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for name in archive.namelist():
        (root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root)
os.chdir(root)
sys.path.insert(0,str(root/'src'))
'''
body='''import json, time, traceback
import numpy as np
import torch
from PIL import Image
from semantic_lossless.config import Config
from semantic_lossless.artifacts import write_json, sha256
from semantic_lossless.sources import fetch_source
from semantic_lossless.diffcom import build_diffcom
assert torch.cuda.is_available()
run=root/'Results/implementation_checks/diffcom_gpu'
run.mkdir(parents=True,exist_ok=False)
try:
    prior=sorted({p.resolve() for p in Path('/kaggle/input').rglob('256x256_diffusion_uncond.pt') if p.is_file()})
    base=sorted({p.resolve() for p in Path('/kaggle/input').rglob('ADJSCC_C*2.pth.tar') if p.is_file()})
    assert len(prior)==len(base)==1, {'prior':[str(p) for p in prior],'base':[str(p) for p in base]}
    assert sha256(prior[0])=='a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1'
    assert sha256(base[0])=='c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'
    fetch_source('diffcom')
    torch.manual_seed(22);np.random.seed(22)
    config=Config('diffcom',device='cuda:0',checkpoint=str(prior[0]),snr_db=10,
        model_options={'base_checkpoint':str(base[0]),'sampling_steps':1000,
                       'adaptive_start':True,'conditioning_method':'hifi_diffcom'})
    write_json(run/'config.json',config.as_dict())
    print('BUILDING_NATIVE_PRIOR_AND_BASE',flush=True)
    codec=build_diffcom(config)
    write_json(run/'model.json',codec.metadata)
    images=sorted(Path('/kaggle/input').rglob('kodim01.png'))
    assert len(images)==1
    with Image.open(images[0]) as handle:
        rgb=handle.convert('RGB');left=(rgb.width-256)//2;top=(rgb.height-256)//2
        rgb=rgb.crop((left,top,left+256,top+256))
        image=torch.from_numpy(np.asarray(rgb,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0).cuda()
    print('STARTING_FULL_POSTERIOR_PILOT',flush=True)
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    output=codec(image)
    torch.cuda.synchronize()
    assert output['reconstruction'].shape==image.shape
    assert torch.isfinite(output['reconstruction']).all()
    reconstruction=output['reconstruction'][0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(reconstruction).save(run/'reconstruction.png')
    rgb.save(run/'input_crop.png')
    write_json(run/'status.json',{'status':'completed','scope':'one Kodak center crop complete posterior runtime check; not matched paper benchmark',
       'elapsed_seconds':time.monotonic()-start,'shape':list(image.shape),
       'denoiser_evaluations':output['denoiser_evaluations'],'complex_uses':int(output['complex_uses'].item()),
       'side_information_bits':output['side_information_bits'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    print((run/'status.json').read_text(),flush=True)
except Exception as error:
    write_json(run/'status.json',{'status':'failed','error':str(error),'scope':'runtime pilot'})
    traceback.print_exc()
    raise
'''
script=bootstrap+body
compile(script,'diffcom_gpu_pilot.py','exec')
folder=root/'Kaggle/diffcom_pilot';folder.mkdir(parents=True,exist_ok=True)
(folder/'kernel.py').write_text(script)
metadata={'id':'satyapaladugu/diffcom-full-posterior-pilot','title':'DiffCom Full Posterior Pilot',
    'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,
    'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':['liushuaiqian/256x256-diffusion-uncond','sherylmehta/kodak-dataset',
                       'satyapaladugu/semantic-lossless-private-adjscc-assets'],
    'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built full posterior runtime pilot')
