"""Build the native CUDA-extension and checkpoint compatibility pilot."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
script='''import hashlib, importlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path('/kaggle/working/mamba_native_pilot');ROOT.mkdir(exist_ok=True)
def record(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,default=str))
def checked(command,cwd=None,timeout=1800):
    print(json.dumps({'stage':'command','command':command}),flush=True)
    subprocess.run(command,cwd=cwd,check=True,timeout=timeout)
try:
    assert torch.cuda.is_available(),'Actual CUDA hardware is required'
    record('environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'scope':'native extension/checkpoint runtime pilot; no training or quality claim'})
    sources=sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob('source_receipt.json') if (p.parent/'models/vmamba.py').is_file()})
    assert len(sources)==1, list(map(str,sources))
    source=ROOT/'source';shutil.copytree(sources[0],source)
    receipt=json.loads((source/'source_receipt.json').read_text())
    assert receipt['commit']=='8323f9799f48c0efddce0521100d5c4e12ebf699'
    for item in receipt['files']:
        if '__pycache__' in Path(item['path']).parts or item['path'].endswith('.pyc'):
            continue  # Kaggle omits upstream Python-3.9 bytecode; execute verified source instead.
        assert hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256']
    checked(['bash','-lc','command -v nvcc && nvcc --version'])
    checked([sys.executable,'-m','pip','install','--no-deps','timm==0.4.12','fvcore','iopath','yacs','termcolor','einops','ninja'])
    os.environ['MAX_JOBS']='2'
    # Build the exact author kernels first; do not replace state-space scans.
    for folder in ['adaptive_selective_scan','selective_scan']:
        checked([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-v','.'],cwd=source/folder)
    sys.path.insert(0,str(source))
    vmamba=importlib.import_module('models.vmamba')
    # The native no-CSI branch refers to this compiled extension although its
    # top-level import is commented out. Supply that import, changing no math.
    vmamba.selective_scan_cuda_core=importlib.import_module('selective_scan_cuda_core')
    channel_class=importlib.import_module('models.channel').Channel
    expected={
      'awgn_encoder.pt':'a9f55f6a5f85c9034cc1e7d1bb9974e6fb377b112a69215d9e70c739125918c5',
      'awgn_decoder.pt':'cf21380246bb1d77b8b62b4dff39640a2c64dc125cd289b8eca961f227cc0316',
      'rayleigh_encoder.pt':'6282cdea3f2e89813d00c25ccde7babbef5d741e3a2a393dabfdea2cfe7d68b9',
      'rayleigh_decoder.pt':'bb394b67804a9de410e21202687ff7579429f5e798c01b057903c894e14e0423'}
    paths={}
    for name,digest in expected.items():
        matches=sorted({p.resolve() for p in Path('/kaggle/input').rglob(name)})
        assert len(matches)==1
        assert hashlib.sha256(matches[0].read_bytes()).hexdigest()==digest
        paths[name]=matches[0]
    pictures=sorted({p.resolve() for p in Path('/kaggle/input').rglob('kodim01.png')})
    assert len(pictures)==1
    for channel_type in ['awgn','rayleigh']:
        print('LOADING_AUTHOR_FULL_MODULES_'+channel_type,flush=True)
        # These hashes identify the explicitly requested, trusted author module
        # artifacts. Full module loading is how native run/eval.py loads them.
        encoder=torch.load(paths[channel_type+'_encoder.pt'],map_location='cpu',weights_only=False).cuda().eval()
        decoder=torch.load(paths[channel_type+'_decoder.pt'],map_location='cpu',weights_only=False).cuda().eval()
        config=encoder.config
        record(channel_type+'_config.json',{'native_config':str(config),
            'encoder_type':type(encoder).__module__+'.'+type(encoder).__name__,
            'decoder_type':type(decoder).__module__+'.'+type(decoder).__name__,
            'filename_tag_resolution':256,'stored_resolution':config.DATA.IMG_SIZE,
            'stored_channel':config.CHANNEL.TYPE,'stored_adaptation':config.CHANNEL.ADAPTIVE})
        assert config.CHANNEL.TYPE==channel_type,'Stored channel conflicts with artifact tag'
        size=int(config.DATA.IMG_SIZE)
        assert size in [128,256,512],f'Unverified stored input resolution {size}'
        with Image.open(pictures[0]) as handle:
            rgb=handle.convert('RGB');assert min(rgb.size)>=size
            x=(rgb.width-size)//2;y=(rgb.height-size)//2
            rgb=rgb.crop((x,y,x+size,y+size))
            image=torch.from_numpy(np.asarray(rgb,dtype=np.float32).copy()/255).permute(2,0,1)[None].cuda()
        channel=channel_class(config)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.no_grad():
            feature=encoder(image,10)
            received,power,h=channel.forward(feature,10)
            if channel_type=='rayleigh':received=torch.conj(h)*received/(torch.abs(h)**2+0.1)
            received=torch.cat((torch.real(received),torch.imag(received)),dim=2)*torch.sqrt(power)
            reconstructed=decoder(received,10)
            torch.cuda.synchronize()
        assert reconstructed.shape==image.shape and torch.isfinite(reconstructed).all()
        record(channel_type+'_runtime.json',{'status':'passed','scope':'one explicit Kodak center crop native runtime; not paper benchmark',
            'input_shape':list(image.shape),'latent_shape':list(feature.shape),'output_shape':list(reconstructed.shape),
            'snr_db':10,'complex_uses':feature.numel()//2,'cbr':feature.numel()/2/image.numel(),
            'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'compatibility_change':'restore commented import of selective_scan_cuda_core; kernels unchanged'})
        # Export data-only checkpoints for later explicit architecture loading.
        torch.save({'state_dict':encoder.state_dict(),'native_config':str(config)},ROOT/(channel_type+'_encoder_state.pt'))
        torch.save({'state_dict':decoder.state_dict(),'native_config':str(decoder.config)},ROOT/(channel_type+'_decoder_state.pt'))
        Image.fromarray(reconstructed[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()).save(ROOT/(channel_type+'_reconstruction.png'))
        del encoder,decoder,feature,received,reconstructed
        torch.cuda.empty_cache()
    record('status.json',{'status':'completed','scope':'native CUDA extension and two checkpoint variants; training not reproduced'})
except Exception as error:
    record('status.json',{'status':'failed','error':str(error)})
    traceback.print_exc();raise
'''
compile(script,'mamba_native_pilot.py','exec')
folder=root/'Kaggle/mamba_pilot';folder.mkdir(parents=True,exist_ok=True)
(folder/'kernel.py').write_text(script)
metadata={'id':'satyapaladugu/mambajscc-native-cuda-pilot','title':'MambaJSCC Native CUDA Pilot',
    'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,
    'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
    'docker_image_pinning_type':'original','docker_image':'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
    'dataset_sources':['satyapaladugu/semantic-lossless-private-mamba-assets','sherylmehta/kodak-dataset'],
    'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built native Mamba CUDA/checkpoint pilot')
