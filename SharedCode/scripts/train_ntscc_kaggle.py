import csv, json, math, os, random, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim, ms_ssim

SEED = 20261016
EPOCHS = 150
BATCH_SIZE = 8
CROP = 256
IMAGES_PER_EPOCH = 128
AMP_ENABLED = True
SNR_DB = 10.0
LAMBDA = 0.18
ETA = 0.2
DIV2K = Path('/kaggle/input/notebooks/jagan028/div2k-dataset-generation-for-isr/Training/HR/hr_images')
KODAK = Path('/kaggle/input/datasets/sherylmehta/kodak-dataset')
SRC = Path('/kaggle/working/NTSCC_plus')
OUT = Path('/kaggle/working/jscc_results/NTSCC_plus')
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not SRC.exists():
    subprocess.run(['git','clone','https://github.com/wsxtyrdd/NTSCC_plus.git',str(SRC)],check=True)
REV=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
sys.path.insert(0,str(SRC))
# NATTEN >=0.21 removed the zero-valued attention-dropout constructor argument.
layer_file=SRC/'layer'/'layers.py'
source=layer_file.read_text()
source=source.replace('from natten import NeighborhoodAttention2D', '''from natten import NeighborhoodAttention2D as _NATTEN_NA2D
class NeighborhoodAttention2D(_NATTEN_NA2D):
    def __init__(self, *args, attn_drop=0.0, **kwargs):
        if attn_drop != 0.0: raise ValueError('Only released default attn_drop=0 is supported')
        super().__init__(*args, **kwargs)''')
layer_file.write_text(source)
from net.ntscc import NTSCC_plus
from channel.channel import Channel

TRAIN_PATHS=sorted(p for p in DIV2K.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
KODAK_PATHS=sorted(KODAK.rglob('kodim*.png'))
assert len(TRAIN_PATHS)==800,(len(TRAIN_PATHS),DIV2K)
assert len(KODAK_PATHS)==24,(len(KODAK_PATHS),KODAK)
assert torch.cuda.device_count() >= 1
print('revision',REV,'DIV2K',len(TRAIN_PATHS),'Kodak',len(KODAK_PATHS),'GPUs',torch.cuda.device_count(),flush=True)

class RandomCropDIV2K(Dataset):
    def __len__(self): return len(TRAIN_PATHS)
    def __getitem__(self,idx):
        im=Image.open(TRAIN_PATHS[idx]).convert('RGB')
        w,h=im.size
        if w<CROP or h<CROP: im=im.resize((max(w,CROP),max(h,CROP)),Image.Resampling.BICUBIC); w,h=im.size
        x=random.randrange(w-CROP+1); y=random.randrange(h-CROP+1)
        a=np.asarray(im.crop((x,y,x+CROP,y+CROP)),dtype=np.float32)/255.
        if random.random()<.5: a=a[:,::-1,:].copy()
        return torch.from_numpy(a.transpose(2,0,1).copy())

class RandomDIV2KSubsetSampler(torch.utils.data.Sampler):
    """Sample a fresh subset without replacement from all 800 DIV2K images each epoch."""
    def __init__(self, dataset, samples_per_epoch, seed):
        self.dataset=dataset; self.samples_per_epoch=samples_per_epoch; self.seed=seed; self.epoch=0
        assert 0 < samples_per_epoch <= len(dataset)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch); self.epoch+=1
        return iter(torch.randperm(len(self.dataset),generator=g)[:self.samples_per_epoch].tolist())
    def __len__(self): return self.samples_per_epoch

def rayleigh_channel_forward(self, channel_in):
    # Added extension: one quasi-static CN(0,1) coefficient per image, perfect receiver CSI/equalization.
    b=channel_in.shape[0]; shape=(b,1)
    hr=torch.randn(shape,device=channel_in.device,dtype=channel_in.real.dtype)/math.sqrt(2)
    hi=torch.randn(shape,device=channel_in.device,dtype=channel_in.real.dtype)/math.sqrt(2)
    h=torch.complex(hr,hi)
    sigma=math.sqrt(1.0/(2*10**(self.chan_param/10)))
    nr=torch.randn_like(channel_in.real)*sigma; ni=torch.randn_like(channel_in.real)*sigma
    n=torch.complex(nr,ni)
    return h.conj()*(h*channel_in+n)/h.abs().square().clamp_min(1e-4)

def make_model(channel_name):
    cfg=SimpleNamespace(eta=ETA,multiple_rate=[16,32,48,64,80,96,102,118,134,160,186,192,208,224,240,256,272,288,304,320],channel_type='awgn' if channel_name=='AWGN' else 'rayleigh',SNR=SNR_DB,device=torch.device('cuda:0'),logger=False)
    if channel_name=='Rayleigh':
        Channel.channel_forward=rayleigh_channel_forward
    return NTSCC_plus(cfg).cuda()

def criterion(out, target):
    b,_,h,w=target.shape; pixels=b*h*w
    py=torch.log(out['likelihoods']['y'].float()).sum()/(-math.log(2)*pixels)
    pz=torch.log(out['likelihoods']['z'].float()).sum()/(-math.log(2)*pixels)
    mse=F.mse_loss(out['x_hat'],target); mse_ntc=F.mse_loss(out['x_hat_ntc'],target)
    return LAMBDA*255**2*(mse+mse_ntc)+py+pz, mse, py+pz

def count_flops(model, channel):
    from torch.profiler import profile, ProfilerActivity
    x=torch.rand(1,3,CROP,CROP,device='cuda')
    first=Image.open(KODAK_PATHS[0]); iw,ih=first.size; first.close()
    x_inf=torch.rand(1,3,ih,iw,device='cuda')
    # torch.profiler's FLOP estimator covers convolution and matrix multiplication operators.
    # NATTEN fused kernels may not expose their internal arithmetic and can be under-counted.
    model.train(); model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],with_flops=True) as prof:
        out=model(x); loss,_,_=criterion(out,x); (loss+model.ntc.aux_loss()).backward()
    train=sum(int(e.flops or 0) for e in prof.key_averages())
    model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad(),profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],with_flops=True) as prof:
        model(x_inf)
    infer=sum(int(e.flops or 0) for e in prof.key_averages())
    torch.cuda.synchronize(); model.train()
    return train,infer

def evaluate(model,channel_name):
    model.eval(); rows=[]
    with torch.no_grad():
        for path in KODAK_PATHS:
            a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255.
            x=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).cuda()
            out=model(x); pred=out['x_hat'].clamp(0,1)
            psnr=float((-10*torch.log10(F.mse_loss(pred,x))).item())
            ss=float(ssim(pred,x,data_range=1.,size_average=True).item())
            mss=float(ms_ssim(pred,x,data_range=1.,size_average=True).item())
            uses=float(out['k'].sum().item())
            h,w=x.shape[-2:]
            rows.append({'model':'NTSCC_plus','channel':channel_name,'snr_db':SNR_DB,'image':path.stem,'psnr_db':psnr,'ssim':ss,'msssim':mss,'complex_channel_uses':uses,'cbr':uses/(3*h*w)})
    with (OUT/f'kodak_{channel_name.lower()}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return rows

summaries=[]
for channel_name in ('AWGN','Rayleigh'):
    model=make_model(channel_name)
    # DataParallel uses both available Kaggle T4 GPUs; batch 4 gives two 256px crops per GPU.
    model=torch.nn.DataParallel(model) if torch.cuda.device_count()>1 else model
    base=model.module if isinstance(model,torch.nn.DataParallel) else model
    flops=count_flops(base,channel_name)
    print('FLOPs training_step,inference',channel_name,flops,flush=True)
    g_params={p for n,p in base.named_parameters() if not n.endswith('.quantiles')}
    aux_params={p for n,p in base.named_parameters() if n.endswith('.quantiles')}
    optimizer=torch.optim.Adam(g_params,lr=1e-4)
    aux_optimizer=torch.optim.Adam(aux_params,lr=1e-3)
    scaler=torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    train_dataset=RandomCropDIV2K()
    sampler=RandomDIV2KSubsetSampler(train_dataset,IMAGES_PER_EPOCH,SEED)
    loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=sampler,num_workers=4,pin_memory=True,drop_last=True)
    run=OUT/channel_name; run.mkdir(exist_ok=True)
    metric_path=run/'epoch_metrics.csv'; rows=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); t0=time.time(); sums=np.zeros(3,dtype=np.float64); count=0
        for x in loader:
            x=x.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True); aux_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16,enabled=AMP_ENABLED):
                out=model(x); loss,mse,bpp=criterion(out,x)
            scaler.scale(loss.float()).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            with torch.amp.autocast('cuda',enabled=False):
                aux=base.ntc.aux_loss()
            aux.backward(); aux_optimizer.step()
            sums += [loss.detach().item(),mse.detach().item(),bpp.detach().item()]; count+=1
        row={'model':'NTSCC_plus','channel':channel_name,'epoch':epoch,'epochs_target':EPOCHS,'snr_db':SNR_DB,'train_loss':sums[0]/count,'train_psnr_db':-10*math.log10(max(sums[1]/count,1e-12)),'estimated_bpp':sums[2]/count,'epoch_seconds':time.time()-t0,'train_flops_per_step':flops[0],'inference_flops_per_image':flops[1],'batch_size':BATCH_SIZE,'crop':CROP,'images_in_epoch':IMAGES_PER_EPOCH,'amp_fp16':AMP_ENABLED,'source_commit':REV}
        rows.append(row)
        with metric_path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=row.keys()); w.writeheader(); w.writerows(rows)
        torch.save({'epoch':epoch,'state_dict':base.state_dict(),'optimizer':optimizer.state_dict(),'aux_optimizer':aux_optimizer.state_dict(),'source_commit':REV},run/'latest.pt')
        print(json.dumps(row),flush=True)
    kodak=evaluate(base,channel_name)
    summary={'model':'NTSCC_plus','channel':channel_name,'epochs':EPOCHS,'snr_db':SNR_DB,'kodak_psnr_db':float(np.mean([r['psnr_db'] for r in kodak])),'kodak_ssim':float(np.mean([r['ssim'] for r in kodak])),'kodak_msssim':float(np.mean([r['msssim'] for r in kodak])),'channel_uses_per_image_mean':float(np.mean([r['complex_channel_uses'] for r in kodak])),'cbr':float(np.mean([r['cbr'] for r in kodak])),'train_flops_per_step':flops[0],'inference_flops_per_image':flops[1],'source_commit':REV}
    summaries.append(summary)
    with (OUT/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=summary.keys()); w.writeheader(); w.writerows(summaries)
    print('SUMMARY',json.dumps(summary),flush=True)
    del model,base; torch.cuda.empty_cache()
print('NTSCC_PLUS_TRAINING_COMPLETE',json.dumps(summaries),flush=True)
