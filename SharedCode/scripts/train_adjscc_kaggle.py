import os,sys,json,csv,time,random,math,subprocess
from pathlib import Path
import numpy as np
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import tensorflow as tf
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

DIV2K=Path('/kaggle/input/notebooks/jagan028/div2k-dataset-generation-for-isr/Training/HR/hr_images')
KODAK=Path('/kaggle/input/datasets/sherylmehta/kodak-dataset')
SRC=Path('/kaggle/working/ADJSCC')
if not SRC.exists(): subprocess.run(['git','clone','https://github.com/alexxu1988/ADJSCC.git',str(SRC)],check=True)
# The author README identifies this vendored tensorflow_compression directory as macOS-only.
VENDORED_TFC=SRC/'tensorflow_compression'
if VENDORED_TFC.exists(): VENDORED_TFC.rename(SRC/'tensorflow_compression_macos_only')
REV=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
sys.path.insert(0,str(SRC))
from util_module import Attention_Encoder, Attention_Decoder
from util_channel import Channel

TRAIN_PATHS=sorted(p for p in DIV2K.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
KODAK_PATHS=sorted(KODAK.glob('kodim*.png'))
assert len(TRAIN_PATHS)==800,(len(TRAIN_PATHS),DIV2K)
assert len(KODAK_PATHS)==24,(len(KODAK_PATHS),KODAK)
gpus=tf.config.list_physical_devices('GPU')
print('ADJSCC commit',REV,'DIV2K',len(TRAIN_PATHS),'Kodak',len(KODAK_PATHS),'GPUs',gpus,flush=True)
assert gpus, 'No Kaggle GPU visible to isolated TensorFlow runtime'

OUT=Path('/kaggle/working/jscc_results/ADJSCC'); OUT.mkdir(parents=True,exist_ok=True)
EPOCHS=150; BATCH=16; CROP=128; SNR=10.; TCN=16; SEED=20261016
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
strategy=tf.distribute.MirroredStrategy()
print('TF replicas',strategy.num_replicas_in_sync,flush=True)

def build(channel):
    image=Input(shape=(None,None,3),name='image')
    snr=Input(shape=(1,),name='snr_db')
    norm=Lambda(lambda z:z/255.,name='normal')(image)
    encoded=Attention_Encoder(norm,snr,TCN)
    if channel=='AWGN':
        z=Channel(channel_type='awgn')(encoded,snr)
        decoded=Attention_Decoder(z,snr)
        model=Model([image,snr],Lambda(lambda q:q*255.,name='denormal')(decoded))
    elif channel=='Rayleigh':
        hr=Input(shape=(1,),name='h_real'); hi=Input(shape=(1,),name='h_imag')
        z=Channel(channel_type='slow_fading_eq')(encoded,snr,hr,hi)
        decoded=Attention_Decoder(z,snr)
        model=Model([image,snr,hr,hi],Lambda(lambda q:q*255.,name='denormal')(decoded))
    else: raise ValueError(channel)
    return model

def make_data(channel):
    specs=[tf.TensorSpec((CROP,CROP,3),tf.float32),tf.TensorSpec((1,),tf.float32)]
    if channel=='Rayleigh': specs += [tf.TensorSpec((1,),tf.float32),tf.TensorSpec((1,),tf.float32)]
    def gen():
        rng=np.random.default_rng(SEED)
        while True:
            for ix in rng.permutation(len(TRAIN_PATHS)):
                p=TRAIN_PATHS[int(ix)]
                a=np.asarray(Image.open(p).convert('RGB'))
                h,w=a.shape[:2]
                if h<CROP or w<CROP:
                    a=np.asarray(Image.fromarray(a).resize((max(w,CROP),max(h,CROP)),Image.Resampling.BICUBIC)); h,w=a.shape[:2]
                y=int(rng.integers(0,h-CROP+1)); x=int(rng.integers(0,w-CROP+1))
                crop=a[y:y+CROP,x:x+CROP].astype(np.float32)
                if rng.random()<.5: crop=crop[:,::-1].copy()
                example=[crop,np.array([SNR],np.float32)]
                if channel=='Rayleigh':
                    example.extend([np.array([rng.normal(0,1/np.sqrt(2))],np.float32),np.array([rng.normal(0,1/np.sqrt(2))],np.float32)])
                yield tuple(example),crop
    return tf.data.Dataset.from_generator(gen,output_signature=(tuple(specs),tf.TensorSpec((CROP,CROP,3),tf.float32))).batch(BATCH,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def flops_for(concrete):
    opts=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    stats=tf.compat.v1.profiler.profile(graph=concrete.graph,options=opts)
    return int(stats.total_float_ops) if stats is not None else None

def profile_flops(model,channel):
    x=tf.zeros((1,CROP,CROP,3),tf.float32); snr=tf.fill((1,1),SNR)
    hs=[tf.zeros((1,1),tf.float32),tf.ones((1,1),tf.float32)] if channel=='Rayleigh' else []
    specs=[tf.TensorSpec((1,CROP,CROP,3),tf.float32),tf.TensorSpec((1,1),tf.float32)] + ([tf.TensorSpec((1,1),tf.float32)]*2 if channel=='Rayleigh' else [])
    @tf.function
    def infer(*args): return model(list(args),training=False)
    inference=flops_for(infer.get_concrete_function(*specs))
    opt=Adam(1e-4)
    @tf.function
    def train_step(*args):
        target=args[0]; inputs=list(args[1:])
        with tf.GradientTape() as tape:
            pred=model(inputs,training=True); loss=tf.reduce_mean(tf.square(pred-target))
        grads=tape.gradient(loss,model.trainable_variables); return loss
    # Build the traced one-step training graph on a single representative sample.
    train_specs=[tf.TensorSpec((1,CROP,CROP,3),tf.float32)] + specs
    training=flops_for(train_step.get_concrete_function(*train_specs))
    return training,inference

class EpochCSV(tf.keras.callbacks.Callback):
    def __init__(self,path,channel,flops): self.path=path; self.channel=channel; self.flops=flops; self.rows=[]
    def on_epoch_begin(self,epoch,logs=None): self.t0=time.time()
    def on_epoch_end(self,epoch,logs=None):
        r={'model':'ADJSCC','channel':self.channel,'epoch':epoch+1,'epochs_target':EPOCHS,'snr_db':SNR,'train_loss_mse_255':float((logs or {}).get('loss',float('nan'))),'epoch_seconds':time.time()-self.t0,'train_flops_per_step':self.flops[0],'inference_flops_per_image':self.flops[1],'tcn':TCN,'nominal_cbr':TCN/96}
        self.rows.append(r)
        with self.path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=r.keys()); w.writeheader(); w.writerows(self.rows)
        self.model.save_weights(self.path.parent/'latest.weights.h5')
        if (epoch+1)%10==0 or epoch==0: print(json.dumps(r),flush=True)

def eval_kodak(model,channel,run):
    rows=[]
    for p in KODAK_PATHS:
        ref=np.asarray(Image.open(p).convert('RGB')).astype(np.float32)[None,...]
        inputs=[ref,np.array([[SNR]],np.float32)]
        if channel=='Rayleigh':
            rng=np.random.default_rng(SEED+int(p.stem[-2:])); inputs.extend([np.array([[rng.normal(0,1/np.sqrt(2))]],np.float32),np.array([[rng.normal(0,1/np.sqrt(2))]],np.float32)])
        pred=np.clip(model.predict(inputs,verbose=0),0,255)/255.; target=ref/255.
        y=tf.convert_to_tensor(target); yh=tf.convert_to_tensor(pred)
        psnr=float(tf.reduce_mean(tf.image.psnr(yh,y,max_val=1.)).numpy())
        ss=float(tf.reduce_mean(tf.image.ssim(yh,y,max_val=1.)).numpy())
        mss=float(tf.reduce_mean(tf.image.ssim_multiscale(yh,y,max_val=1.)).numpy())
        h,w=ref.shape[1:3]; uses=TCN*(h//4)*(w//4)//2
        rows.append({'model':'ADJSCC','channel':channel,'snr_db':SNR,'image':p.stem,'psnr_db':psnr,'ssim':ss,'msssim':mss,'real_symbols':TCN*(h//4)*(w//4),'complex_channel_uses':uses,'cbr':uses/(3*h*w)})
    file=OUT/f'kodak_{channel.lower()}.csv'
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows

summaries=[]
for channel in ['AWGN','Rayleigh']:
    tf.keras.backend.clear_session()
    with strategy.scope():
        model=build(channel); model.compile(Adam(1e-4),loss='mse')
    # Fail before 150 epochs if the official TFC custom SignalConv/GDN path is incompatible.
    first=next(iter(make_data(channel))); smoke=model(first[0],training=True); assert tuple(smoke.shape)==(BATCH,CROP,CROP,3)
    flops=profile_flops(model,channel); print('profiled flops',channel,flops,flush=True)
    run=OUT/channel; run.mkdir(exist_ok=True)
    cb=EpochCSV(run/'epoch_metrics.csv',channel,flops)
    model.fit(make_data(channel),epochs=EPOCHS,steps_per_epoch=len(TRAIN_PATHS)//BATCH,callbacks=[cb],verbose=0)
    rows=eval_kodak(model,channel,run); summary={'model':'ADJSCC','channel':channel,'epochs':EPOCHS,'snr_db':SNR,'kodak_psnr_db':float(np.mean([r['psnr_db'] for r in rows])),'kodak_ssim':float(np.mean([r['ssim'] for r in rows])),'kodak_msssim':float(np.mean([r['msssim'] for r in rows])),'channel_uses_per_image_mean':float(np.mean([r['complex_channel_uses'] for r in rows])),'cbr':float(np.mean([r['cbr'] for r in rows])),'train_flops_per_step':flops[0],'inference_flops_per_image':flops[1],'source_commit':REV}
    summaries.append(summary); print(json.dumps(summary),flush=True)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0].keys()); w.writeheader(); w.writerows(summaries)
    del model; tf.keras.backend.clear_session()
print('ADJSCC completed',json.dumps(summaries),flush=True)
