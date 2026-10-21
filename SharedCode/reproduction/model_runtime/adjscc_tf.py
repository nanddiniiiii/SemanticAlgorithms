"""Faithful ADJSCC TF2.1 builder and MirroredStrategy training extension.

Must execute in the isolated legacy environment. Uses pinned author
Attention_Encoder/Attention_Decoder and the released 255-scale MSE objective.
DIV2K random 128 patches replace the author's pre-extracted ImageNet records.
The physical channel is an explicitly documented matched extension.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
from PIL import Image


def dependency_gate():
    import tensorflow as tf
    if not tf.__version__.startswith("2.1."):
        raise RuntimeError("Unmodified bundled author tensorflow_compression requires TensorFlow 2.1.x")
    import tensorflow_compression as tfc
    if not hasattr(tfc,"SignalConv2D") or not hasattr(tfc,"GDN"):
        raise RuntimeError("Author bundled SignalConv2D/GDN unavailable")
    devices = tf.config.experimental.list_physical_devices("GPU")
    if len(devices)<2:
        raise RuntimeError("ADJSCC matched training requires two visible TensorFlow CUDA GPUs")
    return tf


def build_models(tf, channels=16):
    from util_module import Attention_Encoder, Attention_Decoder
    x = tf.keras.layers.Input(shape=(None,None,3),name="runtime_RGB_255")
    snr = tf.keras.layers.Input(shape=(1,),name="runtime_SNR")
    normalized = tf.keras.layers.Lambda(lambda value:value/255.)(x)
    encoded = Attention_Encoder(normalized,snr,channels)
    encoder = tf.keras.Model([x,snr],encoded)
    latent = tf.keras.layers.Input(shape=(None,None,channels),name="runtime_received_latent")
    decoder_snr = tf.keras.layers.Input(shape=(1,),name="runtime_decoder_SNR")
    decoded = Attention_Decoder(latent,decoder_snr)
    decoded = tf.keras.layers.Lambda(lambda value:value*255.)(decoded)
    decoder = tf.keras.Model([latent,decoder_snr],decoded)
    return encoder,decoder


def matched_transport(tf, features, snr_db, channel, seed):
    shape = tf.shape(features)
    flat = tf.reshape(features,[shape[0],-1])
    half = tf.shape(flat)[1]//2
    tf.debugging.assert_equal(tf.math.floormod(tf.shape(flat)[1],2),0)
    symbols = tf.complex(flat[:,:half],flat[:,half:])
    scale = tf.maximum(tf.sqrt(tf.reduce_mean(tf.abs(symbols)**2,axis=1,keepdims=True)),1e-12)
    tx = symbols/tf.cast(scale,tf.complex64)
    noise_seed = tf.cast(seed,tf.int32)
    noise = tf.random.stateless_normal(tf.concat([tf.shape(tx),[2]],axis=0),noise_seed)
    n = tf.complex(noise[:,:,0],noise[:,:,1])*tf.cast(math.sqrt(10.**(-snr_db/10.)/2),tf.complex64)
    if channel=="rayleigh":
        fade = tf.random.stateless_normal([shape[0],1,2],noise_seed+tf.constant([7919,104729]))
        h = tf.complex(fade[:,:,0],fade[:,:,1])/tf.cast(math.sqrt(2),tf.complex64)
        rx = (h*tx+n)/h
    else:
        rx = tx+n
    rx *= tf.cast(scale,tf.complex64)
    real = tf.concat([tf.math.real(rx),tf.math.imag(rx)],axis=1)
    return tf.reshape(real,shape),half


def source_image(path,crop=None,seed=0,center=False):
    with Image.open(path) as image:
        image=image.convert("RGB")
        if crop:
            w,h=image.size
            if min(w,h)<crop:
                raise ValueError("Image smaller than author patch")
            rng=random.Random(seed)
            x=(w-crop)//2 if center else rng.randrange(w-crop+1)
            y=(h-crop)//2 if center else rng.randrange(h-crop+1)
            image=image.crop((x,y,x+crop,y+crop))
        return np.asarray(image,dtype=np.float32)


def atomic_pickle(payload,path):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload,stream,protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary),str(path))


def diagnostic(output,channel):
    """Real two-replica forward/backward/update/save/load; synthetic only."""
    tf=dependency_gate()
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device,True)
    strategy=tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
    with strategy.scope():
        encoder,decoder=build_models(tf,16)
        optimizer=tf.keras.optimizers.Adam(1e-4)
        variables=encoder.trainable_variables+decoder.trainable_variables
        optimizer._create_all_weights(variables)
    def step(images):
        replica=tf.distribute.get_replica_context().replica_id_in_sync_group
        snr=tf.fill([8,1],10.)
        with tf.GradientTape() as tape:
            features=encoder([images,snr],training=True)
            received,uses=matched_transport(tf,features,10,channel,tf.stack([20261020,replica]))
            rec=decoder([received,snr],training=True)
            loss=tf.nn.compute_average_loss(tf.reduce_mean((images-rec)**2,axis=[1,2,3]),global_batch_size=16)
        gradients=tape.gradient(loss,variables)
        present=[g for g in gradients if g is not None]
        if not present:
            raise RuntimeError("Missing author diagnostic gradients")
        for gradient in present:
            tf.debugging.assert_all_finite(gradient,"Nonfinite diagnostic gradient")
        optimizer.apply_gradients(zip(gradients,variables))
        return loss,replica,uses
    images=strategy.experimental_distribute_values_from_function(
        lambda context:tf.random.stateless_uniform([8,128,128,3],[20261020,int(context.replica_id_in_sync_group)],maxval=255.))
    losses,replicas,uses=strategy.experimental_run_v2(step,args=(images,))
    observed=[int(v.numpy()) for v in strategy.experimental_local_results(replicas)]
    if set(observed)!={0,1}:
        raise RuntimeError("Both TF replicas did not participate")
    loss=float(strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None).numpy())
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite two-GPU diagnostic loss")
    path=output/"diagnostic_roundtrip.pkl"
    before={"encoder":encoder.get_weights(),"decoder":decoder.get_weights(),"optimizer":optimizer.get_weights(),
            "synthetic_diagnostic_only":True,"production_epochs_completed":0}
    atomic_pickle(before,path)
    with path.open("rb") as stream:
        saved=pickle.load(stream)
    encoder.set_weights(saved["encoder"])
    decoder.set_weights(saved["decoder"])
    optimizer.set_weights(saved["optimizer"])
    for key,actual in [("encoder",encoder.get_weights()),("decoder",decoder.get_weights()),("optimizer",optimizer.get_weights())]:
        if len(actual)!=len(saved[key]) or not all(np.array_equal(a,b) for a,b in zip(actual,saved[key])):
            raise RuntimeError("ADJSCC diagnostic checkpoint differs")
    tf.keras.backend.clear_session()
    return {"runtime_validation":"PASSED_TWO_GPU_FORWARD_BACKWARD_OPTIMIZER_ROUNDTRIP",
            "participating_cuda_devices":observed,"diagnostic_global_batch":16,
            "synthetic_diagnostic_only":True,"production_epochs_completed":0,
            "diagnostic_loss":loss,"measured_complex_uses_per_image":[int(v.numpy()) for v in strategy.experimental_local_results(uses)],
            "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "full_kodak_geometry_gpu_validated":False}


def train(args,contract,manifest,train_paths,val_paths,dataset_hash):
    tf=dependency_gate()
    settings=contract["training"]
    if len(train_paths)!=800 or len(val_paths)!=100 or settings["global_batch_size"]!=16:
        raise ValueError("Approved benchmark requires full 800/100 sources and global batch16")
    seed=settings.get("seed",20261020)
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    strategy=tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
    with strategy.scope():
        encoder,decoder=build_models(tf,16) #16 real latent channels => measured CBR1/6
        optimizer=tf.keras.optimizers.Adam(1e-4)
        variables=encoder.trainable_variables+decoder.trainable_variables
        optimizer._create_all_weights(variables) # TF2.1 native Adam slot creation
    def order(epoch):
        result=list(range(800))
        random.Random(seed+epoch*1000003).shuffle(result)
        return result
    identity={"model":"ADJSCC","channel":args.channel,"stage":"main","source_commit":manifest["commit"],
        "dataset_sha256":dataset_hash,"contract_sha256":hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "global_batch":16,"seed":seed,"training_geometry":[128,128],"training_snr_db":settings["snr_db"],
        "transport_extension":True,"tf_version":tf.__version__}
    state={"complete_epochs":0,"cursor":0,"seen":[],"order":order(0),"global_step":0,
        "train_loss_sum":0.,"history":[],"training_complete":False,"best":float("inf"),"bad_epochs":0,"best_epoch":0}
    if args.resume:
        with args.resume.open("rb") as stream:
            checkpoint=pickle.load(stream)
        if checkpoint.get("identity")!=identity:
            raise ValueError("Resume source/data/contract identity mismatch")
        encoder.set_weights(checkpoint["encoder"])
        decoder.set_weights(checkpoint["decoder"])
        optimizer.set_weights(checkpoint["optimizer"])
        state=checkpoint["sampler_and_progress"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    elif (args.output/"latest.pkl").exists():
        raise ValueError("Existing fresh ADJSCC run requires --resume latest.pkl")
    started=time.monotonic()
    last_save=started
    def save(name="latest.pkl"):
        nonlocal last_save
        atomic_pickle({"schema_version":1,"identity":identity,"encoder":encoder.get_weights(),
            "decoder":decoder.get_weights(),"optimizer":optimizer.get_weights(),
            "sampler_and_progress":state,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
            "tf_rng":"Stateless per-step/per-replica seeds derived from checkpoint global_step; no stateful stochastic model layers",
            "schedulers":[],"scaler":None,"fresh_run":True},args.output/name)
        visible={**state,"best":None if math.isinf(state["best"]) else state["best"],"identity":identity,
                 "status":"TRAINING_COMPLETE" if state["training_complete"] else "RESUMABLE_INCOMPLETE"}
        (args.output/"progress.json").write_text(json.dumps(visible,indent=2),encoding="utf-8")
        last_save=time.monotonic()

    def replica_step(images,step):
        replica=tf.distribute.get_replica_context().replica_id_in_sync_group
        snr=tf.fill([tf.shape(images)[0],1],tf.cast(settings["snr_db"],tf.float32))
        with tf.GradientTape() as tape:
            features=encoder([images,snr],training=True)
            received,_=matched_transport(tf,features,settings["snr_db"],args.channel,tf.stack([step,replica+seed]))
            rec=decoder([received,snr],training=True)
            per_image=tf.reduce_mean(tf.square(images-rec),axis=[1,2,3])
            loss=tf.nn.compute_average_loss(per_image,global_batch_size=16)
        gradients=tape.gradient(loss,variables)
        present=[g for g in gradients if g is not None]
        if not present:
            raise RuntimeError("No trainable author gradients")
        for gradient in present:
            tf.debugging.assert_all_finite(gradient,"Nonfinite author gradient")
        optimizer.apply_gradients([(g,v) for g,v in zip(gradients,variables) if g is not None])
        return loss

    while not state["training_complete"]:
        epoch=state["complete_epochs"]
        while state["cursor"]<800:
            indices=state["order"][state["cursor"]:state["cursor"]+16]
            array=np.stack([source_image(train_paths[i],128,seed+epoch*1000003+i) for i in indices])
            distributed=strategy.experimental_distribute_values_from_function(
                lambda context:tf.convert_to_tensor(array[int(context.replica_id_in_sync_group)*8:(int(context.replica_id_in_sync_group)+1)*8]))
            losses=strategy.experimental_run_v2(replica_step,args=(distributed,tf.constant(state["global_step"],dtype=tf.int32)))
            loss=float(strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None).numpy())
            if not math.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            state["train_loss_sum"]+=loss*16
            state["cursor"]+=16
            state["seen"].extend(indices)
            state["global_step"]+=1
            if time.monotonic()-last_save>=min(300,settings["checkpoint_interval_seconds"]):
                save()
            if args.chunk_seconds and time.monotonic()-started>=args.chunk_seconds:
                save()
                return {"status":"RESUMABLE_INCOMPLETE","complete_epochs":state["complete_epochs"]}
        if sorted(state["seen"])!=list(range(800)):
            raise ValueError("Incomplete or duplicated epoch")
        validation=0.
        for i,path in enumerate(val_paths):
            image=tf.convert_to_tensor(source_image(path,128,center=True)[None])
            snr=tf.fill([1,1],tf.cast(settings["snr_db"],tf.float32))
            features=encoder([image,snr],training=False)
            received,_=matched_transport(tf,features,settings["snr_db"],args.channel,[seed+90000000,i])
            rec=decoder([received,snr],training=False)
            validation+=float(tf.reduce_mean(tf.square(rec-image)).numpy())/100.
        if not math.isfinite(validation):
            raise RuntimeError("Nonfinite validation loss")
        complete=epoch+1
        early=settings["early_stopping"]
        improved=validation<state["best"]*(1-early["relative_min_delta"])
        if improved:
            state.update(best=validation,bad_epochs=0,best_epoch=complete)
        else:
            state["bad_epochs"]+=1
        stop=complete>=early["maximum_epochs"] or (complete>=early["minimum_epochs"] and state["bad_epochs"]>=early["patience"])
        state["history"].append({"epoch":complete,"sources":800,"train_objective":state["train_loss_sum"]/800,
                                 "validation_objective":validation,"loss":"released 255-scale MSE"})
        state.update(complete_epochs=complete,cursor=0,seen=[],order=order(complete),train_loss_sum=0.,training_complete=stop)
        save()
        if improved:
            save("best.pkl")
        print(json.dumps(state["history"][-1]),flush=True)
    return {"status":"TRAINING_COMPLETE","training_complete":True,"complete_epochs":state["complete_epochs"],
            "selected_epoch":state["best_epoch"],"training_flops_coverage":"NOT_MEASURED"}


def evaluate(args,contract,manifest,paths):
    tf=dependency_gate()
    import torch
    import lpips
    from pytorch_msssim import ssim,ms_ssim
    if not args.resume:
        raise ValueError("ADJSCC evaluation requires selected best.pkl")
    with args.resume.open("rb") as stream:
        selected=pickle.load(stream)
    with (args.resume.parent/"latest.pkl").open("rb") as stream:
        latest=pickle.load(stream)
    identity=selected["identity"]
    progress=latest["sampler_and_progress"]
    if identity!=latest["identity"] or identity["channel"]!=args.channel or identity["source_commit"]!=manifest["commit"]:
        raise ValueError("Selected checkpoint identity differs")
    if identity["contract_sha256"]!=hashlib.sha256(args.contract.read_bytes()).hexdigest():
        raise ValueError("Contract differs from training")
    if identity["dataset_sha256"]!=args.dataset_sha256:
        raise ValueError("Evaluation training/validation source identity differs")
    if not progress["training_complete"] or progress["complete_epochs"]<50:
        raise ValueError("Fresh full training not complete")
    if selected["sampler_and_progress"]["complete_epochs"]!=progress["best_epoch"]:
        raise ValueError("Expected validation-selected best checkpoint")
    encoder,decoder=build_models(tf,16)
    encoder.set_weights(selected["encoder"])
    decoder.set_weights(selected["decoder"])
    # Keep common scoring on CPU to avoid reserving PyTorch CUDA memory inside
    # the legacy TensorFlow worker. Pretrained LPIPS assets are required.
    perceptual=lpips.LPIPS(net="alex").cpu().eval()
    snrs=contract["evaluation"]["snr_db"]
    trials=contract["evaluation"]["independent_channel_trials_per_image"]
    if snrs!=[0,5,10,15,20] or trials!=10 or len(paths)!=24:
        raise ValueError("Full approved Kodak/SNR/trial coverage required")
    if {p.stem for p in paths}!={"kodim%02d"%i for i in range(1,25)}:
        raise ValueError("Expected original Kodak24 names")
    rows_path=args.output/"measured_results.jsonl"
    if rows_path.exists():
        raise ValueError("Evaluation output exists; use new output directory")
    with rows_path.open("w",encoding="utf-8") as stream,torch.no_grad():
        for path in paths:
            array=source_image(path)
            h,w=array.shape[:2]
            if (h,w) not in {(512,768),(768,512)}:
                raise ValueError("Original Kodak geometry required")
            image=tf.convert_to_tensor(array[None])
            ref=torch.from_numpy(array.transpose(2,0,1).copy()).unsqueeze(0)/255.
            for snr in snrs:
                snr_tensor=tf.fill([1,1],tf.cast(snr,tf.float32))
                # Real warmup at the full native image geometry.
                features=encoder([image,snr_tensor],training=False)
                decoder([features,snr_tensor],training=False).numpy()
                for trial in range(trials):
                    key="%s|%s|%s|%s|%s"%(identity["seed"],path.stem,args.channel,snr,trial)
                    noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],"big")%(2**31-1)
                    begin=time.perf_counter()
                    features=encoder([image,snr_tensor],training=False)
                    features.numpy() # synchronous timing boundary
                    encoded=time.perf_counter()
                    received,uses=matched_transport(tf,features,snr,args.channel,[noise_seed,trial])
                    received.numpy()
                    transported=time.perf_counter()
                    reconstruction=decoder([received,snr_tensor],training=False).numpy()[0]/255.
                    decoded=time.perf_counter()
                    rec=torch.from_numpy(np.clip(reconstruction,0,1).transpose(2,0,1).copy()).unsqueeze(0)
                    mse=float((rec.double()-ref.double()).square().mean())
                    count=int(uses.numpy())
                    row={"measured":True,"model":"ADJSCC","channel":args.channel,"snr_db":snr,
                         "image":path.stem,"trial":trial,"PSNR":-10*math.log10(mse) if mse else None,
                         "SSIM":float(ssim(rec,ref,data_range=1.)),"MS_SSIM":float(ms_ssim(rec,ref,data_range=1.)),
                         "MSE":mse,"LPIPS":float(perceptual(rec*2-1,ref*2-1).mean()),
                         "complex_channel_uses":count,"CBR":count/(3*h*w),"original_height":h,"original_width":w,
                         "encoder_seconds":encoded-begin,"channel_seconds":transported-encoded,
                         "decoder_seconds":decoded-transported,"end_to_end_seconds":decoded-begin,
                         "padding_bottom":0,"padding_right":0,"native_full_image":True,
                         "all_transmitted_streams_counted":True,"serialized_bits":None,
                         "source_commit":manifest["commit"],"forward_flops_supported":None,
                         "flops_coverage":"NOT_MEASURED"}
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
                    stream.flush()
    receipt={"status":"complete","model":"ADJSCC","channel":args.channel,"evaluation_complete":True,
             "measured_rows":1200,"eligible_measured":False,"ranking_eligible":False,
             "validation_pass":True,"identity":identity,"source_provenance":manifest["provenance"],
             "checkpoint_sha256":hashlib.sha256(args.resume.read_bytes()).hexdigest(),
             "complete_training_epochs":progress["complete_epochs"],"selected_epoch":progress["best_epoch"],
             "all_transmitted_streams_counted":True,"flops_coverage":"NOT_MEASURED",
             "stream_ledger":[{"name":"image_latent","type":"complex_symbols","count":"measured per row"}],
             "eligibility_blockers":["Legacy TF forward/backward/optimizer FLOP measurement not implemented"],
             "comparison_track":"explicit_matched_channel_extension","author_paper_reproduction":False}
    (args.output/"measurement_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    return receipt
