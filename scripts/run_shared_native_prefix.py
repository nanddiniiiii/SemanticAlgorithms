"""As-released codebook construction and one-epoch SAC implementation check.

Construction defects are intentionally retained and named in the receipt.
This is not the paper's full proposed-loss method or a convergence experiment.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['construct','train'])
    parser.add_argument('--classifier',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--book',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args()
    import numpy as np
    import torch
    from torchvision.datasets import STL10
    from src.artifacts import write_json,sha256
    from src.sources import fetch_source
    from src.shared_native import native_construction_definitions,build_native_sac
    assert torch.cuda.is_available()
    assert sha256(args.classifier)=='27a3a1cdc119252faffb5633bf122efae3523d613294d637b031c813d6c7dd70'
    args.output.mkdir(parents=True,exist_ok=False)
    fetch_source('shared_semantic_codebook')
    random.seed(1024);np.random.seed(1024);torch.manual_seed(1024)
    definitions=native_construction_definitions()
    data=STL10(args.data_root,split='train',download=False,transform=definitions['data_tf'])
    assert len(data)==5000
    classifier=definitions['GoogLeNet'](3,10).cuda()
    classifier.load_state_dict(torch.load(args.classifier,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    classifier.train()  # Released construction/training uses mutable batch statistics.
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'execution_device':'cuda:0','multi_gpu':False,'precision':'float32'})
    start=time.monotonic()
    if args.phase=='construct':
        loader=torch.utils.data.DataLoader(data,batch_size=10,shuffle=True,drop_last=False,num_workers=0)
        counters=np.ones(10);book=None;updates=0
        protocol={'scope':'literal released construction behavior; known defects retained',
            'classifier_sha256':sha256(args.classifier),'entries':10,'semantic_weight':10,'seed':1024,
            'passes':300,'batch_size':10,'update_batches_per_pass':10,'fit_split':'official5000trainingonly',
            'retained_behaviors':['per-batch minimum instead of per-image minimum',
                'classifier train mode mutates BatchNorm statistics','cached logits per batch',
                'epoch0 initialization only; last native saved book at epoch290'],
            'adaptations':['local dataset/checkpoint paths','unused autograd graph disabled during prototype averaging',
                'extra final299 and status export; native290 artifact remains separate']}
        write_json(args.output/'protocol.json',protocol)
        with torch.no_grad():
            for epoch in range(300):
                for index,(images,labels) in enumerate(loader):
                    images=images.cuda();labels=labels.cuda()
                    if epoch==0:
                        book=images.clone();break
                    minimum=1e8;selected=0  # Deliberately preserve the released assignment defect.
                    logits=classifier(definitions['downsampling'](book,96))
                    for i in range(10):
                        for j in range(10):
                            distance=torch.nn.functional.mse_loss(images[i],book[j])+10*torch.nn.functional.cross_entropy(
                                logits[j].unsqueeze(0).repeat(2,1),labels[i].repeat(2))
                            if distance<minimum:minimum=distance;selected=j
                        counters[selected]+=1
                        book[selected]=book[selected]*(counters[selected]-1)/counters[selected]+images[i]/counters[selected]
                    updates+=1
                    if index+1==10:break
                if epoch%10==0:
                    artifact={'codebook':book.cpu(),'source_epoch':epoch,'update_batches':updates,
                        'construction':'as_released_semantic_weighted_mse_ce','protocol':protocol}
                    torch.save(artifact,args.output/('book_native_e290.pt' if epoch==290 else 'book_latest.pt'))
                    print({'stage':'construction','source_epoch':epoch,'update_batches':updates},flush=True)
                if time.monotonic()-start>args.max_seconds:
                    write_json(args.output/'status.json',{'status':'budget_stopped','source_epoch':epoch,
                        'update_batches':updates,'semantic_codec_trained':False})
                    raise TimeoutError('Construction budget exhausted')
        torch.save({'codebook':book.cpu(),'source_epoch':299,'update_batches':updates,
            'construction':'as_released_semantic_weighted_mse_ce','protocol':protocol},args.output/'book_final_e299.pt')
        assert updates==2990
        write_json(args.output/'status.json',{'status':'completed','update_batches':updates,'image_exposures':29900,
            'native_checkpoint_epoch':290,'final_state_epoch':299,'elapsed_seconds':time.monotonic()-start,
            'semantic_codec_trained':False})
        return
    if not args.book:raise ValueError('Training requires the native290 codebook')
    book_payload=torch.load(args.book,map_location='cpu',weights_only=True)
    assert book_payload['source_epoch']==290 and book_payload['update_batches']==2900
    book=book_payload['codebook'].cuda();assert book.shape==(10,3,256,256)
    model,metadata=build_native_sac();model.train()
    metadata.update(codebook_sha256=sha256(args.book),classifier_sha256=sha256(args.classifier),
        phase='two-step pilot' if args.steps else 'one native epoch prefix',
        data='STL10official5000trainingonly; no augmentation; native bilinear256',
        native_construction_defects_retained=True,heldout_evaluation=False)
    write_json(args.output/'protocol.json',metadata)
    loader=torch.utils.data.DataLoader(data,batch_size=12,shuffle=True,drop_last=True,num_workers=2,pin_memory=True)
    assert len(loader)==416
    # Native group lr overrides the apparent .001 global epoch-zero lr.
    optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':1e-4}],lr=.001)
    shapes={}
    hook=model.encoder.register_forward_hook(lambda module,inputs,out:shapes.update(latent=list(out.shape)))
    total_loss=0.;steps=0;images_seen=0
    for images,labels in loader:
        images=images.cuda();labels=labels.cuda()
        assistance=images.clone()
        for i in range(len(images)):
            minimum=1e8
            for j in range(10):
                distance=torch.nn.functional.mse_loss(images[i],book[j])
                if distance<minimum:minimum=distance;assistance[i]=book[j].clone()
        reconstruction,cbr,snr,mse,loss=model(images,assistance,np.float64(0),given_SNR=10)
        logits=classifier(definitions['downsampling'](reconstruction,96))
        classification_loss=torch.nn.functional.cross_entropy(logits,labels)
        assert torch.isfinite(classification_loss) and torch.isfinite(loss)
        assert reconstruction.shape==images.shape and shapes['latent']==[12,256,4]
        assert abs(cbr-512/(3*256*256))<1e-10
        optimizer.zero_grad();loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step();steps+=1;images_seen+=len(images);total_loss+=float(loss.detach())
        if steps<=2 or steps%25==0:
            print({'stage':'native_epoch0','step':steps,'images_seen':images_seen,
                'mse255_loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-start},flush=True)
        if steps%50==0 or (args.steps and steps>=args.steps) or steps==416:
            state={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'classifier_state_dict':classifier.state_dict(),'steps':steps,'images_seen':images_seen,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'epoch':0,'epoch_complete':steps==416,'protocol':metadata,'resume_supported':False}
            temporary=args.output/'codec.pt.tmp';torch.save(state,temporary);temporary.replace(args.output/'codec.pt')
        if (args.steps and steps>=args.steps) or time.monotonic()-start>args.max_seconds:break
    hook.remove()
    state={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
        'classifier_state_dict':classifier.state_dict(),'steps':steps,'images_seen':images_seen,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'epoch':0,'epoch_complete':steps==416,'protocol':metadata,'resume_supported':False}
    temporary=args.output/'codec.pt.tmp';torch.save(state,temporary);temporary.replace(args.output/'codec.pt')
    loaded=torch.load(args.output/'codec.pt',map_location='cuda:0',weights_only=True)
    model.load_state_dict(loaded['state_dict'],strict=True)
    model.eval()
    with torch.inference_mode():
        recovered=model(images,assistance,np.float64(0),given_SNR=10)[0]
        assert recovered.shape==images.shape and torch.isfinite(recovered).all()
    write_json(args.output/'tensor_shapes.json',{'input':list(images.shape),**shapes,'reconstruction':list(recovered.shape)})
    pilot_complete=bool(args.steps and steps>=args.steps)
    write_json(args.output/'status.json',{'status':'pilot_completed' if pilot_complete else 'budget_stopped' if args.steps else 'epoch_prefix_completed' if steps==416 else 'budget_stopped',
        'steps':steps,'images_seen':images_seen,'mean_training_mse255':total_loss/steps,
        'strict_reload':True,'elapsed_seconds':time.monotonic()-start,'paper_convergence_reproduced':False,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'heldout_quality_measured':False,'scope':'released SAC ablation implementation check; one-epoch prefix only'})
    if args.steps and not pilot_complete:
        raise TimeoutError('Pilot budget exhausted before required optimizer steps; epoch launch must stop')


if __name__=='__main__':main()
