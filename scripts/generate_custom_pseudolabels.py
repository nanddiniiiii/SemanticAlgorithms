"""Frozen teachers for approved SQ-GAN and Shared-Codebook custom adaptations.

Only training/development images are opened. Predictions are never ground truth.
SQ-GAN's eventual native paired augmentation is separate from teacher inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--resolved-sources',type=Path,required=True)
    parser.add_argument('--segmenter-root',type=Path,required=True)
    parser.add_argument('--classifier',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-images',type=int,default=0)
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    import transformers
    from src.artifacts import sha256,write_json
    from src.sources import fetch_source
    from src.shared_native import native_construction_definitions
    from src.metrics import profile_forward
    from src.compute import registered_kernel_flops
    import queue
    assert torch.cuda.device_count()>=2,'DataParallel teacher inference requires two actual CUDA devices'
    manifest=json.loads(args.manifest.read_text())
    canonical=json.dumps(manifest['images'],sort_keys=True,separators=(',',':')).encode()
    assert manifest['status']=='frozen' and hashlib.sha256(canonical).hexdigest()==manifest['manifest_sha256']
    sources=json.loads(args.resolved_sources.read_text())['sources']
    roots={(row['dataset'],row['official_split']):Path(row['root']) for row in sources}
    records=[row for row in manifest['images'] if row['split'] in {'train','val'}]
    if args.max_images:
        records=sorted(records,key=lambda row:row['width']*row['height'],reverse=True)[:args.max_images]
    assert len(records)>=2
    assert sha256(args.classifier)=='27a3a1cdc119252faffb5633bf122efae3523d613294d637b031c813d6c7dd70'
    expected_segmenter={
        'pytorch_model.bin':'027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c',
        'config.json':'649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9',
        'preprocessor_config.json':'d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7'}
    for name,digest in expected_segmenter.items():assert sha256(args.segmenter_root/name)==digest
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'maps').mkdir()
    teacher=SegformerForSemanticSegmentation.from_pretrained(args.segmenter_root,local_files_only=True).cuda().eval()
    processor=SegformerImageProcessor.from_pretrained(args.segmenter_root,local_files_only=True,do_reduce_labels=False)
    expected=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
        'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
    assert teacher.config.num_labels==19
    assert [teacher.config.id2label[i] for i in range(19)]==expected
    fetch_source('shared_semantic_codebook')
    classifier=native_construction_definitions()['GoogLeNet'](3,10).cuda().eval()
    classifier.load_state_dict(torch.load(args.classifier,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    for module in [teacher,classifier]:
        for parameter in module.parameters():parameter.requires_grad_(False)

    class Segment(torch.nn.Module):
        def __init__(self):super().__init__();self.model=teacher
        def forward(self,value):return self.model(pixel_values=value).logits
    work=queue.Queue()
    segment_module=Segment()
    hook=segment_module.register_forward_pre_hook(lambda module,inputs:work.put(str(inputs[0].device)))
    segment_parallel=torch.nn.DataParallel(segment_module).eval()
    classifier_parallel=torch.nn.DataParallel(classifier).eval()
    write_json(args.output/'protocol.json',{'scope':'frozen teacher pseudo-label preparation; no codec/teacher training',
        'manifest_sha256':manifest['manifest_sha256'],'test_opened':False,'teacher_classes':expected,
        'segmenter_id':'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
        'segmenter_revision':'21b3847fae21ddee674abd31129307b6a1235bd9',
        'segmenter_file_hashes':{p.name:sha256(p) for p in args.segmenter_root.iterdir() if p.is_file()},
        'classifier_sha256':sha256(args.classifier),'confidence_threshold':None,
        'segmenter_processor':processor.to_dict(),
        'segmenter_geometry_note':'published pinned processor resizes512square; do not silently replace it with model-card training1024square',
        'pilot_selection':'largest oriented pixel-area training/development images' if args.max_images else None,
        'prediction_policy':'retain all predictions; report confidence and usage; no selective data filtering',
        'map_geometry':'teacher logits bilinear upsampled to original oriented RGB HxW before argmax; native SQ paired transforms later',
        'classifier_geometry':'native96x96 PIL bilinear RGB/255',
        'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'limitations':['Cityscapes-only segmentation taxonomy on diverse RGB images',
            'STL10-only10class predictions are domain-limited pseudo-labels',
            'teacher agreement is not independent semantic accuracy']})
    class_hist=np.zeros(10,dtype=np.int64);semantic_hist=np.zeros(19,dtype=np.int64)
    total_segment=total_classify=0.;first_segment_input=first_classifier_input=None
    start=time.monotonic()
    with (args.output/'annotations.jsonl').open('w') as stream,torch.inference_mode():
        for offset in range(0,len(records),2):
            selected=records[offset:offset+2];images=[]
            for row in selected:
                path=roots[(row['dataset'],row['official_split'])]/row['relative_path']
                assert sha256(path)==row['file_sha256'],'Source image bytes changed: '+str(path)
                with Image.open(path) as handle:images.append(ImageOps.exif_transpose(handle).convert('RGB'))
            segment_input=processor(images=images,return_tensors='pt')['pixel_values'].cuda()
            classifier_input=torch.stack([torch.from_numpy(np.asarray(image.resize((96,96),Image.Resampling.BILINEAR),
                dtype=np.float32).copy()).permute(2,0,1)/255 for image in images]).cuda()
            torch.cuda.synchronize();tick=time.monotonic()
            logits=segment_parallel(segment_input)
            torch.cuda.synchronize();segment_seconds=time.monotonic()-tick;total_segment+=segment_seconds
            tick=time.monotonic();probabilities=classifier_parallel(classifier_input).softmax(1)
            torch.cuda.synchronize();classify_seconds=time.monotonic()-tick;total_classify+=classify_seconds
            if first_segment_input is None:
                first_segment_input=segment_input[:1].clone();first_classifier_input=classifier_input[:1].clone()
            assert torch.isfinite(logits).all() and torch.isfinite(probabilities).all()
            for i,(row,image) in enumerate(zip(selected,images)):
                pixel_logits=torch.nn.functional.interpolate(logits[i:i+1],size=(image.height,image.width),
                    mode='bilinear',align_corners=False)[0]
                labels=pixel_logits.argmax(0).byte().cpu().numpy()
                assert labels.min()>=0 and labels.max()<19
                path=args.output/'maps'/(row['decoded_pixel_sha256']+'.png')
                Image.fromarray(labels).save(path)
                pseudo_class=int(probabilities[i].argmax());confidence=float(probabilities[i].max())
                class_hist[pseudo_class]+=1;semantic_hist+=np.bincount(labels.reshape(-1),minlength=19)
                record={'dataset':row['dataset'],'split':row['split'],'relative_path':row['relative_path'],
                    'image_sha256':row['file_sha256'],'decoded_pixel_sha256':row['decoded_pixel_sha256'],
                    'map_path':str(path.relative_to(args.output)),'map_sha256':sha256(path),
                    'height':image.height,'width':image.width,'classifier_label':pseudo_class,
                    'classifier_confidence':confidence,'classifier_probabilities':probabilities[i].cpu().tolist()}
                stream.write(json.dumps(record)+'\n');stream.flush()
                del pixel_logits
            if offset%50==0:print({'stage':'pseudo_labels','images':offset+len(images),'total':len(records)},flush=True)
    hook.remove();observed=set()
    while not work.empty():observed.add(work.get())
    assert {'cuda:0','cuda:1'}.issubset(observed),'Did not observe teacher inference on both GPUs'
    for name,module,value in [('segmenter',segment_module,first_segment_input),('classifier',classifier,first_classifier_input)]:
        profile=profile_forward(module,value,warmup=3,repeats=20)
        profile['registered_kernel_count']=registered_kernel_flops(module,value)
        profile['memory_note']='Both teachers remain resident; not isolated deployment memory; single CUDA0 profiling'
        write_json(args.output/(name+'_profile.json'),profile)
    write_json(args.output/'status.json',{'status':'completed','images':len(records),
        'segmenter_forward_seconds':total_segment,'classifier_forward_seconds':total_classify,
        'elapsed_seconds':time.monotonic()-start,'observed_segmenter_devices':sorted(observed),
        'classifier_class_counts':class_hist.tolist(),'semantic_pixel_counts':semantic_hist.tolist(),
        'test_opened':False,'codec_training_started':False,
        'timing_scope':'synchronized teacher forwards exclude preprocessing, full-resolution logit interpolation and PNG writing; elapsed includes them'})


if __name__=='__main__':main()
