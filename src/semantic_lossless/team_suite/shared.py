"""Approved pseudo-label semantic prototype construction on common parents."""
from pathlib import Path
import time


def construct(options,run):
    import torch
    import random
    from torch.nn import functional as F
    from .dataset import ParentImages,digest
    from .runtime import write_json
    from semantic_lossless.shared_codebook import native_classifier
    random.seed(int(options.get('seed',22)));torch.manual_seed(int(options.get('seed',22)))
    classifier=native_classifier()
    state=torch.load(options['classifier'],map_location='cpu',weights_only=True)
    classifier.load_state_dict(state.get('state_dict',state),strict=True)
    device=options.get('device','cpu');classifier.to(device).eval()
    for parameter in classifier.parameters():parameter.requires_grad_(False)
    dataset=ParentImages(options['manifest'],'train',256)
    count=int(options.get('codebook_size',32))
    if not 2<=count<=128 or len(dataset)<count:raise ValueError('Invalid semantic codebook size')
    book=torch.stack([dataset[i]['image'] for i in range(count)]).to(device)
    counters=torch.ones(count,device=device)
    started=time.monotonic();confidences=[];hist=torch.zeros(10,dtype=torch.long)
    weight=float(options.get('semantic_weight',10))
    # Explicit bounded common-data construction, distinct from native300 passes.
    with torch.no_grad():
        for index in range(min(len(dataset),int(options.get('construction_images',len(dataset))))):
            image=dataset[index]['image'].unsqueeze(0).to(device)
            probabilities=classifier(F.interpolate(image,size=(96,96),mode='bilinear',align_corners=False)).softmax(1)
            confidence,label=probabilities.max(1)
            logits=classifier(F.interpolate(book,size=(96,96),mode='bilinear',align_corners=False))
            distances=(book-image).square().mean((1,2,3))-weight*logits.log_softmax(1)[:,int(label)]
            selected=int(distances.argmin());counters[selected]+=1
            book[selected]+=(image[0]-book[selected])/counters[selected]
            confidences.append(float(confidence));hist[int(label)]+=1
            if index%50==0:print(f'Prototype construction: {index+1}/{len(dataset)} training parents',flush=True)
            if time.monotonic()-started>float(options.get('construction_seconds',600)):break
    target=Path(run)/'semantic_codebook.pt'
    torch.save({'codebook':book.cpu(),'construction':'semantic_weighted_mse_ce',
        'classifier_sha256':digest(options['classifier']),'fit_split':'common training parents only',
        'pseudo_labels':True,'images_seen':len(confidences),'semantic_weight':weight,
        'adaptations':['frozen native10class pseudo-labels','single bounded prototype pass','per-image minimum reset']},target)
    write_json(Path(run)/'pseudo_label_coverage.json',{'count':len(confidences),
        'class_histogram':hist.tolist(),'mean_confidence':sum(confidences)/len(confidences),
        'teacher_seconds':time.monotonic()-started,'labels_are_ground_truth':False,
        'teacher_flops':'Not yet measured; do not report as zero'})
    return target
