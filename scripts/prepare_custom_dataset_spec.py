"""Resolve mounted original dataset partitions; never infer TextOCR split by folder."""
import argparse
import json
from pathlib import Path
import hashlib
import time


def unique(paths,label):
    values=sorted({path.resolve() for path in paths})
    if len(values)!=1:raise ValueError(f'Expected one {label}, found {values}')
    return values[0]


def resolve_frozen_sources(manifest_path, mount):
    """Remap recorded roots with bounded path probes, never scan image corpora."""
    manifest=json.loads(manifest_path.read_text())
    rows=manifest['images']
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if manifest.get('status')!='frozen' or digest!=manifest['manifest_sha256']:
        raise ValueError('Frozen membership digest mismatch')
    sources=[]
    for source in manifest['spec']['sources']:
        original=Path(source['root'])
        parts=original.as_posix().split('/kaggle/input/',1)
        if len(parts)!=2:
            raise ValueError('Recorded source is not a Kaggle input root')
        suffix=Path(parts[1])
        candidates=[mount/suffix]
        if len(suffix.parts)>=4 and suffix.parts[0]=='datasets':
            candidates.append(mount/suffix.parts[2]/Path(*suffix.parts[3:]))
        key=(source['dataset'],source['official_split'])
        probes=[r['relative_path'] for r in rows
                if (r['dataset'],r['official_split'])==key and r['split'] in ('train','val')][:2]
        found=[]
        for candidate in candidates:
            if candidate.is_dir() and all((candidate/p).is_file() for p in probes):
                resolved=candidate.resolve()
                resolved.relative_to(mount.resolve())
                if resolved not in found:found.append(resolved)
        root=unique(found,source['dataset']+' '+source['official_split'])
        sources.append({**source,'root':str(root)})
        print(json.dumps({'stage':'resolve_frozen_source','dataset':key[0],
            'official_split':key[1],'root':str(root),'path_probes':len(candidates),
            'test_pixels_opened':False}),flush=True)
    return {**manifest['spec'],'sources':sources,'frozen_manifest_sha256':digest,
            'resolution_method':'bounded recorded-path probes; no recursive image traversal',
            'test_pixels_opened':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mount',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-manifest',type=Path)
    args=parser.parse_args();mount=args.mount
    started=time.monotonic()
    print(json.dumps({'stage':'resolve_sources','status':'started',
        'mode':'frozen_path_remap' if args.frozen_manifest else 'initial_discovery'}),flush=True)
    if args.frozen_manifest:
        spec=resolve_frozen_sources(args.frozen_manifest,mount)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(spec,indent=2)+'\n')
        print(json.dumps({'stage':'resolve_sources','status':'completed',
            'elapsed_seconds':time.monotonic()-started,'output':str(args.output)}),flush=True)
        return
    sources=[]
    def add(name,root,official,counts,url,license_note,allowed=None):
        row={'dataset':name,'root':str(root),'official_split':official,'counts':counts,
            'source_url':url,'license_note':license_note}
        if allowed is not None:row['allowed_file_names']=allowed
        sources.append(row)
    for split,counts in [('train',{'train':800}),('valid',{'val':100})]:
        roots=[path for path in mount.rglob('DIV2K_'+split+'_HR')
            if path.is_dir() and any(path.glob('*.png'))]
        add('DIV2K',unique(roots,'DIV2K '+split), 'train' if split=='train' else 'val',counts,
            'https://data.vision.ee.ethz.ch/cvl/DIV2K/','Preserve official dataset terms and attribution; no image redistribution in Git')
    for split,counts in [('train',{'train':2000}),('val',{'val':200,'test':200})]:
        roots=[path for path in mount.rglob(split+'2017') if path.is_dir() and any(path.glob('*.jpg'))]
        add('COCO2017',unique(roots,'COCO '+split),split,counts,
            'https://cocodataset.org/','Images retain individual source/Flickr licenses; annotations licensing separate; do not relicense mixture')
    for split,counts in [('train',{'train':1000}),('val',{'val':100,'test':100})]:
        annotation=unique(mount.rglob('TextOCR_0.1_'+split+'.json'),'TextOCR '+split+' annotation')
        record=json.loads(annotation.read_text())
        images=record['imgs'];images=list(images.values()) if isinstance(images,dict) else images
        allowed=sorted({Path(row['file_name']).name for row in images})
        example=allowed[0]
        image_root=unique([path.parent for path in mount.rglob(example)],'TextOCR image root')
        add('TextOCR',image_root,split,counts,'https://textvqa.org/textocr/dataset/',
            'Preserve TextOCR/TextVQA and underlying image attribution/terms; no bulk publication',allowed)
    # Protect Kodak against accidental exact/near-duplicate inclusion in train.
    kodak=unique([path.parent for path in mount.rglob('kodim01.png')],'Kodak root')
    add('Kodak24',kodak,'test',{'test':24},'https://r0k.us/graphics/kodak/',
        'Evaluation-only Kodak reference images; preserve source attribution')
    spec={'seed':22,'minimum_dimension':256,'sources':sources,
        'scope':'custom RGB adaptation; fixed split membership before training; not a native paper protocol'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(spec,indent=2)+'\n')
    print('Resolved custom split source specification: '+str(args.output),flush=True)


if __name__=='__main__':main()
