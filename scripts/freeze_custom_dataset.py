"""Freeze actual RGB images into deterministic, source-disjoint adaptation splits.

Input JSON lists source directories and requested counts per split. Selection is
model-independent. Exact file and decoded-pixel duplicates are excluded globally.
Near-duplicate scene groups must be supplied through group_by_parent where needed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import PIL
    from PIL import Image, ImageOps
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    if args.output.exists():raise FileExistsError('Frozen manifests are immutable: '+str(args.output))
    seed=spec.get('seed',22);rows=[];seen_files=set();seen_pixels=set();groups={};perceptual=[]
    sources=sorted(spec['sources'],key=lambda source: bool(source['counts'].get('train',0)))
    for source in sources:
        if source['counts'].get('train',0) and source['official_split']!='train':
            raise ValueError('Only official training partitions may populate train')
        if any(not isinstance(value,int) or value<0 for value in source['counts'].values()):
            raise ValueError('Requested counts must be nonnegative integers')
        root=Path(source['root']).resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        candidates=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
        if source.get('allowed_file_names') is not None:
            allowed=set(source['allowed_file_names'])
            candidates=[path for path in candidates if path.name in allowed]
        ranked=sorted(candidates,key=lambda p:hashlib.sha256(
            f"{seed}:{source['dataset']}:{p.relative_to(root).as_posix()}".encode()).hexdigest())
        counts=source['counts'];remaining=dict(counts);selected=0
        for path in ranked:
            if not any(remaining.values()):break
            file_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            if file_hash in seen_files:continue
            with Image.open(path) as handle:
                image=ImageOps.exif_transpose(handle).convert('RGB')
                width,height=image.size
                if min(width,height)<spec.get('minimum_dimension',256):continue
                pixel_hash=hashlib.sha256(f'{width}:{height}:RGB:'.encode()+image.tobytes()).hexdigest()
                small=list(image.convert('L').resize((9,8)).getdata())
                dhash=0
                for y in range(8):
                    for x in range(8):dhash=(dhash<<1)|int(small[y*9+x]>small[y*9+x+1])
            if pixel_hash in seen_pixels:continue
            relative=path.relative_to(root).as_posix()
            group=source['dataset']+':'+(path.parent.relative_to(root).as_posix()
                if source.get('group_by_parent') else relative)
            desired=next((name for name in ['test','val','train'] if remaining.get(name,0)>0),None)
            if any(split!=desired and (dhash^other).bit_count()<=4 for other,split in perceptual):continue
            if group in groups and groups[group]!=desired:continue
            groups[group]=desired;remaining[desired]-=1
            seen_files.add(file_hash);seen_pixels.add(pixel_hash)
            perceptual.append((dhash,desired))
            rows.append({'dataset':source['dataset'],'official_split':source['official_split'],
                'split':desired,'path':str(path),'relative_path':relative,'group_id':group,
                'file_sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'width':width,'height':height,
                'dhash64':f'{dhash:016x}',
                'source_url':source['source_url'],'license_note':source['license_note']})
            selected+=1
        if any(remaining.values()):
            raise ValueError(f"Insufficient eligible unique images for {source['dataset']}: {remaining}; no manifest frozen")
        print({'stage':'source_selection','source':source['dataset'],'selected':selected},flush=True)
    canonical=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    payload={'status':'frozen','seed':seed,'manifest_sha256':hashlib.sha256(canonical).hexdigest(),
        'pillow':PIL.__version__,'selection_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection':'SHA256 seeded relative-path ordering; exact-file and decoded-RGB dedup globally',
        'near_duplicate_limit':'cross-split64bitdHash distance<=4 excluded; heuristic, not a proof of scene independence; known parent groups also respected',
        'spec':spec,'images':rows,'counts':{split:sum(r['split']==split for r in rows) for split in ['train','val','test']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(args.output)
    print({'stage':'frozen','path':str(args.output),'counts':payload['counts'],'sha256':payload['manifest_sha256']},flush=True)


if __name__=='__main__':main()
