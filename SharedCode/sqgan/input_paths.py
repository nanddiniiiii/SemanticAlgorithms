"""Resolve Kaggle's namespaced and legacy mount layouts by actual file counts."""
from pathlib import Path

def resolve_images(root,relative,count):
    matches=[]
    for folder in Path(root).glob('**/'+relative):
        if folder.is_dir():
            files=[p for p in folder.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
            if len(files)==count: matches.append(folder)
    assert len(matches)==1,f'Expected one {relative} with {count} images; found {matches}'
    return matches[0]

def resolve_div2k(root='/kaggle/input'):
    return (resolve_images(root,'Training/HR/hr_images',800),
            resolve_images(root,'Validation/HR/hr_images',100))
