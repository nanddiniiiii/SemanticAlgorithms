"""One source per epoch, deterministic stateless crop schedule; zero worker RNG."""
import hashlib
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch * 1000003).shuffle(order)
    return order


def image_paths(folder, expected):
    if folder is None or not Path(folder).is_dir():
        raise ValueError("A real dataset directory is required")
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} source images, found {len(paths)} in {folder}")
    return paths


def dataset_identity(train, validation):
    """Hash decoded RGB and reject duplicate content and train/validation leakage."""
    groups = []
    for paths in (train, validation):
        hashes = []
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
            hashes.append(hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest())
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate decoded source images")
        groups.append(hashes)
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Training/validation leakage")
    return hashlib.sha256(repr(groups).encode()).hexdigest()


def load_tensor(path, *, crop=None, seed=0, center=False, author_resize=False):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if crop and not author_resize:
            width, height = image.size
            if min(width, height) < crop:
                raise ValueError("Source smaller than author crop; refusing resize")
            rng = random.Random(seed)
            x = (width-crop)//2 if center else rng.randrange(width-crop+1)
            y = (height-crop)//2 if center else rng.randrange(height-crop+1)
            image = image.crop((x, y, x+crop, y+crop))
        array = np.asarray(image, dtype=np.float32) / 255.
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    if author_resize:
        from torchvision.transforms.functional import resize
        tensor = resize(tensor, [128, 128], antialias=True)
    return tensor

