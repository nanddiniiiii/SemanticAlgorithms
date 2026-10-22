"""Native CIFAR-10 or image folders; never use test images for optimization."""
from pathlib import Path
import random
from .paths import resolve_path


def cityscapes_label_lut():
    """Official labelIds -> SQ-GAN's 19 classes plus ignore class 19.

    Source: cityscapesScripts/cityscapesscripts/helpers/labels.py.
    This is label conversion, not a learned segmentation prediction.
    """
    values = [19] * 256
    for train_id, label_id in enumerate([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]):
        values[label_id] = train_id
    return tuple(values)


class Images:
    def __init__(self, directory, config, split):
        self.config = config
        self.split = split
        self.directory = directory
        self.paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
        if not self.paths:
            raise ValueError(f"No images in {directory}")
        if split != "train" and config.max_images:
            self.paths = self.paths[:config.max_images]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        config, split = self.config, self.split
        path = self.paths[index]
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            if split == "train":
                size = config.crop_size
                if min(image.size) < size:
                    raise ValueError(f"Training image {path.name} is smaller than crop_size; resizing must be explicit")
                left = random.randrange(image.width - size + 1)
                top = random.randrange(image.height - size + 1)
                image = image.crop((left, top, left + size, top + size))
                if random.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            values = np.asarray(image, dtype=np.float32).copy() / 255
        return {"image": torch.from_numpy(values).permute(2, 0, 1), "label": -1, "sample": path.relative_to(self.directory).as_posix()}

class CIFAR:
    def __init__(self, directory, config, split):
        self.config = config
        self.split = split
        self.directory = directory
        from torchvision.datasets import CIFAR10
        self.base = CIFAR10(str(directory), train=True, download=config.download_dataset)
        # Preserve the official test set for final testing. Validation is a
        # fixed training-only subset, rather than repeatedly selecting on test.
        ids = list(range(len(self.base)))
        random.Random(config.seed).shuffle(ids)
        boundary = int(0.9 * len(ids))
        self.ids = ids[:boundary] if split == "train" else ids[boundary:]
        if split == "test":
            self.base = CIFAR10(str(directory), train=False, download=config.download_dataset)
            self.ids = list(range(len(self.base)))
        if split != "train" and config.max_images:
            self.ids = self.ids[:config.max_images]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        config, split = self.config, self.split
        actual = self.ids[index]
        image, label = self.base[actual]
        if split == "train" and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        values = np.asarray(image, dtype=np.float32).copy() / 255
        return {"image": torch.from_numpy(values).permute(2, 0, 1), "label": label, "sample": f"cifar10-{split}-{actual}"}



def build_dataset(config, split):
    if config.dataset == "stl10":
        return STL(config, split)
    if config.dataset == "cityscapes":
        return Cityscapes(config, split)
    if config.dataset == "cifar10":
        if not config.data_root:
            raise ValueError("Set data_root to the CIFAR-10 cache directory")
        return CIFAR(resolve_path(config.data_root), config, split)
    if config.dataset != "image_folder":
        raise NotImplementedError(f"Dataset adapter for {config.dataset} is not integrated yet")
    root_value = {"train": config.data_root, "val": config.val_root, "test": config.test_root}[split]
    if not root_value:
        raise ValueError(f"Set the {split} image directory in the config")
    directory = resolve_path(root_value)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return Images(directory, config, split)


def _seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(config, split):
    import torch
    dataset = build_dataset(config, split)
    size = config.batch_size if split == "train" else 1
    generator = torch.Generator().manual_seed(config.seed)
    options = dict(batch_size=size, shuffle=split == "train", num_workers=config.workers,
                   pin_memory=config.device.startswith("cuda"), drop_last=False,
                   generator=generator, worker_init_fn=_seed_worker)
    if config.workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(dataset, **options)


def check_separate_folders(config):
    """Reject duplicate files across folder splits, including copied test images."""
    from .artifacts import sha256
    if config.dataset != "image_folder":
        return
    roots = [(key, resolve_path(value)) for key, value in [("train", config.data_root), ("val", config.val_root), ("test", config.test_root)] if value]
    seen = {}
    for split, directory in roots:
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            identity = sha256(path)
            if identity in seen and seen[identity] != split:
                raise ValueError(f"Duplicate image content across {seen[identity]} and {split}: {path.name}")
            seen[identity] = split


class Cityscapes:
    """Paired Cityscapes trainIds. Official val is evaluation; train has holdout."""
    def __init__(self, config, split):
        self.config = config
        self.split = split
        root = resolve_path(config.data_root)
        official = "val" if split == "test" else "train"
        self.image_root = root / "leftImg8bit" / official
        self.label_root = root / "gtFine" / official
        self.paths = sorted(self.image_root.rglob("*_leftImg8bit.png"))
        if not self.paths:
            raise FileNotFoundError(f"No Cityscapes images in {self.image_root}")
        if split != "test":
            random.Random(config.seed).shuffle(self.paths)
            boundary = int(0.9 * len(self.paths))
            self.paths = self.paths[:boundary] if split == "train" else self.paths[boundary:]
        if split != "train" and config.max_images:
            self.paths = self.paths[:config.max_images]
        if not self.paths:
            raise ValueError("Cityscapes partition is empty")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        path = self.paths[index]
        relative = path.relative_to(self.image_root)
        labels = self.label_root / relative.parent
        label_format = self.config.model_options.get('label_format','train_ids')
        if label_format not in {'train_ids','label_ids'}:
            raise ValueError('label_format must be train_ids or explicit label_ids')
        endings = ['_gtFine_labelIds.png'] if label_format == 'label_ids' else ['_gtFine_trainIds.png', '_gtFine_labelTrainIds.png']
        candidates = [labels / path.name.replace("_leftImg8bit.png", ending) for ending in endings]
        label_path = next((p for p in candidates if p.is_file()),None)
        if label_path is None:
            raise FileNotFoundError(f"Need 0..18/255 trainIds label for {path.name}; raw labelIds must be converted explicitly")
        with Image.open(path) as handle:
            image = handle.convert("RGB").resize((512,256),Image.Resampling.BILINEAR)
        with Image.open(label_path) as handle:
            label = handle.resize((512,256),Image.Resampling.NEAREST)
        if self.split == "train" and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        values = np.asarray(label,dtype=np.int64).copy()
        if values.ndim != 2:
            raise ValueError('Cityscapes labels must be scalar IDs, not color/instance maps')
        if label_format == 'label_ids':
            if not np.isin(values, list(range(34)) + [255]).all():
                raise ValueError('Invalid official Cityscapes labelIds')
            values = np.asarray(cityscapes_label_lut(),dtype=np.int64)[values]
        values[values == 255] = 19
        if values.min() < 0 or values.max() > 19:
            raise ValueError("Expected Cityscapes trainIds 0..18, 19/255 ignored")
        labels = torch.from_numpy(values)
        semantic = torch.nn.functional.one_hot(labels,20).permute(2,0,1).float()
        image = torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1)
        return {"image":image,"label":labels,"semantic":semantic,"sample":relative.as_posix()}


class STL:
    """Native labeled STL-10 with a training-only validation holdout."""
    def __init__(self,config,split):
        from torchvision.datasets import STL10
        self.config,self.split=config,split
        self.base=STL10(str(resolve_path(config.data_root)),split='test' if split=='test' else 'train',download=config.download_dataset)
        self.ids=list(range(len(self.base)))
        if split!='test':
            random.Random(config.seed).shuffle(self.ids)
            boundary=int(.9*len(self.ids))
            self.ids=self.ids[:boundary] if split=='train' else self.ids[boundary:]
        if split!='train' and config.max_images:
            self.ids=self.ids[:config.max_images]
        self.size=int(config.model_options.get('native_image_size',256))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,index):
        import numpy as np
        import torch
        from PIL import Image
        actual=self.ids[index]
        image,label=self.base[actual]
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        if self.split=='train' and random.random()<.5:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1)
        return {'image':tensor,'label':int(label),'sample':f'stl10-{self.split}-{actual}'}
