from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DIV2KWithPseudoLabels(Dataset):
    """Paired RGB and 19-class Cityscapes-ID pseudo-labels for the SQ-GAN adapter."""
    def __init__(self, image_root: str, label_root: str, size: int = 256, train: bool = True):
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.size = int(size)
        self.train = bool(train)
        exts = {'.png', '.jpg', '.jpeg'}
        images = sorted(p for p in self.image_root.rglob('*') if p.suffix.lower() in exts)
        if not images:
            raise FileNotFoundError(f"No images under {self.image_root}")
        self.pairs = []
        missing = []
        for image in images:
            mask = self.label_root / f'{image.stem}.png'
            if mask.is_file():
                self.pairs.append((image, mask))
            else:
                missing.append(str(mask))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} pseudo masks; first: {missing[0]}")
        self.image_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.train:
            scale = max(self.size / image.width, self.size / image.height, 1.0)
            new_size = (max(self.size, round(image.width * scale)), max(self.size, round(image.height * scale)))
            image = image.resize(new_size, Image.Resampling.BICUBIC)
            mask = mask.resize(new_size, Image.Resampling.NEAREST)
            x = np.random.randint(0, image.width - self.size + 1)
            y = np.random.randint(0, image.height - self.size + 1)
            box = (x, y, x + self.size, y + self.size)
            image, mask = image.crop(box), mask.crop(box)
            if np.random.rand() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        image_tensor = self.image_norm(transforms.ToTensor()(image))
        ids = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        if torch.any((ids != 255) & (ids > 18)):
            raise ValueError(f"Expected Cityscapes IDs [0,18] or 255: {mask_path}")
        ids[ids == 255] = 19
        semantic = torch.nn.functional.one_hot(ids, num_classes=20).permute(2, 0, 1).float()
        # Author label transform drops the null channel; all-zero void pixels argmax to class 0.
        label_ids = ids.clone()
        label_ids[label_ids == 19] = 0
        label = label_ids.unsqueeze(0).float()
        return {'image': image_tensor, 'label': label, 'semantic': semantic, 'filename': image_path.name}
