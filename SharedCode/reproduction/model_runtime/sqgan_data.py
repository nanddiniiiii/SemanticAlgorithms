"""DIV2K adapter delegating geometry and normalization to author ImagePaths."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@contextmanager
def sample_rng(seed):
    py, np_state, state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(state)


class DIV2KLabelled(Dataset):
    def __init__(self, source, paths, masks, split, relevant_obj, seed=20261020):
        module = importlib.import_module("data.cityscapes_base")
        if Path(module.__file__).resolve() != Path(source).resolve() / "data" / "cityscapes_base.py":
            raise RuntimeError("Cityscapes transforms resolved outside pinned SQ-GAN source")
        expected = 800 if split == "train" else 100
        if len(paths) != expected or len(masks) != expected:
            raise ValueError(f"SQGAN {split} requires all {expected} labelled DIV2K sources")
        self.native = module.ImagePaths(split=split, is_val=split != "train", paths=list(map(str, paths)),
            elements_input={"ssm_path": list(map(str, masks))}, size=256, ratio_dim=2,
            random_crop=split == "train", relevant_obj=relevant_obj, classes=19)
        self.epoch, self.seed = 0, seed

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        # Replay identical author augmentation after an interrupted epoch.
        with sample_rng(self.seed + self.epoch * 1000003 + index):
            result = self.native[index]
        if result["image"].shape != (3, 256, 512) or result["semantic"].shape != (20, 256, 512):
            raise ValueError("Author fixed 256x512 geometry violated")
        result["source_index"] = index
        return result


class EpochSampler(Sampler):
    """Explicit rank-strided order. Cursor state belongs to the training runner."""
    def __init__(self, dataset, rank, world_size, seed, epoch=0, cursor=0):
        self.dataset, self.rank, self.world_size, self.seed = dataset, rank, world_size, seed
        self.epoch, self.cursor = epoch, cursor

    def set_epoch(self, epoch):
        if self.epoch != epoch:
            self.cursor = 0
        self.epoch = epoch
        self.dataset.epoch = epoch

    def order(self):
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch * 1000003).shuffle(order)
        return order[self.rank::self.world_size]

    def __iter__(self):
        return iter(self.order()[self.cursor:])

    def __len__(self):
        return len(self.order()) - self.cursor

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "order": self.order(),
                "rank": self.rank, "world_size": self.world_size, "seed": self.seed}

    def load_state_dict(self, state):
        if (state["rank"], state["world_size"], state["seed"]) != (self.rank, self.world_size, self.seed):
            raise ValueError("SQGAN resume sampler topology/seed differs")
        self.set_epoch(state["epoch"])
        if state["order"] != self.order() or not 0 <= state["cursor"] <= len(self.order()):
            raise ValueError("SQGAN resume sampler order/cursor corrupt")
        self.cursor = state["cursor"]
