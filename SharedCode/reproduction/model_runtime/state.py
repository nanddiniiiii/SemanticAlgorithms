"""Atomic complete checkpoints and explicit completed-epoch accounting."""
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def assert_epoch_coverage(seen, expected=800):
    if len(seen) != expected or sorted(seen) != list(range(expected)):
        raise ValueError("A completed epoch must cover every unique source exactly once")


@dataclass
class EarlyStop:
    min_epochs: int = 50
    max_epochs: int = 150
    patience: int = 20
    relative_delta: float = .001
    best: float = math.inf
    bad_epochs: int = 0
    best_epoch: int = 0

    def update(self, epoch, loss):
        if not math.isfinite(loss) or epoch < 1:
            raise ValueError("A finite validation loss and complete positive epoch are required")
        improved = loss < self.best * (1 - self.relative_delta)
        if improved:
            self.best, self.bad_epochs, self.best_epoch = loss, 0, epoch
        else:
            self.bad_epochs += 1
        stop = epoch >= self.max_epochs or (epoch >= self.min_epochs and self.bad_epochs >= self.patience)
        return improved, stop

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)

