"""Resumable Base then SA+RA training used exclusively by the single Kaggle runner."""

from __future__ import annotations

import json
import random
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - Kaggle normally provides tqdm
    tqdm = None
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Channels.channel_utils import require_torch
from EncDecPipeline.Models.SwinJSCC.adapter import SwinJSCCAdapter
from EncDecPipeline.Models.SwinJSCC.swin_config import BASE_MODEL_NAME, SA_RA_MODEL_NAME, SwinJSCCConfig
from EncDecPipeline.Models.SwinJSCC.training_utils import (
    DataParallelPlan,
    mse_loss,
    transfer_compatible_weights,
    unwrap_data_parallel,
    wrap_data_parallel,
)
from EncDecPipeline.Preprocessing.image_loader import discover_images, pil_to_tensor


@dataclass(slots=True)
class TrainingPhaseConfig:
    name: str
    epochs: int
    batch_size_per_gpu: int
    learning_rate: float = 1e-4
    crop_size: int = 256
    early_stopping_patience: int = 30
    gradient_clip_norm: float = 1.0
    seed: int = 20260906


@dataclass(slots=True)
class TrainingResult:
    best_validation_mse: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)
    transfer_missing_keys: list[str] = field(default_factory=list)
    transfer_unexpected_keys: list[str] = field(default_factory=list)


class RealImageDataset:
    """Uses only supplied files and deterministic random training crops."""

    def __init__(self, root: str | Path, crop_size: int, training: bool, seed: int) -> None:
        self.paths = discover_images(root)
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        _, Image = __import__("EncDecPipeline.Preprocessing.image_loader", fromlist=["require_torch_and_pillow"]).require_torch_and_pillow()
        image = Image.open(self.paths[index]).convert("RGB")
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(f"Real training image is too small for {self.crop_size}px crop: {self.paths[index]}")
        if self.training:
            generator = random.Random(self.seed + self.epoch * 1_000_003 + index)
            left = generator.randint(0, width - self.crop_size)
            top = generator.randint(0, height - self.crop_size)
        else:
            left = (width - self.crop_size) // 2
            top = (height - self.crop_size) // 2
        return pil_to_tensor(image.crop((left, top, left + self.crop_size, top + self.crop_size)))


def make_loader(dataset: RealImageDataset, batch_size: int, shuffle: bool) -> Any:
    torch = require_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=shuffle,
    )


def _save_resume_checkpoint(path: Path, module: Any, optimizer: Any, epoch: int, best_value: float) -> None:
    torch = require_torch()
    torch.save(
        {
            "epoch": epoch,
            "best_validation_mse": best_value,
            "model_state_dict": unwrap_data_parallel(module).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _load_resume_checkpoint(path: Path, module: Any, optimizer: Any) -> tuple[int, float]:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    unwrap_data_parallel(module).load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["epoch"]) + 1, float(payload["best_validation_mse"])


def train_phase(
    adapter: SwinJSCCAdapter,
    phase: TrainingPhaseConfig,
    train_root: str | Path,
    validation_root: str | Path,
    snr_grid: tuple[int, ...],
    rate_grid: tuple[int, ...],
    data_parallel: DataParallelPlan,
    output_dir: str | Path,
    resume: bool = True,
) -> TrainingResult:
    """Train using MSE, AMP, gradient clipping, valid real data, and early stopping."""

    torch = require_torch()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_data = RealImageDataset(train_root, phase.crop_size, training=True, seed=phase.seed)
    validation_data = RealImageDataset(validation_root, phase.crop_size, training=False, seed=phase.seed)
    global_batch_size = phase.batch_size_per_gpu * len(data_parallel.device_ids)
    train_loader = make_loader(train_data, global_batch_size, shuffle=True)
    validation_loader = make_loader(validation_data, global_batch_size, shuffle=False)
    module = wrap_data_parallel(adapter.build_training_module(), data_parallel)
    optimizer = torch.optim.AdamW(module.parameters(), lr=phase.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint = output / f"{phase.name}_resume.pt"
    start_epoch, best_value = (0, float("inf"))
    if resume and checkpoint.exists():
        start_epoch, best_value = _load_resume_checkpoint(checkpoint, module, optimizer)

    random_generator = random.Random(phase.seed)
    best_epoch = start_epoch - 1
    no_improvement = 0
    history: list[dict[str, float]] = []
    epoch_iterator = range(start_epoch, phase.epochs)
    if tqdm is not None:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc=f"Training {phase.name}",
            unit="epoch",
            dynamic_ncols=True,
        )
    for epoch in epoch_iterator:
        train_data.set_epoch(epoch)
        module.train()
        train_losses: list[float] = []
        for images in train_loader:
            images = images.cuda(non_blocking=True)
            snr = 10 if phase.name == "base" else random_generator.choice(snr_grid)
            rate = adapter.config.base_fixed_c if phase.name == "base" else random_generator.choice(rate_grid)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                reconstruction, _ = module(images, snr, rate)
                loss = mse_loss(images, reconstruction)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(module.parameters(), phase.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        module.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for images in validation_loader:
                images = images.cuda(non_blocking=True)
                snr = 10 if phase.name == "base" else snr_grid[len(snr_grid) // 2]
                rate = adapter.config.base_fixed_c if phase.name == "base" else rate_grid[len(rate_grid) // 2]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    reconstruction, _ = module(images, snr, rate)
                    validation_losses.append(float(mse_loss(images, reconstruction).detach().cpu()))
        validation_mse = sum(validation_losses) / len(validation_losses)
        row = {
            "epoch": float(epoch),
            "train_mse": sum(train_losses) / len(train_losses),
            "validation_mse": validation_mse,
        }
        if tqdm is not None:
            epoch_iterator.set_postfix(
                train_mse=f"{row['train_mse']:.6f}",
                val_mse=f"{row['validation_mse']:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        history.append(row)
        if validation_mse < best_value:
            best_value = validation_mse
            best_epoch = epoch
            no_improvement = 0
            _save_resume_checkpoint(checkpoint, module, optimizer, epoch, best_value)
        else:
            no_improvement += 1
        if no_improvement >= phase.early_stopping_patience:
            break

    start_epoch, _ = _load_resume_checkpoint(checkpoint, module, optimizer)
    (output / f"{phase.name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return TrainingResult(best_validation_mse=best_value, best_epoch=start_epoch - 1, history=history)


def build_base_then_sara(
    config: SwinJSCCConfig,
    upstream_root: str | Path,
    device: str = "cuda:0",
) -> tuple[SwinJSCCAdapter, SwinJSCCAdapter]:
    """Create source-compatible Base and SA+RA networks for explicit transfer learning."""

    from dataclasses import replace

    base = SwinJSCCAdapter(replace(config, variant=BASE_MODEL_NAME), upstream_root).build(device=device)
    sara = SwinJSCCAdapter(replace(config, variant=SA_RA_MODEL_NAME), upstream_root).build(device=device)
    return base, sara
