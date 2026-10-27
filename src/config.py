"""One JSON configuration per experiment; reject misspelled fields early."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from .catalog import get_model
from .paths import resolve_path


@dataclass
class Config:
    algorithm: str
    variant: str = "baseline"
    reproduction_status: str = "unverified"
    dataset: str = "image_folder"
    data_root: str = ""
    val_root: str = ""
    test_root: str = ""
    checkpoint: str = ""
    output_root: str = "Results/experiments"
    device: str = "cpu"
    seed: int = 42
    snr_db: float = 10.0
    rate: int = 96
    epochs: int = 150
    batch_size: int = 2
    accumulation_steps: int = 1
    workers: int = 0
    crop_size: int = 128
    learning_rate: float = 0.0001
    max_seconds: float = 14400
    max_steps: int = 0
    trials: int = 1
    max_images: int = 0
    amp: bool = False
    multi_gpu: bool = False
    download_dataset: bool = False
    save_reconstructions: bool = True
    lpips: bool = False
    model_options: dict = field(default_factory=dict)

    def validate(self):
        spec = get_model(self.algorithm)
        for key in ["snr_db", "learning_rate", "max_seconds"]:
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.variant not in {"baseline", "modified"}:
            raise ValueError("variant must be baseline or modified")
        if self.reproduction_status not in {"unverified", "adapted"}:
            raise ValueError("reproduction_status must be unverified or adapted; paper reproduction is not certified by a config flag")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        for key in ["epochs", "batch_size", "accumulation_steps", "crop_size", "trials"]:
            if not isinstance(getattr(self,key),int) or isinstance(getattr(self,key),bool) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ["workers", "max_steps", "max_images"]:
            if not isinstance(getattr(self,key),int) or isinstance(getattr(self,key),bool):
                raise ValueError(f"{key} must be an integer")
        for key in ["workers", "max_steps", "max_images", "max_seconds"]:
            if getattr(self, key) < 0:
                raise ValueError(f"{key} cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.multi_gpu and not self.device.startswith("cuda"):
            raise ValueError("multi_gpu requires CUDA")
        if spec.backend == "jcm" and (self.dataset != "cifar10" or self.crop_size != 32):
            raise ValueError("JCM currently supports native CIFAR-10 32x32; do not silently resize other benchmarks")
        if not self.variant or "/" in self.variant or "\\" in self.variant:
            raise ValueError("Unsafe variant name")
        return self

    def as_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    values = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return Config(**values).validate()
