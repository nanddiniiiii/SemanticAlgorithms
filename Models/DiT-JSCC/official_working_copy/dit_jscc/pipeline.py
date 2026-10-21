"""Single-image inference pipeline for DiT-JSCC.

The repository ships a minimal inference core alongside this stable public API.
Heavy ML dependencies are imported lazily so that ``--check`` and the web UI
can still provide useful diagnostics before checkpoints are installed.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CORE = REPO_ROOT / "inference_core"
DEFAULT_CONFIGS = {
    256: REPO_ROOT / "configs" / "inference_256.py",
    512: REPO_ROOT / "configs" / "inference_512.py",
}
DEFAULT_CHECKPOINTS = {
    256: REPO_ROOT / "checkpoints" / "dit_jscc_256.pth",
    512: REPO_ROOT / "checkpoints" / "dit_jscc_512.pth",
}
DEFAULT_AUTOENCODER = REPO_ROOT / "checkpoints" / "autoencoder_kl.pth"
SUPPORTED_RATES = {
    256: {
        "semantic": tuple(range(4, 29, 2)),
        "detail": tuple(range(0, 25, 2)),
    },
    512: {
        "semantic": tuple(range(16, 113, 8)),
        "detail": tuple(range(0, 97, 8)),
    },
}


@dataclass(frozen=True)
class InferenceResult:
    """Output image and reproducibility metadata returned by the pipeline."""

    image: Any
    resolution: int
    channel_type: str
    snr_db: float
    semantic_rate: int
    detail_rate: int
    cbr: float
    sampler: str
    steps: int
    guidance_scale: float
    seed: int
    elapsed_seconds: float

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("image")
        data["cbr"] = round(data["cbr"], 8)
        data["elapsed_seconds"] = round(data["elapsed_seconds"], 3)
        return data


def _resolve_device(requested: str | None, torch: Any) -> str:
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Install a CUDA-enabled "
                "PyTorch build or pass device='cpu' for a slow diagnostic run."
            )
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_python_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"dit_jscc_config_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The DiT-JSCC checkpoint must contain a state-dict mapping.")
    for key in ("state_dict", "model", "nnet", "ema"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break

    cleaned: dict[str, Any] = {}
    prefixes = ("module.", "_orig_mod.", "nnet.")
    for key, value in checkpoint.items():
        if "attn_mask" in key:
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def check_environment(resolution: int = 256) -> dict[str, Any]:
    """Return lightweight setup diagnostics without loading any model."""

    if resolution not in DEFAULT_CONFIGS:
        raise ValueError("resolution must be 256 or 512")
    report: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "resolution": resolution,
        "config": str(DEFAULT_CONFIGS[resolution]),
        "config_found": DEFAULT_CONFIGS[resolution].is_file(),
        "checkpoint": str(DEFAULT_CHECKPOINTS[resolution]),
        "checkpoint_found": DEFAULT_CHECKPOINTS[resolution].is_file(),
        "autoencoder": str(DEFAULT_AUTOENCODER),
        "autoencoder_found": DEFAULT_AUTOENCODER.is_file(),
    }
    try:
        import torch

        report.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device=(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        report.update(torch_version=None, cuda_available=False, torch_error=str(exc))
    report["ready"] = all(
        report[key]
        for key in ("config_found", "checkpoint_found", "autoencoder_found")
    ) and bool(report.get("torch_version"))
    return report


class DiTJSCCPipeline:
    """Load DiT-JSCC once and reconstruct PIL images repeatedly.

    Parameters are intentionally explicit so experiments launched from the CLI,
    web demo, and Python API share the same behavior.
    """

    def __init__(
        self,
        resolution: int = 256,
        checkpoint: str | Path | None = None,
        autoencoder: str | Path | None = None,
        config: str | Path | None = None,
        device: str = "auto",
        precision: str = "fp16",
    ) -> None:
        if resolution not in (256, 512):
            raise ValueError("resolution must be either 256 or 512")
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")

        self.resolution = resolution
        self.checkpoint_path = Path(checkpoint or DEFAULT_CHECKPOINTS[resolution])
        self.autoencoder_path = Path(autoencoder or DEFAULT_AUTOENCODER)
        self.config_path = Path(config or DEFAULT_CONFIGS[resolution])
        self.requested_device = device
        self.precision = precision
        self._loaded = False

    def _validate_files(self) -> None:
        missing = [
            path
            for path in (self.config_path, self.checkpoint_path, self.autoencoder_path)
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Required inference files are missing:\n"
                f"{formatted}\nSee checkpoints/README.md for the expected layout."
            )

    def load(self) -> "DiTJSCCPipeline":
        if self._loaded:
            return self
        self._validate_files()
        if str(INFERENCE_CORE) not in sys.path:
            sys.path.insert(0, str(INFERENCE_CORE))

        import numpy as np
        import torch

        import sde
        import utils as legacy_utils
        from dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
        from libs.LVFM_encoder import LVFM_encoder
        from libs.autoencoder import get_model as get_autoencoder

        self.torch = torch
        self.np = np
        self.sde = sde
        self.DPM_Solver = DPM_Solver
        self.NoiseScheduleVP = NoiseScheduleVP
        self.model_wrapper = model_wrapper
        self.device = _resolve_device(self.requested_device, torch)
        self.config = _load_python_config(self.config_path)

        nnet = legacy_utils.get_nnet(**self.config.nnet)
        raw = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(raw)
        incompatible = nnet.load_state_dict(state_dict, strict=False)
        matched = len(nnet.state_dict()) - len(incompatible.missing_keys)
        match_ratio = matched / max(1, len(nnet.state_dict()))
        if match_ratio < 0.8:
            raise RuntimeError(
                f"Checkpoint/model mismatch: only {match_ratio:.1%} of model keys matched. "
                f"Verify that {self.checkpoint_path.name} is the {self.resolution}px "
                "DiT-JSCC checkpoint."
            )
        self.nnet = nnet.to(self.device).eval().requires_grad_(False)
        self.autoencoder = (
            get_autoencoder(str(self.autoencoder_path))
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.semantic_encoder = (
            LVFM_encoder(
                "dinov2-vit-b", device=self.device, resolution=self.resolution
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self._loaded = True
        return self

    def _autocast(self) -> Any:
        if self.device.startswith("cuda") and self.precision != "fp32":
            dtype = (
                self.torch.float16
                if self.precision == "fp16"
                else self.torch.bfloat16
            )
            return self.torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _preprocess(self, image: Any) -> Any:
        from PIL import Image, ImageOps

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (self.resolution, self.resolution),
            method=Image.Resampling.LANCZOS,
        )
        array = self.np.asarray(image, dtype=self.np.float32) / 127.5 - 1.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _calculate_cbr(resolution: int, semantic_rate: int, detail_rate: int) -> float:
        # Both transmitted representations contain a 16 x 16 token grid. Real
        # dimensions are paired into complex channel symbols (factor 1/2).
        complex_symbols = 16 * 16 * (semantic_rate + detail_rate) / 2
        return complex_symbols / (3 * resolution * resolution)

    def _validate_rates(self, semantic_rate: int, detail_rate: int) -> None:
        allowed = SUPPORTED_RATES[self.resolution]
        if semantic_rate not in allowed["semantic"]:
            raise ValueError(
                f"semantic_rate={semantic_rate} is unsupported at {self.resolution}px; "
                f"choose one of {allowed['semantic']}"
            )
        if detail_rate not in allowed["detail"]:
            raise ValueError(
                f"detail_rate={detail_rate} is unsupported at {self.resolution}px; "
                f"choose one of {allowed['detail']}"
            )

    def __call__(
        self,
        image: Any,
        *,
        channel_type: str = "awgn",
        snr_db: float = 0.0,
        semantic_rate: int | None = None,
        detail_rate: int | None = None,
        sampler: str = "dpm_solver",
        steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = 1000,
    ) -> InferenceResult:
        """Transmit and reconstruct one image through the simulated channel."""

        self.load()
        defaults = (8, 8) if self.resolution == 256 else (32, 32)
        semantic_rate = defaults[0] if semantic_rate is None else int(semantic_rate)
        detail_rate = defaults[1] if detail_rate is None else int(detail_rate)
        self._validate_rates(semantic_rate, detail_rate)
        channel_type = channel_type.lower()
        if channel_type not in ("awgn", "rayleigh"):
            raise ValueError("channel_type must be awgn or rayleigh")
        if sampler not in ("dpm_solver", "euler_ode", "euler_sde"):
            raise ValueError("sampler must be dpm_solver, euler_ode, or euler_sde")
        if not 1 <= int(steps) <= 1000:
            raise ValueError("steps must be in [1, 1000]")

        torch = self.torch
        seed = int(seed)
        random.seed(seed)
        self.np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        inputs = self._preprocess(image)
        labels = torch.full((1,), 1000, dtype=torch.long, device=self.device)
        z_init = torch.randn(
            1,
            4,
            self.resolution // 8,
            self.resolution // 8,
            device=self.device,
        )

        with torch.inference_mode(), self._autocast():
            self.nnet.trans_net.channel.chan_type = channel_type
            y_sem = self.semantic_encoder(inputs)

            def guided_model(x: Any, timesteps: Any, **kwargs: Any) -> Any:
                uncond = self.nnet(x, timesteps, mask_all=True, **kwargs)
                cond = self.nnet(x, timesteps, mask_all=False, **kwargs)
                return uncond + guidance_scale * (cond - uncond)

            score_model = self.sde.ScoreModel(
                guided_model,
                pred="noise_pred",
                sde=self.sde.VPSDE(),
            )
            model_kwargs = {
                "y": labels,
                "img": inputs,
                "y_sem": y_sem,
                "given_rate": (semantic_rate, detail_rate),
                "given_snr": float(snr_db),
            }
            if sampler == "dpm_solver":
                schedule = self.NoiseScheduleVP(schedule="linear")
                model_fn = self.model_wrapper(
                    score_model.noise_pred,
                    schedule,
                    time_input_type="0",
                    model_kwargs=model_kwargs,
                )
                latent = self.DPM_Solver(model_fn, schedule).sample(
                    z_init,
                    steps=int(steps),
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            elif sampler == "euler_ode":
                latent = self.sde.euler_maruyama(
                    self.sde.ODE(score_model),
                    z_init,
                    int(steps),
                    verbose=False,
                    **model_kwargs,
                )
            else:
                latent = self.sde.euler_maruyama(
                    self.sde.ReverseSDE(score_model),
                    z_init,
                    int(steps),
                    verbose=False,
                    **model_kwargs,
                )
            reconstruction = self.autoencoder.decode(latent)
            reconstruction = (0.5 * (reconstruction + 1.0)).clamp(0.0, 1.0)

        output = reconstruction[0].float().cpu().permute(1, 2, 0).numpy()
        output = (output * 255.0).round().astype(self.np.uint8)
        from PIL import Image

        return InferenceResult(
            image=Image.fromarray(output),
            resolution=self.resolution,
            channel_type=channel_type,
            snr_db=float(snr_db),
            semantic_rate=semantic_rate,
            detail_rate=detail_rate,
            cbr=self._calculate_cbr(
                self.resolution, semantic_rate, detail_rate
            ),
            sampler=sampler,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=seed,
            elapsed_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        """Release model references and, when applicable, the CUDA cache."""

        if not self._loaded:
            return
        for name in ("nnet", "autoencoder", "semantic_encoder"):
            if hasattr(self, name):
                delattr(self, name)
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        self._loaded = False
