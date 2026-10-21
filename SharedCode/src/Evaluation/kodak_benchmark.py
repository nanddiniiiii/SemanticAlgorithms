"""Model-agnostic Kodak-24 image reconstruction benchmark.

The module deliberately separates source payload bits from complex channel uses.
JPEG/RAW have exact payload sizes but no radio mapping; analog JSCC methods can
report channel uses/CBR even when a serialized payload size is not meaningful.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import socket
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim
from INFRA.Artifacts import ImageArtifact, LatentArtifact


SCHEMA_VERSION = "1.0"
CSV_COLUMNS = (
    "schema_version", "run_id", "timestamp_utc", "contributor", "git_commit",
    "method", "method_version", "operating_point", "device", "image_id",
    "width", "height", "source_bits", "payload_bits", "bits_per_pixel",
    "compression_ratio", "channel_uses", "cbr", "encode_time_ms",
    "decode_time_ms", "total_time_ms", "cpu_process_time_ms",
    "peak_cpu_rss_mb", "peak_cpu_rss_delta_mb", "peak_gpu_allocated_mb",
    "model_parameters", "trainable_parameters", "model_size_mb", "flops",
    "psnr_db", "ssim", "lpips", "timing_repetitions", "notes",
)


@dataclass(slots=True)
class EncodedSample:
    state: Any
    payload_bits: int | None = None
    channel_uses: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkCodec(Protocol):
    name: str
    operating_point: str
    device: str
    version: str

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample: ...
    def decode(self, encoded: EncodedSample) -> Any: ...
    def complexity(self) -> Mapping[str, int | float | None]: ...


def pil_to_tensor(image: Any, device: str = "cpu") -> Any:
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = (
        tensor.detach().float().clamp(0.0, 1.0).squeeze(0)
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def _as_tensor(value: Any, device: str) -> Any:
    import numpy as np
    from PIL import Image
    import torch

    if isinstance(value, ImageArtifact):
        value = value.tensor
    if isinstance(value, Image.Image):
        return pil_to_tensor(value, device)
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not torch.is_tensor(value):
        raise TypeError(f"Decoder returned unsupported type: {type(value).__name__}")
    value = value.detach().float()
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or int(value.shape[1]) != 3:
        raise ValueError(f"Expected reconstructed BCHW RGB tensor, got {tuple(value.shape)}")
    return value.to(device).clamp(0.0, 1.0)


class CallableCodec:
    """Small adapter for arbitrary local models.

    ``encode_fn`` may return either a raw state or ``EncodedSample``. Payload and
    channel-use callbacks receive the raw state and are useful for custom codecs.
    """

    def __init__(
        self,
        name: str,
        encode_fn: Callable[[Any], Any],
        decode_fn: Callable[[Any], Any],
        *,
        operating_point: str = "default",
        device: str = "cpu",
        version: str = "unspecified",
        payload_bits_fn: Callable[[Any], int | None] | None = None,
        channel_uses_fn: Callable[[Any], int | None] | None = None,
        modules: Sequence[Any] = (),
        flops: int | float | None = None,
    ) -> None:
        self.name = name
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.operating_point = operating_point
        self.device = device
        self.version = version
        self.payload_bits_fn = payload_bits_fn
        self.channel_uses_fn = channel_uses_fn
        self.modules = tuple(modules)
        self.reported_flops = flops

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        output = self.encode_fn(tensor)
        if isinstance(output, EncodedSample):
            return output
        return EncodedSample(
            state=output,
            payload_bits=self.payload_bits_fn(output) if self.payload_bits_fn else None,
            channel_uses=self.channel_uses_fn(output) if self.channel_uses_fn else None,
        )

    def decode(self, encoded: EncodedSample) -> Any:
        return self.decode_fn(encoded.state)

    def complexity(self) -> Mapping[str, int | float | None]:
        return module_complexity(self.modules, self.reported_flops)


class EncDecCodec:
    """Adapter for this repository's ``EncDecInterface`` implementations."""

    def __init__(
        self,
        name: str,
        adapter: Any,
        *,
        operating_point: str = "default",
        device: str = "cpu",
        version: str = "unspecified",
        encode_options: Mapping[str, Any] | None = None,
        transport_fn: Callable[[LatentArtifact], Any] | None = None,
        payload_bits_fn: Callable[[LatentArtifact], int | None] | None = None,
        channel_uses_fn: Callable[[LatentArtifact], int | None] | None = None,
        flops: int | float | None = None,
    ) -> None:
        self.name = name
        self.adapter = adapter
        self.operating_point = operating_point
        self.device = device
        self.version = version
        self.encode_options = dict(encode_options or {})
        self.transport_fn = transport_fn
        self.payload_bits_fn = payload_bits_fn
        self.channel_uses_fn = channel_uses_fn
        self.reported_flops = flops

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        artifact = ImageArtifact(tensor=tensor, sample_ids=[image_id], source="Kodak24")
        latent = self.adapter.encode(artifact, **self.encode_options)
        bits = self.payload_bits_fn(latent) if self.payload_bits_fn else None
        if self.channel_uses_fn:
            uses = self.channel_uses_fn(latent)
        elif self.transport_fn:
            uses = None
        else:
            uses = math.ceil(latent.active_real_values() / 2)
        return EncodedSample(latent, bits, uses)

    def decode(self, encoded: EncodedSample) -> Any:
        if self.transport_fn is not None:
            transmission = self.transport_fn(encoded.state)
            if encoded.channel_uses is None and getattr(transmission, "channel_uses_per_image", None):
                encoded.channel_uses = int(transmission.channel_uses_per_image[0])
            return self.adapter.reconstruct_received(transmission, encoded.state)
        return self.adapter.decode(encoded.state, **self.encode_options)

    def complexity(self) -> Mapping[str, int | float | None]:
        modules = [getattr(self.adapter, name, None) for name in ("encoder", "decoder")]
        return module_complexity([module for module in modules if module is not None], self.reported_flops)


class JPEGCodec:
    def __init__(self, quality: int) -> None:
        from PIL import __version__ as pillow_version

        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        self.quality = int(quality)
        self.name = "JPEG"
        self.operating_point = f"quality={quality};subsampling=0"
        self.device = "cpu"
        self.version = f"Pillow-{pillow_version}"

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        stream = BytesIO()
        image.convert("RGB").save(
            stream, format="JPEG", quality=self.quality, optimize=False,
            progressive=False, subsampling=0,
        )
        payload = stream.getvalue()
        return EncodedSample(payload, payload_bits=len(payload) * 8)

    def decode(self, encoded: EncodedSample) -> Any:
        from PIL import Image

        with Image.open(BytesIO(encoded.state)) as image:
            return image.convert("RGB").copy()

    def complexity(self) -> Mapping[str, int | float | None]:
        return module_complexity(())


class RawRGBCodec:
    name = "RAW_RGB24"
    operating_point = "8_bits_per_channel"
    device = "cpu"
    version = "RGB24"

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        rgb = image.convert("RGB")
        payload = rgb.tobytes()
        return EncodedSample((payload, rgb.size), payload_bits=len(payload) * 8)

    def decode(self, encoded: EncodedSample) -> Any:
        from PIL import Image

        payload, size = encoded.state
        return Image.frombytes("RGB", size, payload)

    def complexity(self) -> Mapping[str, int | float | None]:
        return module_complexity(())


def module_complexity(
    modules: Iterable[Any], flops: int | float | None = None
) -> dict[str, int | float | None]:
    parameter_ids: set[int] = set()
    buffer_ids: set[int] = set()
    parameters = trainable = storage_bytes = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in parameter_ids:
                continue
            parameter_ids.add(id(parameter))
            parameters += int(parameter.numel())
            trainable += int(parameter.numel()) if parameter.requires_grad else 0
            storage_bytes += int(parameter.numel() * parameter.element_size())
        for buffer in module.buffers():
            if id(buffer) in buffer_ids:
                continue
            buffer_ids.add(id(buffer))
            storage_bytes += int(buffer.numel() * buffer.element_size())
    return {
        "model_parameters": parameters,
        "trainable_parameters": trainable,
        "model_size_mb": storage_bytes / (1024**2),
        "flops": flops,
    }


class LPIPSMetric:
    def __init__(self, device: str, net: str = "alex") -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS is required; install the project benchmark extra") from error
        self.device = device
        self.metric = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, reference: Any, reconstruction: Any) -> float:
        import torch

        with torch.inference_mode():
            value = self.metric(reference * 2.0 - 1.0, reconstruction * 2.0 - 1.0)
        return float(value.mean().detach().cpu())


class _PeakRSS:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval = interval_seconds
        self.baseline = self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None

    def __enter__(self) -> "_PeakRSS":
        if self._process is None:
            return self
        self.baseline = self.peak = int(self._process.memory_info().rss)

        def sample() -> None:
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, int(self._process.memory_info().rss))

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._process is None:
            return
        self.peak = max(self.peak, int(self._process.memory_info().rss))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def _synchronize(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _git_commit(project_root: Path) -> str:
    head = project_root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            return (project_root / ".git" / ref[5:]).read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


def dataset_files(root: Path) -> list[Path]:
    files = [root / f"kodim{index:02d}.png" for index in range(1, 25)]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Kodak-24 is incomplete in {root}; missing {missing}")
    return files


def benchmark_codec(
    codec: BenchmarkCodec,
    files: Sequence[Path],
    *,
    run_id: str,
    contributor: str,
    project_root: Path,
    timing_repetitions: int = 3,
    warmup_repetitions: int = 1,
    compute_lpips: bool = True,
    lpips_device: str = "cpu",
    notes: str = "",
) -> list[dict[str, Any]]:
    import torch

    if timing_repetitions < 1 or warmup_repetitions < 0:
        raise ValueError("timing_repetitions must be >= 1 and warmup_repetitions >= 0")
    complexity = dict(codec.complexity())
    # Quality evaluation is outside the timed region. Keeping LPIPS on a separate
    # device prevents its AlexNet weights from contaminating codec GPU peaks.
    lpips_metric = LPIPSMetric(lpips_device) if compute_lpips else None
    rows: list[dict[str, Any]] = []
    from PIL import Image

    for file_index, path in enumerate(files):
        with Image.open(path) as opened:
            image = opened.convert("RGB").copy()
        width, height = image.size
        reference = pil_to_tensor(image, codec.device)

        if file_index == 0:
            with torch.inference_mode():
                for _ in range(warmup_repetitions):
                    warm = codec.encode(reference, image, path.stem)
                    codec.decode(warm)
                if warmup_repetitions:
                    del warm
            _synchronize(codec.device)

        if str(codec.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(codec.device)

        encode_ms: list[float] = []
        decode_ms: list[float] = []
        cpu_started = time.process_time_ns()
        with _PeakRSS() as rss:
            with torch.inference_mode():
                for repetition in range(timing_repetitions):
                    if repetition:
                        del encoded, reconstruction_raw
                    _synchronize(codec.device)
                    started = time.perf_counter_ns()
                    encoded = codec.encode(reference, image, path.stem)
                    _synchronize(codec.device)
                    encode_ms.append((time.perf_counter_ns() - started) / 1e6)

                    started = time.perf_counter_ns()
                    reconstruction_raw = codec.decode(encoded)
                    _synchronize(codec.device)
                    decode_ms.append((time.perf_counter_ns() - started) / 1e6)
        cpu_ms = (time.process_time_ns() - cpu_started) / 1e6 / timing_repetitions

        reconstruction = _as_tensor(reconstruction_raw, codec.device)
        if tuple(reference.shape) != tuple(reconstruction.shape):
            raise ValueError(
                f"{codec.name} changed {path.name} shape from {tuple(reference.shape)} "
                f"to {tuple(reconstruction.shape)}; benchmark resizing is forbidden"
            )
        source_bits = width * height * 3 * 8
        payload_bits = encoded.payload_bits
        channel_uses = encoded.channel_uses
        peak_gpu = None
        if str(codec.device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated(codec.device) / (1024**2)

        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "contributor": contributor,
            "git_commit": _git_commit(project_root),
            "method": codec.name,
            "method_version": codec.version,
            "operating_point": codec.operating_point,
            "device": codec.device,
            "image_id": path.stem,
            "width": width,
            "height": height,
            "source_bits": source_bits,
            "payload_bits": payload_bits,
            "bits_per_pixel": payload_bits / (width * height) if payload_bits else None,
            "compression_ratio": source_bits / payload_bits if payload_bits else None,
            "channel_uses": channel_uses,
            "cbr": channel_uses / (3.0 * width * height) if channel_uses is not None else None,
            "encode_time_ms": statistics.fmean(encode_ms),
            "decode_time_ms": statistics.fmean(decode_ms),
            "total_time_ms": statistics.fmean(encode_ms) + statistics.fmean(decode_ms),
            "cpu_process_time_ms": cpu_ms,
            "peak_cpu_rss_mb": rss.peak / (1024**2) if rss.peak else None,
            "peak_cpu_rss_delta_mb": (rss.peak - rss.baseline) / (1024**2) if rss.peak else None,
            "peak_gpu_allocated_mb": peak_gpu,
            "model_parameters": complexity.get("model_parameters"),
            "trainable_parameters": complexity.get("trainable_parameters"),
            "model_size_mb": complexity.get("model_size_mb"),
            "flops": complexity.get("flops"),
            "psnr_db": psnr(reference, reconstruction),
            "ssim": ssim(reference, reconstruction),
            "lpips": lpips_metric(
                reference.to(lpips_device), reconstruction.to(lpips_device)
            ) if lpips_metric else None,
            "timing_repetitions": timing_repetitions,
            "notes": "; ".join(
                value for value in (notes, str(getattr(codec, "notes", ""))) if value
            ),
        }
        rows.append(row)
        print(
            f"{codec.name} {codec.operating_point} {path.stem}: "
            f"PSNR={row['psnr_db']:.3f} dB, SSIM={row['ssim']:.5f}, "
            f"total={row['total_time_ms']:.2f} ms"
        )
    return rows


def make_run_id(prefix: str = "kodak24") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{socket.gethostname()}-{timestamp}-{uuid.uuid4().hex[:8]}"


def append_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Atomically append new logical rows and reject duplicate measurement keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Benchmark CSV is locked: {lock}") from error
    os.close(descriptor)
    try:
        existing: list[dict[str, Any]] = []
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                    raise ValueError(f"Unexpected CSV schema in {path}")
                existing = list(reader)
        keys = {
            (row["run_id"], row["method"], row["operating_point"], row["image_id"])
            for row in existing
        }
        added: list[Mapping[str, Any]] = []
        for row in rows:
            missing = [column for column in CSV_COLUMNS if column not in row]
            if missing:
                raise ValueError(f"Benchmark row is missing columns: {missing}")
            key = (row["run_id"], row["method"], row["operating_point"], row["image_id"])
            if key in keys:
                continue
            keys.add(key)
            added.append(row)

        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
            writer.writerows(added)
        os.replace(temporary, path)
        return len(added)
    finally:
        lock.unlink(missing_ok=True)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["operating_point"])), []).append(row)
    output = []
    metric_names = (
        "payload_bits", "bits_per_pixel", "compression_ratio", "channel_uses", "cbr",
        "encode_time_ms", "decode_time_ms", "total_time_ms", "cpu_process_time_ms",
        "peak_cpu_rss_mb", "peak_cpu_rss_delta_mb", "peak_gpu_allocated_mb",
        "model_parameters", "trainable_parameters", "model_size_mb", "flops",
        "psnr_db", "ssim", "lpips",
    )
    for (method, operating_point), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "method": method,
            "operating_point": operating_point,
            "images": len({str(row["image_id"]) for row in group}),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
        output.append(summary)
    return output


def environment_summary() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
