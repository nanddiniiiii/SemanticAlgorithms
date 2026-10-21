"""Pinned NTSCC+ architecture with an explicit matched transport extension.

No author network/layer/loss file is edited. Native natten 0.14.6 is mandatory.
Training uses the released rate-distortion objective and BOTH Adam optimizers.
Full evaluation fails closed until rate-index physical transport is implemented.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
from types import SimpleNamespace
import zlib

import numpy as np
from PIL import Image
import torch
from torch import nn

from .channels import MatchedChannel


class NTSCCSideInformationRequired(RuntimeError):
    """The analog decoder requires a rate map communicated to the receiver."""


def required_assets(stage, contract, mode="preflight"):
    """Actionable runtime blockers; training has no digital-sideinfo prerequisite."""
    del stage
    missing = []
    if not torch.cuda.is_available():
        missing.append("CUDA PyTorch with native GPU kernels; this interpreter is CPU-only or CUDA unavailable")
    try:
        version = importlib.metadata.version("natten")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "0.14.6":
        missing.append("Original compiled natten==0.14.6 built for this PyTorch/CUDA ABI; no modern API shim or Python fallback")
    else:
        try:
            spec = importlib.util.find_spec("natten._C")
            if spec is None or not any(str(spec.origin).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
                missing.append("natten._C must be the original compiled extension for natten==0.14.6")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            missing.append(f"Loadable native natten==0.14.6 extension: {type(error).__name__}: {error}")
    for package in ("compressai", "timm", "yaml"):
        if importlib.util.find_spec(package) is None:
            missing.append("NTSCC+ runtime dependency: " + ("PyYAML" if package == "yaml" else package))
    if mode in {"evaluate", "eval"}:
        assets = contract.get("model_assets", {}).get("NTSCC_plus", {})
        try:
            resolve_flif(assets.get("flif_executable"))
        except (FileNotFoundError, ValueError) as error:
            missing.append(str(error))
    return missing


def released_loss(source, lmbda):
    """Compile the exact author class without main.py's CLI/W&B side effects."""
    source = Path(source)
    tree = ast.parse((source / "main.py").read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RateDistortionLoss")
    namespace = {"nn": nn, "torch": torch, "math": math}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source / "main.py"), "exec"), namespace)
    return namespace["RateDistortionLoss"](lmbda=lmbda)


def released_config(source, channel, snr_db, device):
    """Read the released YAML operating point and parser's rate-vector default."""
    import yaml
    source = Path(source)
    settings = yaml.safe_load((source / "config" / "ntscc.yaml").read_text(encoding="utf-8"))
    tree = ast.parse((source / "main.py").read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for key in node.keywords:
                    if key.arg == "default":
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(key.value)
                        except (ValueError, TypeError):
                            pass
    return SimpleNamespace(channel_type=channel, SNR=snr_db, device=device, logger=False,
        eta=float(settings["eta"]), multiple_rate=defaults["--multiple-rate"],
        train_lambda=float(settings["train-lambda"]), lr=float(defaults["--lr"]),
        aux_lr=float(defaults["--aux-lr"]), clip_max_norm=float(settings["clip-max-norm"]))


class CheckerboardTransport(nn.Module):
    """Pack only active real coordinates, joining both checkerboards per image.

    The author concatenates [all anchors, all nonanchors] on batch dimension.
    Pair rows i and i+B before calling the common channel, so there is exactly
    one power normalization and one fade per original image. Odd active lengths
    transmit one explicit zero coordinate; measured counts include that pad.
    """
    def __init__(self, kind="awgn", snr_db=10):
        super().__init__()
        self.channel = MatchedChannel(kind, snr_db)
        self.last_uses = None
        self.last_active_real = None
        self.last_padding_real = None

    def forward(self, features, mask):
        if features.shape != mask.shape or features.ndim != 4 or features.shape[0] % 2:
            raise ValueError("Expected equally shaped [2*images,C,H,W] features and binary mask")
        if not bool(torch.isfinite(features).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("NTSCC channel requires finite features and a binary author mask")
        count = features.shape[0] // 2
        if count == 0:
            raise ValueError("Empty checkerboard batch")
        reconstructed_anchor, reconstructed_nonanchor, uses, active, padding = [], [], [], [], []
        for index in range(count):
            pair = torch.stack((features[index], features[index + count]))
            pair_mask = torch.stack((mask[index], mask[index + count])).bool()
            values = pair.masked_select(pair_mask)
            if not values.numel():
                raise ValueError("Author rate mask produced no active symbols for an image; refusing undefined power normalization")
            pad = values.numel() % 2
            tx = torch.cat((values, values.new_zeros(pad))).reshape(1, -1)
            rx = self.channel(tx).reshape(-1)[:values.numel()]
            restored = torch.zeros_like(pair).masked_scatter(pair_mask, rx)
            reconstructed_anchor.append(restored[0])
            reconstructed_nonanchor.append(restored[1])
            uses.append(tx.numel() // 2)
            active.append(values.numel())
            padding.append(pad)
        self.last_uses = torch.tensor(uses, device=features.device, dtype=torch.long)
        self.last_active_real = torch.tensor(active, device=features.device, dtype=torch.long)
        self.last_padding_real = torch.tensor(padding, device=features.device, dtype=torch.long)
        return torch.cat((torch.stack(reconstructed_anchor), torch.stack(reconstructed_nonanchor))), self.last_uses.sum()


class NTSCCAdapter(nn.Module):
    name = "NTSCC_plus"
    crop = 256
    alignment = 128  # author height multiple 128, width 64; common square padding is conservative
    loss_name = "released RateDistortionLoss: lambda*255^2*(MSE_JSCC+MSE_NTC)+bpp_y+bpp_z"
    gradient_clip_max_norm = 1.0
    scheduler_step_per_batch = True

    def __init__(self, source, channel="awgn", snr_db=10, device="cuda"):
        super().__init__()
        blockers = required_assets("train", {}, mode="train")
        if not str(device).startswith("cuda"):
            blockers.append("Pinned author NTSCC+ layers require CUDA; CPU architecture substitution is prohibited")
        if blockers:
            raise RuntimeError("NTSCC+ dependency block: " + "; ".join(blockers))
        source = Path(source).resolve()
        module = importlib.import_module("net.ntscc")
        if Path(module.__file__).resolve() != source / "net" / "ntscc.py":
            raise RuntimeError("net.ntscc resolved outside the pinned NTSCC+ source; use a fresh interpreter")
        config = released_config(source, channel, snr_db, device)

        class MatchedNTSCC(module.NTSCC_plus):
            # This one method is an external subclass extension. All network
            # transforms, rates, checkerboards, likelihoods and loss are native.
            def feature_pass_channel(self, s_masked, mask=None):
                if mask is None:
                    raise ValueError("NTSCC+ matched path requires the native active-rate mask")
                return self.matched_transport(s_masked, mask)

        self.core = MatchedNTSCC(config)
        self.core.matched_transport = CheckerboardTransport(channel, snr_db)
        self.criterion = released_loss(source, config.train_lambda)
        self.snr_db = snr_db
        self.gradient_clip_max_norm = config.clip_max_norm
        self.config = config
        self.index_codec = None
        self.protocol_identity = {"eta": config.eta, "lambda": config.train_lambda,
            "multiple_rate": config.multiple_rate, "main_lr": config.lr,
            "aux_lr": config.aux_lr, "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "main_lr_milestones_steps": [600000], "main_lr_gamma": .1,
            "training_geometry": "256x256 deterministic crop of each DIV2K source; external dataset adaptation",
            "side_information": "oracle indexes during optimization; framed FLIF + repetition3/QPSK + CRC receiver during evaluation"}

    @property
    def transport(self):
        return self.core.matched_transport

    def configure_optimizers(self):
        main, auxiliary = [], []
        for name, parameter in self.core.named_parameters():
            if parameter.requires_grad:
                (auxiliary if name.endswith(".quantiles") else main).append(parameter)
        if not main or not auxiliary:
            raise RuntimeError("NTSCC+ requires disjoint main and entropy-quantile optimizer parameter sets")
        optimizers = [torch.optim.Adam(main, lr=self.config.lr), torch.optim.Adam(auxiliary, lr=self.config.aux_lr)]
        schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[600000], gamma=.1)]
        return optimizers, schedulers

    def auxiliary_loss(self):
        return self.core.ntc.aux_loss()

    def forward(self, x):
        self.transport.channel.snr_db = self.snr_db
        output = self.core(x)
        losses = self.criterion(output, x)
        n = x.shape[0]
        # Reduce native scalars INSIDE each replica; DP gathers only tensors.
        # Sum k/observed uses across replicas, never average replica scalars.
        return {"reconstruction": output["x_hat"],
            "loss_sum": (losses["loss"] * n).reshape(1), "batch_count": x.new_tensor([n]),
            "complex_uses": self.transport.last_uses,
            "active_real_coordinates": self.transport.last_active_real,
            "padding_real_coordinates": self.transport.last_padding_real,
            "bpp_y_sum": (losses["bpp_y"] * n).reshape(1),
            "bpp_z_sum": (losses["bpp_z"] * n).reshape(1)}

    def configure_evaluation(self, contract):
        assets = contract.get("model_assets", {}).get("NTSCC_plus", {})
        self.index_codec = FLIFRateIndexCodec(assets.get("flif_executable"))

    def transmit(self, x, snr_db=None, seed=0):
        """Measure every transmitted stream; use received indexes or gray erasure.

        Native encode/decode remain unchanged. FLIF and the digital PHY are
        explicit benchmark extensions, not the paper's commented LDPC mapping.
        """
        from digital_transport import transmit_bytes
        if self.index_codec is None:
            raise NTSCCSideInformationRequired("Call configure_evaluation(contract) with model_assets.NTSCC_plus.flif_executable before NTSCC+ evaluation")
        if x.shape[0] != 1:
            raise ValueError("NTSCC+ native-image evaluation transmits one complete image/codeword")
        snr = self.snr_db if snr_db is None else snr_db
        self.transport.channel.snr_db = snr

        def sync():
            if x.is_cuda:
                torch.cuda.synchronize(x.device)

        sync()
        start = time.perf_counter()
        captured_masks = []
        handle = self.core.rate_adaption_enc.register_forward_hook(lambda module, inputs, output: captured_masks.append(output[1]))
        try:
            symbols, indexes = self.core.encode(x)
        finally:
            handle.remove()
        if len(captured_masks) != 1:
            raise RuntimeError("Expected one observed native rate-adaption mask per encoder invocation")
        mask = captured_masks[0]
        packet = self.index_codec.encode(indexes, x.shape[-2] // 16, x.shape[-1] // 16)
        sync()
        encoded = time.perf_counter()
        received_symbols, _ = self.transport(symbols, mask)
        sync()
        # CheckerboardTransport calls the common channel once for this image,
        # joining anchor/nonanchor: the same gain is used for the digital map.
        gain = self.transport.channel.last_gain.reshape(-1)[0]
        received_packet, digital = transmit_bytes(packet.packet,
            channel=self.transport.channel.kind, snr_db=snr, seed=int(seed),
            gain=complex(gain.detach().cpu().item()))
        transported = time.perf_counter()
        outage = received_packet is None
        if outage:
            reconstruction = torch.full_like(x, .5)
        else:
            try:
                received_indexes = self.index_codec.decode(received_packet)
            except (ValueError, RuntimeError, subprocess.SubprocessError, OSError):
                # A valid PHY CRC does not justify reusing transmitter indexes
                # if the codec frame cannot be parsed by the receiver.
                outage = True
                reconstruction = torch.full_like(x, .5)
            else:
                receiver_indexes = torch.from_numpy(received_indexes.astype(np.int64).reshape(-1)).to(x.device)
                reconstruction = self.core.decode(received_symbols, receiver_indexes)
        sync()
        decoded = time.perf_counter()
        analog_uses = int(self.transport.last_uses.sum().item())
        digital_uses = int(digital["complex_channel_uses"])
        ledger = {"streams": [
            {"name": "image_latent", "type": "continuous_complex_symbols",
             "complex_channel_uses": analog_uses,
             "active_real_coordinates": int(self.transport.last_active_real.sum().item()),
             "padding_real_coordinates": int(self.transport.last_padding_real.sum().item()),
             "serialized_bits": None, "fading_scope": "one shared coefficient per image/codeword"},
            {**packet.ledger, **digital, "name": "rate_indexes", "serialized_bits": len(packet.packet) * 8,
             "physical_transport": "explicit repetition3/QPSK extension", "all_transmitted_streams_counted": True}],
            "complex_channel_uses": analog_uses + digital_uses,
            "serialized_bits": len(packet.packet) * 8,
            "all_transmitted_streams_counted": True, "side_information_outage": outage,
            "outage_reconstruction": "constant RGB 0.5" if outage else None,
            "hyperprior_transmitted": False,
            "hyperprior_reason": "native JSCC receiver decode(symbols,indexes) does not consume z",
            "author_native_transport": False}
        timing = {"encoder_seconds": encoded - start, "channel_seconds": transported - encoded,
            "decoder_seconds": decoded - transported, "end_to_end_seconds": decoded - start}
        return reconstruction, analog_uses + digital_uses, timing, ledger


def resolve_flif(executable=None):
    candidate = executable or shutil.which("flif")
    if not candidate or not Path(candidate).is_file():
        raise FileNotFoundError("Provide model_assets.NTSCC_plus.flif_executable: a real FLIF encoder/decoder executable; PNG fallback is prohibited")
    return Path(candidate).resolve()


@dataclass(frozen=True)
class SerializedRateIndexes:
    packet: bytes
    rows: int
    columns: int
    codec_sha256: str

    @property
    def ledger(self):
        return {"stream": "rate_indexes", "codec": "FLIF", "serialized_bits": len(self.packet) * 8,
            "framing_bits": FLIFRateIndexCodec.HEADER.size * 8,
            "payload_bits": (len(self.packet) - FLIFRateIndexCodec.HEADER.size) * 8,
            "codec_executable_sha256": self.codec_sha256,
            "complex_channel_uses": None, "physical_transport": "NOT_IMPLEMENTED",
            "all_transmitted_streams_counted": False}


class FLIFRateIndexCodec:
    """Real checked FLIF subprocesses with unique files and an explicit frame.

    A local lossless roundtrip is NOT channel transmission. The ledger therefore
    leaves channel uses unavailable. No original shared output.png/output.flif,
    silent PNG fallback, ideal-capacity conversion, or stale file reuse occurs.
    """
    HEADER = struct.Struct(">4sBIIII")  # magic, version, rows, columns, FLIF bytes, CRC32

    def __init__(self, executable=None, *, timeout_seconds=120, temp_root=None):
        self.executable = resolve_flif(executable)
        self.codec_sha256 = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        self.timeout_seconds = timeout_seconds
        self.temp_root = temp_root

    def _run(self, operation, source, destination):
        if destination.exists():
            raise RuntimeError("FLIF destination must be a unique new file")
        subprocess.run([str(self.executable), "--overwrite", operation, str(source), str(destination)],
            check=True, capture_output=True, timeout=self.timeout_seconds)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("FLIF succeeded without a nonempty new output file")

    def encode(self, indexes, rows, columns):
        if isinstance(rows, bool) or isinstance(columns, bool) or not isinstance(rows, int) or not isinstance(columns, int) or min(rows, columns) <= 0:
            raise ValueError("Positive integer rate-index dimensions required")
        array = indexes.detach().cpu().numpy() if torch.is_tensor(indexes) else np.asarray(indexes)
        if array.size != rows * columns or not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() > 255:
            raise ValueError("Rate indexes must be integer uint8-representable values of the stated shape")
        array = array.astype(np.uint8).reshape(rows, columns)
        with tempfile.TemporaryDirectory(prefix="ntscc-rate-", dir=self.temp_root) as folder:
            folder = Path(folder)
            source, encoded = folder / "indexes.png", folder / "indexes.flif"
            Image.fromarray(array).save(source)
            self._run("-e", source, encoded)
            payload = encoded.read_bytes()
        header = self.HEADER.pack(b"NTSK", 1, rows, columns, len(payload), zlib.crc32(payload))
        result = SerializedRateIndexes(header + payload, rows, columns, self.codec_sha256)
        if not np.array_equal(self.decode(result.packet), array):
            raise RuntimeError("FLIF rate-index receiver roundtrip changed indexes")
        return result

    def decode(self, packet):
        if not isinstance(packet, bytes) or len(packet) <= self.HEADER.size:
            raise ValueError("Truncated rate-index packet")
        magic, version, rows, columns, length, crc = self.HEADER.unpack_from(packet)
        payload = packet[self.HEADER.size:]
        if magic != b"NTSK" or version != 1 or not rows or not columns or rows * columns > 100000000:
            raise ValueError("Invalid rate-index frame or dimensions")
        if len(payload) != length or zlib.crc32(payload) != crc:
            raise ValueError("Corrupted rate-index packet length/CRC32")
        with tempfile.TemporaryDirectory(prefix="ntscc-receiver-", dir=self.temp_root) as folder:
            folder = Path(folder)
            source, decoded = folder / "received.flif", folder / "received.png"
            source.write_bytes(payload)
            self._run("-d", source, decoded)
            with Image.open(decoded) as image:
                if image.mode != "L" or image.size != (columns, rows):
                    raise RuntimeError("FLIF receiver decoded unexpected index image geometry or mode")
                return np.array(image, dtype=np.uint8)
