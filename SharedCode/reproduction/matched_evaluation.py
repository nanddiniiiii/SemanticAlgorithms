"""Common geometry and metric routines around an unchanged model callable.

No model is implemented or silently replaced here. The adapter must return real
reconstructions and measured communication counts. The common tiled protocol is
an explicitly separate benchmark; it is not the authors' native paper protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def dataset_manifest(train_root, validation_root, kodak_root, output):
    pools = {}
    for name, folder, expected in [("train", train_root, 800), ("validation", validation_root, 100), ("kodak", kodak_root, 24)]:
        folder = Path(folder)
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if name == "kodak":
            paths = [p for p in paths if p.stem.startswith("kodim")]
        if len(paths) != expected:
            raise ValueError(f"{name}: expected {expected} images, found {len(paths)}")
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            shape = rgb.shape
            content_hash = hashlib.sha256(str(shape).encode() + rgb.tobytes()).hexdigest()
            rows.append({"relative_path": path.relative_to(folder).as_posix(),
                         "height": shape[0], "width": shape[1],
                         "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "decoded_rgb_sha256": content_hash})
        if len({row["decoded_rgb_sha256"] for row in rows}) != expected:
            raise ValueError(f"Duplicate decoded images in {name}")
        pools[name] = rows
    for a, b in [("train", "validation"), ("train", "kodak"), ("validation", "kodak")]:
        if {r["decoded_rgb_sha256"] for r in pools[a]} & {r["decoded_rgb_sha256"] for r in pools[b]}:
            raise ValueError(f"Data leakage between {a} and {b}")
    encoded = json.dumps(pools, sort_keys=True, separators=(",", ":")).encode()
    manifest = {"pools": pools, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}
    Path(output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def geometry(height, width, tile_height=256, tile_width=512):
    if min(height, width, tile_height, tile_width) <= 0:
        raise ValueError("Dimensions must be positive")
    padded_height = ((height + tile_height - 1) // tile_height) * tile_height
    padded_width = ((width + tile_width - 1) // tile_width) * tile_width
    return {"original_height": height, "original_width": width,
            "padded_height": padded_height, "padded_width": padded_width,
            "tile_height": tile_height, "tile_width": tile_width,
            "tile_count": (padded_height // tile_height) * (padded_width // tile_width),
            "padding": "edge_bottom_right", "resize": False, "discard_original_pixels": False}


def evaluate_native_image(image, transmit, *, image_id, channel, snr_db, trial,
                          seed, alignment=1):
    """Full-image evaluation with only the alignment the selected model needs.

    ``transmit`` receives the whole padded RGB [0,1] image and context and
    must return a same-sized reconstruction plus a measured stream ledger.
    This is separate from the explicitly tiled protocol below.
    """
    from native_geometry import pad_image, unpad_image, measured_cbr
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite RGB [0,1] image")
    padded, info = pad_image(image, multiple=alignment)
    context = {"image_id": image_id, "channel": channel, "snr_db": snr_db,
               "trial": trial, "seed": seed, "geometry": info}
    result = transmit(padded.copy(), context)
    reconstruction = unpad_image(result["reconstruction"], info)
    uses = result["complex_channel_uses"]
    if isinstance(uses, bool) or not isinstance(uses, int) or uses <= 0:
        raise ValueError("Measured positive integer channel uses required")
    if result.get("all_transmitted_streams_counted") is not True:
        raise ValueError("All transmitted streams must be counted")
    bit_count = result.get("serialized_bits")
    if bit_count is not None and (isinstance(bit_count, bool) or not isinstance(bit_count, int) or bit_count < 0):
        raise ValueError("Serialized bits must be a nonnegative integer")
    flop_count = result.get("profiled_forward_flops")
    if flop_count is not None and (isinstance(flop_count, bool) or not isinstance(flop_count, (int, float))
                                   or not math.isfinite(flop_count) or flop_count <= 0):
        raise ValueError("Profiled FLOPs must be finite positive numbers")
    # Metrics have the same RGB range and parameters as the tiled evaluator.
    import torch
    from pytorch_msssim import ssim, ms_ssim
    ref = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float()
    rec = torch.from_numpy(np.clip(reconstruction, 0, 1).transpose(2, 0, 1).copy()).unsqueeze(0).float()
    mse = float(torch.mean((ref.double() - rec.double()) ** 2))
    row = {"image": image_id, "channel": channel, "snr_db": snr_db,
           "trial": trial, **info,
           "psnr_db": float("inf") if mse == 0 else float(-10*np.log10(mse)),
           "ssim": float(ssim(rec, ref, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))),
           "msssim": float(ms_ssim(rec, ref, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))),
           "complex_channel_uses": uses, "cbr": measured_cbr(uses, info),
           "serialized_bits": bit_count,
           "profiled_forward_flops": flop_count,
           "metric_region": "full_original_RGB", "protocol": "native_full_image_v1",
           "author_paper_protocol": bool(result.get("author_paper_protocol", False))}
    return row, np.clip(reconstruction, 0, 1)


def evaluate_image(image, transmit, *, image_id, channel, snr_db, trial, seed,
                   tile_height=256, tile_width=512):
    """transmit(tile_float_HWC, context) -> reconstruction, measured uses, etc.

    The callback applies the actual model's documented normalization. It must
    count both image and semantic streams, padded symbols, digital framing/FEC,
    and any transmitted side information. Unknown counts are rejected.
    """
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite HWC RGB image")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Expected [0,1] RGB")
    h, w = image.shape[:2]
    info = geometry(h, w, tile_height, tile_width)
    padded = np.pad(image, ((0, info["padded_height"] - h), (0, info["padded_width"] - w), (0, 0)), mode="edge")
    restored = np.empty_like(padded, dtype=np.float32)
    covered = np.zeros(padded.shape[:2], dtype=np.uint8)
    uses = 0
    bits = []
    tile_flops = []
    tile_rows = []
    for tile_index, (y, x) in enumerate((y, x) for y in range(0, padded.shape[0], tile_height)
                                      for x in range(0, padded.shape[1], tile_width)):
        key = f"{seed}|{image_id}|{channel}|{snr_db}|{trial}|{tile_index}"
        tile_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
        context = {"image_id": image_id, "channel": channel, "snr_db": snr_db,
                   "trial": trial, "seed": tile_seed, "tile_index": tile_index}
        result = transmit(padded[y:y+tile_height, x:x+tile_width].copy(), context)
        reconstruction = np.asarray(result["reconstruction"], dtype=np.float32)
        if reconstruction.shape != (tile_height, tile_width, 3) or not np.isfinite(reconstruction).all():
            raise ValueError("Model returned wrong geometry or nonfinite reconstruction")
        count = result["complex_channel_uses"]
        if isinstance(count, bool) or int(count) != count or count <= 0:
            raise ValueError("Actual positive integer channel-use count required")
        if result.get("all_transmitted_streams_counted") is not True:
            raise ValueError("Incomplete communication accounting")
        uses += int(count)
        bit_count = result.get("serialized_bits")
        if bit_count is not None and (isinstance(bit_count, bool) or not isinstance(bit_count, int) or bit_count < 0):
            raise ValueError("Serialized bits must be a nonnegative integer")
        flop_count = result.get("profiled_forward_flops")
        if flop_count is not None and (isinstance(flop_count, bool) or not isinstance(flop_count, (int, float))
                                       or not math.isfinite(flop_count) or flop_count <= 0):
            raise ValueError("Profiled FLOPs must be finite positive numbers")
        bits.append(bit_count)
        tile_flops.append(flop_count)
        restored[y:y+tile_height, x:x+tile_width] = np.clip(reconstruction, 0, 1)
        covered[y:y+tile_height, x:x+tile_width] += 1
        tile_rows.append({**context, "complex_channel_uses": int(count)})
    if not np.all(covered == 1):
        raise AssertionError("Every padded pixel must be reconstructed exactly once")
    restored = restored[:h, :w]
    # Shared implementation for all frameworks; identical RGB range and parameters.
    import torch
    from pytorch_msssim import ssim, ms_ssim
    ref = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float()
    rec = torch.from_numpy(restored.transpose(2, 0, 1).copy()).unsqueeze(0).float()
    mse = float(torch.mean((ref.double() - rec.double()) ** 2))
    row = {"image": image_id, "channel": channel, "snr_db": snr_db, "trial": trial,
           **info, "psnr_db": float("inf") if mse == 0 else float(-10*np.log10(mse)),
           "ssim": float(ssim(rec, ref, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))),
           "msssim": float(ms_ssim(rec, ref, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))),
           "complex_channel_uses": uses, "cbr": uses / (3*h*w),
           "serialized_bits": sum(bits) if all(isinstance(v, int) and v >= 0 for v in bits) else None,
           "profiled_forward_flops": sum(tile_flops) if all(v is not None for v in tile_flops) else None,
           "metric_region": "full_original_RGB", "tile_audit": tile_rows,
           "protocol": "matched_tiled_v1", "author_paper_protocol": False}
    return row, restored


def validate_comparison(manifests):
    """Reject mixed cohorts; an empty/ineligible comparison is preferable to a false ranking."""
    if len(manifests) < 2:
        raise ValueError("At least two eligible measured runs required")
    common = ["dataset_manifest_sha256", "evaluation_protocol", "channel_definition", "receiver_csi",
              "snr_grid", "trials", "metric_implementation", "flops_convention",
              "rate_budget", "comparison_track"]
    first = manifests[0]
    for run in manifests:
        if run.get("upstream_authorship_verified") is not True:
            if run.get("implementation_provenance") != "audited_paper_reproduction" or run.get("paper_fidelity_review_verified") is not True:
                raise ValueError(f"Unverified authorship or unreviewed reproduction: {run.get('model')}")
        if run.get("source_integrity_verified") is not True or run.get("model_runtime_patches") != []:
            raise ValueError("Modified or unverified model code")
        if run.get("training_complete") is not True or run.get("evaluation_complete") is not True:
            raise ValueError("Incomplete experiment")
        if run.get("review_receipt_verified") is not True:
            raise ValueError("Missing architecture/training-contract review")
        epochs = run.get("epochs_per_required_stage", 0)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 50 <= epochs <= 150:
            raise ValueError("Completed epochs must be an integer between 50 and 150")
        if run.get("images_per_epoch") != 800:
            raise ValueError("Incomplete source-image exposure")
        if epochs < 150:
            stop = run.get("early_stopping_receipt", {})
            if not (50 <= epochs < 150 and stop.get("stopped_on_plateau") is True
                    and stop.get("minimum_epochs") == 50 and stop.get("patience") == 20
                    and stop.get("relative_min_delta") == 0.001 and stop.get("epochs_completed") == epochs):
                raise ValueError("Insufficient epochs without a validated stopping receipt")
        if run.get("training_image_exposures") != epochs * 800:
            raise ValueError("Epoch count and measured exposures disagree")
        if not run.get("training_geometry"):
            raise ValueError("Training geometry must be disclosed")
        for key in common:
            if key not in run or run[key] is None or run[key] != first.get(key):
                raise ValueError(f"Incompatible or missing {key}: {run.get('model')}")
        if run["evaluation_protocol"] not in {"native_full_image_v1", "matched_tiled_v1"}:
            raise ValueError("Unvalidated evaluation geometry protocol")
    return {"eligible_models": [run["model"] for run in manifests], "comparison_contract": {k: first[k] for k in common}}
