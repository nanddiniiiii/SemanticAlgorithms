"""Real, frozen InternImage Cityscapes predictions; never fabricated labels."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

CONFIG_NAME = "upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py"
CHECKPOINT_NAME = CONFIG_NAME.removesuffix(".py") + ".pth"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assets_from(contract):
    assets = contract.get("model_assets", {})
    return assets.get("SQGAN", assets.get("SQ_GAN", assets.get("SQ-GAN", {})))


def checked_asset(settings, name):
    value, expected = settings.get(name), settings.get(name + "_sha256")
    if not value or not expected or len(expected) != 64:
        raise ValueError(f"SQGAN requires internimage.{name} and independently recorded {name}_sha256")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing InternImage {name}: {path}")
    if sha256(path) != expected.lower():
        raise ValueError(f"InternImage {name} SHA256 mismatch: {path}")
    return path


def validate_assets(settings):
    config = checked_asset(settings, "config")
    checkpoint = checked_asset(settings, "checkpoint")
    if config.name != CONFIG_NAME or checkpoint.name != CHECKPOINT_NAME:
        raise ValueError("SQGAN requires the author's InternImage XL mapillary2cityscapes config/checkpoint pair")
    source = Path(settings.get("source", "")).resolve()
    if not (source / "segmentation" / "mmseg_custom").is_dir():
        raise FileNotFoundError("internimage.source must contain the real InternImage segmentation/mmseg_custom tree")
    return config, checkpoint, source


class InternImagePredictor:
    def __init__(self, settings, device="cuda:0"):
        config, checkpoint, source = validate_assets(settings)
        sys.path.insert(0, str(source / "segmentation"))
        # These imports register the native backbone and compiled DCNv3 operator.
        importlib.import_module("mmseg_custom")
        importlib.import_module("mmcv_custom")
        from mmcv import Config
        from mmseg.apis import init_segmentor, inference_segmentor
        cfg = Config.fromfile(str(config))
        self.model = init_segmentor(cfg, str(checkpoint), device=device)
        self.inference = inference_segmentor
        self.provenance = {"config": str(config), "config_sha256": sha256(config),
                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                           "source": str(source), "source_commit": settings.get("source_commit"),
                           "label_kind": "predicted Cityscapes train IDs, not ground truth"}

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"))
        result = self.inference(self.model, rgb[:, :, ::-1].copy())
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError("Expected one actual mmsegmentation semantic prediction")
        mask = np.asarray(result[0])
        if mask.shape != rgb.shape[:2] or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("InternImage output must retain source geometry and integer train IDs")
        if mask.min() < 0 or mask.max() > 18:
            raise ValueError("Expected Cityscapes 19-class train IDs 0..18")
        return mask.astype(np.uint8)


def generate_labels(paths, output, settings, device="cuda:0"):
    """Materialize real predictions at original resolution with content receipts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictor = InternImagePredictor(settings, device)
    records = []
    for path in paths:
        path = Path(path).resolve()
        source_hash = sha256(path)
        target = output / (source_hash + ".png")
        with Image.open(path) as image:
            mask = predictor(image)
        temporary = target.with_suffix(".tmp.png")
        Image.fromarray(mask).save(temporary)
        temporary.replace(target)
        records.append({"image": str(path), "image_sha256": source_hash,
                        "mask": str(target.resolve()), "mask_sha256": sha256(target),
                        "height": mask.shape[0], "width": mask.shape[1]})
        payload = {"schema_version": 1, "complete": False, "segmenter": predictor.provenance,
                   "records": records}
        temporary_manifest = output / "labels.json.tmp"
        temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_manifest.replace(output / "labels.json")
    payload["complete"] = True
    temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_manifest.replace(output / "labels.json")
    return payload


def verified_masks(paths, manifest, settings):
    config, checkpoint, _ = validate_assets(settings)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("Incomplete SQGAN label manifest")
    provenance = payload["segmenter"]
    if provenance["config_sha256"] != sha256(config) or provenance["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Label producer differs from pinned InternImage assets")
    records = {row["image_sha256"]: row for row in payload["records"]}
    masks = []
    for path in paths:
        digest = sha256(path)
        if digest not in records:
            raise ValueError(f"No actual InternImage prediction recorded for {path}")
        row = records[digest]
        mask_path = Path(row["mask"])
        if sha256(mask_path) != row["mask_sha256"]:
            raise ValueError(f"Semantic mask SHA256 mismatch: {mask_path}")
        with Image.open(path) as image, Image.open(mask_path) as label:
            values = np.asarray(label)
            if label.size != image.size or values.ndim != 2 or values.min() < 0 or values.max() > 18:
                raise ValueError("Semantic mask geometry/class range does not match real source")
        masks.append(mask_path)
    return masks
