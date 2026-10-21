"""Synthetic adapter checks, not model training or Kodak benchmark results."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
from PIL import Image
from native_geometry import pad_image, unpad_image, measured_cbr

root = Path(__file__).resolve().parents[2]
source = root / 'Models/author_releases/SwinJSCC/a6d0e6da53548976acbe9317839a077ef31f190f/data/datasets.py'
spec = importlib.util.spec_from_file_location('swin_author_dataset_probe', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
with tempfile.TemporaryDirectory(prefix='synthetic_swin_geometry_') as folder:
    for height, width in [(512, 768), (768, 512), (513, 769), (769, 513)]:
        rng = np.random.default_rng(height + width)
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = Path(folder) / 'fixture.png'
        Image.fromarray(image).save(path)
        native, _ = module.Datasets([folder])[0]
        padded, geometry = pad_image(image)
        restored = unpad_image(padded, geometry)
        assert np.array_equal(restored, image)
        native_hw = list(native.shape[-2:])
        if height % 128 == 0 and width % 128 == 0:
            assert native_hw == [height, width]
            native_pixels = (native.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            assert np.array_equal(native_pixels, image)
        # C=96 produces one complex use per 16 RGB samples for aligned inputs.
        uses = padded.shape[0] * padded.shape[1] * 96 // 512
        rows.append({**geometry, 'native_author_output_hw': native_hw,
                     'synthetic_roundtrip': 'PASS', 'example_padding_inclusive_uses': uses,
                     'example_cbr_original_pixels': measured_cbr(uses, geometry)})
    training = module.HR_image(SimpleNamespace(image_dims=(3, 256, 256)), [folder])
    assert list(training[0].shape) == [3, 256, 256]
report = {'status': 'PASS', 'synthetic_only': True, 'model_forward_checked': False,
          'author_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'training_transform': repr(training.transform), 'checks': rows}
out = root / 'Results/reproduction_audit/swin_geometry_adapter_checks.json'
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
