"""Frozen membership must not admit validation images as training or changed bytes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FrozenDataset(unittest.TestCase):
    def test_freeze_and_changed_bytes_rejected(self):
        try:
            from PIL import Image
            import numpy as np
            import torch
        except ImportError:
            self.skipTest('Requires model environment')
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);images=folder/'images';images.mkdir()
            rng=np.random.default_rng(1)
            for i in range(4):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(images/f'{i}.png')
            spec={'minimum_dimension':16,'sources':[{'dataset':'fixture','root':str(images),
                'official_split':'train','counts':{'train':2,'val':1,'test':1},
                'source_url':'fixture','license_note':'synthetic'}]}
            path=folder/'spec.json';path.write_text(json.dumps(spec));manifest=folder/'manifest.json'
            command=[sys.executable,str(root/'scripts/freeze_custom_dataset.py'),'--spec',str(path),'--output',str(manifest)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            from semantic_lossless.config import Config
            from semantic_lossless.data import FrozenImages
            config=Config('adjscc',crop_size=16,model_options={'dataset_manifest':str(manifest)})
            data=FrozenImages(config,'train');self.assertEqual(len(data),2)
            self.assertEqual(tuple(data[0]['image'].shape),(3,16,16))
            Path(data.rows[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):FrozenImages(config,'train')
            spec['sources'][0]['official_split']='val';path.write_text(json.dumps(spec));manifest.unlink()
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(manifest.exists())
