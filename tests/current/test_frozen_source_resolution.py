import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_custom_dataset_spec import resolve_frozen_sources


class FrozenSourceResolutionTests(unittest.TestCase):
    def fixture(self,root,flat=False):
        source=root/('release/photos' if flat else 'datasets/owner/release/photos')
        source.mkdir(parents=True)
        (source/'train.png').write_bytes(b'train fixture: no image decoder needed')
        rows=[{'dataset':'example','official_split':'train','split':'train',
               'relative_path':'train.png'}]
        spec={'sources':[{'dataset':'example','official_split':'train',
              'root':'/kaggle/input/datasets/owner/release/photos'}]}
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps({'status':'frozen','manifest_sha256':digest,
                                       'spec':spec,'images':rows}))
        return manifest,source

    def test_both_mount_layouts_need_no_recursive_scan_or_image_read(self):
        for flat in (False,True):
            with self.subTest(flat=flat),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);manifest,source=self.fixture(root,flat)
                before=manifest.read_bytes()
                with patch.object(Path,'rglob',side_effect=AssertionError('Corpus traversal forbidden')):
                    result=resolve_frozen_sources(manifest,root)
                self.assertEqual(Path(result['sources'][0]['root']),source.resolve())
                self.assertFalse(result['test_pixels_opened'])
                self.assertEqual(before,manifest.read_bytes())

    def test_modified_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,_=self.fixture(root)
            record=json.loads(manifest.read_text())
            record['images'][0]['split']='test'
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                resolve_frozen_sources(manifest,root)

    def test_missing_source_fails_instead_of_rescanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,source=self.fixture(root)
            (source/'train.png').unlink()
            with self.assertRaisesRegex(ValueError,'Expected one'):
                resolve_frozen_sources(manifest,root)
