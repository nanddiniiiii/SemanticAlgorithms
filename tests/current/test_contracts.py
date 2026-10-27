import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from src.artifacts import export_manifest, write_json
from src.config import Config, load_config
from src.data import Images, cityscapes_label_lut
from src.events import Events, collect


class Contracts(unittest.TestCase):
    def test_ideal_source_codec_does_not_claim_a_radio_snr(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(Events(folder,Config('sqgan')).identity['snr_db'])
            self.assertEqual(Events(folder,Config('jcm')).identity['snr_db'],10)

    def test_cityscapes_official_labels_and_ignore_classes(self):
        table=cityscapes_label_lut()
        self.assertEqual(len(table),256)
        self.assertEqual((table[7],table[26],table[33]),(0,13,18))
        self.assertEqual((table[0],table[9],table[29],table[255]),(19,19,19,19))
        self.assertEqual(set(table),set(range(20)))

    def test_config_rejects_nonfinite_and_unknown(self):
        with self.assertRaises(ValueError):
            Config('deepjscc', snr_db=float('nan')).validate()
        with self.assertRaises(ValueError):
            Config('deepjscc', reproduction_status='verified').validate()
        self.assertEqual(Config('deepjscc').validate().reproduction_status,'unverified')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps({'algorithm':'deepjscc','epohcs':1}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_collector_deduplicates_and_ignores_partial_line(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                Events(root,Config('deepjscc')).emit('test','completed',{'psnr_db':30})
            with (root/'events.jsonl').open('a') as stream:
                stream.write('{broken')
            self.assertTrue((root/'metrics.csv').is_file())
            csv=root/'combined.csv'
            self.assertEqual(collect(root,csv),1)
            self.assertEqual(collect(root,csv),0)
            lock=csv.with_suffix('.csv.lock'); lock.touch()
            with self.assertRaises(RuntimeError):
                collect(root,csv)

    def test_export_hash_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/'test.json',{'value':float('inf')})
            self.assertIsNone(json.loads((root/'test.json').read_text())['value'])
            self.assertEqual(export_manifest(root)['file_count'],1)
            self.assertEqual(export_manifest(root)['file_count'],1)

    def test_image_dataset_is_picklable_for_windows_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'example.png').touch()
            dataset=Images(root,Config('deepjscc'),'test')
            restored=pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored.paths,dataset.paths)


if __name__=='__main__':
    unittest.main()
