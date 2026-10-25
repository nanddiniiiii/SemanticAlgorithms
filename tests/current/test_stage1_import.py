"""Canonical import must preserve variants and never duplicate measured rows."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


class Stage1Import(unittest.TestCase):
    def test_native_results_import_is_idempotent(self):
        script = Path(__file__).resolve().parents[2]/'scripts/import_stage1_outputs.py'
        spec = importlib.util.spec_from_file_location('stage1_import_test', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); module.ROOT = root
            (root/'Docs/validation').mkdir(parents=True)
            source = root/'Results/stage1_downloads/fixture/sa_ra'; source.mkdir(parents=True)
            (source/'metrics.json').write_text(json.dumps({'status': 'completed', 'images': 1, 'scope': 'test fixture only'}))
            with (source/'per_image.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['sample','sha256','psnr_db','rate_complete'])
                writer.writeheader(); writer.writerow({'sample':'fixture.png','sha256':'fixture','psnr_db':31,'rate_complete':False})
            with contextlib.redirect_stdout(io.StringIO()): module.main([])
            destination = root/'Results/metrics.csv'
            first = destination.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()): module.main([])
            self.assertEqual(destination.read_bytes(), first)
            with destination.open(newline='') as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['algorithm'], rows[0]['metric'], float(rows[0]['value'])), ('swinjscc','psnr_db',31))
            self.assertEqual(rows[0]['variant'], 'baseline_native_sa_ra')
            other = root/'Results/stage1_downloads/fixture/fixed_c32'; other.mkdir()
            (other/'metrics.json').write_bytes((source/'metrics.json').read_bytes())
            (other/'per_image.csv').write_bytes((source/'per_image.csv').read_bytes())
            with contextlib.redirect_stdout(io.StringIO()): module.main([])
            with destination.open(newline='') as stream: both = list(csv.DictReader(stream))
            self.assertEqual({row['variant'] for row in both}, {'baseline_native_sa_ra','baseline_native_fixed_c32'})
            self.assertEqual(len(both), 2)
            events = list((root/'Results/experiments/swinjscc').rglob('events.jsonl'))
            event = next(json.loads(path.read_text().splitlines()[0]) for path in events
                if json.loads(path.read_text().splitlines()[0])['native_variant'] == 'sa_ra')
            self.assertEqual(event['native_variant'], 'sa_ra')
            self.assertFalse(event['training_reproduced'])
