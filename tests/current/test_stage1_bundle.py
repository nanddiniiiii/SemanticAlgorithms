"""Exercise the transported checkout in a fresh process, not the local checkout."""
import ast
import base64
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


class Stage1Bundle(unittest.TestCase):
    def test_payload_resolves_project_root_in_fresh_process(self):
        root = Path(__file__).resolve().parents[2]
        path = root/'Kaggle/stage1_shared_training/kernel.py'
        if not path.is_file():
            self.skipTest('Build Stage1 jobs before testing their actual transport payload')
        tree = ast.parse(path.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'blob' for target in node.targets))
        payload = base64.b64decode(ast.literal_eval(assignment.value.args[0]))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for name in archive.namelist(): (target/name).resolve().relative_to(target.resolve())
                archive.extractall(target)
            self.assertTrue((target/'sources.lock.json').is_file())
            env = {**os.environ, 'SEMANTIC_LOSSLESS_ROOT': str(target), 'PYTHONPATH': str(target/'src')}
            result = subprocess.run([sys.executable, '-c',
                'from semantic_lossless.paths import project_root; print(project_root())'],
                cwd=target, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), target.resolve())
