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
    def test_sqgan_actual_bootstrap_writes_provenance(self):
        root = Path(__file__).resolve().parents[2]
        path = root/'Kaggle/sqgan_evaluation/kernel.py'
        if not path.is_file():
            self.skipTest('Build SQGAN job before testing the actual bootstrap')
        tree = ast.parse(path.read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'snapshots' for t in node.targets):
                break
            nodes.append(node)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'transported'
            class RemapRoot(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if node.value == '/kaggle/working/Semantic_Lossless':
                        return ast.copy_location(ast.Constant(str(target)), node)
                    return node
            bootstrap = ast.unparse(RemapRoot().visit(ast.Module(body=nodes, type_ignores=[])))
            program = Path(temporary)/'bootstrap.py'
            program.write_text(bootstrap, encoding='utf-8')
            env = {k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','SEMANTIC_LOSSLESS_ROOT'}}
            result = subprocess.run([sys.executable, str(program)], cwd=temporary,
                env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((target/'sources.lock.json').is_file())
            self.assertTrue((target/'source_bundle.json').is_file())

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
