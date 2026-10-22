"""Scoped legacy imports and verified external source identities."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import subprocess
import sys
import threading
_lock = threading.RLock()

def verify_revision(root, expected):
    if not expected:
        raise ValueError('External source requires an explicit Git revision')
    root = Path(root).resolve()
    result = subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','-C',str(root),'rev-parse','HEAD'], capture_output=True, text=True, check=True)
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f'External revision mismatch: expected {expected}, found {actual}')
    dirty = subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','-C',str(root),'status','--porcelain'], capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise ValueError('External dependency has uncommitted changes; commit and configure its exact revision')
    return actual

@contextmanager
def scoped_imports(root, prefixes):
    """Restore sys.path and colliding legacy module names after model creation.

    Use subprocess workers for packages with delayed imports or incompatible
    dependencies. This scope is intended for the audited pinned Swin source.
    """
    with _lock:
        old_path = sys.path[:]
        matches = lambda name: any(name == prefix or name.startswith(prefix+'.') for prefix in prefixes)
        saved = {name:module for name,module in sys.modules.items() if matches(name)}
        for name in saved:
            del sys.modules[name]
        sys.path.insert(0, str(Path(root).resolve()))
        try:
            importlib.invalidate_caches()
            yield
        finally:
            sys.path[:] = old_path
            for name in list(sys.modules):
                if matches(name):
                    del sys.modules[name]
            sys.modules.update(saved)
