"""OS-owned advisory locks: process termination releases ownership automatically."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    try:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:yield True
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    finally:stream.close()
