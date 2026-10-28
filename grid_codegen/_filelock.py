"""The flock subset used by GRiD's caches, on POSIX and native Windows.

Locks cover byte zero on Windows and serialize independent open handles,
including handles in the same process. Closing the file releases the lock.
"""
from __future__ import annotations

import os

if os.name != "nt":
    from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
else:
    import errno
    import msvcrt
    import time

    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def flock(file, operation: int) -> None:
        fd = file if isinstance(file, int) else file.fileno()
        position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation == LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return
            if operation not in (LOCK_EX, LOCK_EX | LOCK_NB):
                raise ValueError("only exclusive flock locks are supported")
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if operation & LOCK_NB:
                        raise BlockingIOError(errno.EAGAIN, "file is locked") from error
                    time.sleep(0.05)
        finally:
            os.lseek(fd, position, os.SEEK_SET)
