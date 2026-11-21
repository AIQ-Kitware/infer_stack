"""Signal-safe lifetime for the command and lease owned by ``run``."""

from __future__ import annotations

import signal
import subprocess
import threading
from contextlib import contextmanager


class RunCancelled(SystemExit):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(128 + signum)


@contextmanager
def cancellation_signals():
    """Turn SIGTERM into stack unwinding; protect cleanup from repeat signals.

    Python's default SIGTERM action exits without running finally blocks.
    Install before acquisition so controller rollback also covers readiness.
    Embedded calls from other threads retain their existing signal behavior.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def cancel(signum, frame):
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise RunCancelled(signum)

    try:
        for sig in signals:
            signal.signal(sig, cancel)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_child(command, *, env, stop_timeout=10):
    """Forward cancellation before waiting, then allow the lease to release.

    A signal may target just the wrapper, rather than its whole process group.
    Give a Docker client time to forward it to the container's init process;
    subprocess.run would immediately kill the client on stack unwinding.
    """
    with subprocess.Popen(command, env=env) as proc:
        try:
            return int(proc.wait())
        except BaseException as exc:
            signum = (
                exc.signum
                if isinstance(exc, RunCancelled)
                else signal.SIGINT
                if isinstance(exc, KeyboardInterrupt)
                else signal.SIGTERM
            )
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
