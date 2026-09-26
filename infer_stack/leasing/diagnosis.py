"""Why an engine is not starting: shared by every backend.

A backend supplies the instances of one deployment (containers, or pods) as
:class:`~infer_stack.leasing.residency.Container` records, and a way to read
their recent logs. :func:`diagnose_startup` decides whether waiting could
still help, so the fail-fast wait, the "likely cause" text and the TUI's
error path behave the same on every backend.
"""

from __future__ import annotations

from typing import Callable, Sequence

#: Restarts after which the runtime's own bookkeeping says an engine is
#: looping, not loading. Two is already conclusive: a model that loads does not exit.
CRASH_LOOP_RESTARTS = 2

#: Engine log lines worth quoting verbatim, and what they mean. Each is
#: UNRECOVERABLE: the same container, restarted, fails the same way, so one
#: crash is already conclusive and there is nothing to wait for.
_ENGINE_ERROR_HINTS = (
    ('trust_remote_code', 'the model needs trust_remote_code=True '
     '(set `runtime.trust_remote_code: true` on the endpoint)'),
    ('does not recognize this architecture', 'this engine cannot read the '
     "model's architecture (it may need trust_remote_code=True)"),
    ('are not supported for now', 'this vLLM build does not implement the '
     "model's architecture"),
    ('is not supported', 'this vLLM build does not implement the '
     "model's architecture"),
    ('No supported config format', 'the model repository has no config this '
     'engine can read'),
    ('ValidationError', 'the engine rejected its own configuration'),
    ('error: unrecognized arguments', 'the engine rejected a command-line flag '
     '(check `runtime.extra_args` against this vLLM version)'),
    ('401 Client Error', 'the model is gated: set HF_TOKEN with '
     '`infer-stack env HF_TOKEN=...`'),
    ('403 Client Error', 'the model is gated: set HF_TOKEN with '
     '`infer-stack env HF_TOKEN=...`'),
)

#: Failures a RESTART CAN FIX: the hub was unreachable, a download was cut off,
#: a port was still held by the container we just replaced. `restart:
#: unless-stopped` exists for exactly these, so a restart count alone must not
#: condemn an engine -- the whole point of the policy is that the next attempt
#: succeeds. Weigh a new entry by one question: would running the same container
#: again plausibly work? If yes it belongs here; if no it belongs above.
_TRANSIENT_ENGINE_SIGNATURES = (
    'Max retries exceeded',
    'Connection reset by peer',
    'Connection refused',
    'Temporary failure in name resolution',
    'Failed to resolve',
    'Read timed out',
    'ReadTimeoutError',
    'ConnectionError',
    'IncompleteRead',
    'Consistency check failed',          # a truncated HF download
    '429 Client Error',                  # hub rate limit
    '500 Server Error',
    '502 Server Error',
    '503 Server Error',
    '504 Server Error',
    'Address already in use',
)


def classify_engine_log(logs: str) -> str | None:
    """``'fatal'``, ``'transient'``, or ``None`` when the log says neither.

    Order matters: an unrecoverable signature wins over a transient one, because
    a hub timeout earlier in the same log does not make a rejected config
    loadable. CUDA OOM counts as fatal — the allocation is deterministic, so the
    restart repeats it — and it is the one class with a documented remedy
    (:mod:`infer_stack.leasing.vram`).
    """
    from .vram import looks_like_cuda_oom

    if not logs:
        return None
    if any(needle in logs for needle, _ in _ENGINE_ERROR_HINTS):
        return 'fatal'
    if looks_like_cuda_oom(logs):
        return 'fatal'
    if any(needle in logs for needle in _TRANSIENT_ENGINE_SIGNATURES):
        return 'transient'
    return None


def _engine_error_summary(logs: str) -> str:
    """The engine's own error, quoted, with a hint when we recognise it."""
    from .vram import looks_like_cuda_oom

    if not logs.strip():
        return '; no engine log available (`infer-stack logs` for more)'
    hint = next((note for needle, note in _ENGINE_ERROR_HINTS if needle in logs), None)
    if hint is None and looks_like_cuda_oom(logs):
        hint = 'the GPU ran out of memory for this configuration'
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    quoted = ' | '.join(lines[-3:])[:400]
    summary = f'; last log: {quoted}'
    return f'{summary}; likely cause: {hint}' if hint else summary




def diagnose_startup(instances: Sequence, read_logs: Callable[[], str]) -> str | None:
    """Diagnose an engine that cannot start, or ``None`` if it may still load.

    A restart policy makes an engine that exits immediately restart forever,
    so a model that can never start looks exactly like one that is still
    loading, and an acquire waits out its whole timeout. The runtime's
    bookkeeping says an instance crashed; its LOG says whether another attempt
    could ever work (:func:`classify_engine_log`):

    * an unrecoverable error -- a rejected config or flag, an architecture
      this build does not implement, a gated repo, a CUDA OOM -- is fatal on
      the FIRST crash, because the restart reproduces it exactly;
    * a transient one -- an unreachable hub, a truncated download, a port
      still held -- is never fatal while something will retry it;
    * an unrecognised crash keeps the blunt budget of
      :data:`CRASH_LOOP_RESTARTS` restarts.

    ``instances`` must be exactly one (absent or ambiguous: ``None``). The
    returned string carries the engine's last words, because the cause is in
    its log and nowhere else.
    """
    if len(instances) != 1:
        return None                      # absent (not created yet) or ambiguous
    instance = instances[0]
    exited_for_good = (
        instance.state in {'exited', 'dead'}
        and (instance.exit_code or 0) != 0
    )
    crashed = exited_for_good or instance.restart_count >= 1
    if not crashed:
        return None                      # created, starting, or healthy
    logs = read_logs()
    verdict = classify_engine_log(logs)
    if verdict == 'transient' and instance.will_be_restarted:
        return None                      # a retry is coming, and may work
    if verdict == 'transient':
        return (f'engine is not starting (exited with code '
                f'{instance.exit_code} and will not be restarted)'
                f'{_engine_error_summary(logs)}')
    looping = instance.restart_count >= CRASH_LOOP_RESTARTS
    if verdict != 'fatal' and not (exited_for_good or looping):
        return None                      # unrecognised: keep the restart budget
    why = (f'restarted {instance.restart_count} time(s)' if instance.restart_count
           else f'exited with code {instance.exit_code}')
    return f'engine is not starting ({why}){_engine_error_summary(logs)}'
