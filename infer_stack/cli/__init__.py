#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""kwconf-based CLI for infer-stack.

Each subcommand is a ``kw.Config`` subclass; ``ManageCLI`` composes
them into a single ``kw.ModalCLI`` exposed as the ``infer-stack`` entry
point. Because every subcommand is a ``DataConfig``, the same class can
be invoked from the shell (``infer-stack render --profile X``) or from
Python (``RenderCLI.main(argv=False, profile='X')``).

The layers are:

* ``context``  — path/config/env/override helpers (no other cli deps).
* ``probes``   — pure readiness/model-selection helpers over a deployment.
* ``compose``  — compose + LiteLLM helpers.
* ``options``  — shared ``DataConfig`` mixins for override flags.
* ``commands_catalog`` — catalog editor (models/endpoints/hosts/bundles).
* ``commands_leasing`` — acquire/release/run/leases/test + reconcile.
* ``commands_runtime`` — ``status`` + ``stack`` day-2 compose wrappers.
* ``commands_meta``    — version/help/config introspection.
"""

from __future__ import annotations

import kwconf as kw

from .. import __version__

# Keep submodules importable as attributes (e.g. ``infer_stack.cli.commands_runtime``)
# so tests can patch seams where they are actually looked up.
from . import (  # noqa: F401
    commands_catalog,
    commands_leasing,
    commands_meta,
    commands_mock,
    commands_runtime,
    context,
    options,
)
from .commands_catalog import CatalogModalCLI
from .commands_mock import MockModalCLI
from .commands_leasing import (
    AcquireCLI,
    ApplyCLI,
    CleanCLI,
    EvictCLI,
    GcCLI,
    LeasesCLI,
    MeasureCLI,
    ReleaseCLI,
    RenewCLI,
    NetworkModalCLI,
    RoutesModalCLI,
    SecretsModalCLI,
    RunCLI,
    TestCLI,
    TuiCLI,
    WaitCLI,
)
from .commands_leasing import EnvCLI as LeasingEnvCLI
from .commands_leasing import RenderCLI as LeasingRenderCLI
from .commands_meta import (
    ConfigModalCLI,
    ConfigPathsCLI,
    HelpModalCLI,
    VersionCLI,
)
from .commands_runtime import (
    DoctorCLI,
    LogsCLI,
    PsCLI,
    StackModalCLI,
    StatusCLI,
)

# ---------------------------------------------------------------------------
# Modal CLI + entry point
# ---------------------------------------------------------------------------


class ManageCLI(kw.ModalCLI):
    # The program name in usage and error messages (kwconf would use the class
    # name, "ManageCLI").
    __prog__ = 'infer-stack'

    description = (
        'Lease, acquire, and run LLM endpoints. Primary workflow: '
        'catalog -> acquire/run. See `infer-stack help tree`.'
    )

    __epilog__ = """
    Quickstart:
        infer-stack config init                 # storage + default backend
        infer-stack catalog init                # start a model catalog
        infer-stack catalog model add smol135 \\
            --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
        infer-stack catalog endpoint add --model smol135   # -> smol135-1
        infer-stack acquire smol135-1           # render + bring up + wait
        infer-stack leases                      # what is desired vs running
        infer-stack test smol135-1              # one real generation
        infer-stack release --all               # tear it back down

    Mental model:
        catalog (what can run) -> acquire (declare intent, a lease; --ttl for a
        soft TTL) -> the controller reconciles a compose project (gateway +
        models + Open WebUI) onto your GPUs. `acquire --no-apply` writes that
        project without starting it (then `apply`); `leases` shows desired vs
        actual. Run `infer-stack help tree` for the whole command surface.
    """

    # Backs the modal ``--version`` flag (kwconf reads ``__version__``).
    # The ``version`` *subcommand* is registered below under a non-colliding
    # attribute name; its CLI name comes from ``VersionCLI.__command__``.
    __version__ = __version__

    # Meta / introspection
    version_command = VersionCLI
    help = HelpModalCLI  # `infer-stack help tree` — the whole surface at a glance
    config = ConfigModalCLI
    paths = ConfigPathsCLI  # top-level alias for `config paths`
    status = StatusCLI
    doctor = DoctorCLI  # preflight the configured backend's prerequisites

    # Catalog editor (models / endpoints / hosts / bundles — no raw YAML)
    catalog = CatalogModalCLI

    # Deterministic mock endpoint for tests/dry-runs (no GPU, no API key)
    mock = MockModalCLI

    # Leasing model (acquire/release/run + status)
    acquire = AcquireCLI  # stand up endpoints: lease + up + wait (--ttl for soft TTL)
    release = ReleaseCLI
    evict = EvictCLI  # force-tear-down released (idle) models to free GPUs
    gc = GcCLI  # reclaim TTL-expired (leaked) leases + free their GPUs
    clean = CleanCLI  # dry-run by default; -f releases + tears down everything
    renew = RenewCLI
    run = RunCLI
    # Reconcile primitives (lease-free): render desired -> disk, apply disk -> up
    render = LeasingRenderCLI  # write the compose project for desired, no `up`
    apply = ApplyCLI  # bring the desired set up (the trigger for acquire --no-apply)
    wait = WaitCLI  # block until endpoints are ready (acquire --no-wait fan-out)
    leases = LeasesCLI
    routes = RoutesModalCLI  # inspect/seed/prune the LiteLLM route registry
    network = NetworkModalCLI  # stable per-service addresses + upstream routing check
    tui = TuiCLI  # live Textual monitor + controls (opt-in: infer-stack[tui])
    test = TestCLI  # smoke-test a served endpoint through the front door
    measure = MeasureCLI  # measure an endpoint's real VRAM requirement (placement.min_vram_gib)
    env = LeasingEnvCLI  # managed env-file: path / read KEY / set KEY=VALUE
    secrets = SecretsModalCLI  # rotate the LiteLLM master key

    # Day-2 ops on the running stack, grouped under `stack`; logs/ps also kept
    # at the top level as the two hottest convenience aliases.
    stack = StackModalCLI
    logs = LogsCLI
    ps = PsCLI


def __getattr__(name: str):
    # ``requests`` is imported lazily so a bare ``infer-stack --help`` doesn't
    # pay for it. Tests still patch the seam as ``cli_mod.requests`` — resolving
    # the attribute imports the (shared) module so the patch is global.
    if name == 'requests':
        import requests
        return requests
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def runtime_failure(exc) -> str:
    """A failed ``docker`` / ``kubectl`` command, for a person, not a traceback.

    The command (without its long path arguments), the runtime's own last
    words, and a hint for the failures with a known cause.

    >>> import subprocess
    >>> ex = subprocess.CalledProcessError(1, ['docker', 'compose', '-p', 'x', 'up', '-d'],
    ...     stderr='Error: Bind for :::14042 failed: port is already allocated')
    >>> print(runtime_failure(ex))
    `docker compose up -d` failed:
      Error: Bind for :::14042 failed: port is already allocated
      port 14042 is in use by something else (`ss -ltnp | grep :14042` shows what); stop it, then `infer-stack apply`
    >>> ex.stderr = 'failed to bind host port 0.0.0.0:13000/tcp: address already in use'
    >>> runtime_failure(ex).splitlines()[-1].split(' (')[0]
    '  port 13000 is in use by something else'
    """
    import re

    cmd = [str(a) for a in (exc.cmd if isinstance(exc.cmd, (list, tuple)) else [exc.cmd])]
    # Drop option values that are paths or project names: the verb is what reads.
    shown, skip = [], False
    for arg in cmd:
        if skip:
            skip = False
            continue
        if arg in ('-f', '-p', '--env-file', '-n', '--namespace'):
            skip = True
            continue
        shown.append(arg)
    text = '\n'.join(filter(None, [exc.stderr or '', exc.output or '']))
    last = [ln.strip() for ln in str(text).splitlines() if ln.strip()][-3:]
    lines = [f'`{" ".join(shown)}` failed:'] + [f'  {ln}' for ln in last]
    port = re.search(r'Bind for \S*?:(\d+) failed: port is already allocated'
                     r'|:(\d+)(?:/tcp)?: (?:bind: )?address already in use', str(text))
    if port:
        number = port.group(1) or port.group(2)
        lines.append(f'  port {number} is in use by something else (`ss -ltnp | grep '
                     f':{number}` shows what); stop it, then `infer-stack apply`')
    elif shown and shown[0] in ('docker', 'kubectl'):
        lines.append('  `infer-stack doctor` checks what this backend needs')
    return '\n'.join(lines)


def main(argv=None) -> int:
    import subprocess

    from ..leasing import LeaseLockError
    from ..leasing.backend import BackendTimeout

    try:
        rv = ManageCLI.main(argv=argv)
    except LeaseLockError as exc:
        # A mutating verb (acquire/release/gc/evict/apply) could not get the
        # cross-process lock; surface the actionable diagnosis, not a traceback.
        raise SystemExit(str(exc))
    except subprocess.CalledProcessError as exc:
        # The runtime refused (a port in use, a daemon down): its words and a
        # hint, not our stack. The controller already rolled back what it had
        # committed, or left the change pending for `infer-stack apply`.
        raise SystemExit(runtime_failure(exc))
    except BackendTimeout as exc:
        raise SystemExit(str(exc))
    return int(rv) if rv is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
