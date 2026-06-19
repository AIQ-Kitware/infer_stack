#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""scriptconfig-based CLI for infer-stack.

Each subcommand is a ``scfg.DataConfig`` subclass; ``ManageCLI`` composes
them into a single ``scfg.ModalCLI`` exposed as the ``infer-stack`` entry
point. Because every subcommand is a ``DataConfig``, the same class can
be invoked from the shell (``infer-stack render --profile X``) or from
Python (``RenderCLI.main(argv=False, profile='X')``).

This package was split out of a single ``cli.py`` module. The layers are:

* ``context``  — path/config/env/override/plan helpers (no other cli deps).
* ``probes``   — pure readiness/model-selection helpers over a deployment.
* ``compose``  — compose + LiteLLM + preflight + diagnostics helpers.
* ``options``  — shared ``DataConfig`` mixins for override flags.
* ``commands_profile``  — profile/config management subcommands.
* ``commands_runtime``  — up/down/deploy/status/env + day-2-ops wrappers.
* ``commands_smoke``    — diagnose/wait/smoke-test/benchmark subcommands.

``infer_stack.cli`` re-exports the previously top-level names so existing
``from infer_stack.cli import ...`` imports keep working.
"""

from __future__ import annotations

import scriptconfig as scfg

from .. import __version__
from ..kubeai_ops import (  # noqa: F401
    CommandError,
    deploy_rendered_artifacts,
)
from ..kubeai_ops import (  # noqa: F401
    print_status as kubeai_print_status,
)

# Keep submodules importable as attributes (e.g. ``infer_stack.cli.commands_runtime``)
# so tests can patch seams where they are actually looked up.
from . import (  # noqa: F401
    commands_catalog,
    commands_leasing,
    commands_meta,
    commands_profile,
    commands_runtime,
    commands_smoke,
    compose,
    context,
    options,
    probes,
)
from .commands_catalog import CatalogModalCLI
from .commands_leasing import (
    AcquireCLI,
    ApplyCLI,
    EvictCLI,
    LeasesCLI,
    ReleaseCLI,
    RenewCLI,
    RunCLI,
    ServeCLI,
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
from .commands_profile import (
    DescribeProfileCLI,
    ExplainCLI,
    InitCLI,
    KubeaiSyncResourceProfilesCLI,
    ListModelsCLI,
    ListProfilesCLI,
    LockCLI,
    RenderCLI,
    ResolveCLI,
    SetupCLI,
    SwitchCLI,
    ValidateCLI,
    VerifyProfileCLI,
)
from .commands_runtime import (
    DeployCLI,
    DownCLI,
    EnvCLI,
    LogsCLI,
    OllamaListCLI,
    OllamaPsCLI,
    OllamaPullCLI,
    PsCLI,
    PurgeCLI,
    StackModalCLI,
    StatusCLI,
    UpCLI,
)
from .commands_smoke import (
    BenchmarkCLI,
    DiagnoseCLI,
    SmokeTestCLI,
    WaitReadyCLI,
)
from .compose import (  # noqa: F401
    _compose_has_service,
    _litellm_delete_missed_config_model,
)
from .context import (  # noqa: F401
    apply_config_overrides,
    backend_name,
    build_plan,
    config_for_runtime,
    config_path,
    effective_allow_unsupported,
    effective_inventory,
    ensure_renderable,
    generated_dir,
    has_runtime_overrides,
    kubeai_generated_dir,
    load_config,
    models_path,
    plan_path,
    render_is_stale,
    runtime_dir_for_config,
    save_plan,
)
from .probes import (  # noqa: F401
    _default_model_for_deployment,
    _resolve_smoke_protocol_from_deployment,
)

# ---------------------------------------------------------------------------
# Modal CLI + entry point
# ---------------------------------------------------------------------------


class LegacyModalCLI(scfg.ModalCLI):
    """Pre-leasing profile/active-profile commands (held here; promoted out as
    they gain leasing-native behavior, then this group is removed wholesale)."""

    __command__ = 'legacy'

    # config / profile management
    setup = SetupCLI
    init = InitCLI
    resolve = ResolveCLI
    validate = ValidateCLI
    lock = LockCLI
    render = RenderCLI
    switch = SwitchCLI
    list_models = ListModelsCLI
    list_profiles = ListProfilesCLI
    explain = ExplainCLI
    describe_profile = DescribeProfileCLI
    verify_profile = VerifyProfileCLI
    kubeai_sync_resource_profiles = KubeaiSyncResourceProfilesCLI
    # active-profile runtime lifecycle
    up = UpCLI
    down = DownCLI
    purge = PurgeCLI
    deploy = DeployCLI
    env = EnvCLI
    diagnose = DiagnoseCLI
    wait_ready = WaitReadyCLI
    smoke_test = SmokeTestCLI
    benchmark = BenchmarkCLI
    ollama_pull = OllamaPullCLI
    ollama_list = OllamaListCLI
    ollama_ps = OllamaPsCLI


class ManageCLI(scfg.ModalCLI):
    description = (
        'Lease, serve, and run LLM endpoints. Primary workflow: '
        'catalog -> acquire/serve/run. Pre-leasing profile commands live under '
        '`infer-stack legacy`; see `infer-stack help tree`.'
    )

    __epilog__ = """
    Quickstart:
        infer-stack config init                 # storage + default backend
        infer-stack catalog init                # start a model catalog
        infer-stack catalog model add smol135 \\
            --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
        infer-stack catalog endpoint add --model smol135   # -> smol135-1
        infer-stack serve smol135-1             # render + bring up + wait
        infer-stack leases                      # what is desired vs running
        infer-stack test smol135-1              # one real generation
        infer-stack release --all               # tear it back down

    Mental model:
        catalog (what can run) -> serve/acquire (declare intent, a lease)
        -> the controller reconciles a compose project (gateway + models +
        Open WebUI) onto your GPUs. `serve --no-apply` writes that project
        without starting it (then `apply`); `leases` shows desired vs actual.
        Run `infer-stack help tree` for the whole command surface.
    """

    # Backs the modal ``--version`` flag (scriptconfig reads ``__version__``).
    # The ``version`` *subcommand* is registered below under a non-colliding
    # attribute name; its CLI name comes from ``VersionCLI.__command__``.
    __version__ = __version__

    # Meta / introspection
    version_command = VersionCLI
    help = HelpModalCLI  # `infer-stack help tree` — the whole surface at a glance
    config = ConfigModalCLI
    paths = ConfigPathsCLI  # top-level alias for `config paths`
    status = StatusCLI

    # Catalog editor (models / endpoints / hosts / bundles — no raw YAML)
    catalog = CatalogModalCLI

    # Pre-leasing profile world (grouped; was ~25 top-level verbs)
    legacy = LegacyModalCLI

    # Leasing model (acquire/release/run/serve + status)
    acquire = AcquireCLI
    release = ReleaseCLI
    evict = EvictCLI  # force-tear-down released (idle) models to free GPUs
    renew = RenewCLI
    run = RunCLI
    serve = ServeCLI
    # Reconcile primitives (lease-free): render desired -> disk, apply disk -> up
    render = LeasingRenderCLI  # write the compose project for desired, no `up`
    apply = ApplyCLI  # bring the desired set up (the trigger for serve --no-apply)
    wait = WaitCLI  # block until endpoints are ready (serve --no-wait fan-out)
    leases = LeasesCLI
    tui = TuiCLI  # live Textual monitor + controls (opt-in: infer-stack[tui])
    test = TestCLI  # smoke-test a served endpoint through the front door
    env = LeasingEnvCLI  # managed env-file: path / read KEY / set KEY=VALUE

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


def main(argv=None) -> int:
    rv = ManageCLI.main(argv=argv)
    return int(rv) if rv is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
