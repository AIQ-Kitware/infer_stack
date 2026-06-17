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

import requests  # noqa: F401  (cli_mod.requests is patched in tests)
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
from .commands_leasing import (
    AcquireCLI,
    LeasesCLI,
    ReleaseCLI,
    RenewCLI,
    RunCLI,
    SecretsCLI,
    ServeCLI,
)
from .commands_meta import (
    ConfigModalCLI,
    ConfigPathsCLI,
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
    PullCLI,
    PurgeCLI,
    RestartCLI,
    StartCLI,
    StatusCLI,
    StopCLI,
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


class ManageCLI(scfg.ModalCLI):
    description = (
        'Render and run vLLM serving profiles through the Compose or KubeAI '
        'backends. Primary workflow: setup -> render -> up (or deploy).'
    )

    # Backs the modal ``--version`` flag (scriptconfig reads ``__version__``).
    # The ``version`` *subcommand* is registered below under a non-colliding
    # attribute name; its CLI name comes from ``VersionCLI.__command__``.
    __version__ = __version__

    # Meta / introspection
    version_command = VersionCLI
    config = ConfigModalCLI
    paths = ConfigPathsCLI  # top-level alias for `config paths`

    # Config / profile management
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

    # Runtime
    up = UpCLI
    down = DownCLI
    purge = PurgeCLI
    deploy = DeployCLI
    status = StatusCLI
    env = EnvCLI
    diagnose = DiagnoseCLI
    wait_ready = WaitReadyCLI
    smoke_test = SmokeTestCLI
    benchmark = BenchmarkCLI
    ollama_pull = OllamaPullCLI
    ollama_list = OllamaListCLI
    ollama_ps = OllamaPsCLI

    # Leasing model (acquire/release/run/serve + status)
    acquire = AcquireCLI
    release = ReleaseCLI
    renew = RenewCLI
    run = RunCLI
    serve = ServeCLI
    leases = LeasesCLI
    secrets = SecretsCLI

    # Compose day-2-ops wrappers
    logs = LogsCLI
    ps = PsCLI
    restart = RestartCLI
    pull = PullCLI
    start = StartCLI
    stop = StopCLI


def main(argv=None) -> int:
    rv = ManageCLI.main(argv=argv)
    return int(rv) if rv is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
