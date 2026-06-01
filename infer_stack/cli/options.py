from __future__ import annotations

from ..paths import CONFIG_DIR_ENV
from ..paths import DATA_DIR_ENV
import scriptconfig as scfg

# ---------------------------------------------------------------------------
# DataConfig mixins for common override flags
# ---------------------------------------------------------------------------


class _PathOverridesMixin(scfg.DataConfig):
    """Adds global ``--config-dir`` / ``--data-dir`` to a subcommand."""

    config_dir = scfg.Value(
        None,
        type=str,
        help=(
            f"Directory containing config.yaml / models.yaml. Defaults to "
            f"~/.config/infer_stack (XDG_CONFIG_HOME) or ${CONFIG_DIR_ENV} when set."
        ),
    )
    data_dir = scfg.Value(
        None,
        type=str,
        help=(
            f"Directory for rendered artifacts and bind-mount state. Defaults to "
            f"~/.local/share/infer_stack (XDG_DATA_HOME) or ${DATA_DIR_ENV} when set."
        ),
    )


class _BackendOverrideMixin(scfg.DataConfig):
    backend = scfg.Value(None, choices=["compose", "kubeai"], help="Active backend override.")


class _ComposeOverrideMixin(scfg.DataConfig):
    compose_cmd = scfg.Value(None, type=str, help="Docker compose command override (e.g. 'podman compose').")


class _ProfileOverrideMixin(scfg.DataConfig):
    profile = scfg.Value(None, type=str, help="Active profile override (sets config.active_profile).")


class _PortOverridesMixin(scfg.DataConfig):
    litellm_port = scfg.Value(None, type=int)
    open_webui_port = scfg.Value(None, type=int)
    postgres_port = scfg.Value(None, type=int)


class _ClusterOverridesMixin(scfg.DataConfig):
    namespace = scfg.Value(None, type=str, help="Kubernetes namespace for kubeai deployments.")
    ingress_host = scfg.Value(None, type=str, help="Ingress host (kubeai only).")
    ingress_enabled = scfg.Value(
        None,
        isflag=True,
        alias=["ingress"],
        help="Enable cluster ingress (kubeai only); use --no-ingress to disable.",
    )


class _AllowUnsupportedMixin(scfg.DataConfig):
    allow_unsupported = scfg.Value(False, isflag=True, help="Allow validation errors when planning/rendering.")


class _SimulateHardwareMixin(scfg.DataConfig):
    simulate_hardware = scfg.Value(
        None,
        type=str,
        help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80). Useful for planning on smaller machines.",
    )


class _AllowedGpusMixin(scfg.DataConfig):
    allowed_gpus = scfg.Value(
        None,
        type=str,
        help=(
            "Restrict placement to a comma-separated list of GPU indices "
            "(e.g. '1' or '1,3'). Real indices are preserved — the rendered "
            "compose stack pins device_ids to exactly those GPUs. May also "
            "be set via INFER_STACK_ALLOWED_GPUS. Useful for integration "
            "tests on machines where some GPUs are tied up."
        ),
    )


class _PlanOverridesCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
    _AllowedGpusMixin,
):
    """Standard set of overrides for any command that builds a plan."""

    pass


class _SwitchPathOverridesCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
    _AllowedGpusMixin,
):
    """Overrides for commands that take a positional ``profile`` (no --profile)."""

    pass
