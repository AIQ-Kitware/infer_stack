from __future__ import annotations

from ..paths import CONFIG_DIR_ENV
from ..paths import DATA_DIR_ENV
import kwconf as kw

# ---------------------------------------------------------------------------
# DataConfig mixins for common override flags
# ---------------------------------------------------------------------------


class _PathOverridesMixin(kw.Config):
    """Adds global ``--config-dir`` / ``--data-dir`` to a subcommand."""

    config_dir = kw.Value(
        None,
        type=str,
        help=(
            f'Directory containing config.yaml / models.yaml. Defaults to '
            f'~/.config/infer_stack (XDG_CONFIG_HOME) or ${CONFIG_DIR_ENV} when set.'
        ),
    )
    data_dir = kw.Value(
        None,
        type=str,
        help=(
            f'Directory for rendered artifacts and bind-mount state. Defaults to '
            f'~/.local/share/infer_stack (XDG_DATA_HOME) or ${DATA_DIR_ENV} when set.'
        ),
    )


class _BackendOverrideMixin(kw.Config):
    backend = kw.Value(
        None, type=str, choices=['compose', 'kubeai'], help='Active backend override.'
    )


class _ComposeOverrideMixin(kw.Config):
    compose_cmd = kw.Value(
        None,
        type=str,
        help="Docker compose command override (e.g. 'podman compose').",
    )


class _ProfileOverrideMixin(kw.Config):
    profile = kw.Value(
        None,
        type=str,
        help='Active profile override (sets config.active_profile).',
    )


class _PortOverridesMixin(kw.Config):
    litellm_port = kw.Value(None, type=int)
    open_webui_port = kw.Value(None, type=int)
    postgres_port = kw.Value(None, type=int)


class _ClusterOverridesMixin(kw.Config):
    namespace = kw.Value(
        None, type=str, help='Kubernetes namespace for kubeai deployments.'
    )
    ingress_host = kw.Value(
        None, type=str, help='Ingress host (kubeai only).'
    )
    ingress_enabled = kw.Value(
        None,
        isflag=True,
        alias=['ingress'],
        help='Enable cluster ingress (kubeai only); use --no-ingress to disable.',
    )


class _AllowUnsupportedMixin(kw.Config):
    allow_unsupported = kw.Value(
        False,
        isflag=True,
        help='Allow validation errors when planning/rendering.',
    )


class _SimulateHardwareMixin(kw.Config):
    simulate_hardware = kw.Value(
        None,
        type=str,
        help='Simulate GPUs: comma-separated NxM or M entries (e.g. 4x96, 2x80, "48,16" for a heterogeneous host). Useful for planning on smaller machines.',
    )


class _AllowedGpusMixin(kw.Config):
    allowed_gpus = kw.Value(
        None,
        type=str,
        help=(
            'Restrict placement to a comma-separated list of GPU indices '
            "(e.g. '1' or '1,3'). Real indices are preserved — the rendered "
            'compose stack pins device_ids to exactly those GPUs. Useful for '
            'integration tests on machines where some GPUs are tied up.'
        ),
    )


class _DisplayGpuMixin(kw.Config):
    skip_display_gpus = kw.Value(
        None,
        isflag=True,
        alias=['skip-display-gpus'],
        help=(
            'Skip display-attached GPUs during placement, leaving the GPU '
            'driving a monitor free. OFF by default — placement uses every '
            'GPU, so single-GPU and workstation hosts work out of the box. '
            'Opt in per-command with this flag, or persist it with '
            '`infer-stack config set skip_display_gpus true`.'
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
