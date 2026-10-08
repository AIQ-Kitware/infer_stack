from __future__ import annotations

from ..paths import CONFIG_DIR_ENV
from ..paths import DATA_DIR_ENV
import kwconf as kw

# ---------------------------------------------------------------------------
# DataConfig mixins for common override flags
# ---------------------------------------------------------------------------


#: Values a bare flag may carry (``--yes false``); anything else was a positional.
_BOOL_WORDS = frozenset({'true', 'false', 'yes', 'no', 'on', 'off', '1', '0'})


def reclaim_swallowed_positionals(config) -> None:
    """Give back a positional argument that a preceding flag consumed.

    kwconf flags take an optional value, so ``logs -f qwen`` parsed as
    ``follow='qwen'`` and no names: the command then acted on everything.
    Every infer-stack flag is a boolean, so a flag holding any other string
    was handed a positional. It becomes ``True``, and the string goes back to
    the front of the command's positional list (or its single positional,
    when that is still empty).
    """
    defaults = type(config).__default__
    positional = sorted((k for k, v in defaults.items() if getattr(v, 'position', None)),
                        key=lambda k: defaults[k].position)
    for key, value in defaults.items():
        if not getattr(value, 'isflag', False) or value.isflag == 'counter':
            continue
        got = config[key]
        if not isinstance(got, str) or got.strip().lower() in _BOOL_WORDS:
            continue
        config[key] = True
        if not positional:
            raise SystemExit(f'--{key} takes no value (got {got!r})')
        target = positional[0]
        many = defaults[target].parsekw.get('nargs') in ('*', '+')
        if many:
            config[target] = [got, *(config[target] or [])]
        elif config[target] in (None, ''):
            config[target] = got
        else:
            raise SystemExit(f'--{key} takes no value (got {got!r})')


class _FlagSafeMixin(kw.Config):
    """Parses ``--flag positional`` as a flag and a positional (see above)."""

    @classmethod
    def cli(cls, *args, **kwargs):
        config = super().cli(*args, **kwargs)
        reclaim_swallowed_positionals(config)
        return config


class _PathOverridesMixin(_FlagSafeMixin):
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


class _SimulateHardwareMixin(kw.Config):
    simulate_hardware = kw.Value(
        None,
        type=str,
        help='Simulate GPUs: comma-separated NxM[@CC] or M[@CC] entries (e.g. 4x96@12.0, 2x80@9.0, "48@7.5,16"). Useful for planning on smaller machines and capability classes.',
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
