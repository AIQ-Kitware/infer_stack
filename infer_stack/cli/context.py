from __future__ import annotations

from ..config import CONFIG_FILE
from ..config import MODELS_FILE
from ..config import deep_merge
from ..config import default_output_config
from ..config import default_state_paths
from ..config import generated_dir_for_config
from ..config import initial_config
from ..config import kubeai_generated_dir_for_config
from ..config import kubeai_local_values_path
from ..config import load_yaml
from ..config import plan_path_for_config
from ..config import save_yaml
from ..hardware import detect_inventory
from ..hardware import simulate_inventory
from ..paths import CONFIG_DIR_ENV
from ..paths import config_root
from ..paths import data_root
from ..paths import set_config_root
from ..paths import set_data_root
from ..resolver import resolve
from ..validator import validate_resolved
from copy import deepcopy
from pathlib import Path
from typing import Any
import os

# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------


def config_path() -> Path:
    return config_root() / CONFIG_FILE


def models_path() -> Path:
    return config_root() / MODELS_FILE


def generated_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return generated_dir_for_config(cfg)


def kubeai_generated_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return kubeai_generated_dir_for_config(cfg)


def plan_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return plan_path_for_config(cfg)


def _hydrate_config_defaults(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Merge persisted config on top of current defaults.

    The stack schema is still moving quickly.  Users may already have a
    config.yaml that predates a newly introduced component such as Ollama.
    Loading through this helper keeps those configs valid by filling in new
    default images, ports, provider toggles, state paths, and frontend/gateway
    defaults without rewriting the user's file.
    """
    return deep_merge(initial_config(), cfg or {})


def _safe_load_config() -> dict[str, Any]:
    """Load config.yaml if present; otherwise return defaults."""
    path = config_path()
    if path.exists():
        return _hydrate_config_defaults(load_yaml(path))
    return initial_config()


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit(
            f'No config.yaml found at {path}. Run '
            '`infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, '
            f'or point ${CONFIG_DIR_ENV} / --config-dir at an existing config.'
        )
    return _hydrate_config_defaults(load_yaml(path))


def runtime_dir_for_config(cfg: dict[str, Any]) -> Path:
    state = cfg.get('state', {})
    runtime = state.get('runtime')
    if not runtime:
        return data_root() / 'runtime'
    p = Path(runtime)
    if p.is_absolute():
        return p
    return data_root() / p


def runtime_env_path(cfg: dict[str, Any]) -> Path:
    return generated_dir(cfg) / '.env'


def runtime_litellm_config_path(cfg: dict[str, Any]) -> Path:
    return runtime_dir_for_config(cfg) / 'litellm_config.yaml'


def backend_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get('backend', 'compose')).lower()


# ---------------------------------------------------------------------------
# Env-var / override resolution
# ---------------------------------------------------------------------------


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _env_bool(name: str) -> bool | None:
    value = _env_text(name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {'1', 'true', 'yes', 'on', 'enabled'}:
        return True
    if lowered in {'0', 'false', 'no', 'off', 'disabled'}:
        return False
    raise SystemExit(f'Invalid boolean value for {name}: {value!r}')


def _env_int(name: str) -> int | None:
    value = _env_text(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as ex:
        raise SystemExit(f'Invalid integer value for {name}: {value!r}') from ex


def _as_mapping(args: Any) -> dict[str, Any]:
    """Coerce a CLI args object into a plain dict.

    Works for ``None``, ``argparse.Namespace``, and ``scfg.DataConfig``
    instances. Used to side-step name clashes between user-declared fields
    and ``DataConfig`` builtins (e.g. ``namespace`` is a property on the
    base class; ``getattr(cfg, 'namespace')`` returns the property, not the
    field value, while ``asdict()['namespace']`` returns the field value).
    """
    if args is None:
        return {}
    if hasattr(args, 'asdict'):
        return dict(args.asdict())
    if hasattr(args, '__dict__'):
        return dict(vars(args))
    return dict(args)


def _arg_or_env(
    args_dict: dict[str, Any], attr: str, env_name: str, *, caster=None
):
    """Look up ``attr`` in the args dict, falling back to env var ``env_name``."""
    value = args_dict.get(attr)
    if value is not None:
        return value
    env_value = _env_text(env_name)
    if env_value is None:
        return None
    if caster is None:
        return env_value
    try:
        return caster(env_value)
    except ValueError as ex:
        raise SystemExit(f'Invalid value for {env_name}: {env_value!r}') from ex


def apply_config_overrides(
    cfg: dict[str, Any], args: Any | None
) -> dict[str, Any]:
    """Merge runtime overrides (CLI args + env vars) on top of ``cfg``.

    ``args`` may be an ``argparse.Namespace``, a ``scfg.DataConfig`` instance,
    or any mapping; it is coerced to a plain dict via ``_as_mapping``.
    """
    if args is None:
        return deepcopy(cfg)
    overrides = _as_mapping(args)
    out = deepcopy(cfg)
    out.setdefault('runtime', {})
    out.setdefault('ports', {})
    out.setdefault('state', {})
    out.setdefault('output', {})
    out.setdefault('cluster', {})
    out['cluster'].setdefault('ingress', {})

    backend = _arg_or_env(overrides, 'backend', 'INFER_STACK_BACKEND')
    if backend:
        out['backend'] = backend

    profile = _arg_or_env(overrides, 'profile', 'INFER_STACK_PROFILE')
    if profile:
        out['active_profile'] = profile

    compose_cmd = _arg_or_env(
        overrides, 'compose_cmd', 'INFER_STACK_COMPOSE_CMD'
    )
    if compose_cmd:
        out['runtime']['compose_cmd'] = compose_cmd

    litellm_port = overrides.get('litellm_port')
    if litellm_port is None:
        litellm_port = _env_int('INFER_STACK_LITELLM_PORT')
    if litellm_port is not None:
        out['ports']['litellm'] = litellm_port

    open_webui_port = overrides.get('open_webui_port')
    if open_webui_port is None:
        open_webui_port = _env_int('INFER_STACK_OPEN_WEBUI_PORT')
    if open_webui_port is not None:
        out['ports']['open_webui'] = open_webui_port

    postgres_port = overrides.get('postgres_port')
    if postgres_port is None:
        postgres_port = _env_int('INFER_STACK_POSTGRES_PORT')
    if postgres_port is not None:
        out['ports']['postgres'] = postgres_port

    # All rendered artifacts and bind-mount state live under a single root
    # (``--data-dir`` / ``INFER_STACK_DATA_DIR``), which is baked into the
    # absolute ``state.*`` and ``output.generated_dir`` paths at setup time.
    # Granular per-knob path overrides were removed in favour of that one root;
    # edit ``state.*`` / ``output.generated_dir`` in config.yaml directly for
    # bespoke split layouts.
    #
    # When the data root is *explicitly* relocated via ``--data-dir``,
    # re-anchor the managed state tree + generated dir onto the new root.
    # ``set_data_root`` has already been applied by ``_apply_path_overrides``,
    # so the ``default_*`` helpers reflect the new location. Without this,
    # ``setup --data-dir X`` against an existing config would silently no-op,
    # since these paths are stored absolute and baked once at first setup.
    #
    # Triggered on the CLI flag only, not ``INFER_STACK_DATA_DIR``: the env var
    # is the ambient anchor a fresh config already picks up, and regenerating
    # from defaults on every plain ``setup`` would clobber hand-edited custom
    # ``state.*`` paths. The flag is the unambiguous "relocate now" signal.
    if overrides.get('data_dir'):
        out['state'] = default_state_paths()
        out['output']['generated_dir'] = default_output_config()[
            'generated_dir'
        ]
    elif not out['output'].get('generated_dir'):
        out['output']['generated_dir'] = default_output_config()[
            'generated_dir'
        ]

    namespace = _arg_or_env(overrides, 'namespace', 'INFER_STACK_NAMESPACE')
    if namespace:
        out['cluster']['namespace'] = namespace

    ingress_host = _arg_or_env(
        overrides, 'ingress_host', 'INFER_STACK_INGRESS_HOST'
    )
    if ingress_host:
        out['cluster']['ingress']['host'] = ingress_host

    ingress_enabled = overrides.get('ingress_enabled')
    if ingress_enabled is None:
        ingress_enabled = _env_bool('INFER_STACK_INGRESS_ENABLED')
    if ingress_enabled is not None:
        out['cluster']['ingress']['enabled'] = bool(ingress_enabled)

    return out


_OVERRIDE_ATTRS = (
    'profile',
    'backend',
    'compose_cmd',
    'litellm_port',
    'open_webui_port',
    'postgres_port',
    'namespace',
    'ingress_host',
    'ingress_enabled',
    'simulate_hardware',
    'allowed_gpus',
)

_OVERRIDE_ENVS = (
    'INFER_STACK_BACKEND',
    'INFER_STACK_PROFILE',
    'INFER_STACK_COMPOSE_CMD',
    'INFER_STACK_LITELLM_PORT',
    'INFER_STACK_OPEN_WEBUI_PORT',
    'INFER_STACK_POSTGRES_PORT',
    'INFER_STACK_NAMESPACE',
    'INFER_STACK_INGRESS_HOST',
    'INFER_STACK_INGRESS_ENABLED',
    'INFER_STACK_ALLOWED_GPUS',
)


def has_runtime_overrides(args: Any | None) -> bool:
    if args is None:
        return False
    overrides = _as_mapping(args)
    if any(overrides.get(attr) is not None for attr in _OVERRIDE_ATTRS):
        return True
    return any(_env_text(name) is not None for name in _OVERRIDE_ENVS)


def effective_allow_unsupported(args: Any | None, cfg: dict[str, Any]) -> bool:
    overrides = _as_mapping(args)
    arg_value = bool(overrides.get('allow_unsupported'))
    policy_value = bool(
        cfg.get('policy', {}).get('allow_unsupported_render', False)
    )
    return arg_value or policy_value


def _parse_allowed_gpus(raw: Any) -> list[int] | None:
    """Parse a comma-separated list of GPU indices, or ``None`` if unset.

    Accepts ints (when the value comes from ``data=`` kwargs in the
    programmatic API), as well as strings of the form ``"1"`` or ``"1,3"``.
    """
    if raw is None or raw == '':
        return None
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [x.strip() for x in str(raw).split(',') if x.strip()]
    try:
        return [int(x) for x in items]
    except (TypeError, ValueError) as ex:
        raise SystemExit(
            f'Invalid --allowed-gpus value {raw!r}: expected a comma-separated '
            f"list of integer GPU indices (e.g. '1' or '1,3'). {ex}"
        )


def _filter_inventory_to_allowed(
    inventory: dict[str, Any], allowed: list[int] | None
) -> dict[str, Any]:
    """Return a new inventory containing only the GPUs whose ``index`` is in ``allowed``.

    Real indices are preserved — there is no renumbering — so a profile
    that says ``placement.gpu_indices: [1, 3]`` still pins to physical
    GPUs 1 and 3 after filtering.
    """
    if not allowed:
        return inventory
    allowed_set = set(allowed)
    filtered = [
        g for g in inventory.get('gpus', []) if g.get('index') in allowed_set
    ]
    return {'gpu_count': len(filtered), 'gpus': filtered}


def effective_inventory(args: Any | None) -> dict[str, Any] | None:
    """Build the inventory the resolver should see, honoring CLI / env overrides.

    Returns ``None`` when nothing is constraining the inventory, so the
    resolver falls back to ``detect_inventory()`` at plan time.
    """
    overrides = _as_mapping(args)
    spec = overrides.get('simulate_hardware')
    allowed = _parse_allowed_gpus(
        overrides.get('allowed_gpus') or _env_text('INFER_STACK_ALLOWED_GPUS')
    )
    if not spec and allowed is None:
        return None
    base = simulate_inventory(spec) if spec else detect_inventory()
    return _filter_inventory_to_allowed(base, allowed)


def config_for_runtime(
    args: Any | None, *, allow_missing: bool = False
) -> dict[str, Any]:
    if config_path().exists():
        cfg = load_config()
    elif allow_missing:
        cfg = initial_config()
    else:
        raise SystemExit(
            f'No config.yaml found at {config_path()}. Run '
            '`infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, '
            f'or point ${CONFIG_DIR_ENV} / --config-dir at an existing config.'
        )
    return apply_config_overrides(cfg, args)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def build_plan(
    cfg: dict[str, Any],
    *,
    profile_name: str | None = None,
    allow_unsupported: bool = False,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve(cfg, inventory=inventory, profile_name=profile_name)
    report = validate_resolved(resolved)
    return {
        'schema_version': 1,
        'allow_unsupported': bool(allow_unsupported),
        'validated': report,
        'deployment': resolved,
    }


def save_plan(plan: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    path = plan_path(cfg)
    save_yaml(path, plan)
    return path


def ensure_renderable(plan: dict[str, Any]) -> None:
    validated = plan.get('validated', {}) or {}
    if validated.get('errors') and not plan.get('allow_unsupported', False):
        raise SystemExit(
            'Refusing to render because the resolved plan contains validation errors. '
            'Use `--allow-unsupported` to override.'
        )


def render_is_stale(cfg: dict[str, Any] | None = None) -> bool:
    cfg = load_config() if cfg is None else cfg
    cfg_path = config_path()
    current_plan = plan_path(cfg)
    backend = backend_name(cfg)

    if backend == 'kubeai':
        kubeai_root = kubeai_generated_dir(cfg)
        required_outputs = [
            current_plan,
            kubeai_root / 'namespace.yaml',
            kubeai_root / 'kubeai-values.yaml',
            kubeai_root / 'models.yaml',
        ]
    else:
        required_outputs = [
            current_plan,
            generated_dir(cfg) / 'docker-compose.yml',
            runtime_env_path(cfg),
        ]
        # litellm_config.yaml is optional now; direct Ollama/raw-server profiles
        # intentionally do not render it.

    if any(not p.exists() for p in required_outputs):
        return True

    if cfg_path.exists():
        oldest_generated = min(p.stat().st_mtime for p in required_outputs)
        if cfg_path.stat().st_mtime > oldest_generated:
            return True
        if backend == 'kubeai':
            local_values_path = kubeai_local_values_path()
            if (
                local_values_path.exists()
                and local_values_path.stat().st_mtime > oldest_generated
            ):
                return True

    if any(
        current_plan.stat().st_mtime > p.stat().st_mtime
        for p in required_outputs
        if p != current_plan
    ):
        return True
    return False


# ---------------------------------------------------------------------------
def _apply_path_overrides(config: Any) -> None:
    """Honour ``--config-dir`` / ``--data-dir`` from a parsed subcommand config."""
    overrides = _as_mapping(config)
    if overrides.get('config_dir'):
        set_config_root(overrides['config_dir'])
    if overrides.get('data_dir'):
        set_data_root(overrides['data_dir'])
