#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""scriptconfig-based CLI for infer-stack.

Each subcommand is a ``scfg.DataConfig`` subclass; ``ManageCLI`` composes
them into a single ``scfg.ModalCLI`` exposed as the ``infer-stack`` entry
point. Because every subcommand is a ``DataConfig``, the same class can
be invoked from the shell (``infer-stack render --profile X``) or from
Python (``RenderCLI.main(argv=False, profile='X')``).

Layout:
* Path / config helpers (config_path, generated_dir, runtime_*).
* Override resolution (apply_config_overrides, has_runtime_overrides,
  effective_allow_unsupported, effective_inventory, config_for_runtime).
* Planning helpers (build_plan, save_plan, ensure_renderable,
  render_is_stale).
* Backend-specific helpers (_compose_base_cmd, _kubeai_stub,
  _compose_up_with_router_recreate, _maybe_rerender).
* DataConfig mixins for override flags shared across subcommands.
* Per-command DataConfig classes, grouped by topic.
* ``ManageCLI`` ModalCLI and ``main`` entry point.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests
import scriptconfig as scfg

from .catalog import PROFILE_NAME_ALIASES, profile_summary
from .benchmark import run_benchmark
from .config import (
    CONFIG_FILE,
    MODELS_FILE,
    deep_merge,
    default_output_config,
    generated_dir_for_config,
    initial_config,
    kubeai_generated_dir_for_config,
    kubeai_local_values_path,
    load_kubeai_resource_profiles,
    load_yaml,
    normalized_catalogs,
    normalized_state,
    plan_path_for_config,
    save_kubeai_resource_profiles,
    save_yaml,
)
from .contracts import load_profile_contract
from .docker_utils import (
    DockerCommandError,
    PortInUseError,
    check_ports_available,
    compose_down,
    compose_recreate_router,
    compose_up,
    docker_rm_dirs,
    our_published_ports,
)
from .env_utils import parse_env_file
from .exporters import export_benchmark_bundle
from .hardware import detect_inventory, simulate_inventory
from .kubeai_ops import CommandError, deploy_rendered_artifacts, print_status as kubeai_print_status
from .paths import (
    CONFIG_DIR_ENV,
    DATA_DIR_ENV,
    config_root,
    data_root,
    set_config_root,
    set_data_root,
)
from .profile_runtime import default_base_url
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved
from .verification import verify_profile


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
            f"No config.yaml found at {path}. Run "
            "`infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, "
            f"or point ${CONFIG_DIR_ENV} / --config-dir at an existing config."
        )
    return _hydrate_config_defaults(load_yaml(path))


def runtime_dir_for_config(cfg: dict[str, Any]) -> Path:
    state = cfg.get("state", {})
    runtime = state.get("runtime")
    if not runtime:
        return data_root() / "runtime"
    p = Path(runtime)
    if p.is_absolute():
        return p
    return data_root() / p


def runtime_env_path(cfg: dict[str, Any]) -> Path:
    return generated_dir(cfg) / ".env"


def runtime_litellm_config_path(cfg: dict[str, Any]) -> Path:
    return runtime_dir_for_config(cfg) / "litellm_config.yaml"


def backend_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("backend", "compose")).lower()


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
    if lowered in {"1", "true", "yes", "on", "enabled"}:
        return True
    if lowered in {"0", "false", "no", "off", "disabled"}:
        return False
    raise SystemExit(f"Invalid boolean value for {name}: {value!r}")


def _env_int(name: str) -> int | None:
    value = _env_text(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as ex:
        raise SystemExit(f"Invalid integer value for {name}: {value!r}") from ex


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
    if hasattr(args, "asdict"):
        return dict(args.asdict())
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


def _arg_or_env(args_dict: dict[str, Any], attr: str, env_name: str, *, caster=None):
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
        raise SystemExit(f"Invalid value for {env_name}: {env_value!r}") from ex


def _configured_state_paths(state_root: str) -> dict[str, str]:
    base = Path(state_root)
    return {
        "hf_cache": str(base / "hf-cache"),
        "vllm_cache": str(base / "vllm-cache"),
        "open_webui": str(base / "open-webui"),
        "postgres_open_webui": str(base / "postgres-open-webui"),
        "postgres_litellm": str(base / "postgres-litellm"),
        "ollama": str(base / "ollama"),
        "runtime": str(base / "runtime"),
    }


def apply_config_overrides(cfg: dict[str, Any], args: Any | None) -> dict[str, Any]:
    """Merge runtime overrides (CLI args + env vars) on top of ``cfg``.

    ``args`` may be an ``argparse.Namespace``, a ``scfg.DataConfig`` instance,
    or any mapping; it is coerced to a plain dict via ``_as_mapping``.
    """
    if args is None:
        return deepcopy(cfg)
    overrides = _as_mapping(args)
    out = deepcopy(cfg)
    out.setdefault("runtime", {})
    out.setdefault("ports", {})
    out.setdefault("state", {})
    out.setdefault("output", {})
    out.setdefault("cluster", {})
    out["cluster"].setdefault("ingress", {})

    backend = _arg_or_env(overrides, "backend", "INFER_STACK_BACKEND")
    if backend:
        out["backend"] = backend

    profile = _arg_or_env(overrides, "profile", "INFER_STACK_PROFILE")
    if profile:
        out["active_profile"] = profile

    compose_cmd = _arg_or_env(overrides, "compose_cmd", "INFER_STACK_COMPOSE_CMD")
    if compose_cmd:
        out["runtime"]["compose_cmd"] = compose_cmd

    litellm_port = overrides.get("litellm_port")
    if litellm_port is None:
        litellm_port = _env_int("INFER_STACK_LITELLM_PORT")
    if litellm_port is not None:
        out["ports"]["litellm"] = litellm_port

    open_webui_port = overrides.get("open_webui_port")
    if open_webui_port is None:
        open_webui_port = _env_int("INFER_STACK_OPEN_WEBUI_PORT")
    if open_webui_port is not None:
        out["ports"]["open_webui"] = open_webui_port

    postgres_port = overrides.get("postgres_port")
    if postgres_port is None:
        postgres_port = _env_int("INFER_STACK_POSTGRES_PORT")
    if postgres_port is not None:
        out["ports"]["postgres"] = postgres_port

    state_root = _arg_or_env(overrides, "state_root", "INFER_STACK_STATE_ROOT")
    if state_root:
        out["state"].update(_configured_state_paths(state_root))

    runtime_dir = _arg_or_env(overrides, "runtime_dir", "INFER_STACK_RUNTIME_DIR")
    if runtime_dir:
        out["state"]["runtime"] = runtime_dir

    generated_dir_override = _arg_or_env(overrides, "generated_dir", "INFER_STACK_GENERATED_DIR")
    if generated_dir_override:
        out["output"]["generated_dir"] = generated_dir_override
    elif not out["output"].get("generated_dir"):
        out["output"]["generated_dir"] = default_output_config()["generated_dir"]

    namespace = _arg_or_env(overrides, "namespace", "INFER_STACK_NAMESPACE")
    if namespace:
        out["cluster"]["namespace"] = namespace

    ingress_host = _arg_or_env(overrides, "ingress_host", "INFER_STACK_INGRESS_HOST")
    if ingress_host:
        out["cluster"]["ingress"]["host"] = ingress_host

    ingress_enabled = overrides.get("ingress_enabled")
    if ingress_enabled is None:
        ingress_enabled = _env_bool("INFER_STACK_INGRESS_ENABLED")
    if ingress_enabled is not None:
        out["cluster"]["ingress"]["enabled"] = bool(ingress_enabled)

    return out


_OVERRIDE_ATTRS = (
    "profile",
    "backend",
    "compose_cmd",
    "litellm_port",
    "open_webui_port",
    "postgres_port",
    "namespace",
    "ingress_host",
    "ingress_enabled",
    "simulate_hardware",
    "allowed_gpus",
    "generated_dir",
)

_OVERRIDE_ENVS = (
    "INFER_STACK_BACKEND",
    "INFER_STACK_PROFILE",
    "INFER_STACK_COMPOSE_CMD",
    "INFER_STACK_LITELLM_PORT",
    "INFER_STACK_OPEN_WEBUI_PORT",
    "INFER_STACK_POSTGRES_PORT",
    "INFER_STACK_NAMESPACE",
    "INFER_STACK_INGRESS_HOST",
    "INFER_STACK_INGRESS_ENABLED",
    "INFER_STACK_GENERATED_DIR",
    "INFER_STACK_ALLOWED_GPUS",
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
    arg_value = bool(overrides.get("allow_unsupported"))
    policy_value = bool(cfg.get("policy", {}).get("allow_unsupported_render", False))
    return arg_value or policy_value


def _parse_allowed_gpus(raw: Any) -> list[int] | None:
    """Parse a comma-separated list of GPU indices, or ``None`` if unset.

    Accepts ints (when the value comes from ``data=`` kwargs in the
    programmatic API), as well as strings of the form ``"1"`` or ``"1,3"``.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [x.strip() for x in str(raw).split(",") if x.strip()]
    try:
        return [int(x) for x in items]
    except (TypeError, ValueError) as ex:
        raise SystemExit(
            f"Invalid --allowed-gpus value {raw!r}: expected a comma-separated "
            f"list of integer GPU indices (e.g. '1' or '1,3'). {ex}"
        )


def _filter_inventory_to_allowed(inventory: dict[str, Any], allowed: list[int] | None) -> dict[str, Any]:
    """Return a new inventory containing only the GPUs whose ``index`` is in ``allowed``.

    Real indices are preserved — there is no renumbering — so a profile
    that says ``placement.gpu_indices: [1, 3]`` still pins to physical
    GPUs 1 and 3 after filtering.
    """
    if not allowed:
        return inventory
    allowed_set = set(allowed)
    filtered = [g for g in inventory.get("gpus", []) if g.get("index") in allowed_set]
    return {"gpu_count": len(filtered), "gpus": filtered}


def effective_inventory(args: Any | None) -> dict[str, Any] | None:
    """Build the inventory the resolver should see, honoring CLI / env overrides.

    Returns ``None`` when nothing is constraining the inventory, so the
    resolver falls back to ``detect_inventory()`` at plan time.
    """
    overrides = _as_mapping(args)
    spec = overrides.get("simulate_hardware")
    allowed = _parse_allowed_gpus(
        overrides.get("allowed_gpus") or _env_text("INFER_STACK_ALLOWED_GPUS")
    )
    if not spec and allowed is None:
        return None
    base = simulate_inventory(spec) if spec else detect_inventory()
    return _filter_inventory_to_allowed(base, allowed)


def config_for_runtime(args: Any | None, *, allow_missing: bool = False) -> dict[str, Any]:
    if config_path().exists():
        cfg = load_config()
    elif allow_missing:
        cfg = initial_config()
    else:
        raise SystemExit(
            f"No config.yaml found at {config_path()}. Run "
            "`infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, "
            f"or point ${CONFIG_DIR_ENV} / --config-dir at an existing config."
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
        "schema_version": 1,
        "allow_unsupported": bool(allow_unsupported),
        "validated": report,
        "deployment": resolved,
    }


def save_plan(plan: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    path = plan_path(cfg)
    save_yaml(path, plan)
    return path


def ensure_renderable(plan: dict[str, Any]) -> None:
    validated = plan.get("validated", {}) or {}
    if validated.get("errors") and not plan.get("allow_unsupported", False):
        raise SystemExit(
            "Refusing to render because the resolved plan contains validation errors. "
            "Use `--allow-unsupported` to override."
        )


def render_is_stale(cfg: dict[str, Any] | None = None) -> bool:
    cfg = load_config() if cfg is None else cfg
    cfg_path = config_path()
    current_plan = plan_path(cfg)
    backend = backend_name(cfg)

    if backend == "kubeai":
        kubeai_root = kubeai_generated_dir(cfg)
        required_outputs = [
            current_plan,
            kubeai_root / "namespace.yaml",
            kubeai_root / "kubeai-values.yaml",
            kubeai_root / "models.yaml",
        ]
    else:
        required_outputs = [
            current_plan,
            generated_dir(cfg) / "docker-compose.yml",
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
        if backend == "kubeai":
            local_values_path = kubeai_local_values_path()
            if local_values_path.exists() and local_values_path.stat().st_mtime > oldest_generated:
                return True

    if any(current_plan.stat().st_mtime > p.stat().st_mtime for p in required_outputs if p != current_plan):
        return True
    return False


# ---------------------------------------------------------------------------
# Backend-specific helpers
# ---------------------------------------------------------------------------


def _compose_base_cmd(cfg: dict[str, Any]) -> list[str]:
    """Build the shared ``docker compose -f ... --env-file ...`` prefix.

    Used by every compose-wrapper subcommand so the user doesn't have to
    cd into the rendered-artifacts directory just to run a one-shot
    ``ps`` / ``restart`` / ``pull`` / ``logs``.
    """
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = generated_dir(cfg) / ".env"
    return cfg["runtime"]["compose_cmd"].split() + [
        "-f", str(compose_file),
        "--env-file", str(env_file),
    ]


def _kubeai_stub(cmd_name: str) -> None:
    """Raise for a day-2-ops subcommand that has no kubeai implementation yet.

    These wrappers (logs/ps/restart/pull/start/stop) compose docker-compose
    invocations and have no kubectl equivalent in this CLI. Until somebody
    writes one, surface the gap as ``NotImplementedError`` so callers can
    distinguish "kubeai doesn't do this yet" from a real failure.
    """
    raise NotImplementedError(
        f"`{cmd_name}` is not implemented for the kubeai backend yet. "
        f"Use the equivalent kubectl command in the meantime "
        f"(e.g. `kubectl -n <namespace> ...`)."
    )


def _compose_up_with_router_recreate(
    cfg: dict[str, Any],
    *,
    detach: bool,
) -> None:
    """Run ``compose up`` and refresh LiteLLM's model list to match the new render.

    A compose ``up`` against the rendered stack only restarts vLLM services
    whose specs changed. LiteLLM and Open WebUI keep their existing
    containers and would therefore serve stale model lists until something
    else refreshed them. Two ways to do that:

    1. **Live refresh** (preferred): talk to LiteLLM's admin API
       (``/model/new`` / ``/model/delete``) to diff and apply alias changes
       in-process. LiteLLM and Open WebUI stay up — users hitting unchanged
       models see no disruption. Skipped automatically on cold start or if
       the admin API is unreachable.
    2. **Container recreate** (fallback): force-recreate only the LiteLLM
       container so it reloads the rendered YAML on startup. Open WebUI stays
       up; used only when live refresh fails and Compose did not already
       reload LiteLLM during convergence.
    """
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = generated_dir(cfg) / ".env"
    compose_cmd = cfg["runtime"]["compose_cmd"]

    _preflight_check_ports(cfg)

    litellm_in_render = _compose_has_service(compose_file, "litellm")
    litellm_before = (
        _compose_service_state(compose_cmd, compose_file, env_file, "litellm")
        if litellm_in_render
        else {}
    )

    compose_up(
        compose_cmd,
        compose_file,
        env_file,
        detach=detach,
        remove_orphans=True,
    )

    # If `up` failed it would have already raised; only do the router refresh
    # in detached mode (foreground `up` keeps the user attached to logs and
    # leaves cycling decisions to compose).
    if not detach:
        return

    if not runtime_litellm_config_path(cfg).exists():
        return

    if not litellm_in_render:
        # Direct Ollama / raw-server profiles intentionally do not render a
        # LiteLLM service.  A stale litellm_config.yaml may still exist in the
        # runtime directory from a previous gateway profile, but that must not
        # trigger a router refresh/recreate against a service that is no longer
        # present in the active compose file.
        return

    litellm_after = _compose_service_state(compose_cmd, compose_file, env_file, "litellm")
    litellm_reloaded_by_compose = bool(litellm_after) and (
        litellm_after.get("id") != litellm_before.get("id")
        or litellm_after.get("started_at") != litellm_before.get("started_at")
        or litellm_before.get("running") not in {"true", "True"}
    )
    if litellm_reloaded_by_compose:
        # Compose already created or restarted LiteLLM while converging the
        # stack.  The process has read the freshly rendered YAML, so a second
        # live refresh or forced recreate is redundant churn.
        print("LiteLLM was started/reloaded by compose; skipping extra router refresh.")
        return

    try:
        _litellm_refresh_router_live(cfg)
        return
    except RouterRefreshError as ex:
        print(
            f"Live router refresh skipped ({ex}); "
            "recreating litellm container to reload config from YAML "
            "(open-webui stays up)..."
        )

    compose_recreate_router(
        compose_cmd,
        compose_file,
        env_file,
        detach=True,
    )


def _compose_has_service(compose_file: Path, service_name: str) -> bool:
    """Return true when a rendered compose file contains ``service_name``.

    This is intentionally based on the rendered compose file rather than the
    presence of sidecar artifacts such as ``runtime/litellm_config.yaml``.
    Runtime artifacts are persistent across profile switches, while the compose
    service list is the active source of truth for what ``docker compose up``
    can recreate.
    """
    try:
        doc = load_yaml(compose_file)
    except FileNotFoundError:
        return False
    services = doc.get("services") or {}
    return service_name in services


def _compose_service_container_id(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    service_name: str,
) -> str:
    """Return the current container id for a rendered compose service."""
    return _compose_service_state(compose_cmd, compose_file, env_file, service_name).get("id", "")


def _compose_service_state(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    service_name: str,
) -> dict[str, str]:
    """Return a robust best-effort state snapshot for a compose service.

    Use JSON ``docker inspect`` rather than a Go template.  The template form is
    brittle for containers without a healthcheck: missing ``State.Health`` can
    make ``docker inspect --format`` fail, causing diagnostics to show only an
    id and the convergence logic to misclassify a still-running LiteLLM
    container as newly reloaded.

    The returned fields are also used to diagnose ``exit code 137`` cases: when
    a container disappears or restarts, ``oom_killed`` / ``exit_code`` make it
    clear whether Docker killed it, Compose recreated it, or the process exited
    normally.
    """
    if not compose_file.exists():
        return {}
    ps_cmd = compose_cmd.split() + [
        "-f", str(compose_file), "--env-file", str(env_file),
        "ps", "-q", service_name,
    ]
    try:
        proc = subprocess.run(ps_cmd, capture_output=True, text=True, check=False, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    container_id = proc.stdout.strip().splitlines()[0]
    inspect_cmd = ["docker", "inspect", container_id]
    try:
        proc = subprocess.run(inspect_cmd, capture_output=True, text=True, check=False, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return {"id": container_id}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"id": container_id}
    try:
        payload = json.loads(proc.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return {"id": container_id}

    state = payload.get("State") or {}
    health = state.get("Health") or {}
    return {
        "id": payload.get("Id") or container_id,
        "name": str(payload.get("Name") or "").lstrip("/"),
        "running": str(bool(state.get("Running"))).lower(),
        "status": str(state.get("Status") or ""),
        "health": str(health.get("Status") or "none"),
        "started_at": str(state.get("StartedAt") or ""),
        "finished_at": str(state.get("FinishedAt") or ""),
        "exit_code": str(state.get("ExitCode") if state.get("ExitCode") is not None else ""),
        "oom_killed": str(bool(state.get("OOMKilled"))).lower(),
        "restart_count": str(payload.get("RestartCount") if payload.get("RestartCount") is not None else ""),
    }


def _compose_rendered_service_names(compose_file: Path) -> list[str]:
    """Return service names from the rendered compose file."""
    try:
        doc = load_yaml(compose_file)
    except FileNotFoundError:
        return []
    return sorted((doc.get("services") or {}).keys())


def _short_id(value: str) -> str:
    """Shorten a docker container id for human diagnostics."""
    return value[:12] if value else "-"


def _http_probe_summary(method: str, url: str, *, headers: dict[str, str] | None = None, json_body: Any | None = None, timeout: float = 8.0) -> str:
    """Return a concise one-line summary for a diagnostic HTTP probe."""
    try:
        resp = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
    except requests.exceptions.RequestException as ex:
        return f"ERR {type(ex).__name__}: {ex}"
    body = (resp.text or "").strip().replace("\n", " ")
    if len(body) > 220:
        body = body[:220] + "..."
    return f"HTTP {resp.status_code}: {body}" if resp.status_code >= 400 else f"HTTP {resp.status_code}"


def _print_gateway_diagnostics(cfg: dict[str, Any], deployment: dict[str, Any], *, model: str | None = None, require_generation: bool = False) -> None:
    """Print targeted probes for the active gateway/provider graph."""
    env = parse_env_file(runtime_env_path(cfg)) if runtime_env_path(cfg).exists() else {}
    ports = cfg.get("ports", {}) or {}
    gateways = deployment.get("gateways", {}) or {}
    providers = deployment.get("providers", {}) or {}

    litellm = gateways.get("litellm") or {}
    if litellm.get("enabled"):
        litellm_port = ports.get("litellm")
        base = f"http://127.0.0.1:{litellm_port}"
        key = env.get("LITELLM_MASTER_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        print("\nLiteLLM probes:")
        print(f"  GET {base}/model/info -> {_http_probe_summary('GET', base + '/model/info', headers=headers)}")
        print(f"  GET {base}/v1/models  -> {_http_probe_summary('GET', base + '/v1/models', headers=headers)}")
        if require_generation:
            probe_model = model or _default_model_for_deployment(deployment)
            protocol = _resolve_smoke_protocol_from_deployment(deployment, probe_model)
            ok, msg = _ready_openai_probe(
                base_url=f"{base}/v1",
                headers=headers,
                model=probe_model,
                protocol=protocol,
                prompt="Reply with ready.",
                max_tokens=1,
                require_generation=True,
            )
            status = "OK" if ok else "WAIT"
            print(f"  generation probe ({probe_model}, {protocol}) -> {status}: {msg}")

    vllm = providers.get("vllm") or {}
    runtimes = vllm.get("runtimes") or {}
    if runtimes:
        print("\nvLLM provider probes:")
        for name, rt in runtimes.items():
            service = rt.get("compose_service_name") or f"vllm-{name}"
            host_port = rt.get("host_port") or ports.get("vllm") or 18000
            print(f"  {service}: model={rt.get('served_model_name')} protocol={rt.get('protocol_mode')} gpu={rt.get('gpu_indices')}")
            if host_port:
                url = f"http://127.0.0.1:{host_port}/health"
                print(f"    GET {url} -> {_http_probe_summary('GET', url)}")

    ollama = providers.get("ollama") or {}
    if ollama.get("enabled") and ollama.get("publish_port"):
        port = ports.get("ollama") or 11434
        base = f"http://127.0.0.1:{port}"
        print("\nOllama probes:")
        print(f"  GET {base}/api/tags -> {_http_probe_summary('GET', base + '/api/tags')}")


def _print_compose_diagnostics(cfg: dict[str, Any], *, tail: int = 0) -> None:
    """Print compose state and optionally recent logs for diagnostic purposes."""
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = runtime_env_path(cfg)
    compose_cmd = cfg["runtime"]["compose_cmd"]
    services = _compose_rendered_service_names(compose_file)
    if not services:
        print(f"No rendered compose services found at {compose_file}")
        return
    print("\nCompose services:")
    for svc in services:
        state = _compose_service_state(compose_cmd, compose_file, env_file, svc)
        if not state:
            print(f"  {svc:22s} absent")
            continue
        print(
            f"  {svc:22s} id={_short_id(state.get('id', ''))} "
            f"name={state.get('name', '-')} "
            f"running={state.get('running', '-')} status={state.get('status', '-')} "
            f"health={state.get('health', '-')} exit={state.get('exit_code', '-')} "
            f"oom={state.get('oom_killed', '-')} restarts={state.get('restart_count', '-')} "
            f"started_at={state.get('started_at', '-')}")
    if tail:
        log_services = [svc for svc in services if svc == "litellm" or svc == "open-webui" or svc.startswith("vllm-") or svc == "ollama"]
        if log_services:
            print(f"\nRecent logs (--tail {tail}):")
            cmd = _compose_base_cmd(cfg) + ["logs", "--tail", str(tail), *log_services]
            subprocess.run(cmd, check=False)


def _explain_readiness_message(msg: str) -> str:
    """Add operator-facing interpretation to common readiness failures."""
    lower = msg.lower()
    if "cannot connect to host litellm" in lower or "name or service not known" in lower:
        return (
            msg
            + "\n  hint: the frontend or caller cannot resolve/reach the LiteLLM service. "
            "Run `infer-stack diagnose --logs --tail 80` to check whether the active profile renders LiteLLM and whether the container is running. "
            "If diagnose shows `oom=true` or `exit=137`, Docker killed LiteLLM rather than merely waiting on a vLLM upstream."
        )
    if "connection error" in lower or "connection refused" in lower or "cannot connect to host vllm" in lower:
        return (
            msg
            + "\n  hint: LiteLLM is responding, but the upstream vLLM process is not serving yet. "
            "This is expected while a single vLLM runtime is being replaced; wait-ready will keep polling."
        )
    return msg


class RouterRefreshError(RuntimeError):
    """Live LiteLLM router refresh did not complete; caller should fall back."""


def _resolve_env_refs(obj: Any, env: dict[str, str]) -> Any:
    """Recursively replace ``os.environ/VAR`` strings with the value from ``env``.

    Mirrors LiteLLM's YAML-load substitution so we can feed the admin API
    literal credentials. If a referenced variable isn't in ``env`` the original
    string is left alone — caller will get the upstream error to debug.
    """
    if isinstance(obj, str):
        if obj.startswith("os.environ/"):
            var = obj.removeprefix("os.environ/")
            return env.get(var, obj)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_refs(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_refs(v, env) for v in obj]
    return obj



def _litellm_delete_missed_config_model(resp: requests.Response) -> bool:
    """Return true for LiteLLM's config-model delete miss response.

    ``GET /model/info`` returns both models loaded from config.yaml and
    models inserted into LiteLLM's DB via ``/model/new``.  In current LiteLLM
    releases, ``POST /model/delete`` only deletes DB-backed models.  When the
    reported id belongs to a config-backed model, the delete endpoint returns a
    400/404 response whose body says the model id was not found in the DB.
    That is not a proxy availability failure, so it should not trigger the
    fallback path that restarts the LiteLLM container.
    """
    try:
        payload = resp.json()
    except ValueError:
        payload = resp.text
    text = str(payload).lower()
    return "not found" in text and "db" in text

def _litellm_refresh_router_live(cfg: dict[str, Any]) -> None:
    """Sync the running LiteLLM router's model list to match the rendered YAML.

    Diffs ``GET /model/info`` (current state in the running container) against
    the rendered ``litellm_config.yaml`` (desired state), then applies
    ``POST /model/delete`` and ``POST /model/new`` for the differences. Aliases
    that didn't change keep serving traffic without interruption.

    Raises ``RouterRefreshError`` on any failure (admin API unreachable,
    auth missing, individual call fails); the caller falls back to the
    full container-recreate path.
    """
    import yaml as _yaml

    litellm_port = cfg.get("ports", {}).get("litellm")
    if not litellm_port:
        raise RouterRefreshError("litellm port not configured in cfg")
    base = f"http://127.0.0.1:{litellm_port}"

    env = parse_env_file(runtime_env_path(cfg))
    master_key = env.get("LITELLM_MASTER_KEY", "").strip()
    if not master_key:
        raise RouterRefreshError("LITELLM_MASTER_KEY missing from rendered .env")
    headers = {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}

    config_path_ = runtime_litellm_config_path(cfg)
    if not config_path_.exists():
        raise RouterRefreshError(f"rendered litellm config not found at {config_path_}")
    try:
        desired_doc = _yaml.safe_load(config_path_.read_text(encoding="utf-8")) or {}
    except _yaml.YAMLError as ex:
        raise RouterRefreshError(f"could not parse {config_path_}: {ex}") from ex
    desired_models = desired_doc.get("model_list") or []

    # The rendered YAML keeps secrets as `os.environ/VAR` references so the
    # file itself isn't sensitive. LiteLLM resolves these only at YAML-load
    # time on container startup — the admin API takes literal values. Inline
    # the actual env values now so /model/new gets a usable upstream.
    desired_models = [_resolve_env_refs(m, env) for m in desired_models]

    last_ex: requests.exceptions.RequestException | None = None
    resp = None
    max_attempts = 20
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(f"{base}/model/info", headers=headers, timeout=5)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as ex:
            last_ex = ex
            if attempt < max_attempts:
                import time

                time.sleep(1.5)
                continue
            raise RouterRefreshError(f"GET /model/info failed after {attempt} attempts: {ex}") from ex
    assert resp is not None
    current_models = resp.json().get("data") or []

    # Key by alias (model_name). Within an alias, the "upstream" identity is
    # litellm_params.model (e.g. "openai/qwen3.5-9b"). If that changes, the
    # alias points to a different service and must be re-added; if it matches,
    # the alias is untouched and continues serving.
    def upstream_of(entry):
        return (entry.get("litellm_params") or {}).get("model")

    current_by_alias = {m["model_name"]: m for m in current_models}
    desired_by_alias = {m["model_name"]: m for m in desired_models}

    to_delete: list[tuple[str, str]] = []
    to_add: list[dict] = []

    for alias, current in current_by_alias.items():
        desired = desired_by_alias.get(alias)
        if desired is None:
            to_delete.append((alias, current["model_info"]["id"]))
        elif upstream_of(current) != upstream_of(desired):
            to_delete.append((alias, current["model_info"]["id"]))
            to_add.append(desired)

    for alias, desired in desired_by_alias.items():
        if alias not in current_by_alias:
            to_add.append(desired)

    if not to_delete and not to_add:
        return

    # Delete-before-add so the same alias can transition to a new upstream
    # without LiteLLM rejecting a duplicate model_name.  LiteLLM distinguishes
    # config-file models from DB-backed models: /model/delete only applies to
    # DB-backed rows.  When an existing model was loaded from config.yaml,
    # LiteLLM may report it in /model/info but return "not found in db" from
    # /model/delete.  Treat that as a non-fatal stale-config alias instead of
    # forcing a LiteLLM container restart; the desired new aliases can still be
    # added live, and a later manual LiteLLM restart will clean up the stale
    # config-backed aliases if the operator cares about /v1/models hygiene.
    stale_config_aliases: set[str] = set()
    deleted_count = 0
    for alias, model_id in to_delete:
        try:
            resp = requests.post(
                f"{base}/model/delete",
                headers=headers,
                json={"id": model_id},
                timeout=5,
            )
            if resp.status_code in {400, 404} and _litellm_delete_missed_config_model(resp):
                stale_config_aliases.add(alias)
                continue
            resp.raise_for_status()
            deleted_count += 1
        except requests.exceptions.RequestException as ex:
            raise RouterRefreshError(f"DELETE alias={alias} id={model_id} failed: {ex}") from ex

    skipped_add_aliases: set[str] = set()
    for model in to_add:
        alias = model.get("model_name", "<unknown>")
        if alias in stale_config_aliases:
            # Same alias, changed upstream, and the old alias is config-backed.
            # Adding would collide and deleting would require a container restart.
            skipped_add_aliases.add(alias)
            continue
        try:
            resp = requests.post(
                f"{base}/model/new",
                headers=headers,
                json=model,
                timeout=5,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as ex:
            raise RouterRefreshError(f"POST /model/new alias={alias} failed: {ex}") from ex

    summary_parts = []
    if deleted_count:
        summary_parts.append(f"removed {deleted_count} alias(es)")
    added_count = len(to_add) - len(skipped_add_aliases)
    if added_count:
        summary_parts.append(f"added {added_count} alias(es)")
    if stale_config_aliases:
        summary_parts.append(
            "left "
            f"{len(stale_config_aliases)} stale config-backed alias(es) live "
            "because LiteLLM would not delete them without a restart"
        )
    if skipped_add_aliases:
        summary_parts.append(
            "skipped "
            f"{len(skipped_add_aliases)} same-name update(s); "
            "restart LiteLLM to replace those aliases"
        )
    print(f"Live LiteLLM router refresh: {', '.join(summary_parts)}.")


def _preflight_check_ports(cfg: dict[str, Any]) -> None:
    """Verify only the host ports the current rendered stack will publish."""
    ports = cfg.get("ports", {})
    candidates: list[tuple[str, int, str]] = []
    deployment: dict[str, Any] = {}
    try:
        if plan_path(cfg).exists():
            deployment = load_yaml(plan_path(cfg)).get("deployment", {})
    except Exception:
        deployment = {}

    frontends = deployment.get("frontends", {}) or {}
    gateways = deployment.get("gateways", {}) or {}
    providers = deployment.get("providers", {}) or {}

    if not deployment:
        # Fallback for very old rendered states; keep this conservative.
        if ports.get("litellm"):
            candidates.append(("litellm", int(ports["litellm"]), "0.0.0.0"))
        if ports.get("open_webui"):
            candidates.append(("open-webui", int(ports["open_webui"]), "0.0.0.0"))
    else:
        if (gateways.get("litellm") or {}).get("enabled") and ports.get("litellm"):
            candidates.append(("litellm", int(ports["litellm"]), "0.0.0.0"))
        if (frontends.get("open_webui") or {}).get("enabled") and (frontends.get("open_webui") or {}).get("publish_port", True) and ports.get("open_webui"):
            candidates.append(("open-webui", int(ports["open_webui"]), "0.0.0.0"))
        reverse_proxy = frontends.get("reverse_proxy") or {}
        if reverse_proxy.get("enabled"):
            if reverse_proxy.get("publish_http", True):
                candidates.append(("reverse-proxy-http", int(reverse_proxy.get("http_port") or ports.get("reverse_proxy_http") or 80), reverse_proxy.get("http_bind_host") or "0.0.0.0"))
            if reverse_proxy.get("publish_https", True):
                candidates.append(("reverse-proxy-https", int(reverse_proxy.get("https_port") or ports.get("reverse_proxy_https") or 443), reverse_proxy.get("https_bind_host") or "0.0.0.0"))
        ollama = providers.get("ollama") or {}
        if ollama.get("enabled") and ollama.get("publish_port") and ports.get("ollama"):
            candidates.append(("ollama", int(ports["ollama"]), "127.0.0.1"))
        for name, rt in ((providers.get("vllm") or {}).get("runtimes") or {}).items():
            if rt.get("publish_port"):
                candidates.append((f"vllm-{name}", int(rt.get("host_port") or 18000), "127.0.0.1"))

    owned = our_published_ports(
        cfg["runtime"]["compose_cmd"],
        generated_dir(cfg) / "docker-compose.yml",
        runtime_env_path(cfg),
    )
    to_check = [(svc, port, host) for svc, port, host in candidates if port not in owned]

    try:
        check_ports_available(to_check)
    except PortInUseError as ex:
        raise SystemExit(str(ex)) from ex


def _apply_path_overrides(config: Any) -> None:
    """Honour ``--config-dir`` / ``--data-dir`` from a parsed subcommand config."""
    overrides = _as_mapping(config)
    if overrides.get("config_dir"):
        set_config_root(overrides["config_dir"])
    if overrides.get("data_dir"):
        set_data_root(overrides["data_dir"])


def _maybe_rerender(config: Any, cfg: dict[str, Any]) -> None:
    """Re-run RenderCLI if runtime overrides changed or rendered outputs are stale.

    Both ``up`` and ``deploy`` need this so the rendered artifacts always
    match the current config + overrides before any container action.
    """
    if has_runtime_overrides(config) or render_is_stale(cfg):
        overrides = _as_mapping(config)
        RenderCLI.main(
            argv=False,
            profile=overrides.get("profile"),
            backend=overrides.get("backend"),
            compose_cmd=overrides.get("compose_cmd"),
            litellm_port=overrides.get("litellm_port"),
            open_webui_port=overrides.get("open_webui_port"),
            postgres_port=overrides.get("postgres_port"),
            namespace=overrides.get("namespace"),
            ingress_host=overrides.get("ingress_host"),
            ingress_enabled=overrides.get("ingress_enabled"),
            generated_dir=overrides.get("generated_dir"),
            allow_unsupported=bool(overrides.get("allow_unsupported")),
            simulate_hardware=overrides.get("simulate_hardware"),
            allowed_gpus=overrides.get("allowed_gpus"),
            yes=bool(overrides.get("yes")),
        )


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


class _GeneratedDirOverrideMixin(scfg.DataConfig):
    generated_dir = scfg.Value(
        None,
        type=str,
        help=(
            "Directory to write rendered artifacts (docker-compose.yml, .env, "
            "plan.yaml, kubeai/*) into. Defaults to <data-dir>/generated. May "
            "also be set via INFER_STACK_GENERATED_DIR or output.generated_dir."
        ),
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
    _GeneratedDirOverrideMixin,
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


# ---------------------------------------------------------------------------
# Profile / config management commands
# ---------------------------------------------------------------------------


class InitCLI(_PathOverridesMixin):
    """Write a fresh config.yaml + empty models.yaml under config_root()."""

    force = scfg.Value(False, isflag=True, help="Overwrite an existing config.yaml.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.force:
            raise SystemExit("config.yaml already exists. Use --force to overwrite.")
        save_yaml(cfg_path, initial_config())
        if not models_path().exists():
            save_yaml(models_path(), {"models": {}, "profiles": {}})
        print(f"Wrote {cfg_path}")
        return 0


class SetupCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _GeneratedDirOverrideMixin,
):
    """Create / update config.yaml with the requested overrides applied."""

    reset = scfg.Value(False, isflag=True, help="Start from default config values before applying overrides.")
    state_root = scfg.Value(None, type=str, help="Base directory for state.* paths.")
    runtime_dir = scfg.Value(None, type=str, help="Override state.runtime specifically.")
    resource_profiles_file = scfg.Value(
        None,
        type=str,
        help="For kubeai setups, sync a local Helm values file with resourceProfiles into kubeai-values.local.yaml.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.reset:
            cfg = load_yaml(cfg_path)
        else:
            cfg = initial_config()
        cfg = apply_config_overrides(cfg, config)
        save_yaml(cfg_path, cfg)
        if not models_path().exists():
            save_yaml(models_path(), {"models": {}, "profiles": {}})
        if config.resource_profiles_file:
            # User-supplied path: anchor on CWD so a typed relative path
            # behaves as the user expects.
            source = Path(config.resource_profiles_file)
            if not source.is_absolute():
                source = Path.cwd() / source
            values_doc = load_yaml(source)
            if "resourceProfiles" not in values_doc:
                raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
            target = save_kubeai_resource_profiles(values_doc)
            if plan_path(cfg).exists():
                plan_path(cfg).unlink()
            print(f"Wrote {target}")
        print(f"Wrote {cfg_path}")
        print(
            f"Configured backend={cfg.get('backend', 'compose')} "
            f"active_profile={cfg.get('active_profile', '') or '<unset>'}"
        )
        return 0


class ResolveCLI(_PlanOverridesCLI):
    """Resolve the active profile into a deployment dict and print it."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["deployment"], indent=2))
        return 0


class ValidateCLI(_PlanOverridesCLI):
    """Resolve + validate the active profile; exit non-zero on validation errors."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["validated"], indent=2))
        return 0 if plan["validated"]["ok"] else 2


class LockCLI(_PlanOverridesCLI):
    """Write plan.yaml after resolving + validating the active profile."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        if not plan["validated"]["ok"] and not plan["allow_unsupported"]:
            raise SystemExit(
                "Refusing to write plan.yaml because validation failed. Use --allow-unsupported to override."
            )
        save_plan(plan, cfg)
        print(json.dumps(plan, indent=2))
        return 0


class RenderCLI(_PlanOverridesCLI):
    """Render the active profile's deployment into compose/kubeai artifacts."""

    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=["y"],
        help="Apply rendered changes without prompting. Without this, render shows a per-file diff and asks for confirmation.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        print(f"Wrote {plan_path(cfg)}")
        if backend_name(cfg) == "kubeai":
            print(f"Rendered KubeAI artifacts into {kubeai_generated_dir(cfg)}")
        else:
            print(f"Rendered Compose into {generated_dir(cfg)}")
            print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
        return 0


class SwitchCLI(_SwitchPathOverridesCLI):
    """Persist a new active_profile and re-render (optionally re-applying)."""

    profile = scfg.Value(None, type=str, position=1, help="Profile name to switch to.")
    apply = scfg.Value(False, isflag=True, help="Also apply the new profile to the running stack.")
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="Apply rendered changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit("switch: missing required profile name")
        persisted_cfg = load_config()
        persisted_cfg["active_profile"] = config.profile
        save_yaml(config_path(), persisted_cfg)
        cfg = apply_config_overrides(persisted_cfg, config)

        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        if config.apply:
            if backend_name(cfg) == "compose":
                _compose_up_with_router_recreate(cfg, detach=True)
            else:
                deploy_rendered_artifacts(plan["deployment"])
        print(f"Switched active_profile to {config.profile}")
        return 0


class ListModelsCLI(_PathOverridesMixin):
    """Print every model in the merged catalog."""

    __command__ = "list-models"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        for name, model in cats.get("models", {}).items():
            ref = model.get("hf_model_id") or model.get("url", "")
            print(f"{name}: {ref}")
        return 0


class ListProfilesCLI(_PathOverridesMixin):
    """Print every serving profile in the merged catalog."""

    __command__ = "list-profiles"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        profiles = cats.get("profiles", {})
        hidden_legacy = set(PROFILE_NAME_ALIASES)
        for name, profile in profiles.items():
            if name in hidden_legacy:
                continue
            if profile.get("kind") == "invalid-profile":
                continue
            summary = profile_summary(profile)
            providers = ",".join(summary.get("providers", [])) or "none"
            print(
                f"{name}: providers={providers} gateway={summary['gateway']} "
                f"frontend={summary['frontend']} frontend_provider={summary['frontend_provider']} "
                f"routes={summary['route_count']}"
            )
        return 0


class ExplainCLI(_PathOverridesMixin):
    """Pretty-print a YAML file (defaults to the current plan.yaml)."""

    __command__ = "explain"

    file = scfg.Value(None, type=str, help="Path to read (default: current plan.yaml).")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if config.file:
            target = Path(config.file)
            if not target.is_absolute():
                target = Path.cwd() / target
        else:
            target = plan_path()
        if not target.exists():
            raise SystemExit(f"Missing file: {target}")
        print(json.dumps(load_yaml(target), indent=2))
        return 0


class DescribeProfileCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Print the profile contract for a given profile name."""

    __command__ = "describe-profile"

    profile = scfg.Value(None, type=str, position=1, help="Profile name to describe.")
    format = scfg.Value("yaml", choices=["json", "yaml"])
    output = scfg.Value(None, type=str, help="Write to this file instead of stdout.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        overrides = _as_mapping(config)
        if not overrides.get("profile"):
            raise SystemExit("describe-profile: missing required profile name")
        contract = load_profile_contract(
            overrides["profile"],
            backend=_arg_or_env(overrides, "backend", "INFER_STACK_BACKEND"),
            simulate_hardware_spec=overrides.get("simulate_hardware"),
        )
        return _print_structured(contract, overrides["format"], overrides.get("output"))


def _print_structured(data: dict[str, Any], fmt: str, output: str | None) -> int:
    if fmt == "yaml":
        import yaml

        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(f"Wrote {target}")
        return 0
    print(text)
    return 0


class VerifyProfileCLI(_SwitchPathOverridesCLI):
    """Sanity-check a resolved profile (post-render expectations)."""

    __command__ = "verify-profile"

    profile = scfg.Value(None, type=str, position=1, help="Profile name to verify.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit("verify-profile: missing required profile name")
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        result = verify_profile(plan["deployment"])
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2


class KubeaiSyncResourceProfilesCLI(_PathOverridesMixin):
    """Pull a Helm ``resourceProfiles`` values file into kubeai-values.local.yaml."""

    __command__ = "kubeai-sync-resource-profiles"

    from_file = scfg.Value(None, type=str, required=True, help="Helm values file with a top-level resourceProfiles map.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)
        # User-supplied path: anchor on CWD.
        source = Path(config.from_file)
        if not source.is_absolute():
            source = Path.cwd() / source
        values_doc = load_yaml(source)
        if "resourceProfiles" not in values_doc:
            raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
        target = save_kubeai_resource_profiles(values_doc)
        if plan_path(cfg).exists():
            plan_path(cfg).unlink()
        profiles, _, _ = load_kubeai_resource_profiles()
        print(f"Wrote {target}")
        print(f"Synced {len(profiles)} KubeAI resource profile(s)")
        return 0


# ---------------------------------------------------------------------------
# Export / bundle commands (transitional; helm_audit owns the canonical path)
# ---------------------------------------------------------------------------


class _ExportBundleCLI(_SwitchPathOverridesCLI):
    """Shared body for ``export-benchmark-bundle`` / ``export-helm-bundle``."""

    profile = scfg.Value(None, type=str, position=1, help="Profile name to export.")
    base_url = scfg.Value(None, type=str)
    output_dir = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit(f"{cls.__command__}: missing required profile name")
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        print(
            "Benchmark bundle export here is transitional; prefer the helm_audit "
            "integration layer for CRFM HELM bundle generation."
        )
        output_dir = None
        if config.output_dir:
            output_dir = Path(config.output_dir)
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir
        result = export_benchmark_bundle(
            plan["deployment"],
            base_url=config.base_url,
            output_dir=output_dir,
        )
        print(f"Wrote {result['bundle_path']}")
        print(f"Wrote {result['model_deployments_path']}")
        return 0


class ExportBenchmarkBundleCLI(_ExportBundleCLI):
    """Export a benchmark bundle (transitional; helm_audit owns the canonical path)."""

    __command__ = "export-benchmark-bundle"


class ExportHelmBundleCLI(_ExportBundleCLI):
    """Alias of export-benchmark-bundle that emits the legacy helm/ layout."""

    __command__ = "export-helm-bundle"


# ---------------------------------------------------------------------------
# Runtime commands
# ---------------------------------------------------------------------------


class UpCLI(_PlanOverridesCLI):
    """Bring the rendered compose stack up. Re-renders first if anything changed."""

    detach = scfg.Value(False, isflag=True, short_alias=["d"], help="Run in background instead of attaching to logs.")
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="If `up` triggers a re-render, apply changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`up` only supports the compose backend. Use `deploy` for kubeai.")
        _maybe_rerender(config, cfg)
        _compose_up_with_router_recreate(cfg, detach=bool(config.detach))
        return 0


class DownCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Bring the rendered compose stack down (does not touch volumes)."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`down` only supports the compose backend.")
        # Re-render before down so a stale/invalid compose file from an older
        # package version does not strand containers.  Compose's
        # --remove-orphans still removes services from the previous profile
        # when the project name / generated directory is unchanged.
        _maybe_rerender(config, cfg)
        try:
            compose_down(
                cfg["runtime"]["compose_cmd"],
                generated_dir(cfg) / "docker-compose.yml",
                runtime_env_path(cfg),
            )
        except DockerCommandError as ex:
            raise SystemExit(
                f"compose down failed after re-rendering the current stack: {ex}"
            ) from ex
        return 0


class PurgeCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Stop the stack and delete all Docker-written state directories.

    Uses a temporary Alpine container to remove directories that Docker wrote
    as root, avoiding ``Permission denied`` errors from plain ``rm -rf``.
    """

    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="Skip confirmation prompt.")
    delete_cache = scfg.Value(
        False, isflag=True,
        help="Also delete hf-cache and vllm-cache (model weights). By default those are preserved.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)

        state = normalized_state(cfg.get("state", {}))
        always_delete = ["postgres_litellm", "postgres_open_webui", "open_webui", "ollama", "runtime"]
        model_dirs = ["hf_cache", "vllm_cache"]
        keys = always_delete + model_dirs if config.delete_cache else always_delete
        dirs_to_delete = [Path(state[k]) for k in keys if Path(state[k]).exists()]

        if not dirs_to_delete:
            print("Nothing to purge — state directories do not exist.")
            return 0

        if not config.yes:
            print("The following directories will be deleted (via Docker to handle root-owned files):")
            for d in dirs_to_delete:
                print(f"  {d}")
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Aborted.")
                return 1

        compose_file = generated_dir(cfg) / "docker-compose.yml"
        if compose_file.exists() and backend_name(cfg) == "compose":
            try:
                compose_down(
                    cfg["runtime"]["compose_cmd"],
                    compose_file,
                    runtime_env_path(cfg),
                )
            except Exception as ex:
                print(f"Warning: compose down failed (containers may already be stopped): {ex}")

        compose_cmd = cfg.get("runtime", {}).get("compose_cmd", "docker compose")
        docker_cmd = compose_cmd.split()[0]
        docker_rm_dirs(dirs_to_delete, docker_cmd=docker_cmd)
        print("Purge complete.")
        return 0


class DeployCLI(_PlanOverridesCLI):
    """Apply the rendered stack to its backend (kubeai apply / compose up)."""

    detach = scfg.Value(False, isflag=True, short_alias=["d"])
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="If `deploy` triggers a re-render, apply changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        _maybe_rerender(config, cfg)
        if backend_name(cfg) == "kubeai":
            plan = load_yaml(plan_path(cfg))
            try:
                deploy_rendered_artifacts(plan["deployment"])
            except CommandError as ex:
                namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
                raise SystemExit(
                    f"Failed to deploy to namespace {namespace!r}. Confirm "
                    f"`infer-stack setup --backend kubeai --namespace {namespace}` "
                    "matches the namespace where the KubeAI Helm release is installed.\n"
                    f"Original error: {ex}"
                ) from ex
            return 0
        compose_up(
            cfg["runtime"]["compose_cmd"],
            generated_dir(cfg) / "docker-compose.yml",
            generated_dir(cfg) / ".env",
            detach=bool(config.detach),
            remove_orphans=True,
        )
        return 0


class EnvCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
):
    """Inspect the rendered .env file (path, single value, or eval-friendly export)."""

    key = scfg.Value(None, type=str, position=1, help="Print only this variable's value. Empty = all.")
    export = scfg.Value(False, isflag=True, help="Print `export KEY=value` lines suitable for `eval`.")
    path = scfg.Value(False, isflag=True, help="Print only the absolute path to .env (default if no flags).")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`env` only applies to the compose backend.")
        env_file = runtime_env_path(cfg)
        if not env_file.exists():
            raise SystemExit(
                f"No .env at {env_file}. Run `infer-stack render` first."
            )
        # Default with no flags and no key: print the path so users can do
        #   `source $(infer-stack env)` (with `set -a` if they want export semantics).
        if config.key is None and not config.export:
            print(env_file)
            return 0
        env = parse_env_file(env_file)
        if config.key:
            if config.key not in env:
                raise SystemExit(f"{config.key!r} not found in {env_file}")
            print(env[config.key])
            return 0
        # --export: emit eval-friendly export lines.
        for k, v in env.items():
            print(f"export {k}={shlex.quote(v)}")
        return 0


class StatusCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
):
    """Show the runtime status of the rendered stack."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) == "kubeai":
            namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
            try:
                kubeai_print_status(namespace)
            except CommandError as ex:
                raise SystemExit(
                    f"Failed to query KubeAI resources in namespace {namespace!r}. Confirm "
                    f"`infer-stack setup --backend kubeai --namespace {namespace}` "
                    "matches the namespace where the KubeAI Helm release is installed.\n"
                    f"Original error: {ex}"
                ) from ex
            return 0
        proc = subprocess.run(_compose_base_cmd(cfg) + ["ps"])
        return int(proc.returncode)


# ---------------------------------------------------------------------------
# Compose day-2-ops wrappers (raise NotImplementedError on kubeai)
# ---------------------------------------------------------------------------


class _ComposeWrapperBase(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Common fields for ``docker compose <subcmd>`` wrappers."""

    services = scfg.Value(None, nargs="*", position=1, help="Optional service names to filter (empty = all).")


class LogsCLI(_ComposeWrapperBase):
    """Tail rendered Compose service logs without typing the full docker compose path."""

    follow = scfg.Value(False, isflag=True, short_alias=["f"], help="Stream logs (docker compose logs -f).")
    tail = scfg.Value(None, type=str, help="Tail the last N lines (default: all). Pass a number or 'all'.")
    timestamps = scfg.Value(False, isflag=True)
    no_color = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("logs")
        cmd = _compose_base_cmd(cfg) + ["logs"]
        if config.follow:
            cmd.append("--follow")
        if config.tail is not None:
            cmd.extend(["--tail", str(config.tail)])
        if config.no_color:
            cmd.append("--no-color")
        if config.timestamps:
            cmd.append("--timestamps")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PsCLI(_ComposeWrapperBase):
    """``docker compose ps`` for the rendered stack."""

    all = scfg.Value(False, isflag=True, short_alias=["a"], help="Include stopped containers.")
    services_only = scfg.Value(False, isflag=True, help="Print only service names (passes --services to docker compose).")
    quiet = scfg.Value(False, isflag=True, short_alias=["q"], help="Print only container IDs.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("ps")
        cmd = _compose_base_cmd(cfg) + ["ps"]
        if config.all:
            cmd.append("--all")
        if config.services_only:
            cmd.append("--services")
        if config.quiet:
            cmd.append("--quiet")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class RestartCLI(_ComposeWrapperBase):
    """``docker compose restart [services...]``."""

    timeout = scfg.Value(None, type=int, help="Stop timeout in seconds.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("restart")
        cmd = _compose_base_cmd(cfg) + ["restart"]
        if config.timeout is not None:
            cmd.extend(["--timeout", str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PullCLI(_ComposeWrapperBase):
    """``docker compose pull [services...]``."""

    quiet = scfg.Value(False, isflag=True, short_alias=["q"])
    ignore_pull_failures = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("pull")
        cmd = _compose_base_cmd(cfg) + ["pull"]
        if config.quiet:
            cmd.append("--quiet")
        if config.ignore_pull_failures:
            cmd.append("--ignore-pull-failures")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StartCLI(_ComposeWrapperBase):
    """``docker compose start [services...]``."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("start")
        cmd = _compose_base_cmd(cfg) + ["start"]
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StopCLI(_ComposeWrapperBase):
    """``docker compose stop [services...]``."""

    timeout = scfg.Value(None, type=int)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("stop")
        cmd = _compose_base_cmd(cfg) + ["stop"]
        if config.timeout is not None:
            cmd.extend(["--timeout", str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class OllamaPullCLI(_PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin):
    """Pull an Ollama model into the rendered Ollama model store."""

    __command__ = "ollama-pull"

    model = scfg.Value(None, type=str, position=1, help="Ollama model tag to pull, e.g. smollm2:135m.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.model:
            raise SystemExit("ollama-pull: missing required model tag, e.g. `infer-stack ollama-pull smollm2:135m`")
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`ollama-pull` only supports the compose backend.")
        cmd = _compose_base_cmd(cfg) + ["exec", "ollama", "ollama", "pull", str(config.model)]
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class OllamaListCLI(_PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin):
    """List models installed in the rendered Ollama service."""

    __command__ = "ollama-list"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`ollama-list` only supports the compose backend.")
        cmd = _compose_base_cmd(cfg) + ["exec", "ollama", "ollama", "list"]
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class OllamaPsCLI(_PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin):
    """Show loaded Ollama models for the rendered Ollama service."""

    __command__ = "ollama-ps"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`ollama-ps` only supports the compose backend.")
        cmd = _compose_base_cmd(cfg) + ["exec", "ollama", "ollama", "ps"]
        proc = subprocess.run(cmd)
        return int(proc.returncode)


# ---------------------------------------------------------------------------
# Smoke-test / benchmark commands
# ---------------------------------------------------------------------------




def _default_model_for_deployment(deployment: dict[str, Any], explicit: str | None = None) -> str | None:
    """Pick a reasonable model name for readiness/smoke probes."""
    if explicit:
        return str(explicit)
    litellm_routes = ((deployment.get("gateways", {}) or {}).get("litellm", {}) or {}).get("routes", {}) or {}
    if litellm_routes:
        return str(next(iter(litellm_routes)))
    vllm_runtimes = ((deployment.get("providers", {}) or {}).get("vllm", {}) or {}).get("runtimes", {}) or {}
    if vllm_runtimes:
        first = next(iter(vllm_runtimes.values()))
        return str(first.get("served_model_name") or first.get("logical_model_name") or first.get("runtime_name") or "") or None
    ollama_routes = ((deployment.get("providers", {}) or {}).get("ollama", {}) or {}).get("routes", {}) or {}
    if ollama_routes:
        first = next(iter(ollama_routes.values()))
        return str(first.get("upstream_model") or first.get("model_ref") or "") or None
    return None


def _resolve_smoke_protocol_from_deployment(deployment: dict[str, Any], model_name: str | None) -> str:
    """Resolve chat vs completions from schema-v5 routes/runtimes."""
    if model_name:
        routes = ((deployment.get("gateways", {}) or {}).get("litellm", {}) or {}).get("routes", {}) or {}
        route = routes.get(model_name)
        if route:
            return str(route.get("protocol_mode") or "chat")
        vllm_runtimes = ((deployment.get("providers", {}) or {}).get("vllm", {}) or {}).get("runtimes", {}) or {}
        for rt in vllm_runtimes.values():
            aliases = set(rt.get("served_aliases") or [])
            aliases.add(str(rt.get("served_model_name") or ""))
            if model_name in aliases:
                return str(rt.get("protocol_mode") or "chat")
    return "chat"


def _ready_openai_probe(
    *,
    base_url: str,
    headers: dict[str, str],
    model: str | None,
    protocol: str,
    prompt: str,
    max_tokens: int,
    require_generation: bool,
) -> tuple[bool, str]:
    """Probe an OpenAI-compatible surface once without exiting."""
    try:
        models_resp = requests.get(f"{base_url}/models", headers=headers, timeout=10)
    except requests.exceptions.RequestException as ex:
        return False, f"/models not reachable yet: {ex}"
    if models_resp.status_code >= 400:
        body = (models_resp.text or "").strip()
        return False, f"/models returned HTTP {models_resp.status_code}: {body[:300]}"
    try:
        models_doc = models_resp.json()
    except ValueError:
        return False, "/models returned non-JSON response"
    models = models_doc.get("data") or []
    model_name = model or (models[0].get("id") if models else None)
    if not model_name:
        return False, "/models is reachable but no models are advertised"
    if not require_generation:
        return True, f"/models is ready; selected model {model_name}"
    if protocol == "completions":
        payload = {"model": model_name, "prompt": prompt, "max_tokens": max_tokens}
        endpoint = f"{base_url}/completions"
    else:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        endpoint = f"{base_url}/chat/completions"
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
    except requests.exceptions.RequestException as ex:
        return False, f"{endpoint} not serving yet: {ex}"
    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        return False, f"{endpoint} returned HTTP {resp.status_code}: {body[:300]}"
    return True, f"{model_name} served a {protocol} probe"


def _ready_ollama_probe(
    *,
    base_url: str,
    model: str | None,
    prompt: str,
    max_tokens: int,
    require_generation: bool,
) -> tuple[bool, str]:
    """Probe an Ollama-native surface once without exiting."""
    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=10)
    except requests.exceptions.RequestException as ex:
        return False, f"/api/tags not reachable yet: {ex}"
    if tags_resp.status_code >= 400:
        body = (tags_resp.text or "").strip()
        return False, f"/api/tags returned HTTP {tags_resp.status_code}: {body[:300]}"
    try:
        tags_doc = tags_resp.json()
    except ValueError:
        return False, "/api/tags returned non-JSON response"
    models = tags_doc.get("models") or []
    model_name = model or (models[0].get("name") if models else None)
    if not model_name:
        if require_generation:
            return False, "Ollama is reachable but no model is installed; run `infer-stack ollama-pull <tag>`"
        return True, "Ollama API is reachable"
    if not require_generation:
        return True, f"Ollama API is reachable; selected model {model_name}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=45)
    except requests.exceptions.RequestException as ex:
        return False, f"/api/chat not serving yet: {ex}"
    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        return False, f"/api/chat returned HTTP {resp.status_code}: {body[:300]}"
    return True, f"{model_name} served an Ollama chat probe"


def _wait_until_ready(
    cfg: dict[str, Any],
    config: Any,
    *,
    model: str | None = None,
    timeout: float = 600.0,
    interval: float = 5.0,
    prompt: str = "Reply with ready.",
    max_tokens: int = 1,
    require_generation: bool = True,
    quiet: bool = False,
) -> str:
    """Wait until the active profile can serve a real request.

    Docker Compose health only tells us that a process/container passed its
    healthcheck.  For vLLM, the API can exist before the model path is fully
    ready through LiteLLM.  This probes the user-facing access surface and, by
    default, requires a tiny generation/completion to succeed.
    """
    import time

    plan = _smoke_plan(cfg, config)
    deployment = plan.get("deployment", {})
    access = deployment.get("access", {}).get("default", {}) or {}
    access_kind = str(access.get("kind") or "openai-compatible")
    base_url = _infer_default_base_url(cfg, config, deployment=deployment)
    model_name = _default_model_for_deployment(deployment, explicit=model)
    deadline = time.monotonic() + float(timeout)
    last_message = "not probed yet"
    attempt = 0

    env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
    headers = {"Content-Type": "application/json"}
    if access_kind != "ollama-native":
        auth_env_name = str(access.get("auth_env_name") or "LITELLM_MASTER_KEY")
        api_key = getattr(config, "api_key", None) or env.get(auth_env_name, "") or env.get("LITELLM_MASTER_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    protocol = _resolve_smoke_protocol_from_deployment(deployment, model_name)
    while True:
        attempt += 1
        if access_kind == "ollama-native":
            ok, message = _ready_ollama_probe(
                base_url=base_url,
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                require_generation=require_generation,
            )
        else:
            ok, message = _ready_openai_probe(
                base_url=base_url,
                headers=headers,
                model=model_name,
                protocol=protocol,
                prompt=prompt,
                max_tokens=max_tokens,
                require_generation=require_generation,
            )
        last_message = message
        if ok:
            if not quiet:
                print(f"Ready: {message}")
            return message
        now = time.monotonic()
        if now >= deadline:
            raise SystemExit(
                "Timed out waiting for the active stack to serve requests.\n"
                f"Last probe: {_explain_readiness_message(last_message)}\n"
                "Useful diagnostics:\n"
                "  infer-stack diagnose --logs --tail 80\n"
                "  infer-stack ps\n"
                "  infer-stack logs vllm-* litellm open-webui"
            )
        if not quiet and (attempt == 1 or attempt % 6 == 0):
            print(f"Waiting for readiness: {_explain_readiness_message(last_message)}")
        time.sleep(float(interval))

def _smoke_plan(cfg: dict[str, Any], config: Any) -> dict[str, Any]:
    overrides = _as_mapping(config)
    return build_plan(
        cfg,
        profile_name=overrides.get("profile"),
        allow_unsupported=effective_allow_unsupported(config, cfg),
        inventory=effective_inventory(config),
    )


def _infer_default_base_url(cfg: dict[str, Any], config: Any, deployment: dict[str, Any] | None = None) -> str:
    explicit = _as_mapping(config).get("base_url")
    if explicit:
        return str(explicit).rstrip("/")
    if deployment is None:
        try:
            deployment = _smoke_plan(cfg, config).get("deployment", {})
        except Exception:
            deployment = {
                "backend": backend_name(cfg),
                "cluster": cfg.get("cluster", {}),
                "ports": cfg.get("ports", {}),
            }
    return default_base_url(deployment)


def _smoke_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    timeout: float = 30,
    retries: int = 1,
    retry_delay: float = 2.0,
) -> requests.Response:
    """Wrapper around ``requests.{get,post}`` that emits actionable errors.

    The smoke test runs against a stack that may be (a) not listening yet,
    (b) listening but with an unhealthy upstream that resets connections, or
    (c) returning HTTP errors during model load. Retry transient startup
    failures so ``switch --apply && smoke-test`` is usable immediately after a
    provider container was recreated.
    """
    last_timeout: requests.exceptions.Timeout | None = None
    last_conn: requests.exceptions.ConnectionError | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            break
        except requests.exceptions.Timeout as ex:
            last_timeout = ex
            if attempt < retries:
                import time

                time.sleep(retry_delay)
                continue
            raise SystemExit(
                f"Request to {url} timed out after {timeout}s.\n"
                "The model may still be loading, or the server is overloaded.\n"
                "  infer-stack logs vllm-*"
            ) from ex
        except requests.exceptions.ConnectionError as ex:
            last_conn = ex
            if attempt < retries:
                import time

                time.sleep(retry_delay)
                continue
            # Two distinct sub-cases inside ConnectionError that warrant different
            # remediation: (a) nothing listening on the port, (b) something is
            # listening but it closed the connection without responding (typical
            # of LiteLLM up but a depended-on vLLM container still loading the
            # model and failing the dependency health-check chain).
            cause = ex.args[0] if ex.args else ex
            cause_str = str(cause)
            if "RemoteDisconnected" in cause_str or "Connection aborted" in cause_str:
                raise SystemExit(
                    f"Connection to {url} was closed before a response arrived.\n"
                    "The router is listening but an upstream service is not ready yet.\n"
                    "Check container status and logs:\n"
                    "  infer-stack ps\n"
                    "  infer-stack logs vllm-*"
                ) from ex
            if "Connection refused" in cause_str or "Failed to establish a new connection" in cause_str:
                raise SystemExit(
                    f"Could not connect to {url}: nothing is listening yet.\n"
                    "If you just ran `infer-stack up`, give the router a few seconds.\n"
                    "  infer-stack ps                # confirm the litellm container is running\n"
                    "  infer-stack logs litellm      # check for startup errors"
                ) from ex
            raise SystemExit(f"Connection error reaching {url}: {cause_str}") from ex
    else:  # pragma: no cover - defensive; loop exits via break or raise
        if last_timeout is not None:
            raise last_timeout
        if last_conn is not None:
            raise last_conn
        raise RuntimeError("smoke request failed without an exception")
    status = getattr(resp, "status_code", 200)
    if status >= 400:
        body = getattr(resp, "text", "") or ""
        body = body.strip()
        if len(body) > 500:
            body = body[:500] + "... [truncated]"
        reason = getattr(resp, "reason", "") or ""
        if status in (401, 403):
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "The auth key didn't match what the running container expects.\n"
                "If you re-rendered after the container started, the key in .env "
                "may have changed. Restart with:\n"
                "  infer-stack down && infer-stack up -d\n"
                f"Response: {body}"
            )
        if status == 503:
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "An upstream service is unavailable (commonly the vLLM engine is still loading).\n"
                "  infer-stack logs vllm-*\n"
                f"Response: {body}"
            )
        raise SystemExit(
            f"HTTP {status} {reason} from {url}.\nResponse: {body}"
        )
    return resp


def _resolve_smoke_test_protocol(
    cfg: dict[str, Any],
    config: Any,
    model_name: str,
) -> str:
    """Pick the OpenAI route for smoke-test based on protocol resolution order.

    1. ``--protocol`` CLI override (``chat`` or ``completions``).
    2. Resolved deployment: if the requested model maps to a service whose
       protocol_mode is known, use that.
    3. Active profile's primary service protocol_mode.
    4. Fallback: ``chat``.
    """
    overrides = _as_mapping(config)
    explicit = overrides.get("protocol")
    if explicit:
        return str(explicit)
    try:
        plan = build_plan(
            cfg,
            profile_name=overrides.get("profile"),
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
    except Exception:
        return "chat"
    deployment = plan.get("deployment", {})
    return _resolve_smoke_protocol_from_deployment(deployment, model_name)


def _ollama_smoke_test(
    base_url: str,
    *,
    model: str | None,
    prompt: str,
    max_tokens: int,
    skip_chat: bool,
) -> int:
    """Smoke-test an Ollama-native endpoint without requiring LiteLLM."""
    tags_resp = _smoke_request("GET", f"{base_url}/api/tags", timeout=30, retries=12, retry_delay=5)
    tags_doc = tags_resp.json()
    print(json.dumps(tags_doc, indent=2))
    if skip_chat:
        return 0
    models = tags_doc.get("models") or []
    model_name = model or (models[0].get("name") if models else None)
    if not model_name:
        raise SystemExit(
            "Ollama is reachable, but no models are installed in its model store.\n"
            "Pull one through the CLI wrapper, for example:\n"
            "  infer-stack ollama-pull smollm2:135m\n"
            "Then rerun:\n"
            "  infer-stack smoke-test --model smollm2:135m"
        )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    resp = _smoke_request("POST", f"{base_url}/api/chat", json_body=payload, timeout=120, retries=3, retry_delay=5)
    print(json.dumps(resp.json(), indent=2))
    return 0



class DiagnoseCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Print targeted diagnostics for the active rendered stack.

    This command is intentionally more specific than ``ps`` or ``logs``.  It
    prints the resolved provider/gateway/frontend graph, rendered compose
    service state, LiteLLM route probes, direct provider probes, and optional
    recent logs.  It helps distinguish these cases:

    * LiteLLM container is actually absent/down.
    * LiteLLM is running but its upstream vLLM process is still booting.
    * Open WebUI is polling a provider that is not present in the active
      profile.
    """

    __command__ = "diagnose"

    model = scfg.Value(None, type=str, help="Model/alias to use for optional generation diagnostics.")
    logs = scfg.Value(False, isflag=True, help="Include recent logs for litellm/open-webui/vllm/ollama services.")
    tail = scfg.Value(80, type=int, help="Number of log lines per service when --logs is set.")
    generation = scfg.Value(False, isflag=True, help="Also run a tiny generation probe through the active access surface.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=_as_mapping(config).get("profile"),
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        deployment = plan.get("deployment", {}) or {}
        print(f"active_profile: {deployment.get('source', {}).get('active_profile') or cfg.get('active_profile')}")
        print(f"backend: {deployment.get('backend') or backend_name(cfg)}")
        print(f"plan: {plan_path(cfg)}")
        access = (deployment.get("access", {}) or {}).get("default", {}) or {}
        if access:
            print("default access:")
            print(f"  kind: {access.get('kind')}")
            print(f"  base_url: {access.get('base_url')}")
            if access.get("auth_env_name"):
                print(f"  auth_env_name: {access.get('auth_env_name')}")

        providers = deployment.get("providers", {}) or {}
        gateways = deployment.get("gateways", {}) or {}
        frontends = deployment.get("frontends", {}) or {}
        print("\nresolved graph:")
        print(f"  providers: {', '.join(k for k, v in providers.items() if (v or {}).get('enabled') or (v or {}).get('runtimes')) or 'none'}")
        print(f"  gateways:  {', '.join(k for k, v in gateways.items() if (v or {}).get('enabled')) or 'none'}")
        print(f"  frontends: {', '.join(k for k, v in frontends.items() if (v or {}).get('enabled')) or 'none'}")
        litellm_routes = ((gateways.get("litellm") or {}).get("routes") or {})
        if litellm_routes:
            print("\nLiteLLM routes:")
            for alias, route in litellm_routes.items():
                print(
                    f"  {alias}: provider={route.get('provider')} "
                    f"runtime={route.get('runtime', '-')} upstream={route.get('upstream_model', route.get('model', '-'))} "
                    f"protocol={route.get('protocol_mode', 'chat')}"
                )

        if backend_name(cfg) == "compose":
            _print_compose_diagnostics(cfg, tail=int(config.tail) if config.logs else 0)
            _print_gateway_diagnostics(cfg, deployment, model=config.model, require_generation=bool(config.generation))
        else:
            namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
            kubeai_print_status(namespace)
        return 0


class WaitReadyCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Wait until the active profile can serve a real request.

    This is stronger than Docker Compose health.  It probes the same access
    surface users will hit (LiteLLM, direct Ollama, or direct vLLM) and, by
    default, requires a tiny generation/completion to succeed.
    """

    __command__ = "wait-ready"

    base_url = scfg.Value(None, type=str, help="Override the resolved base URL.")
    api_key = scfg.Value(None, type=str, help="Override the auth key for OpenAI-compatible surfaces.")
    model = scfg.Value(None, type=str, help="Model/alias to probe. Defaults to the first active route/runtime.")
    prompt = scfg.Value("Reply with ready.", type=str)
    max_tokens = scfg.Value(1, type=int)
    timeout = scfg.Value(600, type=float, help="Maximum seconds to wait.")
    interval = scfg.Value(5, type=float, help="Seconds between probes.")
    skip_generation = scfg.Value(False, isflag=True, help="Only wait for the API model listing/tag endpoint, not generation.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        _wait_until_ready(
            cfg,
            config,
            model=config.model,
            timeout=float(config.timeout),
            interval=float(config.interval),
            prompt=config.prompt,
            max_tokens=int(config.max_tokens),
            require_generation=not bool(config.skip_generation),
            quiet=False,
        )
        return 0


class SmokeTestCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Probe the running router with a single chat/completions request."""

    __command__ = "smoke-test"

    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)
    model = scfg.Value(None, type=str)
    prompt = scfg.Value("Say hello in one sentence.", type=str)
    max_tokens = scfg.Value(128, type=int)
    skip_chat = scfg.Value(False, isflag=True)
    no_wait = scfg.Value(False, isflag=True, help="Do not wait for the active access surface to serve a real request before the smoke request.")
    wait_timeout = scfg.Value(600, type=float, help="Seconds to wait for readiness before the smoke request.")
    wait_interval = scfg.Value(5, type=float, help="Seconds between readiness probes.")
    protocol = scfg.Value(
        None,
        choices=["chat", "completions"],
        help="Force the smoke-test endpoint. Defaults to the resolved profile's protocol_mode.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = _smoke_plan(cfg, config)
        deployment = plan.get("deployment", {})
        access = deployment.get("access", {}).get("default", {}) or {}
        env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
        base_url = _infer_default_base_url(cfg, config, deployment=deployment)

        if not bool(config.no_wait):
            _wait_until_ready(
                cfg,
                config,
                model=config.model,
                timeout=float(config.wait_timeout),
                interval=float(config.wait_interval),
                prompt=config.prompt,
                max_tokens=1,
                require_generation=not bool(config.skip_chat),
                quiet=True,
            )

        access_kind = str(access.get("kind") or "openai-compatible")
        explicit_base_url = bool(_as_mapping(config).get("base_url"))
        if access_kind == "ollama-native" and not explicit_base_url:
            return _ollama_smoke_test(
                base_url,
                model=config.model,
                prompt=config.prompt,
                max_tokens=int(config.max_tokens),
                skip_chat=bool(config.skip_chat),
            )

        headers = {"Content-Type": "application/json"}
        auth_env_name = str(access.get("auth_env_name") or "LITELLM_MASTER_KEY")
        api_key = config.api_key or env.get(auth_env_name, "") or env.get("LITELLM_MASTER_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        models_resp = _smoke_request("GET", f"{base_url}/models", headers=headers, timeout=30, retries=12, retry_delay=5)
        models = models_resp.json().get("data", [])
        print(json.dumps(models_resp.json(), indent=2))
        if config.skip_chat:
            return 0
        if not models:
            raise SystemExit("No models returned from /models")
        model_name = config.model or models[0]["id"]
        protocol = _resolve_smoke_test_protocol(cfg, config, model_name)
        if protocol == "completions":
            payload = {
                "model": model_name,
                "prompt": config.prompt,
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/completions"
        else:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": config.prompt}],
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/chat/completions"
        resp = _smoke_request("POST", endpoint, headers=headers, json_body=payload, timeout=120, retries=3, retry_delay=5)
        print(json.dumps(resp.json(), indent=2))
        return 0


class BenchmarkCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Run benchmark_prompts.json against the router."""

    model = scfg.Value(None, type=str, required=True)
    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        # benchmark_prompts.json is a user-supplied fixture. Look for it
        # first in the config dir, then fall back to CWD so an ad-hoc
        # invocation from a checkout still picks up a sibling file.
        prompts_path = config_root() / "benchmark_prompts.json"
        if not prompts_path.exists():
            prompts_path = Path.cwd() / "benchmark_prompts.json"
        if not prompts_path.exists():
            raise SystemExit(
                f"benchmark_prompts.json not found at {config_root() / 'benchmark_prompts.json'} "
                f"or {Path.cwd() / 'benchmark_prompts.json'}"
            )
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        cfg = config_for_runtime(config)
        env = parse_env_file(runtime_env_path(cfg))
        base_url = config.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
        api_key = config.api_key or env.get("LITELLM_MASTER_KEY", "")
        data = run_benchmark(base_url, api_key, config.model, prompts)
        print(json.dumps(data, indent=2))
        return 0


# ---------------------------------------------------------------------------
# Modal CLI + entry point
# ---------------------------------------------------------------------------


class ManageCLI(scfg.ModalCLI):
    description = (
        "Render and run vLLM serving profiles through the Compose or KubeAI "
        "backends. Primary workflow: setup -> render -> up (or deploy)."
    )

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
    export_benchmark_bundle = ExportBenchmarkBundleCLI
    export_helm_bundle = ExportHelmBundleCLI

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


if __name__ == "__main__":
    raise SystemExit(main())
