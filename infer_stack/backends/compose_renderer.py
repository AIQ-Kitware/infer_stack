from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import yaml
from jinja2 import BaseLoader, Environment

from ..config import normalized_output, normalized_state, DEFAULT_PORTS
from ..diff_prompt import confirm_writes
from ..env_utils import ensure_secret, parse_env_file, write_env_file


def _template(name: str) -> str:
    return files("infer_stack").joinpath(f"templates/{name}").read_text(encoding="utf-8")


def _compose_quote(value: object) -> str:
    """Quote scalars for Compose YAML while preserving env interpolation text."""

    return json.dumps("" if value is None else str(value))


def _compose_gpus(value: object, indent: int = 4) -> str:
    """Render the ``gpus:`` service key for either form Compose accepts.

    Compose understands both a scalar (``all``, ``-1``, or a device count) and
    a structured list of device-request mappings.  Scalars render inline and
    quoted; lists/dicts are emitted as indented block YAML so the structured
    "GPU settings" escape hatch round-trips faithfully instead of collapsing to
    a quoted Python repr.
    """

    pad = " " * indent
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = "true" if value else "false"
    if isinstance(value, (str, int, float)):
        scalar = _compose_quote(value) if isinstance(value, str) else json.dumps(value)
        return f"{pad}gpus: {scalar}"
    dumped = yaml.safe_dump({"gpus": value}, default_flow_style=False, sort_keys=False).rstrip("\n")
    return "\n".join(pad + line for line in dumped.splitlines())


def render_compose_artifacts(lock_data: dict, *, assume_yes: bool = True) -> None:
    """Render component-aware Compose artifacts for the resolved stack."""
    deployment = dict(lock_data.get("deployment", {}))
    deployment["state"] = normalized_state(deployment.get("state", {}))
    deployment["output"] = normalized_output(deployment.get("output"))
    generated = Path(deployment["output"]["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(deployment["state"]["runtime"])
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env_path = generated / ".env"
    existing = parse_env_file(env_path)
    env_values: dict[str, str] = {}

    frontends = deployment.get("frontends", {}) or {}
    gateways = deployment.get("gateways", {}) or {}
    providers = deployment.get("providers", {}) or {}

    if (frontends.get("open_webui") or {}).get("enabled"):
        env_values.update(
            {
                "OPENWEBUI_POSTGRES_DB": existing.get("OPENWEBUI_POSTGRES_DB", "openwebui"),
                "OPENWEBUI_POSTGRES_USER": existing.get("OPENWEBUI_POSTGRES_USER", "openwebui"),
                "OPENWEBUI_POSTGRES_PASSWORD": ensure_secret(existing, "OPENWEBUI_POSTGRES_PASSWORD"),
                "WEBUI_SECRET_KEY": ensure_secret(existing, "WEBUI_SECRET_KEY"),
            }
        )
        ldap_defaults = (((frontends.get("open_webui") or {}).get("ldap") or {}).get("env_defaults") or {})
        for key, default in ldap_defaults.items():
            env_values[key] = existing.get(key, str(default))

    if (gateways.get("litellm") or {}).get("enabled"):
        env_values.update(
            {
                "LITELLM_POSTGRES_DB": existing.get("LITELLM_POSTGRES_DB", "litellm"),
                "LITELLM_POSTGRES_USER": existing.get("LITELLM_POSTGRES_USER", "litellm"),
                "LITELLM_POSTGRES_PASSWORD": ensure_secret(existing, "LITELLM_POSTGRES_PASSWORD"),
                "LITELLM_MASTER_KEY": ensure_secret(existing, "LITELLM_MASTER_KEY", prefix="sk-"),
            }
        )

    if (providers.get("vllm") or {}).get("enabled"):
        env_values.update(
            {
                "VLLM_BACKEND_API_KEY": ensure_secret(existing, "VLLM_BACKEND_API_KEY"),
                "HF_TOKEN": existing.get("HF_TOKEN", ""),
            }
        )

    # Ports: expose configured host ports via environment variables so
    # docker-compose can reference them and we persist them into `.env`.
    ports = deployment.get("ports", {}) or {}

    # LiteLLM port
    if (gateways.get("litellm") or {}).get("enabled"):
        litellm_port = ports.get("litellm") or DEFAULT_PORTS.get("litellm", 14042)
        env_values["INFER_STACK_LITELLM_PORT"] = existing.get("INFER_STACK_LITELLM_PORT", str(litellm_port))

    # Open WebUI port
    if (frontends.get("open_webui") or {}).get("enabled"):
        open_webui_port = ports.get("open_webui") or DEFAULT_PORTS.get("open_webui", 13000)
        env_values["INFER_STACK_OPEN_WEBUI_PORT"] = existing.get("INFER_STACK_OPEN_WEBUI_PORT", str(open_webui_port))

    # Reverse proxy ports
    reverse_proxy = frontends.get("reverse_proxy") or {}
    if reverse_proxy.get("enabled"):
        http_port = reverse_proxy.get("http_port") or ports.get("reverse_proxy_http") or DEFAULT_PORTS.get("reverse_proxy_http", 80)
        https_port = reverse_proxy.get("https_port") or ports.get("reverse_proxy_https") or DEFAULT_PORTS.get("reverse_proxy_https", 443)
        env_values["INFER_STACK_REVERSE_PROXY_HTTP_PORT"] = existing.get("INFER_STACK_REVERSE_PROXY_HTTP_PORT", str(http_port))
        env_values["INFER_STACK_REVERSE_PROXY_HTTPS_PORT"] = existing.get("INFER_STACK_REVERSE_PROXY_HTTPS_PORT", str(https_port))

    # Ollama port (if publish enabled)
    if (providers.get("ollama") or {}).get("enabled"):
        # Ollama host_port is resolved in the deployment; fall back to DEFAULT_PORTS
        ollama_port = (providers.get("ollama") or {}).get("host_port") or ports.get("ollama") or DEFAULT_PORTS.get("ollama", 11434)
        env_values["INFER_STACK_OLLAMA_PORT"] = existing.get("INFER_STACK_OLLAMA_PORT", str(ollama_port))

    # vLLM runtimes: enumerate and export per-runtime host ports (index-based)
    vllm_runtimes = (providers.get("vllm") or {}).get("runtimes", {}) or {}
    for idx, (name, svc) in enumerate(vllm_runtimes.items()):
        host_port = svc.get("host_port") or ports.get("vllm") or (18000 + idx)
        env_name = f"INFER_STACK_VLLM_{idx}_PORT"
        env_values[env_name] = existing.get(env_name, str(host_port))

    # Preserve unknown/user-supplied keys, but let managed keys above win.
    for key, value in existing.items():
        env_values.setdefault(key, value)

    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["compose_quote"] = _compose_quote
    env.filters["compose_gpus"] = _compose_gpus
    normalized_lock = dict(lock_data)
    normalized_lock["deployment"] = deployment

    reverse_proxy = ((deployment.get("frontends") or {}).get("reverse_proxy") or {})
    if reverse_proxy.get("enabled"):
        if reverse_proxy.get("config_path"):
            reverse_proxy["nginx_config_path"] = reverse_proxy["config_path"]
        else:
            reverse_proxy["nginx_config_path"] = str(runtime_dir / "nginx.conf")
        deployment["frontends"]["reverse_proxy"] = reverse_proxy

    ctx = {"lock": normalized_lock}
    compose = env.from_string(_template("docker-compose.yml.j2")).render(**ctx) + "\n"

    compose_fpath = generated / "docker-compose.yml"
    planned: dict[Path, str] = {compose_fpath: compose}

    lite_llm_config_fpath = runtime_dir / "litellm_config.yaml"
    if (gateways.get("litellm") or {}).get("enabled"):
        litellm_cfg = env.from_string(_template("litellm_config.yaml.j2")).render(**ctx) + "\n"
        planned[lite_llm_config_fpath] = litellm_cfg

    if reverse_proxy.get("enabled") and not reverse_proxy.get("config_path"):
        nginx_ctx = {"rp": reverse_proxy}
        planned[Path(reverse_proxy["nginx_config_path"])] = env.from_string(_template("nginx.conf.j2")).render(**nginx_ctx) + "\n"

    if not confirm_writes(planned, assume_yes=assume_yes, title="Pending compose render"):
        raise SystemExit("Aborted by user; no files were written.")

    write_env_file(env_path, env_values)
    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
