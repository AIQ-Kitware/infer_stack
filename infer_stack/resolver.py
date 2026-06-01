from __future__ import annotations

from copy import deepcopy
from typing import Any

from .catalog import canonical_profile_name, normalize_ollama_models, normalize_stack_profiles, normalize_vllm_models, sanitize_name
from .config import (
    DEFAULT_PORTS,
    PINNED_IMAGES,
    deep_merge,
    load_kubeai_resource_profiles,
    merged_catalogs,
    normalized_cluster,
    normalized_output,
    normalized_state,
    resource_profiles_to_kubeai_values,
)
from .hardware import detect_inventory


def _available_gpu_indices(inventory: dict[str, Any], reserve_display_gpu: str | bool | None) -> list[int]:
    gpus = deepcopy(inventory.get("gpus", []))
    if reserve_display_gpu == "auto":
        return [g["index"] for g in gpus if not g.get("display_active")]
    if reserve_display_gpu is True:
        return [g["index"] for g in gpus if not g.get("display_active")]
    return [g["index"] for g in gpus]


def _first_fit(available: list[int], count: int) -> tuple[list[int], str | None]:
    if len(available) < count:
        return available[:], f"need {count} GPUs but only {len(available)} available"
    return available[:count], None


def _runtime_value(runtime: dict[str, Any], model: dict[str, Any], key: str, default: Any) -> Any:
    runtime_cfg = runtime.get("runtime", {}) or {}
    if key in runtime_cfg:
        return runtime_cfg[key]
    if key in runtime:
        return runtime[key]
    return model.get("defaults", {}).get(key, default)


def _enabled_value(value: Any, default: bool = False) -> bool:
    if value == "auto":
        return default
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_string_map(value: Any) -> dict[str, str]:
    """Normalize user-supplied environment/label maps without dropping values.

    Compose service override hooks are intentionally permissive.  Keeping the
    values as strings lets users pass literal values, numeric-looking settings,
    booleans, and ``${ENV_VAR}`` substitutions through to Docker Compose.
    """

    if not value:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping, got {type(value).__name__}: {value!r}")
    return {str(k): "" if v is None else str(v) for k, v in value.items()}


def _as_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _bool_string(value: Any, default: bool = False) -> str:
    return "true" if _enabled_value(value, default=default) else "false"


def _service_override_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Common opt-in escape hatches for rendered Compose services."""

    return {
        "extra_env": _as_string_map(raw.get("extra_env", raw.get("environment", {})) or {}),
        "env_file": _as_string_list(raw.get("env_file", [])),
        "extra_volumes": _as_string_list(raw.get("extra_volumes", raw.get("volumes", []))),
        "extra_hosts": _as_string_list(raw.get("extra_hosts", [])),
        "labels": _as_string_map(raw.get("labels", {})),
        "additional_ports": _as_string_list(raw.get("additional_ports", [])),
        "gpus": raw.get("gpus"),
    }


def _resolve_gpu_indices(
    *,
    name: str,
    placement: dict[str, Any],
    topology: dict[str, Any],
    preferred_gpu_count: int,
    available: list[int],
) -> tuple[list[int], str | None]:
    strategy = placement.get("strategy", "first_fit")
    if strategy in {"exact", "multi_gpu", "single_gpu"}:
        gpu_indices = list(placement.get("gpu_indices", []))
        if not gpu_indices:
            return [], f"{name} uses {strategy} placement but no gpu_indices were provided"
        return gpu_indices, None
    gpu_count = int(placement.get("gpu_count", topology.get("tensor_parallel_size", preferred_gpu_count) or 1))
    return _first_fit(available, gpu_count)


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_vllm_runtime(
    *,
    profile: dict[str, Any],
    runtime_name: str,
    runtime: dict[str, Any],
    models: dict[str, Any],
    inventory: dict[str, Any],
    policy: dict[str, Any],
    used: set[int],
    backend: str,
) -> dict[str, Any]:
    model_key = runtime.get("model") or runtime.get("base_model")
    if not model_key and runtime.get("hf_model_id"):
        model_key = sanitize_name(runtime["hf_model_id"])
        models = dict(models)
        models[model_key] = {
            "key": model_key,
            "hf_model_id": runtime["hf_model_id"],
            "url": f"hf://{runtime['hf_model_id']}",
            "served_model_name": runtime.get("served_model_name") or model_key,
            "logical_model_name": runtime.get("logical_model_name") or model_key,
            "tokenizer_name": runtime.get("tokenizer_name") or model_key,
            "supported_protocols": ["chat", "completions"],
            "modalities": ["text"],
            "features": ["TextGeneration"],
            "defaults": {},
            "preferred_gpu_count": 1,
            "min_vram_gib_per_replica": 0,
            "resource_profile": "",
            "notes": [],
            "caveats": [],
        }
    if model_key not in models:
        raise KeyError(f"Unknown vLLM model: {model_key}")
    model = deepcopy(models[model_key])
    placement = deepcopy(runtime.get("placement", {}))
    topology = deepcopy(runtime.get("topology", {}))
    available = [i for i in _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto")) if i not in used]
    if "tp" in topology and "tensor_parallel_size" not in topology:
        topology["tensor_parallel_size"] = topology["tp"]
    if "dp" in topology and "data_parallel_size" not in topology:
        topology["data_parallel_size"] = topology["dp"]
    gpu_indices, placement_error = _resolve_gpu_indices(
        name=f"vLLM runtime {runtime_name}",
        placement=placement,
        topology=topology,
        preferred_gpu_count=int(model.get("preferred_gpu_count", 1) or 1),
        available=available,
    )
    if backend == "compose":
        used.update(gpu_indices)
    tp = int(topology.get("tensor_parallel_size", max(1, len(gpu_indices) or placement.get("gpu_count", 1))))
    dp = int(topology.get("data_parallel_size", 1))

    tool_calling = _merge(model.get("tool_calling", {}), runtime.get("tool_calling", {}))
    tool_call_parser = tool_calling.get("parser")
    tool_calling_on = bool(tool_calling.get("enabled", tool_calling.get("auto", False)))
    enable_auto_tool_choice = bool(tool_calling_on and tool_call_parser)

    reasoning = _merge(model.get("reasoning", {}), runtime.get("reasoning", {}))
    reasoning_enabled = bool(reasoning.get("enabled", False))
    reasoning_parser = reasoning.get("parser") if reasoning_enabled else None
    reasoning_expose_to_openwebui = bool(reasoning.get("expose_to_openwebui", reasoning_enabled))

    chat_compat = _merge(model.get("chat_compat", {}), runtime.get("chat_compat", {}))
    chat_compat_enabled = bool(chat_compat.get("enabled", False))
    chat_compat_strategy = str(chat_compat.get("strategy", "flat_messages")) if chat_compat_enabled else None

    hf_model_id = runtime.get("hf_model_id", model.get("hf_model_id", ""))
    served_model_name = runtime.get("served_model_name") or model.get("served_model_name") or runtime_name
    logical_model_name = runtime.get("logical_model_name") or model.get("logical_model_name") or served_model_name
    public_name = runtime.get("public_name") or runtime_name
    protocol_mode = runtime.get("protocol_mode") or runtime.get("protocol") or "chat"
    model_url = runtime.get("url") or model.get("url") or (f"hf://{hf_model_id}" if hf_model_id else "")

    return {
        "provider": "vllm",
        "runtime_name": runtime_name,
        "service_name": sanitize_name(runtime_name),
        "compose_service_name": f"vllm-{sanitize_name(runtime_name)}",
        "container_name": f"vllm-{sanitize_name(runtime_name)}",
        "profile_name": profile["name"],
        "profile_public_name": public_name,
        "kubernetes_name": sanitize_name(public_name),
        "model_ref": model_key,
        "hf_model_id": hf_model_id,
        "model_url": model_url,
        "logical_model_name": logical_model_name,
        "served_model_name": served_model_name,
        "served_aliases": [],
        "protocol_mode": protocol_mode,
        "supported_protocols": list(model.get("supported_protocols", ["chat", "completions"])),
        "modalities": model.get("modalities", ["text"]),
        "features": deepcopy(model.get("features", ["TextGeneration"])),
        "engine": "VLLM",
        "memory_class_gib": model.get("memory_class_gib"),
        "min_vram_gib_per_replica": model.get("min_vram_gib_per_replica", 0),
        "context_window": model.get("context_window"),
        "tokenizer_name": runtime.get("tokenizer_name", model.get("tokenizer_name", logical_model_name)),
        "notes": deepcopy(model.get("notes", [])) + deepcopy(runtime.get("notes", [])),
        "audit_notes": deepcopy(runtime.get("audit_notes", [])) + deepcopy(model.get("caveats", [])),
        "tags": deepcopy(runtime.get("tags", [])),
        "gpu_indices": gpu_indices,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "resource_profile": runtime.get("resource_profile", model.get("resource_profile", "")),
        "min_replicas": int(runtime.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
        "max_replicas": int(runtime.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
        "priority_class_name": runtime.get("priority_class_name", model.get("priority_class_name")),
        "max_model_len": int(_runtime_value(runtime, model, "max_model_len", 32768)),
        "gpu_memory_utilization": float(_runtime_value(runtime, model, "gpu_memory_utilization", 0.9)),
        "enable_prefix_caching": bool(_runtime_value(runtime, model, "enable_prefix_caching", True)),
        "max_num_batched_tokens": int(_runtime_value(runtime, model, "max_num_batched_tokens", 8192)),
        "max_num_seqs": int(_runtime_value(runtime, model, "max_num_seqs", 16)),
        "thinking_history_policy": model.get("thinking_history_policy", "keep_final_only"),
        "placement": placement,
        "topology": topology,
        "placement_error": placement_error,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "tool_call_parser": tool_call_parser,
        "extra_args": deepcopy(runtime.get("extra_args", model.get("defaults", {}).get("extra_args", []))),
        "reasoning_enabled": reasoning_enabled,
        "reasoning_parser": reasoning_parser,
        "reasoning_expose_to_openwebui": reasoning_expose_to_openwebui,
        "chat_compat_enabled": chat_compat_enabled,
        "chat_compat_strategy": chat_compat_strategy,
        "benchmark_transport": deepcopy(runtime.get("benchmark_transport", runtime.get("transport", {}))),
        "publish_port": bool(runtime.get("publish_port", False)),
        "host_port": runtime.get("host_port"),
        **_service_override_fields(runtime),
    }


def _resolve_ollama_model_tag(model_ref: str, ollama_models: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if model_ref in ollama_models:
        model = deepcopy(ollama_models[model_ref])
        return model["tag"], model
    return str(model_ref), {"key": model_ref, "tag": str(model_ref), "served_model_name": sanitize_name(str(model_ref)), "defaults": {}}


def _collect_ollama_needed(profile: dict[str, Any]) -> bool:
    p = profile.get("providers", {}).get("ollama", {}) or {}
    if _enabled_value(p.get("enabled"), default=False):
        return True
    for route in (profile.get("routes", {}) or {}).values():
        if str(route.get("provider", "")).lower() == "ollama":
            return True
    return False


def _resolve_ollama_provider(profile: dict[str, Any], config: dict[str, Any], inventory: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    profile_ollama = deepcopy(profile.get("providers", {}).get("ollama", {}) or {})
    # Historical configs used the top-level ``ollama`` section for provider
    # details.  Newer configs may keep those details under
    # ``providers.ollama``.  Support both so users can opt into advanced
    # settings without having to rewrite old config files.
    config_ollama = deepcopy(config.get("ollama", {}) or {})
    config_provider_ollama = deepcopy((config.get("providers", {}) or {}).get("ollama", {}) or {})
    merged = _merge(_merge(config_ollama, config_provider_ollama), profile_ollama)
    enabled = _enabled_value(merged.get("enabled"), default=_collect_ollama_needed(profile))
    if not enabled:
        return {"enabled": False, "routes": {}}

    gpu_indices = merged.get("gpu_indices", "auto")
    placement_error = None
    if gpu_indices == "auto" or gpu_indices is None:
        available = _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto"))
        gpu_indices = available
    else:
        gpu_indices = list(gpu_indices)
    return {
        "enabled": True,
        "service_name": "ollama",
        "base_url": "http://ollama:11434",
        "host_port": int(config.get("ports", {}).get("ollama", 11434)),
        "gpu_indices": gpu_indices,
        "publish_port": bool(merged.get("publish_port", False)),
        "host": str(merged.get("host", "0.0.0.0:11434")),
        "keep_alive": str(merged.get("keep_alive", "2m")),
        "context_length": int(merged.get("context_length", 4096)),
        "num_parallel": int(merged.get("num_parallel", 1)),
        "max_loaded_models": int(merged.get("max_loaded_models", 1)),
        "max_queue": int(merged.get("max_queue", 8)),
        "placement_error": placement_error,
        "routes": {},
        **_service_override_fields(merged),
    }


def _resolve_routes(profile: dict[str, Any], vllm_runtimes: dict[str, Any], ollama_models: dict[str, Any]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for alias, raw in (profile.get("routes", {}) or {}).items():
        route = deepcopy(raw) or {}
        provider = str(route.get("provider", "vllm")).lower()
        aliases = route.get("aliases") or [alias]
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a) for a in aliases]
        if alias not in aliases:
            aliases.insert(0, alias)
        for public_alias in aliases:
            if provider == "vllm":
                runtime_name = route.get("runtime") or route.get("service") or route.get("target")
                if runtime_name is None and len(vllm_runtimes) == 1:
                    runtime_name = next(iter(vllm_runtimes))
                if runtime_name not in vllm_runtimes:
                    raise KeyError(f"route {alias!r} references unknown vLLM runtime {runtime_name!r}")
                rt = vllm_runtimes[runtime_name]
                routes[public_alias] = {
                    "alias": public_alias,
                    "provider": "vllm",
                    "runtime": runtime_name,
                    "service_name": rt["service_name"],
                    "upstream_service_name": rt.get("compose_service_name", rt["service_name"]),
                    "served_model_name": rt["served_model_name"],
                    "protocol_mode": rt["protocol_mode"],
                    "max_model_len": rt["max_model_len"],
                    "chat_compat_enabled": rt.get("chat_compat_enabled", False),
                    "chat_compat_strategy": rt.get("chat_compat_strategy"),
                }
            elif provider == "ollama":
                model_ref = route.get("model") or route.get("tag") or public_alias
                tag, model = _resolve_ollama_model_tag(str(model_ref), ollama_models)
                defaults = deepcopy(model.get("defaults", {}))
                defaults.update(deepcopy(route.get("options", {})))
                max_model_len = int(route.get("max_model_len") or defaults.get("num_ctx") or model.get("context_window") or 4096)
                routes[public_alias] = {
                    "alias": public_alias,
                    "provider": "ollama",
                    "upstream_model": tag,
                    "model_ref": str(model_ref),
                    "protocol_mode": "chat",
                    "max_model_len": max_model_len,
                    "options": defaults,
                }
            else:
                raise KeyError(f"route {alias!r} uses unknown provider {provider!r}")
    return routes


def _resolve_litellm(profile: dict[str, Any], routes: dict[str, Any], providers: dict[str, Any], backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if backend == "kubeai":
        return {"enabled": False, "base_url": "", "routes": {}}
    cfg = deepcopy(config.get("litellm", {}) or {})
    cfg = _merge(cfg, (config.get("gateways", {}) or {}).get("litellm", {}) or {})
    cfg = _merge(cfg, profile.get("gateways", {}).get("litellm", {}) or {})
    route_providers = {route.get("provider") for route in routes.values()}
    needs_gateway = bool(routes) and (backend == "compose")
    if len(route_providers) > 1:
        needs_gateway = True
    enabled = _enabled_value(cfg.get("enabled"), default=needs_gateway)
    return {
        "enabled": bool(enabled),
        "base_url": "http://litellm:4000/v1" if enabled else "",
        "routes": routes if enabled else {},
        **_service_override_fields(cfg),
    }


def _resolve_open_webui_ldap(raw: dict[str, Any]) -> dict[str, Any]:
    ldap = deepcopy(raw.get("ldap", {}) or {})
    enabled = _enabled_value(ldap.get("enabled"), default=False)
    if not enabled:
        return {"enabled": False, "env": {}, "env_defaults": {}}

    # Defaults are intentionally expressed through .env placeholders so LDAP
    # credentials and site-local directory settings can be edited after render
    # without changing generated Compose YAML.
    env = {
        "ENABLE_LDAP": _bool_string(True),
        "LDAP_SERVER_LABEL": str(ldap.get("server_label", "OpenLDAP")),
        "LDAP_SERVER_HOST": str(ldap.get("server_host", "${LDAP_HOST}")),
        "LDAP_SERVER_PORT": str(ldap.get("server_port", "${LDAP_PORT}")),
        "LDAP_USE_TLS": str(ldap.get("use_tls", "${LDAP_USE_TLS}")),
        "LDAP_VALIDATE_CERT": str(ldap.get("validate_cert", ldap.get("validate_certs", "${LDAP_VALIDATE_CERTS}"))),
        "LDAP_APP_DN": str(ldap.get("app_dn", "${LDAP_BASEDN}")),
        "LDAP_APP_PASSWORD": str(ldap.get("app_password", "${LDAP_PASSWD}")),
        "LDAP_SEARCH_BASE": str(ldap.get("search_base", "${LDAP_SEARCH_BASE}")),
        "LDAP_ATTRIBUTE_FOR_USERNAME": str(ldap.get("attribute_for_username", "${LDAP_ATTRIBUTE_FOR_USERNAME}")),
        "LDAP_ATTRIBUTE_FOR_MAIL": str(ldap.get("attribute_for_mail", "mail")),
        "LDAP_SEARCH_FILTER": str(ldap.get("search_filter", "${LDAP_SEARCH_FILTER}")),
    }
    env.update(_as_string_map(ldap.get("extra_env", {})))
    env_defaults = {
        "LDAP_HOST": str(ldap.get("env_defaults", {}).get("LDAP_HOST", "")),
        "LDAP_PORT": str(ldap.get("env_defaults", {}).get("LDAP_PORT", "636")),
        "LDAP_USE_TLS": str(ldap.get("env_defaults", {}).get("LDAP_USE_TLS", "true")),
        "LDAP_VALIDATE_CERTS": str(ldap.get("env_defaults", {}).get("LDAP_VALIDATE_CERTS", "true")),
        "LDAP_BASEDN": str(ldap.get("env_defaults", {}).get("LDAP_BASEDN", "")),
        "LDAP_PASSWD": str(ldap.get("env_defaults", {}).get("LDAP_PASSWD", "")),
        "LDAP_SEARCH_BASE": str(ldap.get("env_defaults", {}).get("LDAP_SEARCH_BASE", "")),
        "LDAP_ATTRIBUTE_FOR_USERNAME": str(ldap.get("env_defaults", {}).get("LDAP_ATTRIBUTE_FOR_USERNAME", "uid")),
        "LDAP_SEARCH_FILTER": str(ldap.get("env_defaults", {}).get("LDAP_SEARCH_FILTER", "")),
    }
    return {"enabled": True, "env": env, "env_defaults": env_defaults}


def _resolve_open_webui(profile: dict[str, Any], gateways: dict[str, Any], providers: dict[str, Any], backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if backend == "kubeai":
        return {"enabled": False, "auth": False, "provider": "none", "publish_port": False}
    raw = deepcopy(config.get("open_webui", {}) or {})
    raw = _merge(raw, (config.get("frontends", {}) or {}).get("open_webui", {}) or {})
    raw = _merge(raw, profile.get("frontends", {}).get("open_webui", {}) or {})
    default_enabled = backend == "compose" and (gateways.get("litellm", {}).get("enabled") or providers.get("ollama", {}).get("enabled"))
    enabled = _enabled_value(raw.get("enabled"), default=bool(default_enabled))
    provider = str(raw.get("provider", "auto"))
    if provider == "auto":
        if gateways.get("litellm", {}).get("enabled"):
            provider = "litellm"
        elif providers.get("ollama", {}).get("enabled"):
            provider = "ollama"
        else:
            provider = "none"
    ldap = _resolve_open_webui_ldap(raw)
    return {
        "enabled": enabled,
        "auth": bool(raw.get("auth", False)),
        "provider": provider,
        "publish_port": _enabled_value(raw.get("publish_port"), default=True),
        "webui_url": str(raw.get("webui_url", raw.get("WEBUI_URL", ""))),
        "cors_allow_origin": str(raw.get("cors_allow_origin", raw.get("CORS_ALLOW_ORIGIN", ""))),
        "ldap": ldap,
        **_service_override_fields(raw),
    }


def _resolve_reverse_proxy(profile: dict[str, Any], backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if backend == "kubeai":
        return {"enabled": False}

    raw = deepcopy(config.get("reverse_proxy", {}) or {})
    raw = _merge(raw, (config.get("frontends", {}) or {}).get("reverse_proxy", {}) or {})
    raw = _merge(raw, profile.get("frontends", {}).get("reverse_proxy", {}) or {})
    enabled = _enabled_value(raw.get("enabled"), default=False)
    if not enabled:
        return {"enabled": False}

    ssl = deepcopy(raw.get("ssl", {}) or {})
    hsts = deepcopy(raw.get("hsts", {}) or {})
    ssl_enabled = _enabled_value(ssl.get("enabled"), default=True)
    target = str(raw.get("target", "open_webui"))
    if target in {"open_webui", "open-webui"}:
        target_service = "open-webui"
        target_port = 8080
        target_scheme = "http"
        depends_on = ["open-webui"]
    elif target == "litellm":
        target_service = "litellm"
        target_port = 4000
        target_scheme = "http"
        depends_on = ["litellm"]
    elif target == "ollama":
        target_service = "ollama"
        target_port = 11434
        target_scheme = "http"
        depends_on = ["ollama"]
    else:
        target_service = str(raw.get("target_service", target))
        target_port = int(raw.get("target_port", 8080))
        target_scheme = str(raw.get("target_scheme", "http"))
        depends_on = _as_string_list(raw.get("depends_on", []))

    service_overrides = _service_override_fields(raw)
    ports = config.get("ports", {}) or {}
    return {
        "enabled": True,
        "image": str(raw.get("image") or config.get("images", {}).get("nginx") or PINNED_IMAGES.get("nginx", "nginx:alpine")),
        "service_name": str(raw.get("service_name", "reverse-proxy")),
        "container_name": str(raw.get("container_name", "reverse-proxy")),
        "target": target,
        "target_service": target_service,
        "target_port": target_port,
        "target_scheme": target_scheme,
        "depends_on": depends_on,
        "server_name": str(raw.get("server_name", "localhost")),
        "publish_http": _enabled_value(raw.get("publish_http"), default=True),
        # Publishing :443 is only meaningful when there is a TLS server block to
        # listen on it, so gate it on ssl regardless of how publish_https was
        # defaulted (the top-level config template ships it as true).
        "publish_https": _enabled_value(raw.get("publish_https"), default=True) and ssl_enabled,
        "http_port": int(raw.get("http_port") or ports.get("reverse_proxy_http") or 80),
        "https_port": int(raw.get("https_port") or ports.get("reverse_proxy_https") or 443),
        "http_bind_host": str(raw.get("http_bind_host", "")),
        "https_bind_host": str(raw.get("https_bind_host", "")),
        "force_https": _enabled_value(raw.get("force_https"), default=ssl_enabled),
        "client_max_body_size": str(raw.get("client_max_body_size", "1G")),
        "proxy_connect_timeout": str(raw.get("proxy_connect_timeout", "60s")),
        "proxy_read_timeout": str(raw.get("proxy_read_timeout", "600s")),
        "proxy_send_timeout": str(raw.get("proxy_send_timeout", "600s")),
        "proxy_buffer_size": str(raw.get("proxy_buffer_size", "128k")),
        "proxy_buffers": str(raw.get("proxy_buffers", "4 256k")),
        "proxy_busy_buffers_size": str(raw.get("proxy_busy_buffers_size", "256k")),
        "proxy_buffering": raw.get("proxy_buffering"),
        "proxy_cache": raw.get("proxy_cache"),
        "resolver": _as_string_list(raw.get("resolver", [])),
        "resolver_timeout": str(raw.get("resolver_timeout", "5s")),
        "hsts": {
            "enabled": _enabled_value(hsts.get("enabled"), default=True),
            "max_age": int(hsts.get("max_age", 63072000)),
            "include_subdomains": _enabled_value(hsts.get("include_subdomains"), default=True),
            "preload": _enabled_value(hsts.get("preload"), default=False),
        },
        "ssl": {
            "enabled": ssl_enabled,
            "certificate": str(ssl.get("certificate", "")),
            "certificate_key": str(ssl.get("certificate_key", "")),
            "dhparam": str(ssl.get("dhparam", "")),
            "certificate_container_path": str(ssl.get("certificate_container_path", "/etc/ssl/certs/infer-stack-site.crt")),
            "certificate_key_container_path": str(ssl.get("certificate_key_container_path", "/etc/ssl/private/infer-stack-site.key")),
            "dhparam_container_path": str(ssl.get("dhparam_container_path", "/etc/ssl/certs/dhparam.pem")),
            "protocols": str(ssl.get("protocols", "TLSv1.2 TLSv1.3")),
            "ciphers": str(ssl.get("ciphers", "ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384")),
            "prefer_server_ciphers": _enabled_value(ssl.get("prefer_server_ciphers"), default=True),
            "session_cache": str(ssl.get("session_cache", "shared:SSL:10m")),
            "ecdh_curve": str(ssl.get("ecdh_curve", "secp384r1")),
            "session_tickets": _enabled_value(ssl.get("session_tickets"), default=False),
            "stapling": _enabled_value(ssl.get("stapling"), default=True),
            "stapling_verify": _enabled_value(ssl.get("stapling_verify"), default=True),
        },
        "config_path": str(raw.get("config_path", "")),
        "extra_config": str(raw.get("extra_config", "")),
        **service_overrides,
    }


def _serving_profile(profile: dict[str, Any], vllm_runtimes: dict[str, Any], routes: dict[str, Any]) -> dict[str, Any]:
    first_rt = next(iter(vllm_runtimes.values()), {}) if vllm_runtimes else {}
    first_route = next(iter(routes.values()), {}) if routes else {}
    public = first_route.get("alias") or first_rt.get("profile_public_name") or profile["name"]
    return {
        "name": profile["name"],
        "public_name": public,
        "kind": profile.get("kind", "stack"),
        "description": profile.get("description", ""),
        "base_model": first_rt.get("model_ref", first_route.get("model_ref", "")),
        "logical_model_name": first_rt.get("logical_model_name", first_route.get("alias", "")),
        "served_model_name": first_rt.get("served_model_name", first_route.get("upstream_model", "")),
        "served_aliases": list(routes.keys()),
        "protocol_mode": first_rt.get("protocol_mode", first_route.get("protocol_mode", "chat")),
        "engine": "mixed" if vllm_runtimes and any(r.get("provider") == "ollama" for r in routes.values()) else ("VLLM" if vllm_runtimes else "OLLAMA"),
        "resource_profile": first_rt.get("resource_profile", ""),
        "service_name": first_rt.get("service_name", ""),
        "kubernetes_name": first_rt.get("kubernetes_name", sanitize_name(profile["name"])),
        "tags": deepcopy(profile.get("tags", [])),
        "audit_notes": deepcopy(profile.get("audit_notes", [])),
        "notes": deepcopy(profile.get("notes", [])),
        "benchmark_transport": {},
    }


def _resolve_access(deployment: dict[str, Any]) -> dict[str, Any]:
    ports = deployment.get("ports", {})
    access: dict[str, Any] = {}
    litellm = deployment["gateways"]["litellm"]
    ollama = deployment["providers"]["ollama"]
    vllm_runtimes = deployment["providers"]["vllm"].get("runtimes", {})
    frontends = deployment.get("frontends", {}) or {}
    open_webui = frontends.get("open_webui", {}) or {}
    reverse_proxy = frontends.get("reverse_proxy", {}) or {}
    if deployment.get("backend") == "kubeai":
        ingress = deployment.get("cluster", {}).get("ingress", {}) or {}
        base = f"http://{ingress['host']}/openai/v1" if ingress.get("enabled") and ingress.get("host") else "http://127.0.0.1:8000/openai/v1"
        access["default"] = {"kind": "openai-compatible", "base_url": base, "auth_env_name": "KUBEAI_OPENAI_API_KEY"}
    elif litellm.get("enabled"):
        access["default"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{ports.get('litellm', 14042)}/v1", "auth_env_name": "LITELLM_MASTER_KEY"}
    elif ollama.get("enabled"):
        access["default"] = {"kind": "ollama-native", "base_url": f"http://127.0.0.1:{ports.get('ollama', 11434)}", "auth_required": False}
        access["ollama_openai"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{ports.get('ollama', 11434)}/v1", "auth_required": False}
    elif vllm_runtimes:
        name, rt = next(iter(vllm_runtimes.items()))
        port = rt.get("host_port") or 18000
        access["default"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{port}/v1", "auth_env_name": "VLLM_BACKEND_API_KEY"}
    else:
        access["default"] = {"kind": "none", "base_url": ""}
    if open_webui.get("enabled") and open_webui.get("publish_port"):
        access["open_webui"] = {"kind": "web-ui", "base_url": f"http://127.0.0.1:{ports.get('open_webui', 13000)}", "auth_required": bool(open_webui.get("auth"))}
    if reverse_proxy.get("enabled"):
        scheme = "https" if reverse_proxy.get("ssl", {}).get("enabled", True) and reverse_proxy.get("publish_https", True) else "http"
        host = reverse_proxy.get("server_name") or "127.0.0.1"
        port = reverse_proxy.get("https_port") if scheme == "https" else reverse_proxy.get("http_port")
        default_port = 443 if scheme == "https" else 80
        port_suffix = "" if int(port or default_port) == default_port else f":{port}"
        access["reverse_proxy"] = {"kind": "web-ui", "base_url": f"{scheme}://{host}{port_suffix}", "auth_required": bool(open_webui.get("auth"))}
    for idx, (name, rt) in enumerate(vllm_runtimes.items()):
        port = rt.get("host_port") or (18000 + idx)
        access[f"vllm_{name}"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{port}/v1", "auth_env_name": "VLLM_BACKEND_API_KEY"}
    return access


def resolve(config: dict[str, Any], inventory: dict[str, Any] | None = None, profile_name: str | None = None) -> dict[str, Any]:
    raw_catalogs = merged_catalogs(config)
    vllm_models = normalize_vllm_models(raw_catalogs.get("vllm_models", raw_catalogs.get("models", {})))
    ollama_models = normalize_ollama_models(raw_catalogs.get("ollama_models", {}))
    profiles = normalize_stack_profiles(raw_catalogs.get("profiles", {}), vllm_models, ollama_models)
    inventory = deepcopy(inventory) if inventory is not None else detect_inventory()
    effective_profile_name = canonical_profile_name(profile_name or config.get("active_profile"))
    if effective_profile_name not in profiles:
        raise KeyError(f"Unknown profile: {effective_profile_name}")
    profile = deepcopy(profiles[effective_profile_name])
    if profile.get("kind") == "invalid-profile":
        raise KeyError(f"Profile {effective_profile_name!r} is invalid: {profile.get('catalog_error', 'unknown error')}")

    backend = str(config.get("backend", "compose")).lower()
    merged_policy = _merge(config.get("policy", {}), profile.get("policy", {}))
    images = deep_merge(PINNED_IMAGES, config.get("images", {}) or {})
    ports = deep_merge(DEFAULT_PORTS, config.get("ports", {}) or {})
    state = normalized_state(config.get("state", {}))
    output = normalized_output(config.get("output", {}))

    used: set[int] = set()
    vllm_cfg = deepcopy(profile.get("providers", {}).get("vllm", {}) or {})
    runtime_cfgs = deepcopy(vllm_cfg.get("runtimes", {}) or {})
    vllm_runtimes: dict[str, Any] = {}
    for runtime_name, runtime in runtime_cfgs.items():
        resolved_rt = _resolve_vllm_runtime(
            profile=profile,
            runtime_name=str(runtime_name),
            runtime=runtime,
            models=vllm_models,
            inventory=inventory,
            policy=merged_policy,
            used=used,
            backend=backend,
        )
        vllm_runtimes[str(runtime_name)] = resolved_rt

    ollama_provider = _resolve_ollama_provider(profile, config, inventory, merged_policy)
    routes = _resolve_routes(profile, vllm_runtimes, ollama_models)
    for alias, route in routes.items():
        if route.get("provider") == "vllm" and route.get("runtime") in vllm_runtimes:
            aliases = vllm_runtimes[route["runtime"]].setdefault("served_aliases", [])
            if alias not in aliases:
                aliases.append(alias)
    ollama_routes = {k: v for k, v in routes.items() if v.get("provider") == "ollama"}
    if ollama_routes and not ollama_provider.get("enabled"):
        ollama_provider = _resolve_ollama_provider({**profile, "providers": {**profile.get("providers", {}), "ollama": {"enabled": True}}}, config, inventory, merged_policy)
    ollama_provider["routes"] = ollama_routes

    providers = {
        "ollama": ollama_provider,
        "vllm": {"enabled": bool(vllm_runtimes), "runtimes": vllm_runtimes},
    }
    litellm = _resolve_litellm(profile, routes, providers, backend, config)
    gateways = {"litellm": litellm}
    frontends = {"open_webui": _resolve_open_webui(profile, gateways, providers, backend, config)}
    frontends["reverse_proxy"] = _resolve_reverse_proxy(profile, backend, config)
    serving_profile = _serving_profile(profile, vllm_runtimes, routes)

    if backend == "kubeai":
        resource_profiles, resource_profiles_values, resource_profiles_path = load_kubeai_resource_profiles()
        if resource_profiles:
            resource_profile_source = str(resource_profiles_path)
        else:
            resource_profiles = deepcopy(config.get("resource_profiles", {}))
            resource_profiles_values = deepcopy({"resourceProfiles": resource_profiles_to_kubeai_values(resource_profiles)["resourceProfiles"]})
            resource_profile_source = "config.yaml.resource_profiles"
    else:
        resource_profiles = deepcopy(config.get("resource_profiles", {}))
        resource_profiles_values = deepcopy({"resourceProfiles": resource_profiles_to_kubeai_values(resource_profiles)["resourceProfiles"]})
        resource_profile_source = "config.yaml.resource_profiles"

    # Compatibility aliases. New templates should use providers/gateways/frontends.
    services = list(vllm_runtimes.values())
    router_aliases = {alias: route.get("service_name", route.get("upstream_model", "")) for alias, route in routes.items()}

    deployment = {
        "schema_version": 5,
        "source": {"config_file": "config.yaml", "active_profile": effective_profile_name},
        "backend": backend,
        "images": images,
        "ports": ports,
        "policy": merged_policy,
        "vllm": {
            "enable_responses_api_store": bool(profile.get("vllm", {}).get("enable_responses_api_store", False)),
            "logging_level": str(profile.get("vllm", {}).get("logging_level", "INFO")),
        },
        "state": state,
        "output": output,
        "cluster": normalized_cluster(config.get("cluster", {})),
        "resource_profiles": resource_profiles,
        "resource_profiles_values": resource_profiles_values,
        "resource_profiles_source": resource_profile_source,
        "inventory": inventory,
        "providers": providers,
        "gateways": gateways,
        "frontends": frontends,
        "access": {},
        "profile": serving_profile,
        "serving_profile": deepcopy(serving_profile),
        "services": services,
        "router": {"enabled": litellm.get("enabled"), "type": "litellm" if litellm.get("enabled") else "none", "aliases": router_aliases},
        "open_webui": frontends["open_webui"],
    }
    deployment["access"] = _resolve_access(deployment)
    return deployment
