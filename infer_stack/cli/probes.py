from __future__ import annotations

from typing import Any

# The HTTP probes now live in the shared, layer-neutral ``infer_stack.probe`` so
# the leasing backend can reuse them too. Re-exported here under their original
# names for the legacy ``wait-ready`` / ``switch`` callers.
from ..probe import ollama_ready as _ready_ollama_probe  # noqa: F401
from ..probe import openai_ready as _ready_openai_probe  # noqa: F401


def _default_model_for_deployment(
    deployment: dict[str, Any], explicit: str | None = None
) -> str | None:
    """Pick a reasonable model name for readiness/smoke probes."""
    if explicit:
        return str(explicit)
    litellm_routes = (
        (deployment.get('gateways', {}) or {}).get('litellm', {}) or {}
    ).get('routes', {}) or {}
    if litellm_routes:
        return str(next(iter(litellm_routes)))
    vllm_runtimes = (
        (deployment.get('providers', {}) or {}).get('vllm', {}) or {}
    ).get('runtimes', {}) or {}
    if vllm_runtimes:
        first = next(iter(vllm_runtimes.values()))
        return (
            str(
                first.get('served_model_name')
                or first.get('logical_model_name')
                or first.get('runtime_name')
                or ''
            )
            or None
        )
    ollama_routes = (
        (deployment.get('providers', {}) or {}).get('ollama', {}) or {}
    ).get('routes', {}) or {}
    if ollama_routes:
        first = next(iter(ollama_routes.values()))
        return (
            str(first.get('upstream_model') or first.get('model_ref') or '')
            or None
        )
    return None


def _resolve_smoke_protocol_from_deployment(
    deployment: dict[str, Any], model_name: str | None
) -> str:
    """Resolve chat vs completions from schema-v5 routes/runtimes."""
    if model_name:
        routes = (
            (deployment.get('gateways', {}) or {}).get('litellm', {}) or {}
        ).get('routes', {}) or {}
        route = routes.get(model_name)
        if route:
            return str(route.get('protocol_mode') or 'chat')
        vllm_runtimes = (
            (deployment.get('providers', {}) or {}).get('vllm', {}) or {}
        ).get('runtimes', {}) or {}
        for rt in vllm_runtimes.values():
            aliases = set(rt.get('served_aliases') or [])
            aliases.add(str(rt.get('served_model_name') or ''))
            if model_name in aliases:
                return str(rt.get('protocol_mode') or 'chat')
    return 'chat'


