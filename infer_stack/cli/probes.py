from __future__ import annotations

from typing import Any
import requests

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
