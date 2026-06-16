"""Low-level readiness probes for OpenAI-compatible and Ollama surfaces.

One implementation of "is this endpoint actually serving", shared by the legacy
``wait-ready`` / ``switch`` path (``cli.probes`` re-exports these) and the
leasing Compose backend. Each probe is a pure function over an injected HTTP
client (``http``, defaulting to the ``requests`` module), so it works against a
real server or a fake in tests, and stays out of the CLI layer that depends on
it (no upward imports).

Returns ``(ready, reason)`` and never raises for ordinary network errors — a
not-yet-ready endpoint is a normal, retryable outcome.
"""

from __future__ import annotations

from typing import Any

import requests


def openai_ready(
    *,
    base_url: str,
    headers: dict[str, str] | None = None,
    model: str | None = None,
    protocol: str = 'chat',
    prompt: str = 'Reply with ready.',
    max_tokens: int = 1,
    require_generation: bool = True,
    require_listed: bool = False,
    http: Any = requests,
) -> tuple[bool, str]:
    """Probe an OpenAI-compatible surface once without exiting.

    ``require_listed`` additionally insists the requested ``model`` appears in
    ``/models`` (used by the leasing front door to confirm an alias is routable).
    """
    headers = headers or {}
    try:
        models_resp = http.get(f'{base_url}/models', headers=headers, timeout=10)
    except requests.exceptions.RequestException as ex:
        return False, f'/models not reachable yet: {ex}'
    if models_resp.status_code >= 400:
        body = (models_resp.text or '').strip()
        return (
            False,
            f'/models returned HTTP {models_resp.status_code}: {body[:300]}',
        )
    try:
        models_doc = models_resp.json()
    except ValueError:
        return False, '/models returned non-JSON response'
    models = models_doc.get('data') or []
    listed = {m.get('id') for m in models}
    model_name = model or (models[0].get('id') if models else None)
    if not model_name:
        return False, '/models is reachable but no models are advertised'
    if require_listed and model is not None and listed and model not in listed:
        return False, f'{model} is not advertised by the gateway yet'
    if not require_generation:
        return True, f'/models is ready; selected model {model_name}'
    if protocol == 'completions':
        payload: dict[str, Any] = {
            'model': model_name,
            'prompt': prompt,
            'max_tokens': max_tokens,
        }
        endpoint = f'{base_url}/completions'
    else:
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
        }
        endpoint = f'{base_url}/chat/completions'
    try:
        resp = http.post(endpoint, headers=headers, json=payload, timeout=45)
    except requests.exceptions.RequestException as ex:
        return False, f'{endpoint} not serving yet: {ex}'
    if resp.status_code >= 400:
        body = (resp.text or '').strip()
        return (
            False,
            f'{endpoint} returned HTTP {resp.status_code}: {body[:300]}',
        )
    return True, f'{model_name} served a {protocol} probe'


def ollama_ready(
    *,
    base_url: str,
    model: str | None = None,
    prompt: str = 'Reply with ready.',
    max_tokens: int = 1,
    require_generation: bool = True,
    http: Any = requests,
) -> tuple[bool, str]:
    """Probe an Ollama-native surface once without exiting."""
    try:
        tags_resp = http.get(f'{base_url}/api/tags', timeout=10)
    except requests.exceptions.RequestException as ex:
        return False, f'/api/tags not reachable yet: {ex}'
    if tags_resp.status_code >= 400:
        body = (tags_resp.text or '').strip()
        return (
            False,
            f'/api/tags returned HTTP {tags_resp.status_code}: {body[:300]}',
        )
    try:
        tags_doc = tags_resp.json()
    except ValueError:
        return False, '/api/tags returned non-JSON response'
    models = tags_doc.get('models') or []
    model_name = model or (models[0].get('name') if models else None)
    if not model_name:
        if require_generation:
            return (
                False,
                'Ollama is reachable but no model is installed; run `infer-stack ollama-pull <tag>`',
            )
        return True, 'Ollama API is reachable'
    if not require_generation:
        return True, f'Ollama API is reachable; selected model {model_name}'
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'num_predict': max_tokens},
    }
    try:
        resp = http.post(f'{base_url}/api/chat', json=payload, timeout=45)
    except requests.exceptions.RequestException as ex:
        return False, f'/api/chat not serving yet: {ex}'
    if resp.status_code >= 400:
        body = (resp.text or '').strip()
        return (
            False,
            f'/api/chat returned HTTP {resp.status_code}: {body[:300]}',
        )
    return True, f'{model_name} served an Ollama chat probe'
