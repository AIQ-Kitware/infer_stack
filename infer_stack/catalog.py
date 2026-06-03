from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILE_NAME_ALIASES = {
    'helm-qwen2-72b-instruct': 'qwen2-72b-instruct-tp2-balanced',
    'helm-qwen2.5-7b-instruct': 'qwen2-5-7b-instruct-turbo-default',
    'helm-qwen2.5-72b-instruct': 'qwen2-5-72b-instruct-tp2-balanced',
    'helm-gpt-oss-20b': 'gpt-oss-20b-completions',
    'helm-vicuna-7b-v1.3': 'vicuna-7b-v1-3-no-chat-template',
}


def sanitize_name(value: str) -> str:
    value = str(value).strip().lower()
    out: list[str] = []
    prev_dash = False
    for char in value:
        if char.isalnum():
            out.append(char)
            prev_dash = False
            continue
        if not prev_dash:
            out.append('-')
            prev_dash = True
    sanitized = ''.join(out).strip('-')
    return sanitized or 'profile'


def canonical_profile_name(name: str | None) -> str | None:
    if name is None:
        return None
    return PROFILE_NAME_ALIASES.get(name, name)


def _list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return deepcopy(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_vllm_models(catalog: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw in (catalog or {}).items():
        entry = deepcopy(raw) or {}
        hf_model_id = entry.get('hf_model_id', '')
        url = entry.get('url') or (f'hf://{hf_model_id}' if hf_model_id else '')
        defaults = deepcopy(entry.get('defaults', {}))
        supported_protocols = entry.get('supported_protocols')
        if supported_protocols is None:
            supported_protocols = ['chat', 'completions']
        normalized[key] = {
            'key': key,
            'provider': 'vllm',
            'canonical_key': sanitize_name(entry.get('canonical_key', key)),
            'hf_model_id': hf_model_id,
            'url': url,
            'family': entry.get('family', ''),
            'modalities': _list(entry.get('modalities'), ['text']),
            'supported_protocols': [str(p) for p in supported_protocols],
            'reasoning': deepcopy(entry.get('reasoning', {})),
            'tokenizer_name': entry.get('tokenizer_name')
            or entry.get('tokenizer')
            or entry.get('served_model_name')
            or key,
            'logical_model_name': entry.get('logical_model_name')
            or entry.get('served_model_name')
            or key,
            'served_model_name': entry.get('served_model_name')
            or entry.get('logical_model_name')
            or key,
            'memory_class_gib': entry.get('memory_class_gib'),
            'min_vram_gib_per_replica': entry.get(
                'min_vram_gib_per_replica', 0
            ),
            'preferred_gpu_count': entry.get('preferred_gpu_count', 1),
            'context_window': entry.get('context_window'),
            'defaults': defaults,
            'engine': 'VLLM',
            'resource_profile': entry.get('resource_profile', ''),
            'priority_class_name': entry.get('priority_class_name'),
            'tool_calling': deepcopy(entry.get('tool_calling', {})),
            'thinking_history_policy': entry.get(
                'thinking_history_policy', 'keep_final_only'
            ),
            'features': deepcopy(entry.get('features', ['TextGeneration'])),
            'safe_defaults': deepcopy(entry.get('safe_defaults', defaults)),
            'notes': _list(entry.get('notes')),
            'caveats': _list(entry.get('caveats')),
        }
    return normalized


def normalize_ollama_models(catalog: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw in (catalog or {}).items():
        entry = deepcopy(raw) or {}
        tag = (
            entry.get('tag')
            or entry.get('ollama_model')
            or entry.get('model')
            or key
        )
        normalized[key] = {
            'key': key,
            'provider': 'ollama',
            'tag': str(tag),
            'served_model_name': entry.get('served_model_name')
            or sanitize_name(str(tag)),
            'logical_model_name': entry.get('logical_model_name')
            or entry.get('served_model_name')
            or sanitize_name(str(tag)),
            'modalities': _list(entry.get('modalities'), ['text']),
            'supported_protocols': [
                str(p)
                for p in _list(entry.get('supported_protocols'), ['chat'])
            ],
            'context_window': entry.get('context_window'),
            'defaults': deepcopy(entry.get('defaults', {})),
            'notes': _list(entry.get('notes')),
            'caveats': _list(entry.get('caveats')),
        }
    return normalized


# Compatibility view used by older helper commands/tests. Prefer provider-specific
# catalogs in new code.
def normalize_model_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    return normalize_vllm_models(catalog)


def _infer_protocol_mode(
    profile_name: str, logical_model_name: str, raw: dict[str, Any]
) -> str:
    explicit = raw.get('protocol_mode') or raw.get('protocol')
    if explicit:
        return str(explicit)
    tags = {str(tag) for tag in raw.get('tags', [])}
    if 'completions' in tags:
        return 'completions'
    if 'chat' in tags:
        return 'chat'
    name = f'{profile_name} {logical_model_name}'.lower()
    if 'completion' in name:
        return 'completions'
    return 'chat'


def _normalize_bool_map(
    raw: Any, default_enabled: bool = False
) -> dict[str, Any]:
    if isinstance(raw, dict):
        return deepcopy(raw)
    if raw in (None, 'auto'):
        return {'enabled': 'auto'}
    return {'enabled': bool(raw) if raw is not None else default_enabled}


def _normalize_components(
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    providers = deepcopy(profile.get('providers', {}))
    gateways = deepcopy(profile.get('gateways', {}))
    frontends = deepcopy(profile.get('frontends', {}))

    # Convenience/older shape: components: {ollama: true, litellm: false, ...}
    components = deepcopy(profile.get('components', {}) or {})
    for name in ['ollama', 'vllm']:
        if name in components and name not in providers:
            providers[name] = _normalize_bool_map(components[name])
    if 'litellm' in components and 'litellm' not in gateways:
        gateways['litellm'] = _normalize_bool_map(components['litellm'])
    if 'open_webui' in components and 'open_webui' not in frontends:
        frontends['open_webui'] = _normalize_bool_map(components['open_webui'])

    providers.setdefault('vllm', {})
    providers.setdefault('ollama', {'enabled': 'auto'})
    gateways.setdefault('litellm', {'enabled': 'auto'})
    frontends.setdefault('open_webui', {'enabled': 'auto', 'provider': 'auto'})
    return providers, gateways, frontends


def _route_aliases(route_name: str, raw: dict[str, Any]) -> list[str]:
    aliases = raw.get('aliases')
    if aliases is None:
        aliases = raw.get('served_aliases')
    if aliases is None:
        aliases = [route_name]
    elif isinstance(aliases, str):
        aliases = [aliases]
    else:
        aliases = list(aliases)
    if route_name not in aliases:
        aliases.insert(0, route_name)
    ordered: list[str] = []
    for alias in aliases:
        if alias and alias not in ordered:
            ordered.append(str(alias))
    return ordered


def _legacy_profile_to_stack(
    name: str, raw: dict[str, Any], vllm_models: dict[str, Any]
) -> dict[str, Any]:
    services = deepcopy(raw.get('services', []))
    if not services and (raw.get('model') or raw.get('base_model')):
        services = [deepcopy(raw)]
    aliases = deepcopy(raw.get('router', {}).get('aliases', {}))
    runtimes: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    for index, service in enumerate(services):
        runtime_name = sanitize_name(
            service.get('service_name')
            or service.get('runtime')
            or service.get('name')
            or (name if len(services) == 1 else f'runtime-{index + 1}')
        )
        model_key = service.get('base_model') or service.get('model')
        if model_key not in vllm_models:
            raise KeyError(f'Unknown vLLM model: {model_key}')
        model = vllm_models[model_key]
        public_name = sanitize_name(
            service.get('public_name')
            or (name if len(services) == 1 else f'{name}-{runtime_name}')
        )
        logical_model_name = (
            service.get('logical_model_name')
            or model.get('logical_model_name')
            or model.get('served_model_name')
            or model_key
        )
        served_model_name = (
            service.get('served_model_name')
            or model.get('served_model_name')
            or logical_model_name
        )
        protocol_mode = _infer_protocol_mode(
            public_name, logical_model_name, service
        )
        runtimes[runtime_name] = {
            'model': model_key,
            'public_name': public_name,
            'logical_model_name': logical_model_name,
            'served_model_name': served_model_name,
            'protocol_mode': protocol_mode,
            'placement': deepcopy(service.get('placement', {})),
            'topology': deepcopy(service.get('topology', {})),
            'runtime': deepcopy(service.get('runtime', {})),
            'extra_args': deepcopy(service.get('extra_args', [])),
            'reasoning': deepcopy(service.get('reasoning', {})),
            'tool_calling': deepcopy(service.get('tool_calling', {})),
            'chat_compat': deepcopy(service.get('chat_compat', {})),
            'resource_profile': service.get(
                'resource_profile', model.get('resource_profile', '')
            ),
            'min_replicas': int(
                service.get(
                    'min_replicas',
                    model.get('defaults', {}).get('min_replicas', 0),
                )
            ),
            'max_replicas': int(
                service.get(
                    'max_replicas',
                    model.get('defaults', {}).get('max_replicas', 1),
                )
            ),
            'priority_class_name': service.get(
                'priority_class_name', model.get('priority_class_name')
            ),
            'tags': list(service.get('tags', raw.get('tags', [])) or []),
            'audit_notes': list(
                service.get('audit_notes', raw.get('audit_notes', [])) or []
            ),
            'notes': list(service.get('notes', raw.get('notes', [])) or []),
            'benchmark_transport': deepcopy(
                service.get(
                    'benchmark_transport',
                    service.get(
                        'transport',
                        raw.get(
                            'benchmark_transport', raw.get('transport', {})
                        ),
                    ),
                )
            ),
            'publish_port': bool(service.get('publish_port', False)),
        }
    # Convert old router alias map to route map.
    if aliases:
        for alias, target in aliases.items():
            routes[str(alias)] = {
                'provider': 'vllm',
                'runtime': sanitize_name(str(target)),
            }
    else:
        for runtime_name, rt in runtimes.items():
            route_name = str(
                rt.get('public_name')
                or rt.get('logical_model_name')
                or rt.get('served_model_name')
                or runtime_name
            )
            routes[route_name] = {'provider': 'vllm', 'runtime': runtime_name}
    return {
        'name': name,
        'description': raw.get('description', ''),
        'kind': 'stack',
        'providers': {
            'vllm': {'enabled': True, 'runtimes': runtimes},
            'ollama': {'enabled': False},
        },
        'gateways': {'litellm': {'enabled': True}},
        'frontends': {'open_webui': {'enabled': True, 'provider': 'litellm'}},
        'routes': routes,
        'policy': deepcopy(raw.get('policy', {})),
        'vllm': deepcopy(raw.get('vllm', {})),
        'tags': list(raw.get('tags', []) or []),
        'audit_notes': list(raw.get('audit_notes', []) or []),
        'notes': list(raw.get('notes', []) or []),
    }


def _normalize_route_map(routes_raw: Any) -> dict[str, Any]:
    if routes_raw is None:
        return {}
    if isinstance(routes_raw, list):
        out: dict[str, Any] = {}
        for item in routes_raw:
            raw = deepcopy(item)
            name = str(
                raw.pop('name', raw.get('alias', raw.get('model', 'route')))
            )
            out[name] = raw
        return out
    return deepcopy(routes_raw)


def normalize_stack_profiles(
    catalog: dict[str, Any],
    vllm_models: dict[str, Any],
    ollama_models: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for original_name, raw in (catalog or {}).items():
        name = original_name
        profile = deepcopy(raw) or {}
        try:
            if (
                'providers' not in profile
                and 'routes' not in profile
                and 'components' not in profile
            ):
                stack = _legacy_profile_to_stack(name, profile, vllm_models)
            else:
                providers, gateways, frontends = _normalize_components(profile)
                stack = {
                    'name': name,
                    'description': profile.get('description', ''),
                    'kind': profile.get('kind', 'stack'),
                    'providers': providers,
                    'gateways': gateways,
                    'frontends': frontends,
                    'routes': _normalize_route_map(profile.get('routes', {})),
                    'policy': deepcopy(profile.get('policy', {})),
                    'vllm': deepcopy(profile.get('vllm', {})),
                    'tags': list(profile.get('tags', []) or []),
                    'audit_notes': list(profile.get('audit_notes', []) or []),
                    'notes': list(profile.get('notes', []) or []),
                }
            normalized[name] = stack
        except KeyError as ex:
            normalized[name] = {
                'name': name,
                'description': profile.get('description', ''),
                'kind': 'invalid-profile',
                'catalog_error': str(ex),
                'providers': {},
                'gateways': {},
                'frontends': {},
                'routes': {},
                'policy': deepcopy(profile.get('policy', {})),
                'vllm': deepcopy(profile.get('vllm', {})),
                'tags': list(profile.get('tags', []) or []),
                'audit_notes': list(profile.get('audit_notes', []) or []),
                'notes': list(profile.get('notes', []) or []),
            }
    return normalized


# Compatibility function name used in a few imports.
def normalize_profile_catalog(
    catalog: dict[str, Any], models: dict[str, Any]
) -> dict[str, Any]:
    return normalize_stack_profiles(catalog, normalize_vllm_models(models), {})


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    providers = []
    p = profile.get('providers', {}) or {}
    if (p.get('ollama') or {}).get('enabled') not in (False, 'false', None):
        providers.append('ollama')
    vllm_runtimes = (p.get('vllm') or {}).get('runtimes') or {}
    if vllm_runtimes or (p.get('vllm') or {}).get('enabled') is True:
        providers.append('vllm')
    litellm = (profile.get('gateways', {}).get('litellm') or {}).get(
        'enabled', 'auto'
    )
    open_webui = profile.get('frontends', {}).get('open_webui', {}) or {}
    return {
        'name': profile['name'],
        'public_name': profile.get('name', ''),
        'kind': profile.get('kind', 'stack'),
        'providers': providers,
        'gateway': 'litellm'
        if litellm is True
        else ('auto' if litellm == 'auto' else 'none'),
        'frontend': 'open_webui'
        if open_webui.get('enabled', 'auto') not in (False, 'false')
        else 'none',
        'frontend_provider': open_webui.get('provider', 'auto'),
        'route_count': len(profile.get('routes', {}) or {}),
        'description': profile.get('description', ''),
        # Old fields kept so older list formatting doesn't crash.
        'base_model': '',
        'logical_model_name': '',
        'served_model_name': '',
        'protocol_mode': 'stack',
        'engine': ','.join(providers) or 'none',
        'resource_profile': '',
        'tags': profile.get('tags', []),
    }
