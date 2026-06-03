from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import CONFIG_FILE, initial_config, load_yaml
from .hardware import simulate_inventory
from .paths import config_root
from .resolver import resolve
from .profile_runtime import default_base_url


def _default_access(
    service: dict[str, Any], deployment: dict[str, Any]
) -> dict[str, Any]:
    backend = str(deployment.get('backend', 'compose')).lower()
    if backend == 'kubeai':
        return {
            'name': 'kubeai-frontdoor',
            'kind': 'openai-compatible',
            'base_url': default_base_url(deployment),
            'request_model_name': service['profile_public_name'],
            'auth_env_name': 'KUBEAI_OPENAI_API_KEY',
            'auth_placeholder': 'EMPTY',
            'auth_required': False,
            'notes': [
                'Use the KubeAI OpenAI-compatible front door for routed requests.'
            ],
        }
    return {
        'name': 'compose-router',
        'kind': 'openai-compatible',
        'base_url': default_base_url(deployment),
        'request_model_name': service['served_model_name'],
        'auth_env_name': 'LITELLM_MASTER_KEY',
        'auth_placeholder': 'SET_LITELLM_MASTER_KEY_IN_ENV',
        'auth_required': True,
        'notes': ['Use the LiteLLM router front door for routed requests.'],
    }


def _additional_accesses(
    service: dict[str, Any], deployment: dict[str, Any]
) -> list[dict[str, Any]]:
    benchmark_transport = deepcopy(service.get('benchmark_transport', {}))
    if not benchmark_transport:
        return []
    default = _default_access(service, deployment)
    default_auth_env = (
        'VLLM_API_KEY'
        if benchmark_transport.get('kind') == 'vllm-direct'
        else default['auth_env_name']
    )
    default_auth_placeholder = (
        'EMPTY'
        if benchmark_transport.get('kind') == 'vllm-direct'
        else default['auth_placeholder']
    )
    access = {
        'name': benchmark_transport.get('name')
        or benchmark_transport.get('kind')
        or 'compatibility-access',
        'kind': benchmark_transport.get('kind') or default['kind'],
        'base_url': benchmark_transport.get('base_url') or default['base_url'],
        'request_model_name': benchmark_transport.get('request_model_name')
        or service.get('hf_model_id')
        or service['served_model_name'],
        'auth_env_name': benchmark_transport.get('api_key_env')
        or default_auth_env,
        'auth_placeholder': benchmark_transport.get('api_key_placeholder')
        or default_auth_placeholder,
        'auth_required': bool(
            benchmark_transport.get(
                'api_key_required', default_auth_placeholder != 'EMPTY'
            )
        ),
        'notes': [
            'Optional compatibility access hint retained for external integrations.'
        ],
    }
    dedupe_key = (
        access['kind'],
        access['base_url'],
        access['request_model_name'],
    )
    default_key = (
        default['kind'],
        default['base_url'],
        default['request_model_name'],
    )
    if dedupe_key == default_key:
        return []
    return [access]


def build_profile_contract(deployment: dict[str, Any]) -> dict[str, Any]:
    services = []
    for service in deployment.get('services', []):
        services.append(
            {
                'profile_name': service['profile_name'],
                'public_name': service['profile_public_name'],
                'service_name': service['service_name'],
                'kubernetes_name': service['kubernetes_name'],
                'model': {
                    'model_ref': service['model_ref'],
                    'hf_model_id': service['hf_model_id'],
                    'logical_model_name': service['logical_model_name'],
                    'served_model_name': service['served_model_name'],
                    'served_aliases': deepcopy(
                        service.get('served_aliases', [])
                    ),
                    'tokenizer_name': service['tokenizer_name'],
                    'modalities': deepcopy(service.get('modalities', [])),
                },
                'protocol': {
                    'mode': service['protocol_mode'],
                    'engine': service['engine'],
                    'features': deepcopy(service.get('features', [])),
                },
                'access': {
                    'default': _default_access(service, deployment),
                    'additional': _additional_accesses(service, deployment),
                },
                'runtime': {
                    'resource_profile': service['resource_profile'],
                    'priority_class_name': service.get('priority_class_name'),
                    'min_replicas': service['min_replicas'],
                    'max_replicas': service['max_replicas'],
                    'tensor_parallel_size': service['tensor_parallel_size'],
                    'data_parallel_size': service['data_parallel_size'],
                    'max_model_len': service['max_model_len'],
                    'gpu_memory_utilization': service['gpu_memory_utilization'],
                    'max_num_batched_tokens': service['max_num_batched_tokens'],
                    'max_num_seqs': service['max_num_seqs'],
                    'enable_prefix_caching': service['enable_prefix_caching'],
                    'extra_args': deepcopy(service.get('extra_args', [])),
                },
                'placement': {
                    'gpu_indices': deepcopy(service.get('gpu_indices', [])),
                    'placement': deepcopy(service.get('placement', {})),
                    'topology': deepcopy(service.get('topology', {})),
                },
                'notes': deepcopy(service.get('notes', [])),
                'caveats': deepcopy(service.get('audit_notes', [])),
            }
        )
    return {
        'schema_version': 1,
        'kind': 'serving-profile-contract',
        'backend': deployment['backend'],
        'profile': {
            'name': deployment['serving_profile']['name'],
            'public_name': deployment['serving_profile']['public_name'],
            'description': deployment['serving_profile'].get('description', ''),
            'kubernetes_name': deployment['serving_profile'].get(
                'kubernetes_name', ''
            ),
            'service_name': deployment['serving_profile'].get(
                'service_name', ''
            ),
            'protocol_mode': deployment['serving_profile'].get(
                'protocol_mode', ''
            ),
            'engine': deployment['serving_profile'].get('engine', ''),
            'resource_profile': deployment['serving_profile'].get(
                'resource_profile', ''
            ),
            'notes': deepcopy(deployment['serving_profile'].get('notes', [])),
            'caveats': deepcopy(
                deployment['serving_profile'].get('audit_notes', [])
            ),
        },
        'router': {
            'type': deployment.get('router', {}).get('type', ''),
            'aliases': deepcopy(
                deployment.get('router', {}).get('aliases', {})
            ),
        },
        'services': services,
    }


def describe_profile_contract(
    config: dict[str, Any],
    *,
    resolve_fn,
    profile_name: str | None = None,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deployment = resolve_fn(
        config, inventory=inventory, profile_name=profile_name
    )
    return build_profile_contract(deployment)


def load_profile_contract(
    profile_name: str,
    *,
    backend: str | None = None,
    simulate_hardware_spec: str | None = None,
) -> dict[str, Any]:
    config_path = config_root() / CONFIG_FILE
    if config_path.exists():
        config = load_yaml(config_path)
    else:
        config = initial_config()
    config.setdefault('catalog', {})
    config['catalog']['builtin_models'] = True
    config['catalog']['builtin_profiles'] = True
    if backend is not None:
        config['backend'] = backend
    inventory = (
        simulate_inventory(simulate_hardware_spec)
        if simulate_hardware_spec
        else None
    )
    return describe_profile_contract(
        config,
        resolve_fn=resolve,
        profile_name=profile_name,
        inventory=inventory,
    )
