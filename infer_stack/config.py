from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any
import os

import yaml

from .catalog import (
    normalize_ollama_models,
    normalize_stack_profiles,
    normalize_vllm_models,
)
from .paths import MODEL_PATH_ENV, config_root, data_root

CONFIG_FILE = Path('config.yaml')
MODELS_FILE = Path('models.yaml')

# Filenames/sub-paths inside whatever the resolved generated directory is.
# The directory itself defaults to ``<data-dir>/generated`` and is relocated
# by pointing ``--data-dir`` / ``INFER_STACK_DATA_DIR`` at a new root at setup
# time (or by editing ``output.generated_dir`` in config.yaml directly).
GENERATED_DIR_NAME = 'generated'
PLAN_FILENAME = 'plan.yaml'
KUBEAI_GENERATED_SUBDIR = 'kubeai'
KUBEAI_VALUES_FILENAME = 'kubeai-values.yaml'
KUBEAI_LOCAL_VALUES_FILENAME = 'kubeai-values.local.yaml'

PINNED_IMAGES = {
    'postgres': 'postgres:16.8',
    'open_webui': 'ghcr.io/open-webui/open-webui:v0.8.6',
    'litellm': 'ghcr.io/berriai/litellm:v1.82.3-stable',
    'vllm': 'vllm/vllm-openai:v0.19.1',
    'ollama': 'ollama/ollama:latest',
    'nginx': 'nginx:1.29.7-alpine',
}

DEFAULT_PORTS = {
    'litellm': 14042,
    'open_webui': 13000,
    'postgres': 15432,
    'ollama': 11434,
    'reverse_proxy_http': 80,
    'reverse_proxy_https': 443,
}


def _default_storage_root() -> Path:
    """Default parent for ``state.*`` paths (hf-cache, postgres volumes, etc.)."""
    return data_root()


def default_state_paths() -> dict[str, str]:
    storage_root = _default_storage_root()
    return {
        'hf_cache': str(storage_root / 'hf-cache'),
        'vllm_cache': str(storage_root / 'vllm-cache'),
        'torch_cache': str(storage_root / 'torch-cache'),
        'triton_cache': str(storage_root / 'triton-cache'),
        'cuda_cache': str(storage_root / 'cuda-cache'),
        'open_webui': str(storage_root / 'open-webui'),
        'postgres_open_webui': str(storage_root / 'postgres-open-webui'),
        'postgres_litellm': str(storage_root / 'postgres-litellm'),
        'ollama': str(storage_root / 'ollama'),
        'runtime': str(storage_root / 'runtime'),
    }


def _default_generated_dir() -> Path:
    return data_root() / GENERATED_DIR_NAME


def default_output_config() -> dict[str, str]:
    return {'generated_dir': str(_default_generated_dir())}


def normalized_output(output_cfg: dict[str, Any] | None) -> dict[str, str]:
    """Resolve the output section to absolute paths.

    Relative ``generated_dir`` values are anchored on ``data_root()`` so
    that a config that says ``generated_dir: generated`` lands at
    ``<data_root>/generated`` regardless of where ``infer-stack`` is
    invoked from.
    """
    normalized = deepcopy(default_output_config())
    raw = (output_cfg or {}).get('generated_dir')
    candidate = Path(raw) if raw else Path(normalized['generated_dir'])
    if not candidate.is_absolute():
        candidate = data_root() / candidate
    normalized['generated_dir'] = str(candidate)
    return normalized


def generated_dir_for_config(cfg: dict[str, Any]) -> Path:
    return Path(normalized_output(cfg.get('output', {}))['generated_dir'])


def plan_path_for_config(cfg: dict[str, Any]) -> Path:
    return generated_dir_for_config(cfg) / PLAN_FILENAME


def kubeai_generated_dir_for_config(cfg: dict[str, Any]) -> Path:
    return generated_dir_for_config(cfg) / KUBEAI_GENERATED_SUBDIR


def kubeai_values_path_for_config(cfg: dict[str, Any]) -> Path:
    return kubeai_generated_dir_for_config(cfg) / KUBEAI_VALUES_FILENAME


def default_cluster_config() -> dict[str, Any]:
    return {
        'namespace': 'kubeai',
        'kubeai_release_name': 'kubeai',
        'kubeai_chart': 'kubeai/kubeai',
        'service_name': 'kubeai',
        'ingress': {
            'enabled': False,
            'class_name': 'traefik',
            'host': '',
            'path_prefix': '/',
            'tls_secret_name': '',
        },
    }


def default_resource_profiles() -> dict[str, Any]:
    return {
        'gpu-single-default': {
            'limits': {'nvidia.com/gpu': 1},
            'requests': {'nvidia.com/gpu': 1},
        },
        'gpu-tp2-balanced': {
            'limits': {'nvidia.com/gpu': 2},
            'requests': {'nvidia.com/gpu': 2},
        },
        'gpu-tp2-maxctx': {
            'limits': {'nvidia.com/gpu': 2},
            'requests': {'nvidia.com/gpu': 2},
        },
    }


def kubeai_local_values_path() -> Path:
    """Location of the user-editable ``kubeai-values.local.yaml``."""
    return config_root() / KUBEAI_LOCAL_VALUES_FILENAME


def resource_profiles_to_kubeai_values(
    resource_profiles: dict[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {'resourceProfiles': {}}
    for name, spec in (resource_profiles or {}).items():
        item: dict[str, Any] = {}
        if spec.get('node_selector'):
            item['nodeSelector'] = deepcopy(spec['node_selector'])
        if spec.get('requests'):
            item['requests'] = deepcopy(spec['requests'])
        if spec.get('limits'):
            item['limits'] = deepcopy(spec['limits'])
        if spec.get('tolerations'):
            item['tolerations'] = deepcopy(spec['tolerations'])
        if spec.get('runtime_class_name'):
            item['runtimeClassName'] = spec['runtime_class_name']
        if spec.get('scheduler_name'):
            item['schedulerName'] = spec['scheduler_name']
        if spec.get('image_name'):
            item['imageName'] = spec['image_name']
        values['resourceProfiles'][name] = item
    return values


def kubeai_values_to_resource_profiles(
    values_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for name, spec in (
        (values_doc or {}).get('resourceProfiles', {}) or {}
    ).items():
        profiles[name] = deepcopy(spec)
    return profiles


def load_kubeai_resource_profiles() -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    path = kubeai_local_values_path()
    if not path.exists():
        return {}, {}, path
    values_doc = load_yaml(path)
    return kubeai_values_to_resource_profiles(values_doc), values_doc, path


def save_kubeai_resource_profiles(values_doc: dict[str, Any]) -> Path:
    path = kubeai_local_values_path()
    save_yaml(path, values_doc)
    return path


def normalized_state(state: dict[str, Any] | None) -> dict[str, str]:
    """Resolve ``state.*`` to absolute paths.

    Relative values are anchored on ``data_root()`` so that bind-mount
    locations don't depend on where ``infer-stack`` was invoked from.
    """
    normalized = deepcopy(default_state_paths())
    anchor = data_root()
    for key, value in (state or {}).items():
        if value in (None, ''):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = anchor / p
        normalized[key] = str(p)
    return normalized


def normalized_cluster(config: dict[str, Any] | None) -> dict[str, Any]:
    return deep_merge(default_cluster_config(), config or {})


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _load_template_yaml(name: str) -> dict[str, Any]:
    text = (
        files('infer_stack')
        .joinpath(f'templates/{name}')
        .read_text(encoding='utf-8')
    )
    return yaml.safe_load(text) or {}


def builtin_vllm_models_catalog() -> dict[str, Any]:
    return _load_template_yaml('default-vllm-models.yaml')


def builtin_ollama_models_catalog() -> dict[str, Any]:
    return _load_template_yaml('default-ollama-models.yaml')


def builtin_profiles_catalog() -> dict[str, Any]:
    return _load_template_yaml('default-profiles.yaml')


# Backwards-compatible helper name for callers that only know about vLLM models.
def builtin_models_catalog() -> dict[str, Any]:
    return builtin_vllm_models_catalog()


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out



def _as_list(value: Any) -> list[Any]:
    """Return ``value`` as a list while treating strings as one entry."""
    if value in (None, ''):
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_catalog_path(raw: Any, *, anchor: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path


def _split_model_path_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part for part in raw.split(os.pathsep) if part]


def _catalog_files_from_entry(entry: Path) -> list[Path]:
    """Expand one catalog search-path entry into model/profile YAML files.

    Explicit file entries are loaded regardless of name. Directory entries are
    scanned non-recursively for the intentionally narrow catalog names used by
    infer-stack overlays: ``models.yaml``, ``*.models.yaml``,
    ``*.profiles.yaml`` and their ``.yml`` variants. This avoids accidentally
    loading unrelated YAML files
    such as ``config.yaml`` or generated runtime files.
    """
    if entry.is_file():
        return [entry]
    if not entry.is_dir():
        return []
    files: list[Path] = []
    for primary_name in (str(MODELS_FILE), 'models.yml'):
        primary = entry / primary_name
        if primary.exists() and primary.is_file() and primary not in files:
            files.append(primary)
    for pattern in (
        '*.models.yaml',
        '*.models.yml',
        '*.profiles.yaml',
        '*.profiles.yml',
    ):
        for candidate in sorted(entry.glob(pattern)):
            if candidate.is_file() and candidate not in files:
                files.append(candidate)
    return files


def _catalog_entries_from_config(
    catalog_cfg: dict[str, Any], *, anchor: Path
) -> list[Path]:
    """Return catalog entries declared in config.yaml merge order.

    ``user_models_file`` preserves the historical single-catalog behavior.
    ``model_path`` / ``model_paths`` add PATH-like overlay entries without
    replacing the user catalog. Relative config entries are anchored on
    ``config_root()`` because they are part of the infer-stack configuration,
    not the caller's shell working directory.
    """
    entries: list[Path] = []

    raw_user_models = catalog_cfg.get('user_models_file', str(MODELS_FILE))
    for raw in _as_list(raw_user_models):
        entries.append(_resolve_catalog_path(raw, anchor=anchor))

    for key in ('model_path', 'model_paths'):
        for raw in _as_list(catalog_cfg.get(key)):
            entries.append(_resolve_catalog_path(raw, anchor=anchor))

    return entries


def _catalog_entries_from_env(raw_env: str | None, *, anchor: Path) -> list[Path]:
    """Return catalog entries declared by ``INFER_STACK_MODEL_PATH``.

    Relative environment entries are anchored on the process working directory,
    matching the ergonomics of shell variables such as ``PATH`` and
    ``PYTHONPATH``.
    """
    return [
        _resolve_catalog_path(raw, anchor=anchor)
        for raw in _split_model_path_env(raw_env)
    ]


def catalog_files(config: dict[str, Any]) -> list[Path]:
    """Return model/profile catalog overlay files in merge order.

    Merge precedence is deterministic and backward compatible:

    1. ``catalog.user_models_file`` (defaults to ``models.yaml`` under
       ``config_root()``)
    2. optional ``catalog.model_path`` / ``catalog.model_paths`` entries
    3. optional ``INFER_STACK_MODEL_PATH`` entries, split like ``PATH``

    Later files override earlier ones via ``deep_merge``.
    """
    catalog_cfg = config.get('catalog', {}) or {}
    entries: list[Path] = []
    entries.extend(
        _catalog_entries_from_config(catalog_cfg, anchor=config_root())
    )
    entries.extend(
        _catalog_entries_from_env(
            os.environ.get(MODEL_PATH_ENV), anchor=Path.cwd()
        )
    )

    files: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        for candidate in _catalog_files_from_entry(entry):
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(candidate)
    return files

def load_catalog_overlays(config: dict[str, Any]) -> dict[str, Any]:
    """Load and merge all configured model/profile catalog overlay files."""
    merged: dict[str, Any] = {}
    for path in catalog_files(config):
        merged = deep_merge(merged, load_yaml(path))
    return merged

def merged_catalogs(config: dict[str, Any]) -> dict[str, Any]:
    catalog_cfg = config.get('catalog', {})
    built_vllm = (
        builtin_vllm_models_catalog()
        if catalog_cfg.get('builtin_models', True)
        else {}
    )
    built_ollama = (
        builtin_ollama_models_catalog()
        if catalog_cfg.get('builtin_models', True)
        else {}
    )
    built_profiles = (
        builtin_profiles_catalog()
        if catalog_cfg.get('builtin_profiles', True)
        else {}
    )
    user = load_catalog_overlays(config)

    # User files may use the new provider-specific keys or the old generic
    # `models` key, which is interpreted as vLLM models.
    vllm_models = deep_merge(
        built_vllm.get('vllm_models', built_vllm.get('models', {})),
        user.get('vllm_models', user.get('models', {})),
    )
    ollama_models = deep_merge(
        built_ollama.get('ollama_models', {}), user.get('ollama_models', {})
    )
    profiles = deep_merge(
        built_profiles.get('profiles', {}), user.get('profiles', {})
    )
    profiles = deep_merge(profiles, config.get('profiles', {}))
    return {
        'vllm_models': vllm_models,
        'ollama_models': ollama_models,
        'profiles': profiles,
        # Compatibility view.
        'models': vllm_models,
    }


def normalized_catalogs(config: dict[str, Any]) -> dict[str, Any]:
    catalogs = merged_catalogs(config)
    vllm_models = normalize_vllm_models(catalogs.get('vllm_models', {}))
    ollama_models = normalize_ollama_models(catalogs.get('ollama_models', {}))
    profiles = normalize_stack_profiles(
        catalogs.get('profiles', {}), vllm_models, ollama_models
    )
    return {
        'vllm_models': vllm_models,
        'ollama_models': ollama_models,
        'models': vllm_models,
        'profiles': profiles,
    }


def initial_config() -> dict[str, Any]:
    # SmolLM2-135M is tiny (~270 MB) and runs on a single small GPU, so it is a
    # safe out-of-the-box default regardless of detected hardware.
    return {
        'name': 'local-llm-stack',
        'backend': 'compose',
        'active_profile': 'smollm2-135m-single',
        'catalog': {
            'builtin_models': True,
            'builtin_profiles': True,
            'user_models_file': str(MODELS_FILE),
        },
        'policy': {
            'require_fit_validation': True,
            'reserve_display_gpu': 'auto',
            'forbid_reserved_gpu_use': False,
            'require_homogeneous_multi_gpu_groups': True,
            'minimum_vram_headroom_gib': 2,
            'allow_unsupported_render': False,
        },
        'runtime': {
            'compose_cmd': 'docker compose',
            'target_inventory': 'auto',
        },
        'providers': {
            'ollama': {'enabled': 'auto'},
            'vllm': {'enabled': 'auto'},
        },
        'gateways': {
            'litellm': {'enabled': 'auto'},
        },
        'frontends': {
            'open_webui': {'enabled': 'auto'},
            'reverse_proxy': {'enabled': False},
        },
        'ollama': {
            'publish_port': False,
            'host': '0.0.0.0:11434',
            'keep_alive': '2m',
            'context_length': 4096,
            'num_parallel': 1,
            'max_loaded_models': 1,
            'max_queue': 8,
            'gpu_indices': 'auto',
            'extra_env': {},
            'env_file': [],
            'extra_volumes': [],
            'extra_hosts': [],
            'labels': {},
        },
        'ports': deepcopy(DEFAULT_PORTS),
        'images': deepcopy(PINNED_IMAGES),
        'state': default_state_paths(),
        'output': default_output_config(),
        'open_webui': {
            'auth': False,
            'provider': 'auto',
            'publish_port': True,
            'extra_env': {},
            'env_file': [],
            'extra_volumes': [],
            'extra_hosts': [],
            'labels': {},
            'ldap': {'enabled': False},
        },
        'reverse_proxy': {
            'enabled': False,
            'image': '',
            'container_name': 'reverse-proxy',
            'target': 'open_webui',
            'server_name': 'localhost',
            'publish_http': True,
            'publish_https': True,
            'http_port': None,
            'https_port': None,
            'http_bind_host': '',
            'https_bind_host': '',
            'force_https': True,
            'client_max_body_size': '1G',
            'proxy_connect_timeout': '60s',
            'proxy_read_timeout': '600s',
            'proxy_send_timeout': '600s',
            'proxy_buffer_size': '128k',
            'proxy_buffers': '4 256k',
            'proxy_busy_buffers_size': '256k',
            'proxy_buffering': None,
            'proxy_cache': None,
            'resolver': [],
            'resolver_timeout': '5s',
            'hsts': {
                'enabled': True,
                'max_age': 63072000,
                'include_subdomains': True,
                'preload': False,
            },
            'ssl': {
                'enabled': True,
                'certificate': '',
                'certificate_key': '',
                'dhparam': '',
                'certificate_container_path': '/etc/ssl/certs/infer-stack-site.crt',
                'certificate_key_container_path': '/etc/ssl/private/infer-stack-site.key',
                'dhparam_container_path': '/etc/ssl/certs/dhparam.pem',
                'protocols': 'TLSv1.2 TLSv1.3',
                'ciphers': 'ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384',
                'prefer_server_ciphers': True,
                'session_cache': 'shared:SSL:10m',
                'ecdh_curve': 'secp384r1',
                'session_tickets': False,
                'stapling': True,
                'stapling_verify': True,
            },
            'config_path': '',
            'extra_config': '',
            'extra_env': {},
            'env_file': [],
            'extra_volumes': [],
            'extra_hosts': [],
            'labels': {},
        },
        'cluster': default_cluster_config(),
        'resource_profiles': default_resource_profiles(),
        'profiles': {},
    }
