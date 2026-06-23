from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import config_root, data_root

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


def dump_yaml(data: dict[str, Any]) -> str:
    """Serialize ``data`` to the canonical YAML text ``save_yaml`` writes.

    Exposed so diff-preview call sites can render the exact bytes that will
    land on disk before committing the write.
    """
    return yaml.safe_dump(data, sort_keys=False)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data), encoding='utf-8')




def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out





