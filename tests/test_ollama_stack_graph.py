from __future__ import annotations

from pathlib import Path

import yaml

from infer_stack.backends.compose_renderer import render_compose_artifacts
from infer_stack.cli import _compose_has_service
from infer_stack.config import initial_config
from infer_stack.hardware import simulate_inventory
from infer_stack.resolver import resolve
from infer_stack.validator import validate_resolved


def _cfg(tmp_path: Path, profile: str, backend: str = 'compose') -> dict:
    cfg = initial_config()
    cfg['backend'] = backend
    cfg['active_profile'] = profile
    cfg['output']['generated_dir'] = str(tmp_path / 'generated')
    for key in [
        'hf_cache',
        'vllm_cache',
        'torch_cache',
        'triton_cache',
        'cuda_cache',
        'open_webui',
        'postgres_open_webui',
        'postgres_litellm',
        'ollama',
        'runtime',
    ]:
        cfg['state'][key] = str(tmp_path / 'state' / key)
    return cfg


def test_ollama_direct_renders_no_litellm(tmp_path: Path) -> None:
    dep = resolve(
        _cfg(tmp_path, 'ollama-direct'), inventory=simulate_inventory('2x11')
    )
    assert dep['providers']['ollama']['enabled'] is True
    assert dep['gateways']['litellm']['enabled'] is False
    assert dep['frontends']['open_webui']['provider'] == 'ollama'
    assert dep['services'] == []
    assert validate_resolved(dep)['ok'] is True
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'ollama:' in compose
    assert 'litellm:' not in compose
    assert 'postgres-litellm:' not in compose
    assert 'OLLAMA_BASE_URL: http://ollama:11434' in compose
    assert not (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').exists()


def test_ollama_gateway_routes_through_litellm(tmp_path: Path) -> None:
    dep = resolve(
        _cfg(tmp_path, 'ollama-qwen3.5-4b-gateway'),
        inventory=simulate_inventory('2x11'),
    )
    assert dep['gateways']['litellm']['enabled'] is True
    assert (
        dep['gateways']['litellm']['routes']['qwen3.5-4b']['provider']
        == 'ollama'
    )
    render_compose_artifacts({'deployment': dep})
    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    entry = cfg_doc['model_list'][0]
    assert entry['model_name'] == 'qwen3.5-4b'
    assert entry['litellm_params']['model'] == 'ollama_chat/qwen3.5:4b'


def test_mixed_gateway_renders_ollama_vllm_and_litellm(tmp_path: Path) -> None:
    dep = resolve(
        _cfg(tmp_path, 'mixed-ollama-smollm'),
        inventory=simulate_inventory('3x11'),
    )
    assert dep['providers']['ollama']['enabled'] is True
    assert dep['providers']['vllm']['runtimes']
    assert dep['gateways']['litellm']['enabled'] is True
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'ollama:' in compose
    assert 'vllm-smollm:' in compose
    assert 'litellm:' in compose
    litellm = (
        tmp_path / 'state' / 'runtime' / 'litellm_config.yaml'
    ).read_text()
    assert 'ollama_chat/qwen3.5:4b' in litellm
    assert 'openai/' in litellm


def test_kubeai_rejects_ollama_provider(tmp_path: Path) -> None:
    dep = resolve(
        _cfg(tmp_path, 'ollama-direct', backend='kubeai'),
        inventory=simulate_inventory('2x11'),
    )
    report = validate_resolved(dep)
    assert report['ok'] is False
    assert any('Ollama provider' in err for err in report['errors'])


def test_smollm_gpu1_vllm_profile_is_builtin_and_pins_gpu1(
    tmp_path: Path,
) -> None:
    dep = resolve(
        _cfg(tmp_path, 'smollm2-135m-vllm-gpu1'),
        inventory=simulate_inventory('2x24'),
    )
    rt = dep['providers']['vllm']['runtimes']['chat']
    assert rt['gpu_indices'] == [1]
    assert rt['compose_service_name'] == 'vllm-chat'
    assert rt['container_name'] == 'vllm-chat'
    assert dep['gateways']['litellm']['enabled'] is True
    assert dep['frontends']['open_webui']['provider'] == 'litellm'
    assert (
        dep['gateways']['litellm']['routes']['smollm2-135m']['provider']
        == 'vllm'
    )
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'vllm-chat:' in compose
    assert 'container_name: vllm-chat' in compose
    assert 'container_name: chat' not in compose


def test_smollm_gpu1_ollama_profile_is_builtin_and_has_no_routes(
    tmp_path: Path,
) -> None:
    dep = resolve(
        _cfg(tmp_path, 'smollm2-135m-ollama-gpu1'),
        inventory=simulate_inventory('2x24'),
    )
    assert dep['providers']['ollama']['enabled'] is True
    assert dep['providers']['ollama']['gpu_indices'] == [1]
    assert dep['providers']['ollama']['publish_port'] is True
    assert dep['providers']['vllm']['runtimes'] == {}
    assert dep['gateways']['litellm']['enabled'] is False
    assert dep['frontends']['open_webui']['provider'] == 'ollama'
    assert dep['gateways']['litellm']['routes'] == {}
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'CUDA_VISIBLE_DEVICES: "1"' in compose
    assert 'device_ids: ["1"]' in compose
    assert 'litellm:' not in compose


def test_existing_config_without_new_ollama_defaults_is_hydrated(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, 'smollm2-135m-ollama-gpu1')
    cfg['images'].pop('ollama', None)
    cfg['ports'].pop('ollama', None)
    dep = resolve(cfg, inventory=simulate_inventory('2x24'))
    assert dep['images']['ollama'] == 'ollama/ollama:latest'
    assert dep['ports']['ollama'] == 11434
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'image: ollama/ollama:latest' in compose
    assert '"127.0.0.1:11434:11434"' in compose


def test_compose_has_service_ignores_stale_litellm_runtime_file(
    tmp_path: Path,
) -> None:
    compose_file = tmp_path / 'docker-compose.yml'
    compose_file.write_text(
        """
services:
  postgres-open-webui:
    image: postgres:16.8
  ollama:
    image: ollama/ollama:latest
  open-webui:
    image: ghcr.io/open-webui/open-webui:v0.8.6
""".lstrip(),
        encoding='utf-8',
    )

    assert _compose_has_service(compose_file, 'ollama') is True
    assert _compose_has_service(compose_file, 'open-webui') is True
    assert _compose_has_service(compose_file, 'litellm') is False


def test_vllm_compose_persists_obvious_startup_caches(tmp_path: Path) -> None:
    dep = resolve(
        _cfg(tmp_path, 'smollm2-135m-vllm-gpu1'),
        inventory=simulate_inventory('2x24'),
    )
    render_compose_artifacts({'deployment': dep})
    compose = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'HF_HOME: /root/.cache/huggingface' in compose
    assert 'VLLM_CACHE_ROOT: /root/.cache/vllm' in compose
    assert 'TORCHINDUCTOR_CACHE_DIR: /root/.cache/torch/inductor' in compose
    assert 'TRITON_CACHE_DIR: /root/.cache/triton' in compose
    assert 'CUDA_CACHE_PATH: /root/.cache/nvidia/ComputeCache' in compose
    assert ':/root/.cache/huggingface' in compose
    assert ':/root/.cache/vllm' in compose
    assert ':/root/.cache/torch' in compose
    assert ':/root/.cache/triton' in compose
    assert ':/root/.cache/nvidia/ComputeCache' in compose


def test_litellm_does_not_depend_on_provider_health_for_model_swaps(
    tmp_path: Path,
) -> None:
    dep = resolve(
        _cfg(tmp_path, 'mixed-ollama-smollm'),
        inventory=simulate_inventory('3x11'),
    )
    render_compose_artifacts({'deployment': dep})
    doc = yaml.safe_load(
        (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    )
    depends_on = doc['services']['litellm'].get('depends_on', {})
    assert 'postgres-litellm' in depends_on
    assert 'ollama' not in depends_on
    assert 'vllm-smollm' not in depends_on


def test_litellm_config_model_delete_miss_is_nonfatal() -> None:
    from infer_stack.cli import _litellm_delete_missed_config_model

    class DummyResponse:
        text = ''

        def json(self):
            return {'error': 'Model with id=abc not found in db'}

    assert _litellm_delete_missed_config_model(DummyResponse()) is True

    class OtherResponse:
        text = 'permission denied'

        def json(self):
            raise ValueError

    assert _litellm_delete_missed_config_model(OtherResponse()) is False


def test_schema_v5_default_model_and_protocol_helpers(tmp_path: Path) -> None:
    from infer_stack.cli import (
        _default_model_for_deployment,
        _resolve_smoke_protocol_from_deployment,
    )

    smol = resolve(
        _cfg(tmp_path, 'smollm2-135m-vllm-gpu1'),
        inventory=simulate_inventory('2x24'),
    )
    assert _default_model_for_deployment(smol) == 'smollm2-135m'
    assert (
        _resolve_smoke_protocol_from_deployment(smol, 'smollm2-135m') == 'chat'
    )

    gpt2 = resolve(
        _cfg(tmp_path, 'gpt2-single'), inventory=simulate_inventory('2x24')
    )
    assert _default_model_for_deployment(gpt2) == 'gpt2'
    assert (
        _resolve_smoke_protocol_from_deployment(gpt2, 'gpt2') == 'completions'
    )
