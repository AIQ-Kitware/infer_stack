from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from infer_stack.backends.compose_renderer import render_compose_artifacts
from infer_stack.backends.kubeai_renderer import render_kubeai_artifacts
from infer_stack.config import initial_config
from infer_stack.config import save_yaml
from infer_stack.contracts import build_profile_contract, load_profile_contract
from infer_stack.hardware import simulate_inventory
from infer_stack.paths import set_config_root, set_data_root
from infer_stack.resolver import resolve
from infer_stack.validator import validate_resolved


@pytest.fixture(autouse=True)
def _anchor_paths(tmp_path: Path):
    """Anchor config_root() / data_root() at tmp_path for every test.

    The resolver and renderers read from data_root() (for generated/
    and state/), and load_profile_contract() reads from config_root().
    Pointing both at tmp_path keeps each test hermetic.
    """
    set_config_root(tmp_path)
    set_data_root(tmp_path)
    yield
    set_config_root(None)
    set_data_root(None)


def _cfg(tmp_path: Path, *, backend: str = 'compose') -> dict:
    cfg = initial_config()
    cfg['backend'] = backend
    cfg['state'] = {
        'hf_cache': 'state/hf-cache',
        'vllm_cache': 'state/vllm-cache',
        'open_webui': 'state/open-webui',
        'postgres': 'state/postgres',
        'runtime': 'state/runtime',
    }
    cfg['ports'] = {'litellm': 14042, 'open_webui': 13000, 'postgres': 15432}
    # Relative paths in output.generated_dir / state.* anchor on data_root(),
    # which the autouse fixture pins to tmp_path.
    cfg['output'] = {'generated_dir': 'generated'}
    return cfg


def _write_root_config(tmp_path: Path, *, backend: str = 'compose') -> Path:
    cfg = _cfg(tmp_path, backend=backend)
    save_yaml(tmp_path / 'config.yaml', cfg)
    save_yaml(tmp_path / 'models.yaml', {'models': {}, 'profiles': {}})
    return tmp_path


def _deployment(
    tmp_path: Path,
    profile_name: str,
    *,
    backend: str = 'compose',
    inventory: str = '4x96',
) -> dict:
    cfg = _cfg(tmp_path, backend=backend)
    deployment = resolve(
        cfg, inventory=simulate_inventory(inventory), profile_name=profile_name
    )
    validated = validate_resolved(deployment)
    assert validated['ok'], validated
    return deployment


def test_profile_resolution_uses_named_serving_profile(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, 'qwen2-72b-instruct-tp2-balanced')
    assert (
        deployment['serving_profile']['public_name']
        == 'qwen2-72b-instruct-tp2-balanced'
    )
    assert (
        deployment['serving_profile']['logical_model_name']
        == 'qwen/qwen2-72b-instruct'
    )
    assert deployment['services'][0]['tensor_parallel_size'] == 2
    assert 'qwen2-72b-instruct-tp2-balanced' in deployment['router']['aliases']


def test_legacy_profile_alias_resolves_to_canonical_profile(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'helm-qwen2-72b-instruct')
    assert (
        deployment['serving_profile']['name']
        == 'qwen2-72b-instruct-tp2-balanced'
    )


def test_kubeai_render_uses_profile_identity(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path, 'qwen2-72b-instruct-tp2-balanced', backend='kubeai'
    )
    render_kubeai_artifacts({'deployment': deployment})
    models = list(
        yaml.safe_load_all(
            (tmp_path / 'generated' / 'kubeai' / 'models.yaml').read_text()
        )
    )
    assert models[0]['metadata']['name'] == 'qwen2-72b-instruct-tp2-balanced'
    assert (
        models[0]['metadata']['annotations']['infer-stack/logical-model-name']
        == 'qwen/qwen2-72b-instruct'
    )
    assert models[0]['spec']['resourceProfile'] == 'gpu-tp2-balanced:2'
    assert '--tensor-parallel-size=2' in models[0]['spec']['args']
    assert (
        '--served-model-name=qwen2-72b-instruct-tp2-balanced'
        in models[0]['spec']['args']
    )


def test_compose_render_includes_profile_labels_and_aliases(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    litellm_text = (
        tmp_path / 'state' / 'runtime' / 'litellm_config.yaml'
    ).read_text()
    assert 'infer_stack.public_name: "gpt-oss-20b-chat"' in compose_text
    assert 'openai/gpt-oss-20b' in litellm_text
    assert 'gpt-oss-20b-chat' in litellm_text


def test_profile_contract_is_generic_and_backend_agnostic(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'qwen2-72b-instruct-tp2-balanced')
    contract = build_profile_contract(deployment)
    service = contract['services'][0]
    assert contract['kind'] == 'serving-profile-contract'
    assert (
        contract['profile']['public_name'] == 'qwen2-72b-instruct-tp2-balanced'
    )
    assert service['access']['default']['kind'] == 'openai-compatible'
    assert service['access']['additional'][0]['kind'] == 'vllm-direct'
    assert service['access']['additional'][0]['auth_env_name'] == 'VLLM_API_KEY'
    assert 'client_spec' not in str(contract)
    assert 'model_deployments' not in str(contract)


def test_profile_contract_for_kubeai_uses_public_profile_name(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'qwen2-72b-instruct-tp2-balanced', backend='kubeai'
    )
    contract = build_profile_contract(deployment)
    access = contract['services'][0]['access']['default']
    assert access['kind'] == 'openai-compatible'
    assert access['request_model_name'] == 'qwen2-72b-instruct-tp2-balanced'
    assert access['base_url'].endswith('/openai/v1')


def test_load_profile_contract_uses_public_loader_for_qwen(
    tmp_path: Path,
) -> None:
    _write_root_config(tmp_path)
    contract = load_profile_contract(
        'qwen2-72b-instruct-tp2-balanced',
        simulate_hardware_spec='2x96',
    )
    assert (
        contract['profile']['public_name'] == 'qwen2-72b-instruct-tp2-balanced'
    )
    assert (
        contract['services'][0]['model']['logical_model_name']
        == 'qwen/qwen2-72b-instruct'
    )


def _split_compose_blocks(compose_text: str) -> dict[str, str]:
    """Split a rendered docker-compose.yml into a {service_name: body} dict.

    Service blocks start at column 2 (two-space indent) followed by the name
    and a colon. Splitting this way avoids substring confusion between e.g.
    ``open-webui:`` and ``postgres-open-webui:``.
    """
    import re

    pattern = re.compile(r'(?m)^  ([a-zA-Z0-9_-]+):\s*$')
    matches = list(pattern.finditer(compose_text))
    blocks: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(compose_text)
        )
        blocks[name] = compose_text[start:end]
    return blocks


def test_compose_uses_separate_postgres_services(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)

    assert 'postgres-open-webui' in blocks
    assert 'postgres-litellm' in blocks
    assert 'postgres-init' not in blocks
    assert 'postgres-init' not in compose_text
    assert 'shared-postgress' not in compose_text

    owui_pg = blocks['postgres-open-webui']
    ll_pg = blocks['postgres-litellm']
    assert '${OPENWEBUI_POSTGRES_DB}' in owui_pg
    assert '${OPENWEBUI_POSTGRES_USER}' in owui_pg
    assert '${OPENWEBUI_POSTGRES_PASSWORD}' in owui_pg
    assert '${LITELLM_POSTGRES_DB}' not in owui_pg
    assert '${LITELLM_POSTGRES_DB}' in ll_pg
    assert '${LITELLM_POSTGRES_USER}' in ll_pg
    assert '${LITELLM_POSTGRES_PASSWORD}' in ll_pg
    assert '${OPENWEBUI_POSTGRES_DB}' not in ll_pg


def test_compose_router_and_ui_point_at_their_own_postgres(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)

    litellm_block = blocks['litellm']
    openwebui_block = blocks['open-webui']
    assert '@postgres-litellm:5432/${LITELLM_POSTGRES_DB}' in litellm_block
    assert '@postgres-open-webui:' not in litellm_block
    assert (
        '@postgres-open-webui:5432/${OPENWEBUI_POSTGRES_DB}' in openwebui_block
    )
    assert '@postgres-litellm:' not in openwebui_block


def test_compose_env_uses_distinct_db_names_users_and_passwords(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    env_text = (tmp_path / 'generated' / '.env').read_text()
    env_kv = dict(
        line.split('=', 1)
        for line in env_text.splitlines()
        if line and '=' in line and not line.startswith('#')
    )
    assert env_kv['OPENWEBUI_POSTGRES_DB'] == 'openwebui'
    assert env_kv['LITELLM_POSTGRES_DB'] == 'litellm'
    assert env_kv['OPENWEBUI_POSTGRES_USER'] == 'openwebui'
    assert env_kv['LITELLM_POSTGRES_USER'] == 'litellm'
    assert env_kv['OPENWEBUI_POSTGRES_PASSWORD']
    assert env_kv['LITELLM_POSTGRES_PASSWORD']
    assert (
        env_kv['OPENWEBUI_POSTGRES_PASSWORD']
        != env_kv['LITELLM_POSTGRES_PASSWORD']
    )
    # Old shared schema must not be emitted automatically.
    assert 'POSTGRES_DB' not in env_kv
    assert 'POSTGRES_USER' not in env_kv
    assert 'POSTGRES_PASSWORD' not in env_kv


def test_env_rewrite_preserves_unknown_key_value_pairs(tmp_path: Path) -> None:
    generated = tmp_path / 'generated'
    generated.mkdir()
    (generated / '.env').write_text(
        '# user comment\n'
        'OPENWEBUI_POSTGRES_PASSWORD=keepme\n'
        'VERBOSE=1\n'
        'CUSTOM_THING=abc\n'
        'HTTP_PROXY=http://proxy:3128\n'
        'NO_PROXY=localhost,127.0.0.1\n',
        encoding='utf-8',
    )
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    text = (generated / '.env').read_text()
    assert 'VERBOSE=1' in text
    assert 'CUSTOM_THING=abc' in text
    assert 'HTTP_PROXY=http://proxy:3128' in text
    assert 'NO_PROXY=localhost,127.0.0.1' in text
    assert '# user comment' in text
    env_kv = dict(
        line.split('=', 1)
        for line in text.splitlines()
        if line and '=' in line and not line.startswith('#')
    )
    # Existing managed values are preserved (password not rotated).
    assert env_kv['OPENWEBUI_POSTGRES_PASSWORD'] == 'keepme'


def test_helm_pythia_profiles_render_as_completions(tmp_path: Path) -> None:
    for profile_name in (
        'helm-pythia-6.9b',
        'helm-pythia-2.8b-v0',
        'helm-pythia-1b-v0',
    ):
        deployment = _deployment(tmp_path, profile_name, inventory='1x96')
        render_compose_artifacts({'deployment': deployment})
        compose_text = (
            tmp_path / 'generated' / 'docker-compose.yml'
        ).read_text()
        assert 'infer_stack.protocol_mode: "completions"' in compose_text, (
            profile_name
        )
        assert deployment['services'][0]['protocol_mode'] == 'completions', (
            profile_name
        )


def test_helm_base_model_profiles_render_as_completions(tmp_path: Path) -> None:
    for profile_name in (
        'helm-llama-2-7b',
        'helm-mistral-7b-v0.1',
        'helm-falcon-7b',
    ):
        deployment = _deployment(tmp_path, profile_name, inventory='1x96')
        assert deployment['services'][0]['protocol_mode'] == 'completions', (
            profile_name
        )


def test_validation_rejects_chat_protocol_for_completions_only_model(
    tmp_path: Path,
) -> None:
    from infer_stack.config import initial_config

    cfg = initial_config()
    cfg['backend'] = 'compose'
    cfg['state'] = {
        'hf_cache': 'state/hf-cache',
        'vllm_cache': 'state/vllm-cache',
        'open_webui': 'state/open-webui',
        'postgres_open_webui': 'state/postgres-open-webui',
        'postgres_litellm': 'state/postgres-litellm',
        'runtime': 'state/runtime',
    }
    cfg['ports'] = {'litellm': 14042, 'open_webui': 13000, 'postgres': 15432}
    cfg['profiles'] = {
        'broken-pythia-chat': {
            'description': 'Synthetic invalid profile: chat on a completions-only model.',
            'base_model': 'pythia-1b-v0',
            'public_name': 'broken-pythia-chat',
            'protocol_mode': 'chat',
        }
    }
    deployment = resolve(
        cfg,
        inventory=simulate_inventory('1x96'),
        profile_name='broken-pythia-chat',
    )
    report = validate_resolved(deployment)
    assert report['ok'] is False
    joined = ' | '.join(report['errors'])
    assert 'broken-pythia-chat' in joined
    assert 'completions' in joined
    assert 'pythia-1b-v0' in joined


def test_validation_accepts_chat_for_chat_capable_profile(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'qwen2-5-7b-instruct-turbo-default', inventory='1x96'
    )
    report = validate_resolved(deployment)
    assert report['ok'] is True, report
    assert deployment['services'][0]['protocol_mode'] == 'chat'


def test_litellm_completions_profile_uses_text_completion_provider(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, 'helm-pythia-6.9b', inventory='1x96')
    render_compose_artifacts({'deployment': deployment})
    litellm_text = (
        tmp_path / 'state' / 'runtime' / 'litellm_config.yaml'
    ).read_text()
    assert 'text-completion-openai/eleutherai/pythia-6.9b' in litellm_text
    # Make sure we're not also emitting the bare openai/ provider for the same model.
    cfg_doc = yaml.safe_load(litellm_text)
    pythia_entry = next(
        m
        for m in cfg_doc['model_list']
        if m['model_name'] == 'eleutherai/pythia-6.9b'
    )
    assert (
        pythia_entry['litellm_params']['model']
        == 'text-completion-openai/eleutherai/pythia-6.9b'
    )
    # Even though it's completions-only, the alias must remain in model_list
    # so Open WebUI can still see/select it.
    advertised = {m['model_name'] for m in cfg_doc['model_list']}
    assert 'eleutherai/pythia-6.9b' in advertised


def test_litellm_chat_profile_does_not_use_text_completion_provider(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'qwen2-5-7b-instruct-turbo-default', inventory='1x96'
    )
    render_compose_artifacts({'deployment': deployment})
    litellm_text = (
        tmp_path / 'state' / 'runtime' / 'litellm_config.yaml'
    ).read_text()
    assert 'text-completion-openai' not in litellm_text
    assert 'openai/qwen/qwen2.5-7b-instruct-turbo' in litellm_text
    # Chat models keep merge_reasoning_content_in_choices so Open WebUI can
    # display reasoning when LiteLLM normalizes it into message content.
    assert 'merge_reasoning_content_in_choices: true' in litellm_text


def test_pythia_routing_end_to_end_unit_check(tmp_path: Path) -> None:
    """One-shot regression covering the Pythia routing contract.

    Compose label, LiteLLM provider, alias visibility, and the smoke-test
    protocol picker should agree that this profile is completions.
    """
    deployment = _deployment(tmp_path, 'helm-pythia-6.9b', inventory='1x96')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    litellm_text = (
        tmp_path / 'state' / 'runtime' / 'litellm_config.yaml'
    ).read_text()
    cfg_doc = yaml.safe_load(litellm_text)

    # Compose label.
    assert 'infer_stack.protocol_mode: "completions"' in compose_text
    # LiteLLM provider.
    pythia_entry = next(
        m
        for m in cfg_doc['model_list']
        if m['model_name'] == 'eleutherai/pythia-6.9b'
    )
    assert (
        pythia_entry['litellm_params']['model']
        == 'text-completion-openai/eleutherai/pythia-6.9b'
    )
    # Alias is still advertised (Open WebUI can see it).
    assert any(
        m['model_name'] == 'eleutherai/pythia-6.9b'
        for m in cfg_doc['model_list']
    )
    # Resolved deployment service is completions and matches the alias target.
    svc = deployment['services'][0]
    assert svc['protocol_mode'] == 'completions'
    assert 'eleutherai/pythia-6.9b' in svc['served_aliases']


def test_qwen3_6_reasoning_profile_emits_reasoning_flags(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    qwen_block = compose_text.split('vllm-qwen36-35b:', 1)[1].split(
        'vllm-pythia-69b:', 1
    )[0]
    assert '--reasoning-parser' in qwen_block
    assert '"qwen3"' in qwen_block
    # vLLM CLI does not accept --enable-reasoning; the parser flag alone enables it.
    assert '--enable-reasoning' not in qwen_block


def test_pythia_profile_does_not_emit_reasoning_flags(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, 'helm-pythia-6.9b', inventory='1x96')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert '--enable-reasoning' not in compose_text
    assert '--reasoning-parser' not in compose_text


def test_mixed_profile_qwen_renders_tool_call_flags_pythia_does_not(
    tmp_path: Path,
) -> None:
    """Qwen3.6-35B-A3B model card requires:

      --reasoning-parser qwen3
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --language-model-only

    Pythia services must not get any tool-call flags.
    """
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)

    qwen_block = blocks['vllm-qwen36-35b']
    # All four model-card-recommended flags are present.
    assert '--enable-auto-tool-choice' in qwen_block
    assert '--tool-call-parser' in qwen_block
    assert '"qwen3_coder"' in qwen_block
    assert '--reasoning-parser' in qwen_block
    assert '"qwen3"' in qwen_block
    assert '--language-model-only' in qwen_block
    # qwen3_xml is only a manual fallback, never rendered for this profile.
    assert 'qwen3_xml' not in qwen_block

    for pythia_name in ('vllm-pythia-69b', 'vllm-pythia-28b'):
        block = blocks[pythia_name]
        assert '--enable-auto-tool-choice' not in block, pythia_name
        assert '--tool-call-parser' not in block, pythia_name
        assert 'qwen3_coder' not in block, pythia_name
        assert 'qwen3_xml' not in block, pythia_name


def test_mixed_profile_qwen_command_arg_order_matches_model_card(
    tmp_path: Path,
) -> None:
    """Lock down the exact command-line tokens for the Qwen service.

    The Qwen3.6-35B-A3B model card lists these together; we want each
    parser flag's value to immediately follow the flag (vLLM positional
    contract). This catches regressions where a value gets dropped or
    reordered into another argument.
    """
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    qwen_block = blocks['vllm-qwen36-35b']

    # Each flag is immediately followed by its value on the next rendered line.
    for flag, value in (
        ('--reasoning-parser', 'qwen3'),
        ('--tool-call-parser', 'qwen3_coder'),
    ):
        assert flag in qwen_block
        flag_idx = qwen_block.index(flag)
        following = qwen_block[flag_idx:]
        first_value_line = next(
            line.strip() for line in following.splitlines()[1:] if line.strip()
        )
        assert first_value_line == f'- "{value}"', (flag, first_value_line)


def test_mixed_profile_litellm_config_unchanged_by_tool_calling(
    tmp_path: Path,
) -> None:
    """Adding tool_calling on a service must not perturb the LiteLLM config."""
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    by_alias = {
        m['model_name']: m['litellm_params'] for m in cfg_doc['model_list']
    }

    qwen = by_alias['qwen3.6-35b-a3b']
    assert qwen['model'] == 'openai/qwen3.6-35b-a3b'
    assert qwen.get('merge_reasoning_content_in_choices') is True
    # No tool-calling-specific keys leak into LiteLLM params.
    assert 'tools' not in qwen
    assert 'tool_choice' not in qwen
    assert 'enable_auto_tool_choice' not in qwen

    assert (
        by_alias['eleutherai/pythia-6.9b']['model']
        == 'text-completion-openai/eleutherai/pythia-6.9b'
    )
    assert (
        by_alias['eleutherai/pythia-2.8b-v0']['model']
        == 'text-completion-openai/eleutherai/pythia-2.8b-v0'
    )


def test_mixed_profile_compose_service_names_are_stable(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    blocks = _split_compose_blocks(
        (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    )
    infer_stacks = sorted(name for name in blocks if name.startswith('vllm-'))
    assert infer_stacks == [
        'vllm-pythia-28b',
        'vllm-pythia-69b',
        'vllm-qwen36-35b',
    ]


def test_default_pythia_profiles_have_no_chat_compat(tmp_path: Path) -> None:
    for profile_name in (
        'helm-pythia-6.9b',
        'helm-pythia-2.8b-v0',
        'helm-pythia-1b-v0',
    ):
        deployment = _deployment(tmp_path, profile_name, inventory='1x96')
        render_compose_artifacts({'deployment': deployment})
        cfg_doc = yaml.safe_load(
            (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
        )
        for entry in cfg_doc['model_list']:
            params = entry['litellm_params']
            assert 'roles' not in params, profile_name
            assert 'initial_prompt_value' not in params, profile_name
            assert 'final_prompt_value' not in params, profile_name
        # Resolved service has the flag off.
        for svc in deployment['services']:
            assert svc['chat_compat_enabled'] is False, profile_name


def test_pythia_inspect_mmlu_compat_renders_completions_with_litellm_template(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-inspect-mmlu-compat', inventory='2x96'
    )
    render_compose_artifacts({'deployment': deployment})

    # Both Pythia services keep protocol_mode=completions.
    for svc in deployment['services']:
        assert svc['protocol_mode'] == 'completions', svc['service_name']
        assert svc['chat_compat_enabled'] is True, svc['service_name']
        assert svc['chat_compat_strategy'] == 'flat_messages', svc[
            'service_name'
        ]

    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    by_alias = {
        m['model_name']: m['litellm_params'] for m in cfg_doc['model_list']
    }

    for alias, served in (
        ('eleutherai/pythia-6.9b', 'eleutherai/pythia-6.9b'),
        ('eleutherai/pythia-2.8b-v0', 'eleutherai/pythia-2.8b-v0'),
    ):
        params = by_alias[alias]
        # Still a text-completion upstream — chat_compat does not change that.
        assert params['model'] == f'text-completion-openai/{served}'
        # Flat-messages prompt template fields are present and use newline-only post_message.
        assert params['initial_prompt_value'] == ''
        assert params['final_prompt_value'] == ''
        assert set(params['roles']) == {'system', 'user', 'assistant'}
        for role, role_cfg in params['roles'].items():
            assert role_cfg['pre_message'] == '', (alias, role)
            assert role_cfg['post_message'] == '\n', (alias, role)
        # No reasoning passthrough on completions entries.
        assert 'merge_reasoning_content_in_choices' not in params


def test_pythia_inspect_mmlu_compat_does_not_render_chat_template_flags_in_compose(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-inspect-mmlu-compat', inventory='2x96'
    )
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert '--chat-template' not in compose_text
    assert '--chat-template-content-format' not in compose_text


def test_no_compose_service_renders_chat_template_flags(tmp_path: Path) -> None:
    """chat_compat is LiteLLM-only; Compose must never render --chat-template."""
    for profile_name, inventory in (
        ('pythia-inspect-mmlu-compat', '2x96'),
        ('pythia-qwen3.6-mixed-4x96', '4x96'),
        ('qwen3.6-35b-a3b-dual-tp2-4x96', '4x96'),
        ('helm-pythia-6.9b', '1x96'),
        ('qwen2-5-7b-instruct-turbo-default', '1x96'),
    ):
        deployment = _deployment(tmp_path, profile_name, inventory=inventory)
        render_compose_artifacts({'deployment': deployment})
        compose_text = (
            tmp_path / 'generated' / 'docker-compose.yml'
        ).read_text()
        assert '--chat-template' not in compose_text, profile_name


def test_mixed_profile_pythia_services_get_chat_compat_qwen_does_not(
    tmp_path: Path,
) -> None:
    """In `pythia-qwen3.6-mixed-4x96`, only the two Pythia LiteLLM entries
    should carry the flat-messages prompt-template fields. The Qwen chat
    entry must remain a plain `openai/...` route with no `roles` block."""
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})

    services = {s['service_name']: s for s in deployment['services']}
    assert services['pythia-69b']['chat_compat_enabled'] is True
    assert services['pythia-69b']['chat_compat_strategy'] == 'flat_messages'
    assert services['pythia-28b']['chat_compat_enabled'] is True
    assert services['pythia-28b']['chat_compat_strategy'] == 'flat_messages'
    assert services['qwen36-35b']['chat_compat_enabled'] is False
    # vLLM commands are unchanged: no --chat-template anywhere.
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert '--chat-template' not in compose_text

    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    by_alias = {
        m['model_name']: m['litellm_params'] for m in cfg_doc['model_list']
    }

    for alias, served in (
        ('eleutherai/pythia-6.9b', 'eleutherai/pythia-6.9b'),
        ('eleutherai/pythia-2.8b-v0', 'eleutherai/pythia-2.8b-v0'),
    ):
        params = by_alias[alias]
        assert params['model'] == f'text-completion-openai/{served}'
        assert params['initial_prompt_value'] == ''
        assert params['final_prompt_value'] == ''
        assert set(params['roles']) == {'system', 'user', 'assistant'}
        for role_cfg in params['roles'].values():
            assert role_cfg['pre_message'] == ''
            assert role_cfg['post_message'] == '\n'

    qwen_params = by_alias['qwen3.6-35b-a3b']
    assert qwen_params['model'] == 'openai/qwen3.6-35b-a3b'
    assert 'roles' not in qwen_params
    assert 'initial_prompt_value' not in qwen_params
    assert 'final_prompt_value' not in qwen_params


def test_qwen_chat_profiles_are_unaffected_by_chat_compat(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'qwen2-5-7b-instruct-turbo-default', inventory='1x96'
    )
    render_compose_artifacts({'deployment': deployment})
    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    for entry in cfg_doc['model_list']:
        params = entry['litellm_params']
        assert params['model'].startswith('openai/')
        assert 'roles' not in params
        assert 'initial_prompt_value' not in params
        assert 'final_prompt_value' not in params
    assert deployment['services'][0]['chat_compat_enabled'] is False


def test_litellm_config_for_chat_compat_profile_is_valid_yaml(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-inspect-mmlu-compat', inventory='2x96'
    )
    render_compose_artifacts({'deployment': deployment})
    text = (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    cfg_doc = yaml.safe_load(text)
    assert isinstance(cfg_doc, dict)
    assert 'model_list' in cfg_doc
    assert all('litellm_params' in m for m in cfg_doc['model_list'])


def test_no_profile_renders_unsupported_enable_reasoning_flag(
    tmp_path: Path,
) -> None:
    """Regression: vLLM CLI rejects --enable-reasoning."""
    for profile_name, inventory in (
        ('pythia-qwen3.6-mixed-4x96', '4x96'),
        ('qwen3.6-35b-a3b-dual-tp2-4x96', '4x96'),
        ('helm-pythia-6.9b', '1x96'),
        ('qwen2-5-7b-instruct-turbo-default', '1x96'),
    ):
        deployment = _deployment(tmp_path, profile_name, inventory=inventory)
        render_compose_artifacts({'deployment': deployment})
        compose_text = (
            tmp_path / 'generated' / 'docker-compose.yml'
        ).read_text()
        assert '--enable-reasoning' not in compose_text, profile_name


def test_litellm_keeps_merge_reasoning_for_reasoning_models(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    qwen_entry = next(
        m for m in cfg_doc['model_list'] if m['model_name'] == 'qwen3.6-35b-a3b'
    )
    assert (
        qwen_entry['litellm_params']['merge_reasoning_content_in_choices']
        is True
    )


def test_pythia_qwen3_6_mixed_profile_resolves_on_4x96(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    services = {s['service_name']: s for s in deployment['services']}
    assert set(services) == {'qwen36-35b', 'pythia-69b', 'pythia-28b'}

    qwen = services['qwen36-35b']
    assert qwen['protocol_mode'] == 'chat'
    assert qwen['gpu_indices'] == [0, 1]
    assert qwen['tensor_parallel_size'] == 2
    assert qwen['reasoning_enabled'] is True
    assert qwen['reasoning_parser'] == 'qwen3'

    p69 = services['pythia-69b']
    assert p69['protocol_mode'] == 'completions'
    assert p69['gpu_indices'] == [2]
    assert p69['reasoning_enabled'] is False

    p28 = services['pythia-28b']
    assert p28['protocol_mode'] == 'completions'
    assert p28['gpu_indices'] == [3]


def test_pythia_qwen3_6_mixed_profile_renders_compose_and_litellm(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'pythia-qwen3.6-mixed-4x96', inventory='4x96'
    )
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)

    # Three vLLM services and two Postgres services.
    infer_stacks = [name for name in blocks if name.startswith('vllm-')]
    assert sorted(infer_stacks) == [
        'vllm-pythia-28b',
        'vllm-pythia-69b',
        'vllm-qwen36-35b',
    ]
    assert 'postgres-open-webui' in blocks
    assert 'postgres-litellm' in blocks
    assert 'postgres-init' not in blocks

    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    by_alias = {
        m['model_name']: m['litellm_params']['model']
        for m in cfg_doc['model_list']
    }
    assert by_alias['qwen3.6-35b-a3b'].startswith('openai/')
    assert (
        by_alias['eleutherai/pythia-6.9b']
        == 'text-completion-openai/eleutherai/pythia-6.9b'
    )
    assert (
        by_alias['eleutherai/pythia-2.8b-v0']
        == 'text-completion-openai/eleutherai/pythia-2.8b-v0'
    )


def test_qwen3_6_dual_tp2_4x96_profile_resolves_and_renders(
    tmp_path: Path,
) -> None:
    deployment = _deployment(
        tmp_path, 'qwen3.6-35b-a3b-dual-tp2-4x96', inventory='4x96'
    )
    services = {s['service_name']: s for s in deployment['services']}
    assert set(services) == {'qwen36-35b-gpu01', 'qwen36-35b-gpu23'}

    gpu01 = services['qwen36-35b-gpu01']
    gpu23 = services['qwen36-35b-gpu23']
    assert gpu01['gpu_indices'] == [0, 1]
    assert gpu23['gpu_indices'] == [2, 3]
    assert (
        gpu01['tensor_parallel_size'] == 2
        and gpu23['tensor_parallel_size'] == 2
    )
    assert gpu01['protocol_mode'] == 'chat' and gpu23['protocol_mode'] == 'chat'
    assert (
        gpu01['reasoning_enabled'] is True
        and gpu01['reasoning_parser'] == 'qwen3'
    )

    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    assert 'vllm-qwen36-35b-gpu01' in blocks
    assert 'vllm-qwen36-35b-gpu23' in blocks
    for name in ('vllm-qwen36-35b-gpu01', 'vllm-qwen36-35b-gpu23'):
        assert '--reasoning-parser' in blocks[name]
        assert '"qwen3"' in blocks[name]
        assert '--language-model-only' in blocks[name]
        assert '--enable-reasoning' not in blocks[name]

    cfg_doc = yaml.safe_load(
        (tmp_path / 'state' / 'runtime' / 'litellm_config.yaml').read_text()
    )
    by_alias = {
        m['model_name']: m['litellm_params']['model']
        for m in cfg_doc['model_list']
    }
    assert by_alias['qwen3.6-35b-a3b-gpu01'].startswith('openai/')
    assert by_alias['qwen3.6-35b-a3b-gpu23'].startswith('openai/')


def test_kubeai_protocol_validation_applies(tmp_path: Path) -> None:
    from infer_stack.config import initial_config

    cfg = initial_config()
    cfg['backend'] = 'kubeai'
    cfg['resource_profiles'] = {
        'gpu-single-default': {
            'node_selector': {'nvidia.com/gpu.product': 'X'},
            'requests': {'nvidia.com/gpu': 1},
            'limits': {'nvidia.com/gpu': 1},
        },
    }
    cfg['profiles'] = {
        'broken-pythia-chat-kubeai': {
            'description': 'Synthetic invalid profile: chat on a completions-only model.',
            'base_model': 'pythia-1b-v0',
            'public_name': 'broken-pythia-chat-kubeai',
            'protocol_mode': 'chat',
            'resource_profile': 'gpu-single-default:1',
        }
    }
    deployment = resolve(
        cfg,
        inventory=simulate_inventory('1x96'),
        profile_name='broken-pythia-chat-kubeai',
    )
    report = validate_resolved(deployment)
    assert report['ok'] is False
    joined = ' | '.join(report['errors'])
    assert 'broken-pythia-chat-kubeai' in joined
    assert 'completions' in joined


def test_load_profile_contract_uses_public_loader_for_gpt_oss_variants(
    tmp_path: Path,
) -> None:
    _write_root_config(tmp_path)
    completions = load_profile_contract(
        'gpt-oss-20b-completions',
        simulate_hardware_spec='1x96',
    )
    chat = load_profile_contract(
        'gpt-oss-20b-chat',
        simulate_hardware_spec='1x96',
    )
    assert completions['services'][0]['protocol']['mode'] == 'completions'
    assert chat['services'][0]['protocol']['mode'] == 'chat'


def test_compose_infer_stack_persists_caches_and_uses_host_ipc(
    tmp_path: Path,
) -> None:
    """Warm-restart caches and shared memory must be wired into every vLLM service.

    - Hugging Face cache mount (existing behaviour) avoids re-downloading weights.
    - vLLM cache mount + ``VLLM_CACHE_ROOT`` env let torch.compile artifacts
      survive container restarts.
    - ``ipc: host`` is required by vLLM's Docker guidance to give the engine
      adequate shared memory for tensor-parallel and worker IPC.
    """
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    vllm_blocks = {
        name: body for name, body in blocks.items() if name.startswith('vllm-')
    }
    assert vllm_blocks, 'expected at least one rendered vLLM service'

    expected_hf_mount = f'{tmp_path}/state/hf-cache:/root/.cache/huggingface'
    expected_vllm_mount = f'{tmp_path}/state/vllm-cache:/root/.cache/vllm'
    for name, body in vllm_blocks.items():
        assert 'VLLM_CACHE_ROOT: /root/.cache/vllm' in body, name
        assert expected_hf_mount in body, name
        assert expected_vllm_mount in body, name
        assert 'ipc: host' in body, name


def test_compose_non_infer_stacks_do_not_get_vllm_cache_mount(
    tmp_path: Path,
) -> None:
    """The vLLM cache mount and host IPC must not leak into Postgres / litellm / open-webui."""
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    for name in (
        'postgres-open-webui',
        'postgres-litellm',
        'litellm',
        'open-webui',
    ):
        body = blocks[name]
        assert '/root/.cache/vllm' not in body, name
        assert 'VLLM_CACHE_ROOT' not in body, name
        assert 'ipc: host' not in body, name


def test_default_state_paths_include_vllm_cache(tmp_path: Path) -> None:
    from infer_stack.config import default_state_paths, normalized_state

    # data_root() is pinned to tmp_path by the autouse fixture.
    paths = default_state_paths()
    assert 'vllm_cache' in paths
    assert paths['vllm_cache'].endswith('/vllm-cache')
    assert paths['vllm_cache'].startswith(str(tmp_path))

    normalized = normalized_state({'vllm_cache': 'custom/vllm'})
    assert normalized['vllm_cache'] == str(tmp_path / 'custom' / 'vllm')


def test_allowed_gpus_filters_inventory_to_real_indices(tmp_path: Path) -> None:
    """``effective_inventory(--allowed-gpus 1,3)`` against a 4-GPU sim keeps real indices.

    ``placement.strategy: first_fit`` then picks from {1, 3}; the rendered
    compose stack pins ``device_ids`` to those physical indices (no
    renumbering).
    """
    from infer_stack.cli import effective_inventory

    inv = effective_inventory(
        {'simulate_hardware': '4x24', 'allowed_gpus': '1,3'}
    )
    assert inv is not None
    assert inv['gpu_count'] == 2
    assert [g['index'] for g in inv['gpus']] == [1, 3]

    cfg = _cfg(tmp_path)
    deployment = resolve(
        cfg,
        inventory=inv,
        profile_name='test-multi-gpu',
    )
    svc = deployment['services'][0]
    assert svc['gpu_indices'] == [1, 3]
    assert svc['tensor_parallel_size'] == 2

    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    assert 'device_ids: ["1", "3"]' in compose_text


def test_allowed_gpus_too_few_for_profile_surfaces_placement_error(
    tmp_path: Path,
) -> None:
    """Asking for 2-GPU placement when only 1 GPU is allowed must fail validation cleanly."""
    from infer_stack.cli import effective_inventory

    inv = effective_inventory(
        {'simulate_hardware': '4x24', 'allowed_gpus': '1'}
    )
    assert inv is not None and inv['gpu_count'] == 1

    cfg = _cfg(tmp_path)
    deployment = resolve(cfg, inventory=inv, profile_name='test-multi-gpu')
    report = validate_resolved(deployment)
    assert report['ok'] is False
    joined = ' | '.join(report['errors'])
    assert 'need 2 GPUs but only 1 available' in joined


def test_output_generated_dir_redirects_compose_render(tmp_path: Path) -> None:
    """Configured output.generated_dir takes precedence over the default."""
    cfg = _cfg(tmp_path)
    cfg['output'] = {'generated_dir': str(tmp_path / 'custom-out')}
    deployment = resolve(
        cfg,
        inventory=simulate_inventory('1x96'),
        profile_name='qwen2-5-7b-instruct-turbo-default',
    )
    render_compose_artifacts({'deployment': deployment})
    assert (tmp_path / 'custom-out' / 'docker-compose.yml').exists()
    assert (tmp_path / 'custom-out' / '.env').exists()
    # The repo-local generated/ is not touched.
    assert not (tmp_path / 'generated').exists()


def test_output_generated_dir_redirects_kubeai_render(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, backend='kubeai')
    cfg['output'] = {'generated_dir': str(tmp_path / 'custom-out')}
    deployment = resolve(
        cfg,
        inventory=simulate_inventory('1x96'),
        profile_name='qwen2-5-7b-instruct-turbo-default',
    )
    render_kubeai_artifacts({'deployment': deployment})
    assert (tmp_path / 'custom-out' / 'kubeai' / 'models.yaml').exists()
    assert (tmp_path / 'custom-out' / 'kubeai' / 'kubeai-values.yaml').exists()
    assert not (tmp_path / 'generated').exists()


def test_openwebui_auth_defaults_to_disabled_in_compose_render(
    tmp_path: Path,
) -> None:
    """Auth-less Open WebUI is the out-of-the-box experience for compose."""
    deployment = _deployment(tmp_path, 'gpt-oss-20b-chat')
    assert deployment['open_webui']['auth'] is False
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    assert 'WEBUI_AUTH: "False"' in blocks['open-webui']


def test_openwebui_auth_can_be_enabled_via_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg['open_webui'] = {'auth': True}
    deployment = resolve(
        cfg,
        inventory=simulate_inventory('1x96'),
        profile_name='qwen2-5-7b-instruct-turbo-default',
    )
    assert deployment['open_webui']['auth'] is True
    render_compose_artifacts({'deployment': deployment})
    compose_text = (tmp_path / 'generated' / 'docker-compose.yml').read_text()
    blocks = _split_compose_blocks(compose_text)
    assert 'WEBUI_AUTH: "True"' in blocks['open-webui']


def test_normalized_output_anchors_relative_paths_on_data_root(
    tmp_path: Path,
) -> None:
    from infer_stack.config import normalized_output

    # data_root() is pinned to tmp_path by the autouse fixture.
    normalized = normalized_output({'generated_dir': 'out'})
    assert normalized['generated_dir'] == str(tmp_path / 'out')

    normalized = normalized_output({'generated_dir': '/abs/path'})
    assert normalized['generated_dir'] == '/abs/path'


def test_catalog_model_path_env_and_directory_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infer_stack.config import catalog_files, merged_catalogs

    save_yaml(
        tmp_path / 'models.yaml',
        {
            'vllm_models': {
                'base-only': {'hf_model_id': 'example/base-only'},
                'shared': {'hf_model_id': 'example/base-shared'},
            },
            'profiles': {
                'base-profile': {'description': 'from base models.yaml'}
            },
        },
    )
    config_path_dir = tmp_path / 'repo-catalogs'
    config_path_dir.mkdir()
    save_yaml(
        config_path_dir / 'incubilate.models.yaml',
        {
            'vllm_models': {
                'config-only': {'hf_model_id': 'example/config-only'},
                'shared': {'hf_model_id': 'example/config-shared'},
            },
            'profiles': {
                'config-profile': {'description': 'from config model_path'}
            },
        },
    )
    # Directory expansion is intentionally narrow; unrelated YAML should not be
    # treated as a catalog just because it lives beside catalog overlays.
    save_yaml(
        config_path_dir / 'config.yaml',
        {'vllm_models': {'should-not-load': {'hf_model_id': 'bad/bad'}}},
    )

    env_path_dir = tmp_path / 'env-catalogs'
    env_path_dir.mkdir()
    save_yaml(
        env_path_dir / 'experiment.profiles.yaml',
        {
            'vllm_models': {
                'env-only': {'hf_model_id': 'example/env-only'},
                'shared': {'hf_model_id': 'example/env-shared'},
            },
            'profiles': {
                'env-profile': {'description': 'from INFER_STACK_MODEL_PATH'}
            },
        },
    )

    cfg = initial_config()
    cfg['catalog']['model_path'] = 'repo-catalogs'
    monkeypatch.setenv('INFER_STACK_MODEL_PATH', str(env_path_dir))

    catalog_paths = [path.name for path in catalog_files(cfg)]
    assert catalog_paths == [
        'models.yaml',
        'incubilate.models.yaml',
        'experiment.profiles.yaml',
    ]

    cats = merged_catalogs(cfg)
    assert 'base-only' in cats['vllm_models']
    assert 'config-only' in cats['vllm_models']
    assert 'env-only' in cats['vllm_models']
    assert 'should-not-load' not in cats['vllm_models']
    assert cats['vllm_models']['shared']['hf_model_id'] == 'example/env-shared'
    assert 'base-profile' in cats['profiles']
    assert 'config-profile' in cats['profiles']
    assert 'env-profile' in cats['profiles']
