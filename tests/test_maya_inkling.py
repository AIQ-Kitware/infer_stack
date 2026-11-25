"""Managed Maya profile: one GPU, upstream alias/context, and gateway contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'maya_inkling_launcher', ROOT / 'infer_stack/runtime_images/maya_inkling/launcher.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize('variant, context, draft', [
    ('', 196608, True), ('-long', 262144, True), ('-huge', 344064, False),
])
def test_maya_managed_profile(variant, context, draft):
    from infer_stack.leasing import Catalog, render_compose
    from infer_stack.leasing.compose import vllm_service_name
    from infer_stack.leasing.gateway import compose_catalog_route
    from infer_stack.leasing.models import Deployment, DeploymentState
    from infer_stack.leasing.placement import required_gpu_count
    from infer_stack.leasing.suggest import suggest_catalog

    base = 'inkling-2-mini-maya'
    name = base + variant
    inv = {'gpu_count': 2, 'gpus': [
        {'index': i, 'name': 'RTX 3090', 'memory_gib': 24, 'compute_cap': 8.6}
        for i in range(2)]}
    assert name not in suggest_catalog(inv)['endpoints']
    frag = suggest_catalog(inv, include=[base])
    req = Catalog.from_dict(frag).resolve_endpoint(name).to_request()
    assert req.spec['hf_model_id'] == launcher.MODEL_REPO
    assert req.spec['placement'] == {'min_vram_gib': 23.9}
    assert req.spec['runtime']['max_model_len'] == context
    dep = Deployment('grp-maya', 'ck-maya', 'vllm', 'shared-compatible',
                     req.capacity, req.spec, {name: req.served},
                     DeploymentState.LIVE, 0, 0)
    assert required_gpu_count(dep) == 1
    rc = render_compose([dep], {'grp-maya': [1]}, images={
        'vllm': 'vllm/test', 'ollama': 'ollama/test', 'litellm': 'litellm/test'},
        ports={'ollama': 11434}, state={'hf_cache': '/cache/hf',
        'ollama': '/cache/ollama', 'runtime': '/cache/runtime'})
    svc = rc.compose['services'][vllm_service_name(dep)]
    from infer_stack.runtime_images import MAYA_IMAGE
    assert svc['image'] == MAYA_IMAGE
    assert svc['command'] == ['serve']
    assert svc['volumes'] == ['/cache/runtime/maya/inkling-2-mini/models:/models']
    devices = svc['deploy']['resources']['reservations']['devices']
    assert devices[0]['device_ids'] == ['1']
    assert svc['healthcheck']['test'][-1] == 'http://localhost:8000/health'
    assert launcher.settings(svc['environment'])[:4] == (draft, context, 8000, name)
    route = compose_catalog_route(name, req).entry()
    assert route['litellm_params']['model'] == 'openai/' + name
    assert route['litellm_params']['allowed_openai_params'] == ['reasoning_effort']
    assert route['model_info']['max_input_tokens'] == context
    assert route['model_info']['reasoning_effort_levels'] == [
        'none', 'low', 'medium', 'high', 'xhigh']


def test_maya_suggestion_gpu_gate():
    from infer_stack.leasing.suggest import suggest_catalog

    base = 'inkling-2-mini-maya'
    for vram, cap in [(16, 8.6), (24, 7.5), (24, 8.0), (24, 8.9)]:
        inv = {'gpu_count': 1, 'gpus': [{'index': 0, 'name': 'GPU',
               'memory_gib': vram, 'compute_cap': cap}]}
        assert not any(name.startswith(base) for name in
                       suggest_catalog(inv, include=[base])['endpoints'])


@pytest.mark.parametrize('draft, context', [(True, 196608), (False, 344064)])
def test_maya_launcher_config(tmp_path, draft, context):
    root = tmp_path / 'maya'
    models = tmp_path / 'models'
    configs = root / 'docs/inkling2-mini'
    configs.mkdir(parents=True)
    # Model the upstream config shapes, including the tuning and vision fields.
    upstream = {'exe': 'build/maya-ik2', 'args': ['--model', 'old',
                '--max-context', '262144'], 'env': {'IK2_KV': 'fp8'},
                'tokenizer': 'ik2/tokenizer', 'model_name': 'inkling-2-mini'}
    (configs / 'ik2-mini-text.json').write_text(json.dumps(upstream))
    upstream['vision'] = {'exe': 'build-vision/bin/maya-vision'}
    upstream['env']['IK2_DRAFT_CTX'] = '65536'
    (configs / 'ik2-mini.json').write_text(json.dumps(upstream))
    name = 'custom-served-alias'
    cfg = launcher.make_config(root, models, draft=draft, context=context, name=name)
    assert cfg['args'][:4] == ['--model', str(models / launcher.MODEL_FILE),
                              '--max-context', str(context)]
    assert cfg['args'][4:] == (['--draft', str(models / launcher.DRAFT_FILE)] if draft else [])
    assert cfg['env']['IK2_KV'] == 'fp8'
    assert ('IK2_DRAFT_CTX' in cfg['env']) == draft
    assert 'vision' not in cfg
    assert cfg['model_name'] == name
    assert cfg['reasoning_effort'] == 'high'
    assert cfg['model_management'] is False
    assert cfg['idle_unload_s'] == 0
    assert cfg['tokenizer'] == str(root / 'ik2/tokenizer')


@pytest.mark.parametrize('env', [
    {'MAYA_DSPARK': 'maybe'}, {'MAYA_MAX_CONTEXT': '344064'},
    {'MAYA_DSPARK': '0', 'MAYA_MAX_CONTEXT': '344065'},
    {'MAYA_MAX_CONTEXT': '0'}, {'PORT': '0'}, {'MAYA_MODEL_NAME': ''},
])
def test_maya_rejects_bad_settings(env):
    with pytest.raises(ValueError):
        launcher.settings(env)


def test_maya_prepare_reuses_local_files_without_hf(tmp_path, monkeypatch):
    monkeypatch.setenv('MAYA_MODEL_DIR', str(tmp_path))
    monkeypatch.setenv('MAYA_DSPARK', '1')
    for name in [launcher.MODEL_FILE, launcher.DRAFT_FILE]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test fixture')
    monkeypatch.setattr(launcher, 'validate_weights', lambda path: None)
    # No huggingface_hub dependency or network is needed for an existing cache.
    launcher.main(['prepare'])


@pytest.mark.parametrize('metadata', [
    {'general.architecture': 'inkling2_loop', 'inkling2_loop.block_count': 24},
    {'general.architecture': 'inkling2', 'inkling2.block_count': 24},
    {},
])
def test_maya_rejects_incompatible_gguf_metadata(metadata):
    with pytest.raises(ValueError, match='incompatible GGUF'):
        launcher.require_maya_layout(metadata, 'bad.gguf')
    launcher.require_maya_layout({'general.architecture': 'inkling2',
                                  'inkling2.block_count': 48}, 'good.gguf')


def test_maya_download_uses_pins_and_separate_cache(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs['local_dir']) / kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
        return str(path)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(launcher, 'validate_weights', lambda path: None)
    monkeypatch.setenv('MAYA_MODEL_DIR', str(tmp_path))
    monkeypatch.setenv('MAYA_DSPARK', '1')
    incompatible = tmp_path / 'Inkling-2-Mini-UD-Q4_K_XL.gguf'
    incompatible.write_bytes(b'mainline layout')
    launcher.main(['prepare'])
    assert calls == [dict(repo_id=launcher.MODEL_REPO, filename=launcher.MODEL_FILE,
                          revision=launcher.MODEL_REV, local_dir=tmp_path),
                     dict(repo_id=launcher.DRAFT_REPO, filename=launcher.DRAFT_FILE,
                          revision=launcher.DRAFT_REV, local_dir=tmp_path)]
    assert launcher.MODEL_FILE.startswith('Old_Support/')
    assert launcher.MODEL_FILE.endswith('.gguf_file')
    assert incompatible.read_bytes() == b'mainline layout'
    launcher.main(['prepare'])
    assert len(calls) == 2
