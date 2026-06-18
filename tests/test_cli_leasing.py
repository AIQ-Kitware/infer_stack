"""Tests for the leasing CLI verbs (dry-run / NullBackend)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from infer_stack.cli.commands_leasing import (
    AcquireCLI,
    LeasesCLI,
    ReleaseCLI,
    RenewCLI,
    RunCLI,
    ServeCLI,
)

CATALOG = {
    'models': {
        'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'},
        'rr': {'source': 'hf://BAAI/bge-reranker-base'},
    },
    'endpoints': {
        'qwen-coder': {
            'engine': 'vllm',
            'model': 'qc',
            'runtime': {'max_model_len': 32768},
        },
        'reranker': {'engine': 'vllm', 'model': 'rr'},
    },
    'bundles': {'both': ['qwen-coder', 'reranker']},
}


@pytest.fixture
def env(tmp_path):
    cat = tmp_path / 'catalog.yaml'
    cat.write_text(yaml.safe_dump(CATALOG))
    return SimpleNamespace(
        cat=str(cat), db=str(tmp_path / 'ledger.db'), tmp=tmp_path
    )


def _base(env):
    return ['--ledger', env.db, '--catalog', env.cat]


def _leases_json(env, capsys):
    capsys.readouterr()
    LeasesCLI.main(argv=['--ledger', env.db, '--json'])
    return json.loads(capsys.readouterr().out)


def test_acquire_writes_env_file(env):
    envf = env.tmp / 'is.env'
    rc = AcquireCLI.main(
        argv=['qwen-coder', *_base(env), '--env-file', str(envf),
              '--base-url', 'http://x:1/v1']
    )
    assert rc == 0
    text = envf.read_text()
    assert 'INFER_STACK_SESSION_ID=sess-' in text
    assert 'OPENAI_BASE_URL=http://x:1/v1' in text
    assert 'INFER_STACK_ENDPOINT_QWEN_CODER=qwen-coder' in text
    assert 'INFER_STACK_API_KEY_ENV=LITELLM_MASTER_KEY' in text


def test_acquire_coalesces_demand(env, capsys):
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'alice'])
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'bob'])
    data = _leases_json(env, capsys)
    assert len(data['leases']) == 2
    assert len(data['groups']) == 1
    assert data['groups'][0]['demand'] == 2


def test_dedicated_makes_separate_group(env, capsys):
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    AcquireCLI.main(argv=['qwen-coder', '--dedicated', *_base(env)])
    data = _leases_json(env, capsys)
    assert len(data['groups']) == 2


def test_bundle_acquire_expands(env, capsys):
    capsys.readouterr()
    AcquireCLI.main(argv=['both', *_base(env), '--json'])
    data = json.loads(capsys.readouterr().out)
    assert set(data['descriptor']['endpoints']) == {'qwen-coder', 'reranker'}


def test_release_via_env_file(env, capsys):
    envf = env.tmp / 'is.env'
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--env-file', str(envf)])
    rc = ReleaseCLI.main(argv=['--ledger', env.db, '--env-file', str(envf)])
    assert rc == 0
    data = _leases_json(env, capsys)
    assert data['leases'][0]['state'] == 'released'
    # default reclaim is keep-warm, so the group is idle but not torn down
    assert data['groups'][0]['state'] == 'idle'
    assert data['groups'][0]['demand'] == 0


def test_serve_is_standing_lease(env, capsys):
    ServeCLI.main(argv=['qwen-coder', *_base(env)])
    data = _leases_json(env, capsys)
    assert data['leases'][0]['owner'] == 'manual'
    assert data['leases'][0]['expires_at'] is None


def test_renew_extends(env):
    envf = env.tmp / 'is.env'
    AcquireCLI.main(
        argv=['qwen-coder', *_base(env), '--ttl', '1h', '--env-file', str(envf)]
    )
    rc = RenewCLI.main(
        argv=['--ledger', env.db, '--env-file', str(envf), '--ttl', '3h']
    )
    assert rc == 0


def test_run_propagates_env_and_exit(env):
    out = env.tmp / 'out.txt'
    code = (
        'import os, pathlib; '
        f'pathlib.Path({str(out)!r}).write_text('
        "os.environ.get('OPENAI_BASE_URL', ''))"
    )
    rc = RunCLI.main(
        argv=['--endpoint', 'qwen-coder', *_base(env),
              '--base-url', 'http://run:9/v1', '--', sys.executable, '-c', code]
    )
    assert rc == 0
    assert out.read_text() == 'http://run:9/v1'


def test_run_returns_child_exit_code(env):
    rc = RunCLI.main(
        argv=['--endpoint', 'qwen-coder', *_base(env), '--',
              sys.executable, '-c', 'import sys; sys.exit(3)']
    )
    assert rc == 3


def test_run_releases_on_exit(env, capsys):
    RunCLI.main(
        argv=['--endpoint', 'qwen-coder', *_base(env), '--',
              sys.executable, '-c', 'pass']
    )
    data = _leases_json(env, capsys)
    assert data['leases'][0]['state'] == 'released'


def test_unimplemented_backend_errors(env):
    with pytest.raises(SystemExit):
        AcquireCLI.main(argv=['qwen-coder', '--backend', 'kubeai', *_base(env)])


def test_unknown_endpoint_errors(env):
    with pytest.raises(SystemExit):
        AcquireCLI.main(argv=['ghost', *_base(env)])


def test_missing_catalog_errors(env):
    with pytest.raises(SystemExit):
        AcquireCLI.main(
            argv=['qwen-coder', '--ledger', env.db, '--catalog',
                  str(env.tmp / 'nope.yaml')]
        )


def test_secret_set_get_list_roundtrip(tmp_path, monkeypatch):
    import contextlib
    import io

    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    from infer_stack.cli.commands_leasing import (
        SecretGetCLI,
        SecretsCLI,
        SecretSetCLI,
    )

    # set works before any serve (creates the managed .env), merges keys
    SecretSetCLI.main(argv=['HF_TOKEN=hf_demo'])
    SecretSetCLI.main(argv=['OTHER=x'])
    env_file = tmp_path / 'leasing' / 'compose' / '.env'
    text = env_file.read_text()
    assert 'HF_TOKEN=hf_demo' in text and 'OTHER=x' in text   # both preserved

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SecretGetCLI.main(argv=['HF_TOKEN'])
        SecretsCLI.main(argv=['HF_TOKEN'])           # alias
    assert buf.getvalue().split() == ['hf_demo', 'hf_demo']


def test_secret_get_missing_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    from infer_stack.cli.commands_leasing import SecretGetCLI

    with pytest.raises(SystemExit):
        SecretGetCLI.main(argv=['NOPE'])             # no .env yet


def test_include_display_gpus_flag_controls_skip_display(env, monkeypatch):
    """--include-display-gpus must flip the compose backend's skip_display."""
    import infer_stack.cli.commands_leasing as mod

    seen = {}

    class FakeCompose:
        def __init__(self, **kw):
            seen.update(kw)

    import infer_stack.hardware as hw

    monkeypatch.setattr(mod, 'ComposeBackend', FakeCompose)
    # _make_backend does `from ..hardware import detect_inventory` lazily.
    monkeypatch.setattr(hw, 'detect_inventory', lambda: {})

    cfg = AcquireCLI.cli(
        argv=['qwen-coder', '--backend', 'compose', '--include-display-gpus'],
        strict=False,
    )
    mod._make_backend(cfg)
    assert seen['skip_display'] is False

    seen.clear()
    cfg = AcquireCLI.cli(argv=['qwen-coder', '--backend', 'compose'],
                         strict=False)
    mod._make_backend(cfg)
    assert seen['skip_display'] is True
