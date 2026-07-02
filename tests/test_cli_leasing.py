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
    _default_owner,
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


def test_load_catalog_without_flag_uses_default_path(tmp_path, monkeypatch):
    """Regression: release/gc/evict have no --catalog flag, so converging their
    compose backend must fall back to the DEFAULT-path catalog via getattr —
    `config.catalog` raised AttributeError and crashed `gc --backend compose`."""
    from infer_stack.cli import commands_leasing as cl
    from infer_stack.cli.commands_leasing import GcCLI, _load_catalog
    from infer_stack.leasing.catalog import Catalog

    cfg = GcCLI.cli(argv=['--backend', 'compose'], strict=False)
    assert not hasattr(cfg, 'catalog') or cfg.catalog is None

    # Default-path catalog present -> loaded (proves no AttributeError + fallback).
    (tmp_path / 'catalog.yaml').write_text(yaml.safe_dump(CATALOG))
    monkeypatch.setattr(cl, 'config_root', lambda: tmp_path)
    assert isinstance(_load_catalog(cfg), Catalog)

    # No default-path catalog -> SystemExit (caught by _make_backend), NOT AttributeError.
    monkeypatch.setattr(cl, 'config_root', lambda: tmp_path / 'nope')
    with pytest.raises(SystemExit):
        _load_catalog(cfg)


def test_release_gc_evict_accept_catalog_flag(tmp_path):
    """release/gc/evict take --catalog so pipelines can pass the superset
    explicitly (keeps the no-blip gateway without relying on the default path)."""
    from infer_stack.cli.commands_leasing import (
        EvictCLI, GcCLI, ReleaseCLI, _load_catalog,
    )
    from infer_stack.leasing.catalog import Catalog

    cat_path = tmp_path / 'catalog.yaml'
    cat_path.write_text(yaml.safe_dump(CATALOG))
    for cls in (GcCLI, ReleaseCLI, EvictCLI):
        cfg = cls.cli(
            argv=['--backend', 'compose', '--catalog', str(cat_path)],
            strict=False,
        )
        assert isinstance(_load_catalog(cfg), Catalog), cls.__name__


def test_acquire_writes_env_file(env):
    envf = env.tmp / 'is.env'
    rc = AcquireCLI.main(
        argv=['qwen-coder', *_base(env), '--env-file', str(envf),
              '--base-url', 'http://x:1/v1']
    )
    assert rc == 0
    text = envf.read_text()
    assert 'INFER_STACK_LEASE_ID=lease-' in text
    assert 'OPENAI_BASE_URL=http://x:1/v1' in text
    assert 'INFER_STACK_ENDPOINT_QWEN_CODER=qwen-coder' in text
    assert 'INFER_STACK_API_KEY_ENV=LITELLM_MASTER_KEY' in text


def test_acquire_timeout_releases_lease(env, capsys, monkeypatch):
    """A readiness timeout must not leave a phantom ACTIVE lease pinning a GPU:
    the controller releases it and ``acquire`` exits non-zero (--timeout 0 makes
    the never-ready wait fail on the first poll)."""
    from infer_stack.cli import commands_leasing as cl
    from infer_stack.leasing import MemoryBackend

    monkeypatch.setattr(
        cl, '_make_backend',
        lambda config, *, interactive=False: MemoryBackend(ready=False),
    )
    rc = AcquireCLI.main(argv=['qwen-coder', *_base(env), '--timeout', '0'])
    assert rc == 2
    assert 'released' in capsys.readouterr().out
    data = _leases_json(env, capsys)
    assert data['leases'][0]['state'] == 'released'


def test_acquire_timeout_skips_env_file(env, capsys, monkeypatch):
    """A released-on-timeout lease has no standing endpoint, so --env-file is not
    written (pointing a sourceable file at a torn-down lease would be a trap)."""
    from infer_stack.cli import commands_leasing as cl
    from infer_stack.leasing import MemoryBackend

    monkeypatch.setattr(
        cl, '_make_backend',
        lambda config, *, interactive=False: MemoryBackend(ready=False),
    )
    envf = env.tmp / 'is.env'
    rc = AcquireCLI.main(
        argv=['qwen-coder', *_base(env), '--timeout', '0',
              '--env-file', str(envf)]
    )
    assert rc == 2
    assert not envf.exists()


def test_acquire_coalesces_demand(env, capsys):
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'alice'])
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'bob'])
    data = _leases_json(env, capsys)
    assert len(data['leases']) == 2
    assert len(data['deployments']) == 1
    assert data['deployments'][0]['demand'] == 2


def test_dedicated_makes_separate_deployment(env, capsys):
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    AcquireCLI.main(argv=['qwen-coder', '--dedicated', *_base(env)])
    data = _leases_json(env, capsys)
    assert len(data['deployments']) == 2


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
    # default reclaim is keep-warm, so the deployment is idle but not torn down
    assert data['deployments'][0]['state'] == 'idle'
    assert data['deployments'][0]['demand'] == 0


def test_release_all_releases_every_active_lease(env, capsys):
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'a'])
    AcquireCLI.main(argv=['reranker', *_base(env), '--owner', 'b'])
    rc = ReleaseCLI.main(argv=['--ledger', env.db, '--all'])
    assert rc == 0
    data = _leases_json(env, capsys)
    assert data['leases']                                  # leases still listed
    assert all(le['state'] == 'released' for le in data['leases'])


def test_release_all_with_no_leases_is_friendly(env, capsys):
    rc = ReleaseCLI.main(argv=['--ledger', env.db, '--all'])
    assert rc == 0
    assert 'no active leases' in capsys.readouterr().out


def test_evict_idle_deployment_by_endpoint(env, capsys):
    from infer_stack.cli.commands_leasing import EvictCLI

    # keep-warm: releasing leaves the deployment resident (idle), not stopped.
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    ReleaseCLI.main(argv=['--ledger', env.db, '--all'])
    assert _leases_json(env, capsys)['deployments'][0]['state'] == 'idle'

    # evict by the served endpoint alias -> the deployment is stopped (GPU freed).
    rc = EvictCLI.main(argv=['qwen-coder', '--ledger', env.db])
    assert rc == 0
    assert _leases_json(env, capsys)['deployments'][0]['state'] == 'stopped'


def test_evict_requires_target_or_all(env):
    from infer_stack.cli.commands_leasing import EvictCLI

    with pytest.raises(SystemExit):
        EvictCLI.main(argv=['--ledger', env.db])      # neither name nor --all


def test_release_evict_tears_down_immediately(env, capsys):
    # --evict overrides keep-warm: the released deployment is stopped, not idle.
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    rc = ReleaseCLI.main(argv=['--ledger', env.db, '--all', '--evict'])
    assert rc == 0
    assert _leases_json(env, capsys)['deployments'][0]['state'] == 'stopped'


def test_acquire_without_ttl_is_standing_lease(env, capsys):
    # No --ttl -> an infinite (standing-service) lease owned by the caller.
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    data = _leases_json(env, capsys)
    assert data['leases'][0]['owner'] == _default_owner()
    assert data['leases'][0]['expires_at'] is None


def test_wait_after_parallel_no_wait_acquire(env, capsys):
    from infer_stack.cli.commands_leasing import WaitCLI

    # fan out: kick both off without blocking, then wait for both
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--no-wait'])
    AcquireCLI.main(argv=['reranker', *_base(env), '--no-wait'])
    capsys.readouterr()
    rc = WaitCLI.main(argv=['qwen-coder', 'reranker', '--ledger', env.db])
    assert rc == 0                                    # null backend: ready now
    assert 'ready' in capsys.readouterr().out


def test_wait_unknown_endpoint_errors(env):
    from infer_stack.cli.commands_leasing import WaitCLI

    with pytest.raises(SystemExit):
        WaitCLI.main(argv=['ghost', '--ledger', env.db])   # nothing serving it


def test_acquire_no_apply_stages_without_applying(env, capsys):
    capsys.readouterr()
    rc = AcquireCLI.main(argv=['qwen-coder', *_base(env), '--no-apply', '--json'])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data['applied'] is False                   # staged, not brought up
    assert data['lease_id'].startswith('lease-')
    # the lease is recorded (staged intent), so it shows up in `leases`...
    ls = _leases_json(env, capsys)
    assert ls['leases'][0]['state'] == 'active'
    # ...and can be discarded
    ReleaseCLI.main(argv=['--ledger', env.db, '--all'])
    assert _leases_json(env, capsys)['leases'][0]['state'] == 'released'


def test_release_evict_apply_accept_yes_flag(env, capsys):
    # release/evict/apply now gate the compose diff on a TTY; --yes skips it.
    # (null backend never prompts, so this is the plumbing + batched-release path.)
    from infer_stack.cli.commands_leasing import ApplyCLI, EvictCLI

    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--owner', 'a'])
    AcquireCLI.main(argv=['reranker', *_base(env), '--owner', 'b'])
    assert ReleaseCLI.main(argv=['--ledger', env.db, '--all', '--yes']) == 0
    assert all(le['state'] == 'released'
               for le in _leases_json(env, capsys)['leases'])
    assert EvictCLI.main(argv=['--ledger', env.db, '--all', '--yes']) == 0
    assert ApplyCLI.main(argv=['--ledger', env.db, '--yes']) == 0


def test_reverse_proxy_resolution_flag_bool_block(monkeypatch):
    import infer_stack.cli.commands_leasing as mod
    import infer_stack.paths as paths

    def resolve(argv, setting=None):
        monkeypatch.setattr(
            paths, 'get_setting',
            lambda k: setting if k == 'reverse_proxy' else None,
        )
        cfg = AcquireCLI.cli(argv=argv, strict=False)
        return mod._resolve_reverse_proxy(cfg)

    base = ['qwen-coder']
    assert resolve(base) == (False, 80, None)                  # default off
    assert resolve([*base, '--reverse-proxy']) == (True, 80, None)   # flag
    assert resolve(base, setting=True) == (True, 80, None)     # bool setting
    # a block carries port + BYO config; flag still overrides enabled
    block = {'enabled': True, 'port': 8080, 'config_path': '/etc/n.conf'}
    assert resolve(base, setting=block) == (True, 8080, '/etc/n.conf')
    assert resolve([*base, '--no-reverse-proxy'], setting=block) == (
        False, 8080, '/etc/n.conf')


def test_render_and_apply_are_lease_free(env, capsys):
    from infer_stack.cli.commands_leasing import ApplyCLI, RenderCLI

    # declare intent without applying, then the lease-free verbs operate on it
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--no-apply'])
    before = len(_leases_json(env, capsys)['leases'])
    assert RenderCLI.main(argv=['--ledger', env.db]) == 0   # render: no `up`
    assert ApplyCLI.main(argv=['--ledger', env.db]) == 0    # apply: brings up
    after = _leases_json(env, capsys)
    # neither render nor apply minted a lease — still exactly one
    assert len(after['leases']) == before == 1


def test_leases_reports_running_and_gpus(env, capsys):
    # NullBackend serves nothing, so a leased deployment is desired-live but not
    # actually running and has no GPU assignment — the view must say so.
    AcquireCLI.main(argv=['qwen-coder', *_base(env)])
    data = _leases_json(env, capsys)
    g = data['deployments'][0]
    assert g['state'] == 'live'        # desired (ledger)
    assert g['running'] is False       # actual (backend.observe)
    assert g['gpus'] is None           # no placement on the dry-run backend


def test_leases_gpu_and_running_labels():
    from infer_stack.cli.commands_leasing import _gpu_label, _running_label

    observed = {'g1'}
    assignments = {'g1': [0, 1], 'g2': [2], 'g3': []}
    assert _running_label('g1', observed) == 'running'
    assert _running_label('g2', observed) == '—'
    assert _gpu_label('g1', observed, assignments) == '0,1'   # on these GPUs
    assert _gpu_label('g2', observed, assignments) == '→2'    # slated, not up
    assert _gpu_label('g3', observed, assignments) == '→cpu'  # cpu, slated
    assert _gpu_label('gX', observed, assignments) == '-'     # unknown


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
    from infer_stack.cli.commands_leasing import _make_backend
    from types import SimpleNamespace
    with pytest.raises(SystemExit, match='not implemented'):
        _make_backend(SimpleNamespace(backend='nomad'))


def test_kubeai_backend_constructs(env, monkeypatch, tmp_path):
    """--backend kubeai builds a KubeaiBackend wired from settings."""
    from types import SimpleNamespace

    from infer_stack.backends.kubeai import KubeaiBackend
    from infer_stack.cli import commands_leasing as cl

    settings = {
        'kubeai_namespace': 'serving',
        'kubeai_base_url': 'http://10.0.0.5:8000/openai/v1',
        'kubeai_resource_profile': 'rtx-4090',
    }
    monkeypatch.setattr(cl, 'data_root', lambda: tmp_path)
    monkeypatch.setattr(
        'infer_stack.paths.get_setting', lambda key: settings.get(key)
    )
    be = cl._make_backend(SimpleNamespace(backend='kubeai', yes=True))
    assert isinstance(be, KubeaiBackend)
    assert be.namespace == 'serving'
    assert be.base_url == 'http://10.0.0.5:8000/openai/v1'
    assert be.default_resource_profile == 'rtx-4090'
    assert be.state_dir == tmp_path / 'leasing' / 'kubeai'


def test_unknown_endpoint_errors(env):
    with pytest.raises(SystemExit):
        AcquireCLI.main(argv=['ghost', *_base(env)])


def test_missing_catalog_errors(env):
    with pytest.raises(SystemExit):
        AcquireCLI.main(
            argv=['qwen-coder', '--ledger', env.db, '--catalog',
                  str(env.tmp / 'nope.yaml')]
        )


def test_env_set_read_list_roundtrip(tmp_path, monkeypatch):
    """`env` does it all: set KEY=VALUE, read KEY, --export, path."""
    import contextlib
    import io

    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    from infer_stack.cli.commands_leasing import EnvCLI

    # `env KEY=VALUE` works before any serve (creates the .env), merges keys
    EnvCLI.main(argv=['HF_TOKEN=hf_demo'])
    EnvCLI.main(argv=['OTHER=x'])
    env_file = tmp_path / 'leasing' / 'compose' / '.env'
    text = env_file.read_text()
    assert 'HF_TOKEN=hf_demo' in text and 'OTHER=x' in text   # both preserved

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        EnvCLI.main(argv=['HF_TOKEN'])               # `env KEY` prints the value
    assert buf.getvalue().strip() == 'hf_demo'

    # `--export` dumps every entry as shell-sourceable export lines
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        EnvCLI.main(argv=['--export'])
    out = buf.getvalue()
    assert 'export HF_TOKEN=hf_demo' in out and 'export OTHER=x' in out


def test_env_prints_path_first(tmp_path, monkeypatch):
    import contextlib
    import io

    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    from infer_stack.cli.commands_leasing import EnvCLI

    # `env` with no args prints the .env path even before it exists.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        EnvCLI.main(argv=[])
    assert buf.getvalue().strip() == str(
        tmp_path / 'leasing' / 'compose' / '.env'
    )


def test_env_read_missing_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    from infer_stack.cli.commands_leasing import EnvCLI

    with pytest.raises(SystemExit):
        EnvCLI.main(argv=['NOPE'])                    # no .env yet


def test_skip_display_gpus_default_off_flag_and_setting(env, monkeypatch):
    """Display GPUs are used by default; --skip-display-gpus / the setting opt in."""
    import infer_stack.cli.commands_leasing as mod
    import infer_stack.hardware as hw
    import infer_stack.paths as paths

    seen = {}

    class FakeCompose:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(mod, 'ComposeBackend', FakeCompose)
    # _make_backend / _resolve_skip_display import these lazily from their home
    # modules, so patch them there.
    monkeypatch.setattr(hw, 'detect_inventory', lambda: {})

    def skip_display_for(argv, setting=None):
        seen.clear()
        monkeypatch.setattr(
            paths, 'get_setting',
            lambda k: setting if k == 'skip_display_gpus' else None,
        )
        cfg = AcquireCLI.cli(argv=argv, strict=False)
        mod._make_backend(cfg)
        return seen['skip_display']

    base = ['qwen-coder', '--backend', 'compose']
    assert skip_display_for(base) is False                       # default: use all
    assert skip_display_for([*base, '--skip-display-gpus']) is True   # flag opts in
    assert skip_display_for(base, setting='true') is True        # setting opts in
    # explicit flag wins over the setting
    assert skip_display_for([*base, '--no-skip-display-gpus'], setting='true') is False


def test_ui_flag_and_setting_resolution(env, monkeypatch):
    """Open WebUI: default on, --no-ui off, and `config set ui` honored."""
    import infer_stack.cli.commands_leasing as mod
    import infer_stack.hardware as hw

    seen = {}

    class FakeCompose:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(mod, 'ComposeBackend', FakeCompose)
    monkeypatch.setattr(hw, 'detect_inventory', lambda: {})

    def ui_for(argv, setting=None):
        # _make_backend / _resolve_ui both do `from ..paths import get_setting`.
        monkeypatch.setattr(
            'infer_stack.paths.get_setting',
            lambda k: {'backend': 'compose', 'ui': setting}.get(k),
        )
        seen.clear()
        mod._make_backend(mod.AcquireCLI.cli(argv=argv, strict=False))
        return seen['ui']

    assert ui_for(['e', '--backend', 'compose']) is True          # default on
    assert ui_for(['e', '--backend', 'compose', '--no-ui']) is False
    assert ui_for(['e', '--backend', 'compose'], setting=False) is False  # setting
    # explicit flag overrides the setting
    assert ui_for(['e', '--backend', 'compose', '--ui'], setting=False) is True


def test_test_command_smokes_endpoint(tmp_path, monkeypatch, capsys):
    """`infer-stack test <alias>` posts a chat completion and prints the reply."""
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    import requests

    from infer_stack.cli.commands_leasing import EnvCLI, TestCLI

    EnvCLI.main(argv=['LITELLM_MASTER_KEY=sk-test'])
    capsys.readouterr()

    captured = {}

    class Resp:
        status_code = 200
        text = '{}'

        def json(self):
            return {'choices': [{'message': {'content': 'ready'}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured['url'] = url
        captured['auth'] = headers.get('Authorization')
        captured['model'] = json['model']
        return Resp()

    monkeypatch.setattr(requests, 'post', fake_post)
    rc = TestCLI.main(argv=['chat'])
    out = capsys.readouterr().out
    assert rc == 0
    assert captured['url'] == 'http://127.0.0.1:14042/v1/chat/completions'
    assert captured['auth'] == 'Bearer sk-test'   # managed key applied
    assert captured['model'] == 'chat'            # asks for the alias
    assert 'ready' in out


def test_test_command_reports_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    import requests

    from infer_stack.cli.commands_leasing import TestCLI

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError('refused')

    monkeypatch.setattr(requests, 'post', boom)
    rc = TestCLI.main(argv=['chat'])
    out = capsys.readouterr().out
    assert rc == 1
    assert 'FAILED' in out and 'chat' in out


def test_release_unknown_lease_fails(env):
    """Regression: `release <typo>` printed `released 1 lease(s)` and exited 0
    while the real lease kept pinning its GPU."""
    with pytest.raises(SystemExit, match='no such lease'):
        ReleaseCLI.main(argv=['lease-nope', '--ledger', env.db])


def test_renew_released_lease_fails(env):
    """Regression: renew silently resurrected a released lease (state back to
    ACTIVE) without re-realizing anything behind it."""
    envf = env.tmp / 'is.env'
    AcquireCLI.main(argv=['qwen-coder', *_base(env), '--env-file', str(envf)])
    ReleaseCLI.main(argv=['--ledger', env.db, '--env-file', str(envf)])
    with pytest.raises(SystemExit, match='no active lease'):
        RenewCLI.main(
            argv=['--ledger', env.db, '--env-file', str(envf), '--ttl', '3h']
        )


def test_evict_json_stdout_is_pure_json(env, capsys):
    """Regression: the `no idle deployment for: ...` diagnostic printed to stdout
    ahead of the JSON document, breaking `json.loads`/jq consumers."""
    from infer_stack.cli.commands_leasing import EvictCLI

    capsys.readouterr()
    rc = EvictCLI.main(argv=['ghost', '--ledger', env.db, '--json'])
    out, err = capsys.readouterr()
    data = json.loads(out)  # must parse: no human text mixed into stdout
    assert data == {'evicted': [], 'torn_down': [], 'missing': ['ghost']}
    assert 'no idle deployment for: ghost' in err
    assert rc == 0


def test_acquire_json_redacts_api_key(env, capsys, monkeypatch):
    """Regression: `acquire --json` printed the real LiteLLM master key inside
    the descriptor — stdout lands in job logs that get collected and rsynced.
    The env-file (the delivery mechanism) must still carry the real key."""
    from infer_stack.cli import commands_leasing as cl
    from infer_stack.leasing import MemoryBackend

    class KeyedBackend(MemoryBackend):
        def access(self, endpoints):
            return {
                'base_url': 'http://x:1/v1',
                'api_key_env': 'LITELLM_MASTER_KEY',
                'api_key': 'sk-secret123',
            }

    monkeypatch.setattr(
        cl, '_make_backend',
        lambda config, *, interactive=False: KeyedBackend(ready=True),
    )
    envf = env.tmp / 'is.env'
    capsys.readouterr()
    rc = AcquireCLI.main(
        argv=['qwen-coder', *_base(env), '--json', '--env-file', str(envf)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert 'sk-secret123' not in out
    assert 'redacted' in data['descriptor']['api_key']
    assert 'sk-secret123' in envf.read_text()
