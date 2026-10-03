"""Real Compose/KubeAI controllers with stateful fake runtimes and SQLite WAL."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_leasing_compose import FakeDocker
from test_leasing_kubeai import FakeHttp, FakeKubectl
from test_leasing_profile import backend as compose_backend

from infer_stack.backends.kubeai import KubeaiBackend
from infer_stack.cli import ManageCLI
from infer_stack.cli.commands_leasing import AcquireCLI, GcCLI
from infer_stack.cli.commands_ledger import LedgerRotateCLI
from infer_stack.cli.commands_runtime import StatusCLI
from infer_stack.leasing import Catalog, Controller, Ledger, SqliteStore
from infer_stack.leasing.profile import ProfileMismatch
from infer_stack.leasing.transition import rotate_backend
from infer_stack.paths import config_root, data_root


@pytest.fixture
def environment(monkeypatch):
    from infer_stack._log import logger
    logger.disable('infer_stack')
    monkeypatch.setattr('infer_stack._log.configure_logging', lambda *args, **kwargs: None)
    from infer_stack.leasing import default_ledger_path
    catalog_data = {'models': {'m': {'source': 'hf://org/model'}}, 'endpoints': {
        'smol135-kube': {'engine': 'vllm', 'model': 'm',
                         'runtime': {'resource_profile': 'gpu-single-default'}}}}
    catalog = Catalog.from_dict(catalog_data)
    config_root().mkdir(parents=True)
    (config_root() / 'catalog.yaml').write_text(yaml.safe_dump(catalog_data))
    settings = config_root() / 'settings.yaml'
    settings.write_text('backend: kubeai\n')
    docker = FakeDocker()
    kubectl = FakeKubectl()
    path = default_ledger_path()
    def make_backend(config=None, **kwargs):
        kind = getattr(config, 'backend', None) or 'kubeai'
        if kind == 'compose':
            return compose_backend(data_root() / 'leasing/compose', catalog=catalog, docker=docker, ui=False)
        result = KubeaiBackend(state_dir=data_root() / 'leasing/kubeai', namespace='default',
                               run=kubectl, http=FakeHttp(kubectl), default_resource_profile='gpu-single-default')
        result.catalog = catalog
        return result
    def controller(kind):
        return Controller(Ledger(SqliteStore(path)), make_backend(SimpleNamespace(backend=kind)))
    monkeypatch.setattr('infer_stack.cli.commands_leasing._make_backend', make_backend)
    monkeypatch.setattr('infer_stack.cli.commands_ledger._make_backend', make_backend)
    return SimpleNamespace(controller=controller, make_backend=make_backend, catalog=catalog,
                           path=path, docker=docker, kubectl=kubectl, settings=settings)


def history(env, *, active=False, live=False):
    old = env.controller('compose')
    request = env.catalog.resolve_requests(['smol135-kube'])
    outcome = old.acquire('old-owner', request, wait=False)
    if not active:
        old.release(outcome.lease.id)
    if not live:
        old.backend.down()
    return old, outcome


def test_full_real_machine_migration_sequence(environment, capsys):
    env = environment
    old, outcome = history(env)
    profile = old.ledger.profile()
    assert profile['backend'] == 'compose'
    leases, deployments = old.ledger.status(virtual_expiry=True)
    assert len(leases) == len(deployments) == 1
    assert str(leases[0].state) == 'released'
    assert not env.docker.containers
    config_before = env.settings.read_bytes()
    catalog_path = config_root() / 'catalog.yaml'
    catalog_before = catalog_path.read_bytes()
    with pytest.raises(SystemExit, match='ledger rotate'):
        AcquireCLI.main(argv=False, names=['smol135-kube'], wait=False, yes=True)
    capsys.readouterr()
    with pytest.raises(SystemExit, match='gc:.*ledger rotate'):
        GcCLI.main(argv=False, yes=True)
    assert not env.kubectl.applied  # no reinterpretation of Compose rows
    assert old.ledger.profile() == profile
    assert LedgerRotateCLI.main(argv=False, json=True) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan['changed'] and not Path(plan['archive']).exists()
    assert LedgerRotateCLI.main(argv=False, yes=True, json=True) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['changed']
    archived = Ledger(SqliteStore(result['archive']))
    assert archived.profile() == profile
    assert archived.get_lease(outcome.lease.id).state == leases[0].state
    assert archived.get_deployment(outcome.deployments[0].id) is not None
    current = env.controller('kubeai')
    assert current.ledger.status() == ([], [])
    assert current.ledger.profile()['backend'] == 'kubeai'
    assert current.ledger.publication_pending() is None
    assert env.settings.read_bytes() == config_before
    assert catalog_path.read_bytes() == catalog_before
    assert AcquireCLI.main(argv=False, names=['smol135-kube'], wait=False, yes=True) == 0
    capsys.readouterr()
    assert current.ledger.profile()['backend'] == 'kubeai'
    assert current.ledger.profile()['catalogs']
    (doc,) = env.kubectl.applied.values()
    assert doc['metadata']['namespace'] == 'default'
    assert doc['metadata']['name'] == 'smol135-kube'
    assert doc['spec']['resourceProfile'] == 'gpu-single-default:1'
    for flag in ('tensor', 'pipeline', 'data'):
        assert f'--{flag}-parallel-size=1' in doc['spec']['args']
    assert not any(a.startswith('--served-model-name') for a in doc['spec']['args'])
    assert archived.get_deployment(outcome.deployments[0].id) is not None
    assert ManageCLI.main(argv=['ledger', 'archives', '--json']) == 0
    assert json.loads(capsys.readouterr().out)[0]['path'] == result['archive']
    # A process opened against the old backend cannot mutate the new epoch.
    with pytest.raises(ProfileMismatch):
        old.gc()


@pytest.mark.parametrize('active,live,match', [(False, True, 'runtime objects'), (True, False, 'active leases')])
def test_transition_refuses_live_objects_or_leases(environment, active, live, match):
    env = environment
    old, _ = history(env, active=active, live=live)
    before = old.ledger.profile()
    with pytest.raises(ProfileMismatch, match=match):
        rotate_backend(env.controller('kubeai'), env.make_backend(SimpleNamespace(backend='compose')), apply=True)
    assert old.ledger.profile() == before
    assert not list(env.path.parent.glob('archives/*.db'))


def test_rotation_retry_after_archive_before_reset(environment, monkeypatch):
    env = environment
    history(env)
    current = env.controller('kubeai')
    store = current.ledger.store
    original = store._write_profile
    def interrupt(profile):
        raise RuntimeError('simulated power loss inside reset transaction')
    monkeypatch.setattr(store, '_write_profile', interrupt)
    old = env.make_backend(SimpleNamespace(backend='compose'))
    with pytest.raises(RuntimeError, match='power loss'):
        rotate_backend(current, old, apply=True)
    assert store.profile()['backend'] == 'compose'
    assert len(store.list_leases()) == len(store.list_deployments(now=current.clock())) == 1
    archives = list(env.path.parent.glob('archives/*.db'))
    assert len(archives) == 1
    monkeypatch.setattr(store, '_write_profile', original)
    result = rotate_backend(env.controller('kubeai'), old, apply=True)
    assert result['changed']
    assert Path(result['archive']) == archives[0]
    assert len(list(env.path.parent.glob('archives/*.db'))) == 1
    retry = rotate_backend(env.controller('kubeai'), old, apply=True)
    assert not retry['changed']
    assert len(retry['archives']) == 1


def test_unknown_old_runtime_refuses_rotation(environment, monkeypatch):
    history(environment)
    old = environment.make_backend(SimpleNamespace(backend='compose'))
    def unavailable():
        raise RuntimeError('Docker unreachable')
    monkeypatch.setattr(old, 'instances', unavailable)
    with pytest.raises(ProfileMismatch, match='Cannot verify old backend quiescence'):
        rotate_backend(environment.controller('kubeai'), old, apply=True)


def test_gc_matching_and_history_only_mismatch(environment, capsys):
    history(environment)
    assert GcCLI.main(argv=False, backend='compose', yes=True) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit, match='gc:.*ledger rotate'):
        GcCLI.main(argv=False, yes=True)
    assert 'Traceback' not in capsys.readouterr().err
    # Explicit history pruning does not adopt/configure the wrong backend.
    assert GcCLI.main(argv=False, forget=True, json=True) == 0
    assert json.loads(capsys.readouterr().out)['leases'] == 1


@pytest.mark.parametrize('snapshot,configured', [(None, 'kubeai'), ('compose', 'compose'), ('compose', 'kubeai')])
def test_status_names_configured_and_recovery_backends(environment, capsys, snapshot, configured):
    env = environment
    if snapshot:
        history(env)
    env.settings.write_text(f'backend: {configured}\n')
    assert StatusCLI.main(argv=False) == 0
    text = capsys.readouterr().out
    assert f'configured backend: {configured}' in text
    assert f'active recovery backend: {snapshot or "(none — no active snapshot)"}' in text
    assert ('transition required' in text) == bool(snapshot and snapshot != configured)


def test_ledger_modal_real_help():
    for args in (['ledger', '--help'], ['ledger', 'rotate', '--help'], ['ledger', 'archives', '--help']):
        proc = subprocess.run([sys.executable, '-m', 'infer_stack', *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'ledger' in proc.stdout
