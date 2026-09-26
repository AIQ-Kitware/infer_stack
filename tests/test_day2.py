"""Day-2 verbs (ps, logs, status, stack) read the backend, on either backend."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infer_stack.leasing.instances import (
    DOCKER,
    KUBERNETES,
    MEMORY,
    Instance,
    UnknownTarget,
    follow_argv,
    from_residency,
    history_argv,
    resolve,
)
from infer_stack.leasing.residency import Container, Residency, ResidencyUnknown

POD = Instance('model-qwen-0-abc', 'model-qwen-0-abc', 'grp-1', 'running',
               runtime=KUBERNETES, namespace='kubeai')
GATEWAY = Instance('litellm', 'c0ffee1234', '', 'running', ports='14042->4000/tcp')


def test_log_commands_follow_the_runtime():
    assert history_argv(POD, tail=50) == [
        'kubectl', '-n', 'kubeai', 'logs', '--tail', '50', 'model-qwen-0-abc']
    assert follow_argv(POD) == [
        'kubectl', '-n', 'kubeai', 'logs', '--tail', '0', 'model-qwen-0-abc', '--follow']
    assert history_argv(GATEWAY, tail='all', timestamps=True) == [
        'docker', 'logs', '--tail', 'all', '--timestamps', 'c0ffee1234']
    assert follow_argv(GATEWAY)[:2] == ['docker', 'attach']
    memory = Instance('grp-1', 'grp-1', 'grp-1', 'running', runtime=MEMORY)
    assert history_argv(memory) is None and follow_argv(memory) is None


def test_instances_come_from_residency_engines_first():
    residency = Residency(
        by_deployment={'grp-1': (Container('pod-1', 'grp-1', 'restarting',
                                           restart_count=3, reason='CrashLoopBackOff'),)},
        others=(Container('gw', '', 'running', service='litellm'),),
    )
    pods = from_residency(residency, runtime=KUBERNETES, namespace='kubeai')
    assert [i.name for i in pods] == ['pod-1', 'gw']      # pods are named by pod
    assert pods[0].status == 'restarting (CrashLoopBackOff, 3 restarts)'
    containers = from_residency(residency, runtime=DOCKER)
    assert [i.name for i in containers] == ['pod-1', 'litellm']   # service names


def test_a_target_resolves_by_alias_deployment_name_or_id():
    served = {'grp-1': ['qwen']}
    both = [POD, GATEWAY]
    assert resolve(both, ['qwen'], served) == [POD]
    assert resolve(both, ['grp-1', 'litellm'], served) == [POD, GATEWAY]
    assert resolve(both, ['c0ff'], served) == [GATEWAY]
    with pytest.raises(UnknownTarget, match='running: litellm, model-qwen-0-abc'):
        resolve(both, ['nope'], served)


class _Backend:
    """A backend as the day-2 verbs see it."""

    def __init__(self, instances, *, residency=None):
        self._instances = instances
        self._residency = residency
        self.down_calls = 0

    def instances(self):
        if self._instances is None:
            raise ResidencyUnknown('kubectl get pods failed')
        return list(self._instances)

    def residency(self):
        if self._residency is None:
            raise ResidencyUnknown('kubectl get pods failed')
        return self._residency

    def down(self):
        self.down_calls += 1


def _cli(monkeypatch, backend, served=None):
    from infer_stack.cli import commands_runtime as rt

    monkeypatch.setattr(rt, '_day2_backend', lambda config: backend)
    monkeypatch.setattr(rt, '_served_by_deployment', lambda: dict(served or {}))
    return rt


def test_ps_has_one_shape_and_names_what_each_serves(monkeypatch, capsys):
    rt = _cli(monkeypatch, _Backend([POD, GATEWAY]), {'grp-1': ['qwen']})
    assert rt.PsCLI.main(argv=[]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ['NAME', 'STATUS', 'SERVES', 'GPUS', 'STARTED', 'ID', 'PORTS']
    assert 'qwen' in out[1] and 'model-qwen-0-abc' in out[1]
    assert '(front door)' in out[2] and '14042->4000/tcp' in out[2]

    assert rt.PsCLI.main(argv=['qwen', '--json']) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r['name'] for r in rows] == ['model-qwen-0-abc']
    assert rows[0]['serves'] == ['qwen'] and rows[0]['runtime'] == KUBERNETES


def test_ps_hides_finished_instances_unless_all(monkeypatch, capsys):
    done = Instance('old', 'dead0000', 'grp-2', 'exited')
    rt = _cli(monkeypatch, _Backend([POD, done]))
    rt.PsCLI.main(argv=['-q'])
    assert capsys.readouterr().out.split() == ['model-qwen-0-abc']
    rt.PsCLI.main(argv=['-q', '--all'])
    assert capsys.readouterr().out.split() == ['model-qwen-0-abc', 'dead0000']


def test_ps_says_why_when_the_runtime_cannot_be_read(monkeypatch):
    rt = _cli(monkeypatch, _Backend(None))
    with pytest.raises(SystemExit, match='cannot read what is running'):
        rt.PsCLI.main(argv=[])


def test_logs_reads_the_named_endpoint_through_its_runtime(monkeypatch, capsys):
    import subprocess

    rt = _cli(monkeypatch, _Backend([POD, GATEWAY]), {'grp-1': ['qwen']})
    ran = []

    def fake_run(argv, **kw):
        ran.append(argv)
        return SimpleNamespace(stdout=b'INFO: loaded\n', returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert rt.LogsCLI.main(argv=['qwen', '--tail', '5']) == 0
    assert ran == [['kubectl', '-n', 'kubeai', 'logs', '--tail', '5', 'model-qwen-0-abc']]
    assert capsys.readouterr().out == 'INFO: loaded\n'      # one instance: no prefix
    with pytest.raises(SystemExit, match="no instance matches 'nope'"):
        rt.LogsCLI.main(argv=['nope'])


def test_status_health_comes_from_residency(monkeypatch):
    from infer_stack.cli.commands_runtime import _served_models
    from infer_stack.leasing import DeploymentState

    def dep(gid):
        return SimpleNamespace(id=gid, state=DeploymentState.LIVE, engine='vllm',
                               served={gid: {'hf_model_id': 'org/m'}}, spec={})

    residency = Residency(by_deployment={
        'up': (Container('c1', 'up', 'running'),),
        'loading': (Container('c3', 'loading', 'running', health='starting'),),
        'crashing': (Container('c2', 'crashing', 'exited'),),
    })
    rows = _served_models([dep('up'), dep('loading'), dep('crashing'), dep('gone')],
                          _Backend([], residency=residency))
    assert {r[0]: r[3] for r in rows} == {'up': 'up', 'loading': 'starting',
                                          'crashing': 'exited', 'gone': 'STALE'}
    rows = _served_models([dep('up')], _Backend([], residency=None))
    assert rows[0][3] == 'unverified'
    # Recorded but not applied yet (an apply is running): not STALE.
    rows = _served_models([dep('gone')], _Backend([], residency=residency), pending=True)
    assert rows[0][3] == 'pending'


def test_stack_down_stops_the_backend_on_either_backend(monkeypatch):
    backend = _Backend([])
    rt = _cli(monkeypatch, backend)
    assert rt.StackDownCLI.main(argv=[]) == 0
    assert backend.down_calls == 1


def test_stack_compose_targets_the_compose_project_on_this_host(monkeypatch, tmp_path):
    from infer_stack.cli import commands_runtime as rt

    compose_file = tmp_path / 'docker-compose.yml'
    project = SimpleNamespace(compose_file=compose_file,
                              compose_argv=lambda: ['docker', 'compose', '-p', 'gw'])
    kubeai = SimpleNamespace(compose_project=lambda: project)
    monkeypatch.setattr(rt, '_day2_backend', lambda config: kubeai)
    with pytest.raises(SystemExit, match='nothing rendered yet'):
        rt._compose_argv(SimpleNamespace())
    compose_file.write_text('services: {}\n')
    assert rt._compose_argv(SimpleNamespace()) == ['docker', 'compose', '-p', 'gw']

    monkeypatch.setattr(rt, '_day2_backend', lambda config: SimpleNamespace())
    with pytest.raises(SystemExit, match='no Compose project on this host'):
        rt._compose_argv(SimpleNamespace())


@pytest.mark.parametrize(('argv', 'follow', 'names'), [
    (['-f', 'qwen'], True, ['qwen']),
    (['qwen', '-f'], True, ['qwen']),
    (['--follow', 'qwen', 'litellm'], True, ['qwen', 'litellm']),
    (['-f', 'false'], False, []),
])
def test_a_flag_never_swallows_the_positional_after_it(argv, follow, names):
    """kwconf flags take an optional value; `logs -f qwen` followed everything."""
    from infer_stack.cli.commands_runtime import LogsCLI

    config = LogsCLI.cli(argv=argv)
    assert config.follow is follow and list(config.services or []) == names


def test_acquire_yes_before_the_endpoint_still_names_it():
    from infer_stack.cli.commands_leasing import AcquireCLI

    config = AcquireCLI.cli(argv=['--yes', 'qwen'])
    assert config.yes is True and config.names == ['qwen']


def test_no_color_turns_color_off():
    """kwconf reads a leading `no-` as negation: `--no-color` must negate `color`."""
    from infer_stack.cli.commands_runtime import LogsCLI

    assert LogsCLI.cli(argv=[]).color is True
    assert LogsCLI.cli(argv=['--no-color']).color is False
