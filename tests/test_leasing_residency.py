"""Strict physical residency (``ComposeBackend.residency``).

The lenient ``observe()`` returns an empty set when Docker cannot be read; that
is a tested contract (``test_observe_tolerates_unreadable_compose_file``). These
tests pin the opposite contract for the strict snapshot, which later decisions
that stop, remove or hand over a GPU will rely on: failure is *unknown*,
duplicates are *ambiguous*, and GPUs come from the container, never the render.
"""

import json
import shutil
import subprocess
import uuid

import pytest

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.compose import ComposeBackend
from infer_stack.leasing.residency import (
    COMPOSE_PROJECT_LABEL,
    DEPLOYMENT_LABEL,
    WARM_STATES,
    Residency,
    ResidencyUnknown,
    residency_from_inspect,
)

PROJECT = 'infer-stack'


def container(cid, gid, *, state='running', device_ids=('0',), count=0,
              project=PROJECT, requests=None):
    """One ``docker inspect`` entry, shaped as Docker 2x reports a Compose
    ``deploy.resources.reservations.devices[].device_ids`` reservation."""
    if requests is None:
        requests = (
            [{'Driver': 'nvidia', 'Count': count, 'DeviceIDs': list(device_ids),
              'Capabilities': [['gpu']], 'Options': None}]
            if (device_ids or count) else None
        )
    labels = {COMPOSE_PROJECT_LABEL: project}
    if gid is not None:
        labels[DEPLOYMENT_LABEL] = gid
    return {
        'Id': cid,
        'State': {'Status': state},
        'Config': {'Labels': labels},
        'HostConfig': {'DeviceRequests': requests},
    }


def snap(*entries):
    return residency_from_inspect(json.dumps(list(entries)), project=PROJECT)


# -- states ---------------------------------------------------------------


@pytest.mark.parametrize('state', sorted(WARM_STATES))
def test_warm_states_are_resident(state):
    res = snap(container('c1', 'grp-a', state=state))
    assert res.resident('grp-a').container_id == 'c1'


@pytest.mark.parametrize('state', ['created', 'exited', 'removing', 'dead'])
def test_non_warm_states_are_not_resident_but_still_occupy_their_gpu(state):
    # Not warm cache -- but a created/exited container still carries a device
    # request that `up` could start, so nothing may be started on top of it.
    res = snap(container('c1', 'grp-a', state=state, device_ids=('3',)))
    assert res.resident('grp-a') is None
    assert [c.container_id for c in res.occupants(3)] == ['c1']


# -- GPUs -----------------------------------------------------------------


def test_gpus_come_from_device_request_indices():
    res = snap(container('c1', 'grp-a', device_ids=('2', '1')))
    c = res.resident('grp-a')
    assert c.gpus == (1, 2) and c.all_gpus is False
    assert {g for g in range(4) if res.occupants(g)} == {1, 2}


def test_container_without_a_gpu_reservation_occupies_nothing():
    res = snap(container('c1', 'grp-a', device_ids=()))   # DeviceRequests: None
    assert res.resident('grp-a').gpus == ()
    assert all(res.occupants(g) == () for g in range(4))


@pytest.mark.parametrize('requests', [
    [{'Driver': 'nvidia', 'Count': -1, 'DeviceIDs': None}],          # --gpus all
    [{'Driver': 'nvidia', 'Count': 2, 'DeviceIDs': []}],             # --gpus 2
    [{'Driver': 'nvidia', 'Count': 0, 'DeviceIDs': ['GPU-3f1e...']}],  # UUIDs
])
def test_unmappable_reservation_is_treated_as_every_gpu(requests):
    # Never guess which GPU a container holds: fail closed for every GPU.
    res = snap(container('c1', 'grp-a', requests=requests))
    assert res.resident('grp-a').all_gpus is True
    assert all(
        [c.container_id for c in res.occupants(g)] == ['c1'] for g in range(8)
    )


# -- scoping and ambiguity (plan tests 39, 41) ------------------------------


def test_other_projects_and_unlabelled_containers_are_ignored():
    res = snap(
        container('mine', 'grp-a'),
        container('theirs', 'grp-a', project='someone-else'),
        container('nolabel', None),
    )
    assert [c.container_id for c in res.all_containers()] == ['mine']


def test_duplicate_deployment_containers_are_kept_and_ambiguous():
    # A dict keyed by deployment must not silently drop one of two containers.
    res = snap(
        container('c2', 'grp-a', device_ids=('1',)),
        container('c1', 'grp-a', state='exited', device_ids=('0',)),
    )
    assert res.ambiguous('grp-a') is True
    assert res.resident('grp-a') is None
    assert [c.container_id for c in res.containers('grp-a')] == ['c1', 'c2']
    assert [c.container_id for c in res.occupants(0)] == ['c1']
    assert [c.container_id for c in res.occupants(1)] == ['c2']


def test_absent_deployment_is_not_resident_and_not_ambiguous():
    res = snap(container('c1', 'grp-a'))
    assert res.resident('grp-b') is None
    assert res.ambiguous('grp-b') is False
    assert res.containers('grp-b') == ()


@pytest.mark.parametrize('raw', ['not json', '{"Id": "c1"}', '[1, 2]',
                                 json.dumps([{'Config': {'Labels': {
                                     COMPOSE_PROJECT_LABEL: PROJECT,
                                     DEPLOYMENT_LABEL: 'grp-a'}}}])])
def test_unparseable_inspect_output_is_unknown(raw):
    with pytest.raises(ResidencyUnknown):
        residency_from_inspect(raw, project=PROJECT)


# -- ComposeBackend.residency (plan test 40) --------------------------------


class ScriptedDocker:
    """Answers `docker ps` / `docker inspect` from fixtures; records calls."""

    def __init__(self, inspect_entries, *, fail=None):
        self.entries = inspect_entries
        self.fail = fail
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        verb = args[1] if len(args) > 1 else ''
        if self.fail == verb:
            raise RuntimeError(f'docker {verb}: permission denied')
        if args[:2] == ['docker', 'ps']:
            return '\n'.join(e['Id'] for e in self.entries) + '\n'
        if args[:2] == ['docker', 'inspect']:
            wanted = set(args[2:])
            return json.dumps([e for e in self.entries if e['Id'] in wanted])
        if 'ps' in args:          # the lenient observe() path: docker compose ps
            if self.fail == 'compose-ps':
                raise RuntimeError('compose ps failed')
            return '[]'
        return ''


def backend(tmp_path, run):
    return ComposeBackend(state_dir=tmp_path, inventory=simulate_inventory('4x80'),
                          run=run, project=PROJECT)


def test_residency_lists_by_both_labels_in_every_state(tmp_path):
    docker = ScriptedDocker([container('c1', 'grp-a', device_ids=('1',))])
    res = backend(tmp_path, docker).residency()
    ps = docker.calls[0]
    assert ps[:4] == ['docker', 'ps', '-a', '--no-trunc']
    assert f'label={COMPOSE_PROJECT_LABEL}={PROJECT}' in ps
    assert f'label={DEPLOYMENT_LABEL}' in ps
    assert docker.calls[1] == ['docker', 'inspect', 'c1']
    assert res.resident('grp-a').gpus == (1,)


def test_residency_with_no_containers_is_empty_and_skips_inspect(tmp_path):
    docker = ScriptedDocker([])
    res = backend(tmp_path, docker).residency()
    assert res == Residency({})
    assert [c[:2] for c in docker.calls] == [['docker', 'ps']]


@pytest.mark.parametrize('verb', ['ps', 'inspect'])
def test_docker_failure_is_unknown_never_empty(tmp_path, verb):
    docker = ScriptedDocker([container('c1', 'grp-a')], fail=verb)
    with pytest.raises(ResidencyUnknown):
        backend(tmp_path, docker).residency()


def test_observe_contract_is_unchanged_by_the_strict_path(tmp_path):
    # The same failure the strict path reports as unknown stays lenient in
    # observe(): its empty-on-error result is relied on by acquire.
    be = backend(tmp_path, ScriptedDocker([], fail='compose-ps'))
    be.compose_file.parent.mkdir(parents=True, exist_ok=True)
    be.compose_file.write_text('services: {}\n')
    assert be.observe() == set()


# -- against a real Docker daemon ------------------------------------------


def _docker_daemon_usable() -> bool:
    # `docker compose version` succeeds without daemon access; `docker info`
    # does not, which is what this test actually needs.
    if shutil.which('docker') is None:
        return False
    try:
        subprocess.run(['docker', 'info'], capture_output=True, timeout=15, check=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _docker_daemon_usable(), reason='docker daemon not usable')
def test_residency_against_real_docker_created_containers(tmp_path):
    """Creates (never starts) two containers; no GPU is needed to create one."""
    project = f'isres-{uuid.uuid4().hex[:8]}'
    image = 'busybox:latest'
    compose = tmp_path / 'docker-compose.yml'
    compose.write_text(
        f'name: {project}\n'
        'services:\n'
        '  model-a:\n'
        f'    image: {image}\n'
        f'    labels: {{{DEPLOYMENT_LABEL}: grp-aaa}}\n'
        '    deploy: {resources: {reservations: {devices: '
        '[{driver: nvidia, device_ids: ["1", "2"], capabilities: [gpu]}]}}}\n'
        '  cpu-b:\n'
        f'    image: {image}\n'
        f'    labels: {{{DEPLOYMENT_LABEL}: grp-bbb}}\n'
    )

    def run(args):
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout

    base = ['docker', 'compose', '-p', project, '-f', str(compose)]
    try:
        subprocess.run(['docker', 'pull', '-q', image], capture_output=True, timeout=300)
        subprocess.run([*base, 'create'], capture_output=True, check=True, timeout=300)
        be = ComposeBackend(state_dir=tmp_path, inventory=simulate_inventory('4x80'),
                            run=run, project=project)
        res = be.residency()
        a, b = res.containers('grp-aaa'), res.containers('grp-bbb')
        assert len(a) == 1 and a[0].gpus == (1, 2) and a[0].state == 'created'
        assert len(b) == 1 and b[0].gpus == ()
        assert res.resident('grp-aaa') is None           # created is not warm
        assert [c.deployment_id for c in res.occupants(2)] == ['grp-aaa']
    finally:
        subprocess.run([*base, 'down'], capture_output=True, timeout=300)
