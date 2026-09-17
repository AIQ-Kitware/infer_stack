"""Selective apply (plan step P8, tests 39-49): ownership, fingerprints, barrier."""

from __future__ import annotations

import pytest
import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Controller, Ledger, SqliteStore
from infer_stack.leasing.compose import ApplyAborted, ComposeBackend, stamp_fingerprints
from infer_stack.leasing.placement import PlacementInputs
from infer_stack.leasing.residency import DEPLOYMENT_LABEL, FINGERPRINT_LABEL, SERVICE_LABEL
from test_leasing_compose import IMAGES, PORTS, STATE, FakeDocker, FakeHttp, vllm


def backend(tmp_path, docker=None, **kw):
    kw.setdefault('litellm', False)
    kw.setdefault('ui', False)
    return ComposeBackend(state_dir=tmp_path, inventory=simulate_inventory('2x80'),
                          run=docker or FakeDocker(), http=kw.pop('http', FakeHttp(tmp_path)),
                          images=IMAGES, ports=PORTS, state=STATE, **kw)


def render(be, deps, **inputs):
    placement = PlacementInputs(**inputs) if inputs else None
    be.converge(deps, apply=False, placement=placement)
    return yaml.safe_load(be.compose_file.read_text())['services']


def only(fake, service):
    found = [cid for cid, c in fake.containers.items() if c['service'] == service]
    assert len(found) == 1, found
    return found[0]


def test_a_started_service_dropped_later_is_removed_as_managed(tmp_path):        # 39
    be = backend(tmp_path)
    be.converge([vllm('b')])
    fake = be.run
    assert fake.running and all(c['labels'].get(FINGERPRINT_LABEL) for c in fake.containers.values())
    be.converge([])
    assert fake.containers == {}                         # removed, never an orphan


def test_crash_leftovers_are_kept_if_wanted_and_removed_otherwise(tmp_path):     # 40
    be = backend(tmp_path)
    svc = render(be, [vllm('a')])
    name = next(iter(svc))
    fake = be.run
    labels = svc[name]['labels']
    fake.add_container(name, labels=labels, device_ids=[0])                  # wanted
    fake.add_container(name + '-old', labels={**labels, SERVICE_LABEL: name + '-old'},
                       device_ids=[1])                                         # not wanted
    be.apply()
    assert [c['service'] for c in fake.containers.values()] == [name]
    assert not any('up' in c for c in fake.calls)       # nothing started: kept


def test_generated_file_content_drives_the_fingerprint():                           # 41
    from pathlib import Path

    doc = {'services': {'litellm': {'image': 'x', 'volumes': ['/s/cfg.yaml:/etc/c.yaml:ro']}}}
    first = stamp_fingerprints(doc, files={Path('/s/cfg.yaml'): 'routes: 1'})
    same = stamp_fingerprints(doc, files={Path('/s/cfg.yaml'): 'routes: 1'})
    changed = stamp_fingerprints(doc, files={Path('/s/cfg.yaml'): 'routes: 2'})
    assert first == same and first != changed


@pytest.mark.parametrize('state', ['exited', 'dead', 'created'])
def test_a_managed_container_that_does_not_serve_is_replaced(tmp_path, state):     # 42
    be = backend(tmp_path)
    be.converge([vllm('a')])
    fake = be.run
    old = next(iter(fake.containers))
    fake.containers[old]['state'] = state
    be.apply()
    assert old not in fake.containers and fake.running


def test_two_containers_with_the_wanted_key_fail_closed(tmp_path):                  # 43
    be = backend(tmp_path)
    svc = render(be, [vllm('a')])
    name = next(iter(svc))
    for _ in range(2):
        be.run.add_container(name, labels=svc[name]['labels'], device_ids=[0])
    with pytest.raises(ApplyAborted, match='2 containers'):
        be.apply()


def test_paused_required_is_unpaused_and_optional_stays_paused(tmp_path):          # 44
    be = backend(tmp_path)
    import dataclasses

    live = vllm('live', t=0)
    warm = dataclasses.replace(vllm('warm', hf='org/other', t=1), state='idle')
    svc = render(be, [live, warm], required_ids={'live'}, hard={'live': [0]},
                 optional_hints={'warm': [1]})
    fake = be.run
    for name, s in svc.items():
        gpu = [0] if s['labels'][DEPLOYMENT_LABEL] == 'live' else [1]
        fake.add_container(name, labels=s['labels'], device_ids=gpu, state='paused')
    be.apply()
    states = {c['labels'][DEPLOYMENT_LABEL]: c['state'] for c in fake.containers.values()}
    assert states == {'live': 'running', 'warm': 'paused'}


class BarrierDocker(FakeDocker):
    """Refuses to start a container on a GPU another container still holds."""

    def _up_service(self, name, svc):
        devices = (((svc.get('deploy') or {}).get('resources') or {})
                   .get('reservations') or {}).get('devices') or []
        wanted = {str(d) for dev in devices for d in (dev.get('device_ids') or [])}
        for c in self.containers.values():
            if wanted & set(c['device_ids']):
                raise RuntimeError(f'GPU {sorted(wanted)} still held by {c["service"]}')
        super()._up_service(name, svc)


def test_barrier_removes_the_previous_occupant_first_even_for_recreation(tmp_path):  # 45
    be = backend(tmp_path, docker=BarrierDocker())
    be.converge([vllm('a', max_len=1024)])
    be.converge([vllm('a', max_len=2048)])               # same deployment, new fingerprint
    assert len(be.run.containers) == 1
    be.converge([vllm('b', hf='org/b')])                 # a different model onto GPU 0
    assert [c['labels'][DEPLOYMENT_LABEL] for c in be.run.containers.values()] == ['b']


def test_an_unmanaged_occupant_blocks_the_apply(tmp_path):                          # 46
    be = backend(tmp_path)
    render(be, [vllm('a')])
    be.run.add_container('someone-elses', labels={}, device_ids=[0])
    with pytest.raises(ApplyAborted, match='unmanaged container'):
        be.apply()


def test_degraded_deployment_is_never_started_or_removed(tmp_path):                 # 47
    # With the gateway on, upstreams publish no host ports (which would shift
    # with the live set and change unrelated fingerprints).
    be = backend(tmp_path, litellm=True)
    broken, other = vllm('broken', t=0), vllm('other', hf='org/o', t=1)
    be.converge([broken, other])
    fake = be.run
    before = {i for i, c in fake.containers.items() if DEPLOYMENT_LABEL in c['labels']}
    # broken's allocation is now invalid (GPU 7 does not exist): it is degraded.
    be.converge([broken, other], placement=PlacementInputs(
        required_ids={'broken', 'other'}, hard={'broken': [7], 'other': [1]}))
    now = {i for i, c in fake.containers.items() if DEPLOYMENT_LABEL in c['labels']}
    assert now == before                                 # nothing removed, nothing new
    be.converge([broken], placement=PlacementInputs(     # `other` released elsewhere
        required_ids={'broken'}, hard={'broken': [7]}))
    assert [c['labels'][DEPLOYMENT_LABEL] for c in fake.containers.values()
            if DEPLOYMENT_LABEL in c['labels']] == ['broken']


def test_fresh_dynamic_bootstrap_starts_postgres_before_the_gateway(tmp_path):      # 48
    from test_leasing_dynamic_routing import RecordingGateway

    be = backend(tmp_path, litellm=True, dynamic_routing=True, http=RecordingGateway())
    be.converge([vllm('a')])
    order = [s for batch in be.run.started for s in batch]
    assert order.index('postgres-litellm') < order.index('litellm')


def test_gc_orphans_removes_exactly_the_reported_containers(tmp_path):             # 49
    be = backend(tmp_path / 'state')
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    ctl = Controller(ledger, be)
    ctl.gc()
    stray = be.run.add_container('stray', labels={})
    seen = []
    assert ctl.remove_orphans(lambda found: seen.extend(found) or False) == []
    assert stray in be.run.containers                    # declined: nothing removed
    removed = ctl.remove_orphans(lambda found: True)
    assert [c.container_id for c in removed] == [c.container_id for c in seen] == [stray]
    assert stray not in be.run.containers
