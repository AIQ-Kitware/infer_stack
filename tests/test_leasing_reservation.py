"""Tests for the GPU-reservation lease (``acquire --reserve-gpus N``).

A reservation holds N *available* GPUs (count-based first-fit — infer-stack picks
which) without launching any server, so an external process (e.g. HELM's
in-process ``HuggingFaceClient``) can run on exactly the reserved GPU under the
same admission accounting as served runs. These exercise the ledger / placement /
render / controller path with the compose backend's fake docker seam — no real
docker or GPUs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from infer_stack.cli.commands_leasing import _descriptor_for
from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import (
    ComposeBackend,
    Controller,
    Ledger,
    SqliteStore,
    is_reservation,
    plan_placement,
    render_compose,
    reservation_request,
    vllm_structural,
)
from infer_stack.leasing.envfile import descriptor_env
from infer_stack.leasing.models import (
    RESERVED_ENGINE,
    Deployment,
    DeploymentState,
    EndpointRequest,
)
from infer_stack.leasing.placement import required_gpu_count

IMAGES = {'vllm': 'vllm:test', 'ollama': 'ollama:test', 'litellm': 'litellm:test'}
PORTS = {'ollama': 11434}
STATE = {'hf_cache': '/cache/hf', 'ollama': '/cache/ollama'}


class FakeDocker:
    """Stateful docker-compose stand-in: ``up`` reflects the rendered file."""

    def __init__(self):
        self.running: list[str] = []
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        cfile = args[args.index('-f') + 1] if '-f' in args else None
        if 'up' in args:
            data = yaml.safe_load(Path(cfile).read_text()) or {}
            self.running = sorted((data.get('services') or {}).keys())
            return ''
        if 'down' in args:
            self.running = []
            return ''
        if 'ps' in args:
            return json.dumps(
                [{'Service': s, 'State': 'running'} for s in self.running]
            )
        return ''


def make_backend(tmp_path, *, spec='2x80', **kw):
    return ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory(spec),
        run=FakeDocker(),
        images=IMAGES,
        ports=PORTS,
        state=STATE,
        litellm=False,  # a lean stack: no gateway needed to hold a GPU
        ui=False,
        **kw,
    )


def _reservation_deployment(gid, *, count=1, t=0.0):
    req = reservation_request(count)
    return Deployment(
        gid, req.compat_key, req.engine, req.sharing, dict(req.capacity),
        dict(req.spec), {req.endpoint: {}}, DeploymentState.LIVE, t, t,
    )


def _vllm_deployment(gid, *, t=0.0):
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {'engine': 'vllm', 'served_model_name': gid,
         'runtime': {'tensor_parallel_size': 1, 'max_model_len': 4096}},
        {gid: {'served_model_name': gid, 'protocol': 'chat'}},
        DeploymentState.LIVE, t, t,
    )


# -- primitives ------------------------------------------------------------


def test_reservation_request_shape():
    req = reservation_request(2)
    assert is_reservation(req)
    assert req.engine == RESERVED_ENGINE
    assert req.sharing == 'dedicated'          # never coalesce two reservations
    assert req.spec['reserved_gpu_count'] == 2
    assert req.spec['reclaim'] != 'keep-warm'  # frees its GPU on release


def test_reservation_request_floors_at_one():
    assert reservation_request(0).spec['reserved_gpu_count'] == 1


def test_required_gpu_count_honors_reservation():
    assert required_gpu_count(_reservation_deployment('r', count=3)) == 3
    assert required_gpu_count(_vllm_deployment('v')) == 1


# -- placement: the shared-machine property --------------------------------


def test_reservation_first_fits_and_withholds_from_vllm():
    res = _reservation_deployment('res', count=1, t=0.0)
    served = _vllm_deployment('vv', t=1.0)
    plan = plan_placement([res, served], simulate_inventory('2x80'))
    assert not plan.errors
    assert plan.assignments['res'] == [0]   # first-fit picks a free GPU (not pinned)
    assert plan.assignments['vv'] == [1]    # the served run avoids the reserved GPU


def test_multi_gpu_reservation_first_fits_a_block():
    res = _reservation_deployment('res', count=2)
    plan = plan_placement([res], simulate_inventory('4x80'))
    assert plan.assignments['res'] == [0, 1]


# -- render: no container, but the GPU stays booked ------------------------


def test_reservation_renders_no_service_and_is_not_unrenderable():
    res = _reservation_deployment('res', count=1)
    rc = render_compose(
        [res], {'res': [0]}, images=IMAGES, ports=PORTS, state=STATE,
    )
    assert res.id not in rc.unrenderable       # not treated as a placement failure
    assert rc.services == {}                    # no vllm/ollama container emitted


# -- controller end-to-end -------------------------------------------------


def test_reservation_acquire_is_ready_and_reports_the_gpu(tmp_path):
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path, spec='2x80')
    ctl = Controller(ledger, backend)

    out = ctl.acquire('alice', [reservation_request(1)])
    assert out.wait.ready is True              # nothing to probe -> ready at once
    gid = out.deployments[0].id
    assert out.reconcile.assignments[gid] == [0]

    # the env-file descriptor carries the reserved GPU index
    cfg = SimpleNamespace(base_url='http://x/v1', api_key_env='K')
    descriptor = _descriptor_for(
        ctl, out.lease, out.deployments, cfg,
        assignments=out.reconcile.assignments,
    )
    assert descriptor['cuda_visible_devices'] == '0'
    assert descriptor_env(descriptor)['CUDA_VISIBLE_DEVICES'] == '0'


def test_reservation_frees_its_gpu_on_release(tmp_path):
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path, spec='1x80')  # a single GPU: reuse proves it freed
    ctl = Controller(ledger, backend)

    first = ctl.acquire('alice', [reservation_request(1)])
    assert first.reconcile.assignments[first.deployments[0].id] == [0]

    ctl.release(first.lease.id)

    # the one GPU is free again -> a second reservation gets it (not a placement error)
    second = ctl.acquire('bob', [reservation_request(1)])
    assert second.reconcile.assignments[second.deployments[0].id] == [0]


def test_reservation_and_served_run_coschedule_on_distinct_gpus(tmp_path):
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path, spec='2x80')
    ctl = Controller(ledger, backend)

    reserved = ctl.acquire('alice', [reservation_request(1)])
    res_gpu = reserved.reconcile.assignments[reserved.deployments[0].id]

    served_req = EndpointRequest(
        endpoint='served', engine='vllm',
        structural=vllm_structural(model_ref='org/m'),
        capacity={'max_model_len': 4096},
        spec={'engine': 'vllm', 'hf_model_id': 'org/m', 'served_model_name': 'served',
              'runtime': {'tensor_parallel_size': 1, 'max_model_len': 4096}},
        served={'served': {'served_model_name': 'served', 'protocol': 'chat'}},
    )
    served = ctl.acquire('bob', [served_req], wait=False)
    served_gpu = served.reconcile.assignments[served.deployments[0].id]

    assert res_gpu and served_gpu
    assert set(res_gpu).isdisjoint(served_gpu)  # never share the reserved card
