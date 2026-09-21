"""An engine that can never start must fail fast, with its own error.

Reported from a real run: `inclusionAI/Ling-3.0-flash` (an architecture this
vLLM build does not implement) exits immediately, `restart: unless-stopped`
restarts it forever, and `observe()` only counts *running* services -- so the
acquire waited out its whole 1800 s timeout holding a GPU, and the engine's
error ("set trust_remote_code=True") never reached the operator.
"""

from __future__ import annotations

import pytest

from infer_stack.leasing.compose import CRASH_LOOP_RESTARTS
from infer_stack.leasing.residency import DEPLOYMENT_LABEL
from test_leasing_admission import CAT, acquire, make

CRASH_LOG = (
    'INFO 09-21 10:00:00 api_server.py:1 vLLM API server version 0.25.1\n'
    'ValueError: The checkpoint you are trying to load has model type '
    '`bailing_moe_v3` but Transformers does not recognize this architecture. '
    'If the model is custom, set trust_remote_code=True.\n'
)


def crash_loop(ctl, docker, deployment_id, *, restarts=CRASH_LOOP_RESTARTS,
               state='restarting', exit_code=1, logs=CRASH_LOG):
    """Make the deployment's container look like a crash loop to Docker."""
    for cid, container in docker.containers.items():
        if container['labels'].get(DEPLOYMENT_LABEL) == deployment_id:
            container.update(state=state, restart_count=restarts, exit_code=exit_code)
    ctl.backend.deployment_logs = lambda deployment, tail=400: logs


def test_a_crash_looping_engine_is_diagnosed_not_awaited(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')                       # --no-wait: nothing probed yet
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid)

    probe = ctl.backend.probe_ready(ledger.get_deployment(gid), 'one')
    assert probe.ready is False and probe.fatal is True
    assert 'engine is not starting' in probe.detail
    assert 'restarted 2 time(s)' in probe.detail
    assert 'trust_remote_code' in probe.detail          # the engine's own words
    assert 'likely cause' in probe.detail


def test_the_wait_ends_at_once_instead_of_holding_the_gpu(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid)
    started = ctl.clock()

    result = ctl.wait_ready([ledger.get_deployment(gid)], timeout=1800.0, interval=5.0)

    assert result.ready is False
    assert ctl.clock() == started                       # not one interval slept
    assert [(g, ep) for g, ep, _ in result.failures] == [(gid, 'one')]
    assert 'trust_remote_code' in result.failures[0][2]


def test_acquire_releases_the_lease_as_soon_as_the_engine_is_hopeless(tmp_path):
    from infer_stack.leasing import LeaseState

    ledger, ctl, docker = make(tmp_path)
    warm = acquire(ctl, 'one')                          # occupies the GPU
    gid = warm.deployments[0].id
    crash_loop(ctl, docker, gid)
    ctl.release_leases([warm.lease.id])
    started = ctl.clock()

    out = ctl.acquire('bob', CAT.resolve_names(['one']), wait=True,
                      timeout=1800.0, interval=5.0)

    assert out.released_on_timeout is True              # the lease did not linger
    assert ctl.clock() - started < 60                   # and it did not wait 1800 s
    assert out.wait.failures and 'trust_remote_code' in out.wait.failures[0][2]
    assert [le.state for le in ledger.status()[0]] == [LeaseState.RELEASED] * 2


@pytest.mark.parametrize('state,restarts,exit_code,expected', [
    ('exited', 0, 1, 'exited with code 1'),             # died, no restart policy
    ('restarting', CRASH_LOOP_RESTARTS, 1, 'restarted'),
])
def test_both_shapes_of_a_failed_start_are_recognised(tmp_path, state, restarts,
                                                      exit_code, expected):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, state=state, restarts=restarts, exit_code=exit_code)
    assert expected in ctl.backend.startup_failure(ledger.get_deployment(gid))


@pytest.mark.parametrize('state,restarts,exit_code', [
    ('running', 0, 0),                                  # loading normally
    ('created', 0, 0),                                  # not started yet
    ('exited', 0, 0),                                   # finished cleanly
    ('restarting', CRASH_LOOP_RESTARTS - 1, 1),         # one restart is not a loop
])
def test_a_model_that_may_still_load_is_left_alone(tmp_path, state, restarts, exit_code):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, state=state, restarts=restarts, exit_code=exit_code)
    assert ctl.backend.startup_failure(ledger.get_deployment(gid)) is None


def test_unreadable_docker_never_declares_an_engine_hopeless(tmp_path):
    from infer_stack.leasing.residency import ResidencyUnknown

    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id

    def unknown():
        raise ResidencyUnknown('docker ps failed')

    ctl.backend.residency = unknown
    assert ctl.backend.startup_failure(ledger.get_deployment(gid)) is None


def test_an_oom_crash_loop_says_so(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid,
               logs='torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n')
    detail = ctl.backend.startup_failure(ledger.get_deployment(gid))
    assert 'ran out of memory' in detail
