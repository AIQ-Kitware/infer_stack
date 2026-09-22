"""An engine that can never start must fail fast, with its own error.

Reported from a real run: an endpoint whose architecture this vLLM build does
not implement exits immediately, `restart: unless-stopped`
restarts it forever, and `observe()` only counts *running* services -- so the
acquire waited out its whole 1800 s timeout holding a GPU, and the engine's
error ("set trust_remote_code=True") never reached the operator.
"""

from __future__ import annotations

import pytest

from infer_stack.leasing.compose import CRASH_LOOP_RESTARTS, classify_engine_log
from infer_stack.leasing.residency import DEPLOYMENT_LABEL
from test_leasing_admission import CAT, acquire, make

# A synthetic log in the shape the engine really emits. The architecture name
# is invented on purpose: which models an evaluation runs is not this public
# repo's to publish (see docs/repo_visibility.md in the superproject).
CRASH_LOG = (
    'INFO 09-21 10:00:00 api_server.py:1 vLLM API server version 0.25.1\n'
    'ValueError: The checkpoint you are trying to load has model type '
    '`example_moe_v1` but Transformers does not recognize this architecture. '
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
    # An UNRECOGNISED crash: one restart is not yet a loop. (A recognised
    # unrecoverable error is conclusive on the first crash -- see below.)
    crash_loop(ctl, docker, gid, state=state, restarts=restarts, exit_code=exit_code,
               logs='INFO starting\nSegmentation fault\n')
    assert ctl.backend.startup_failure(ledger.get_deployment(gid)) is None


HUB_TIMEOUT_LOG = (
    'INFO 09-21 10:00:00 api_server.py:1 vLLM API server version 0.25.1\n'
    'requests.exceptions.ConnectionError: HTTPSConnectionPool(host='
    "'huggingface.co', port=443): Max retries exceeded\n"
)


def test_a_transient_failure_is_left_to_the_restart_policy(tmp_path):
    """`restart: unless-stopped` exists so an unreachable hub resolves itself.

    Condemning a download that can succeed on the next attempt would waste a
    working model and make the restart policy pointless -- so a transient error
    is never fatal here, however many times it has restarted.
    """
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, restarts=CRASH_LOOP_RESTARTS + 3,
               logs=HUB_TIMEOUT_LOG)
    assert ctl.backend.startup_failure(ledger.get_deployment(gid)) is None


def test_an_unrecoverable_error_is_fatal_on_the_very_first_crash(tmp_path):
    """A rejected config fails identically on every restart, so do not wait."""
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, restarts=1)            # ONE restart, fatal log
    detail = ctl.backend.startup_failure(ledger.get_deployment(gid))
    assert detail is not None
    assert 'trust_remote_code' in detail


@pytest.mark.parametrize('logs,expected', [
    (CRASH_LOG, 'fatal'),
    ('ValueError: quantization fp8 is not supported on this device', 'fatal'),
    ('torch.OutOfMemoryError: CUDA out of memory', 'fatal'),
    (HUB_TIMEOUT_LOG, 'transient'),
    ('OSError: [Errno 98] Address already in use', 'transient'),
    ('INFO starting\nSegmentation fault', None),
    ('', None),
    # An unrecoverable error wins over a transient one earlier in the same log:
    # a hub timeout does not make a rejected config loadable.
    (HUB_TIMEOUT_LOG + CRASH_LOG, 'fatal'),
])
def test_the_log_classifier_separates_hopeless_from_retryable(logs, expected):
    assert classify_engine_log(logs) == expected


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


# -- a transient cause only buys time if something will actually retry ---------


def test_a_transient_failure_nothing_will_retry_is_still_hopeless(tmp_path):
    """`restart: unless-stopped` is what makes a hub timeout worth waiting for.

    A container that exited and will NOT be restarted has no next attempt, so
    waiting for one is exactly the 1800 s of held GPU this all exists to avoid.
    """
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, state='exited', restarts=0, exit_code=1,
               logs='ConnectionError: Max retries exceeded with url: /api/models\n')
    for container in docker.containers.values():
        container['restart_policy'] = 'no'

    detail = ctl.backend.startup_failure(ledger.get_deployment(gid))
    assert detail is not None
    assert 'will not be restarted' in detail
    assert 'Max retries exceeded' in detail          # the cause is still reported


def test_a_transient_failure_that_docker_will_retry_is_left_alone(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, state='exited', restarts=3, exit_code=1,
               logs='ConnectionError: Max retries exceeded with url: /api/models\n')
    for container in docker.containers.values():
        container['restart_policy'] = 'unless-stopped'

    assert ctl.backend.startup_failure(ledger.get_deployment(gid)) is None


def test_an_exhausted_on_failure_budget_counts_as_no_retry(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    out = acquire(ctl, 'one')
    gid = out.deployments[0].id
    crash_loop(ctl, docker, gid, state='exited', restarts=3, exit_code=1,
               logs='Consistency check failed: file should be of size 100\n')
    for container in docker.containers.values():
        container.update(restart_policy='on-failure', restart_max=3)

    assert 'will not be restarted' in ctl.backend.startup_failure(
        ledger.get_deployment(gid))


def test_an_interrupted_wait_does_not_leave_the_lease_holding_a_gpu(tmp_path):
    """Ctrl-C during the readiness wait must release, like a timeout does.

    Measured on aiq-gpu: an interrupted `run` left its lease ACTIVE and its
    deployment LIVE, so `evict --all` could not reclaim the GPU (eviction is for
    idle deployments) and every later acquire was refused -- the stack was not
    quiescent, so the catalog it had been edited to no longer matched the frozen
    epoch. It stayed that way until the 2 h TTL.
    """
    ledger, ctl, docker = make(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    ctl.wait_ready = interrupted
    with pytest.raises(KeyboardInterrupt):
        out = acquire(ctl, 'one', wait=True)
        assert out is None                      # never returns
    # The lease is gone and nothing is left LIVE holding the GPU.
    with ledger.store.transaction() as conn:
        held = conn.execute(
            "select id from leases where state != 'released'").fetchall()
        live = conn.execute(
            "select id from deployments where state = 'live'").fetchall()
    assert held == [] and live == []
