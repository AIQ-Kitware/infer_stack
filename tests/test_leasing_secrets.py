"""`infer-stack secrets rotate`: replacing the LiteLLM master key.

LiteLLM encrypts the credentials it stores in its database with
LITELLM_SALT_KEY, or with the master key when that is unset. So a rotation that
only changed the master key would leave every DB-stored route (dynamic
routing) undecryptable. The first rotation pins the salt to the old key.
"""

from __future__ import annotations

import pytest
import yaml

from infer_stack.env_utils import parse_env_file
from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Controller, Ledger, SqliteStore
from infer_stack.leasing.compose import ComposeBackend
from infer_stack.leasing.gateway import (
    API_KEY_ENV,
    SALT_KEY_ENV,
    set_master_key,
)
from infer_stack.leasing.profile import ProfileMismatch
from infer_stack.leasing.residency import FINGERPRINT_LABEL
from test_leasing_admission import CAT, Clock, acquire
from test_leasing_compose import IMAGES, PORTS, STATE, FakeDocker, FakeResp


class Gateway:
    """Accepts exactly the key currently in the .env, like a recreated LiteLLM."""

    def __init__(self, env_path):
        self.env_path = env_path
        self.down = False

    def get(self, url, headers=None, **kw):
        if self.down:
            raise ConnectionError('refused')
        key = parse_env_file(self.env_path).get(API_KEY_ENV)
        if headers and headers.get('Authorization') == f'Bearer {key}':
            return FakeResp(200, {'data': []})
        return FakeResp(400, {'error': 'Authentication Error'})   # what LiteLLM sends


def make(tmp_path):
    state = tmp_path / 'state'
    clock = Clock()
    backend = ComposeBackend(
        state_dir=state, inventory=simulate_inventory('2x80'), run=FakeDocker(),
        http=Gateway(state / '.env'), images=IMAGES, ports=PORTS, state=STATE,
        catalog=CAT, litellm=True, ui=False, sleep=clock.sleep, clock=clock,
    )
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')), clock=clock)
    return ledger, Controller(ledger, backend, clock=clock, sleep=clock.sleep)


def env(ctl):
    return parse_env_file(ctl.backend.gateway._env_path)


def gateway(ctl):
    doc = yaml.safe_load(ctl.backend.compose_file.read_text())
    return doc['services']['litellm']


def test_rotation_replaces_the_key_and_recreates_the_gateway(tmp_path):
    ledger, ctl = make(tmp_path)
    ctl.release(acquire(ctl, 'one').lease.id)
    old = env(ctl)[API_KEY_ENV]
    fingerprint = gateway(ctl)['labels'][FINGERPRINT_LABEL]

    ctl.rotate_gateway_key()

    new = env(ctl)[API_KEY_ENV]
    assert new != old and new.startswith('sk-')
    assert gateway(ctl)['labels'][FINGERPRINT_LABEL] != fingerprint   # recreated
    assert ctl.backend.front_door().gateway_accepts(new) is True
    assert ctl.backend.front_door().gateway_accepts(old) is False


def test_the_first_rotation_pins_the_salt_to_the_old_key(tmp_path):
    ledger, ctl = make(tmp_path)
    ctl.release(acquire(ctl, 'one').lease.id)
    assert SALT_KEY_ENV not in gateway(ctl)['environment']
    old = env(ctl)[API_KEY_ENV]

    ctl.rotate_gateway_key()
    assert env(ctl)[SALT_KEY_ENV] == old          # what stored routes were encrypted with
    assert gateway(ctl)['environment'][SALT_KEY_ENV] == '${LITELLM_SALT_KEY}'

    ctl.rotate_gateway_key()
    assert env(ctl)[SALT_KEY_ENV] == old          # never moves again


def test_no_salt_is_rendered_until_one_exists(tmp_path):
    """LiteLLM uses an EMPTY salt as the key, so no `${...:-}` default."""
    ledger, ctl = make(tmp_path)
    acquire(ctl, 'one')
    assert SALT_KEY_ENV not in gateway(ctl)['environment']
    assert SALT_KEY_ENV not in ctl.backend.compose_file.read_text()


def test_rotation_is_refused_while_a_lease_holds_the_key(tmp_path):
    ledger, ctl = make(tmp_path)
    acquire(ctl, 'one')
    old = env(ctl)[API_KEY_ENV]
    with pytest.raises(ProfileMismatch, match='active'):
        ctl.rotate_gateway_key()
    assert env(ctl)[API_KEY_ENV] == old
    ctl.rotate_gateway_key(force=True)
    assert env(ctl)[API_KEY_ENV] != old


def test_a_declined_apply_keeps_the_old_key(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    ledger, ctl = make(tmp_path)
    ctl.release(acquire(ctl, 'one').lease.id)
    before = env(ctl)

    def decline(planned):
        raise ConvergeAborted('no')

    ctl.backend._approve_changes = decline
    with pytest.raises(ConvergeAborted):
        ctl.rotate_gateway_key()
    assert env(ctl) == before                     # key restored, no salt added


def test_no_gateway_means_nothing_to_rotate(tmp_path):
    ledger, ctl = make(tmp_path)
    ctl.backend.litellm = False
    with pytest.raises(ProfileMismatch, match='no LiteLLM gateway'):
        ctl.rotate_gateway_key()


def test_a_hand_set_key_must_look_like_a_litellm_key(tmp_path):
    path = tmp_path / '.env'
    set_master_key(path, 'sk-first')
    with pytest.raises(ValueError, match='sk-'):
        set_master_key(path, 'not-a-key')         # master_key() would replace it
    set_master_key(path, 'sk-second')
    assert parse_env_file(path) == {API_KEY_ENV: 'sk-second', SALT_KEY_ENV: 'sk-first'}


def test_an_unreachable_gateway_is_not_a_rejection(tmp_path):
    ledger, ctl = make(tmp_path)
    ctl.backend.http.down = True
    assert ctl.backend.front_door().gateway_accepts('sk-anything', wait=5.0) is None


# -- rotation is a transaction (queue item 16) --------------------------------------


class StartedGateway(Gateway):
    """Accepts the key its container STARTED with, as a real LiteLLM does.

    The key is read from the .env when the litellm container is (re)created,
    not per request: a key written to the file afterwards is not the one the
    gateway checks.
    """

    def __init__(self, env_path, docker):
        super().__init__(env_path)
        self.docker = docker
        self.container = None
        self.key = None

    def get(self, url, headers=None, **kw):
        current = next((cid for cid, c in self.docker.containers.items()
                        if c['service'] == 'litellm'), None)
        if current != self.container:              # (re)created: read the key now
            self.container = current
            self.key = parse_env_file(self.env_path).get(API_KEY_ENV)
        if headers and headers.get('Authorization') == f'Bearer {self.key}':
            return FakeResp(200, {'data': []})
        return FakeResp(400, {'error': 'Authentication Error'})


def make_started(tmp_path):
    ledger, ctl = make(tmp_path)
    ctl.backend.http = StartedGateway(ctl.backend.gateway._env_path, ctl.backend.run)
    ctl.backend.gateway.http = ctl.backend.http
    ctl.release(acquire(ctl, 'one').lease.id)
    return ledger, ctl


def test_a_render_failure_before_apply_keeps_file_and_gateway_on_the_old_key(tmp_path):
    from infer_stack.leasing.residency import ResidencyUnknown

    ledger, ctl = make_started(tmp_path)
    old = env(ctl)[API_KEY_ENV]
    assert ctl.backend.front_door().gateway_accepts(old)
    real = ctl.backend.residency
    ctl.backend.residency = lambda: (_ for _ in ()).throw(ResidencyUnknown('docker down'))
    with pytest.raises(ResidencyUnknown):
        ctl.rotate_gateway_key()
    assert env(ctl)[API_KEY_ENV] == old            # the file says what the gateway runs
    ctl.backend.residency = real
    assert ctl.backend.front_door().gateway_accepts(old)


def test_a_declined_rotation_keeps_file_and_gateway_on_the_old_key(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    ledger, ctl = make_started(tmp_path)
    old = env(ctl)[API_KEY_ENV]
    ctl.backend._approve_changes = lambda planned: (_ for _ in ()).throw(ConvergeAborted('no'))
    with pytest.raises(ConvergeAborted):
        ctl.rotate_gateway_key()
    assert env(ctl)[API_KEY_ENV] == old and ctl.backend.front_door().gateway_accepts(old)


def test_a_failure_after_apply_began_keeps_the_new_key_and_converges(tmp_path):
    from infer_stack.leasing.backend import BackendTimeout

    ledger, ctl = make_started(tmp_path)
    old = env(ctl)[API_KEY_ENV]
    real = ctl.backend.apply
    ctl.backend.apply = lambda: (_ for _ in ()).throw(BackendTimeout('up timed out'))
    ctl.backend.settle_snapshot = lambda: ()
    with pytest.raises(BackendTimeout):
        ctl.rotate_gateway_key()
    new = env(ctl)[API_KEY_ENV]
    assert new != old                              # not reverted blindly
    assert ledger.publication_pending()['apply_requested']
    ctl.backend.apply = real
    ctl.apply_now()                                # the pending publication converges
    assert ctl.backend.front_door().gateway_accepts(new) and not ctl.backend.front_door().gateway_accepts(old)
    assert ledger.publication_pending() is None


def test_an_apply_that_cleanly_misses_the_runtime_keeps_the_old_key(tmp_path):
    """Re-review 2: `ApplyResult(runtime=False)` returned without an
    exception; the gateway was not recreated, so .env must not move ahead."""
    from infer_stack.leasing.backend import ApplyResult

    ledger, ctl = make_started(tmp_path)
    old = env(ctl)[API_KEY_ENV]
    ctl.backend.apply = lambda: ApplyResult(runtime=False, detail='render unreadable')
    with pytest.raises(ProfileMismatch, match='key was not changed.*render unreadable'):
        ctl.rotate_gateway_key()
    assert env(ctl)[API_KEY_ENV] == old
    assert ctl.backend.front_door().gateway_accepts(env(ctl)[API_KEY_ENV])   # what clients get works


# -- rotation under dynamic routing (queue item 24) ---------------------------------


class AdminGateway(StartedGateway):
    """A dynamic-routing LiteLLM: the admin API, answering only the key its
    container started with (read from the container's environment)."""

    def __init__(self, env_path, docker):
        super().__init__(env_path, docker)
        self.routes: dict[str, dict] = {}

    def _started_key(self):
        current = next((c for c in self.docker.containers.values()
                        if c['service'] == 'litellm'), None)
        return (current or {}).get('env', {}).get(API_KEY_ENV)

    def _authorized(self, headers):
        return bool(headers) and headers.get('Authorization') == f'Bearer {self._started_key()}'

    def get(self, url, headers=None, **kw):
        if url.endswith('/v1/model/info'):
            if not self._authorized(headers):
                return FakeResp(400, {'error': 'Authentication Error'})
            return FakeResp(200, {'data': list(self.routes.values())})
        if headers and headers.get('Authorization') == f'Bearer {self._started_key()}':
            return FakeResp(200, {'data': []})
        return FakeResp(400, {'error': 'Authentication Error'})

    def post(self, url, headers=None, **kw):
        if not self._authorized(headers):
            return FakeResp(400, {'error': 'Authentication Error'})
        body = kw.get('json') or {}
        if url.endswith('/model/new'):
            self.routes[body['model_info']['id']] = body
        elif url.endswith('/model/delete'):
            self.routes.pop(body['id'], None)
        return FakeResp(200, {})


def test_rotation_completes_under_a_running_dynamic_routing_gateway(tmp_path):
    """The running gateway holds K0; rotation writes K1 first. The apply's
    route-retirement phase must still talk to the gateway with K0, the
    recreated gateway then runs K1, and the routes verify with K1."""
    state = tmp_path / 'state'
    clock = Clock()
    docker = FakeDocker()
    backend = ComposeBackend(
        state_dir=state, inventory=simulate_inventory('2x80'), run=docker,
        http=None, images={**IMAGES, 'postgres': 'pg:test'}, ports=PORTS, state=STATE,
        catalog=CAT, litellm=True, ui=False, dynamic_routing=True,
        sleep=clock.sleep, clock=clock,
    )
    backend.http = backend.gateway.http = AdminGateway(backend.gateway._env_path, docker)
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')), clock=clock)
    ctl = Controller(ledger, backend, clock=clock, sleep=clock.sleep)
    ctl.release(acquire(ctl, 'one').lease.id)       # a keep-warm model, routed
    old = env(ctl)[API_KEY_ENV]
    assert backend.front_door().gateway_accepts(old) and backend.http.routes

    rec = ctl.rotate_gateway_key()

    new = env(ctl)[API_KEY_ENV]
    assert new != old and not rec.publication_pending
    assert backend.front_door().gateway_accepts(new) and not backend.front_door().gateway_accepts(old)
    assert backend.http.routes                      # verified with the new key
    assert ledger.publication_pending() is None


def test_rotation_completes_behind_kubeai_with_a_dynamic_host_gateway(tmp_path):
    """The same interaction on KubeAI: its host gateway is the same Compose
    machinery, so route retirement must use the running gateway's key."""
    from infer_stack.backends.kubeai import KubeaiBackend
    from test_leasing_kubeai import FakeKubectl

    clock = Clock()
    docker = FakeDocker()
    gateway = ComposeBackend(
        state_dir=tmp_path / 'gateway', inventory={'gpu_count': 0, 'gpus': []},
        run=docker, http=None, project='infer-stack-gateway',
        images={**IMAGES, 'postgres': 'pg:test'}, ports=PORTS, state=STATE,
        litellm=True, ui=False, dynamic_routing=True, sleep=clock.sleep, clock=clock,
    )
    gateway.http = gateway.gateway.http = AdminGateway(gateway.gateway._env_path, docker)
    backend = KubeaiBackend(state_dir=tmp_path / 'kubeai', run=FakeKubectl(),
                            http=gateway.http, gateway=gateway,
                            gateway_upstream='http://10.43.0.9/openai/v1',
                            default_resource_profile='gpu')
    backend.catalog = CAT
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')), clock=clock)
    ctl = Controller(ledger, backend, clock=clock, sleep=clock.sleep)
    ctl.release(acquire(ctl, 'one').lease.id)
    env_path = gateway.gateway._env_path
    old = parse_env_file(env_path)[API_KEY_ENV]
    assert gateway.front_door().gateway_accepts(old)

    ctl.rotate_gateway_key()

    new = parse_env_file(env_path)[API_KEY_ENV]
    assert new != old
    assert gateway.front_door().gateway_accepts(new) and not gateway.front_door().gateway_accepts(old)
    assert ledger.publication_pending() is None


def test_the_old_key_may_linger_briefly_behind_a_rollout(monkeypatch):
    """A rollout completes while the old pod still answers for a moment; the
    rotation's check waits for the rejection instead of failing on it."""
    import time

    from infer_stack.cli.commands_leasing import _old_key_rejected

    monkeypatch.setattr(time, 'sleep', lambda s: None)

    class Lingering:
        def __init__(self, answers):
            self.answers = list(answers)

        def gateway_accepts(self, key, wait=0.0):
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    assert _old_key_rejected(Lingering([True, True, False]), 'k')
    assert not _old_key_rejected(Lingering([True]), 'k', within=0.0)
