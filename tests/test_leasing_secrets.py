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
    assert ctl.backend.gateway_accepts(new) is True
    assert ctl.backend.gateway_accepts(old) is False


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
    assert ctl.backend.gateway_accepts('sk-anything', wait=5.0) is None
