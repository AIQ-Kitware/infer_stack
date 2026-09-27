"""The published endpoint set commits with publication intent, never before.

``profile.catalogs`` is the published catalog union, from which gateway routes
are derived (docs/planning/external-endpoints.md, decision 1). Writing it is
publishing, so it commits in the same transaction as the publication marker
(and, for an acquire, the lease): a declined approval or a crash before that
commit leaves nothing published. Queue item 42.
"""

from __future__ import annotations

import pytest

from infer_stack.env_utils import write_env_file
from infer_stack.leasing import Catalog
from infer_stack.leasing.backend import ConvergeAborted
from test_external_endpoints import REMOTE, catalog
from test_leasing_profile import controller


def _keyed(tmp_path, cat):
    ledger, ctl = controller(tmp_path, catalog=cat)
    write_env_file(ctl.backend.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    return ledger, ctl


def _decline(ctl):
    def decline(planned):
        raise ConvergeAborted('declined')

    ctl.backend._approve_changes = decline


def _published(ledger):
    return sorted(n for s in (ledger.profile() or {}).get('catalogs') or []
                  for n in (s.get('endpoints') or {}))


def test_a_declined_first_external_access_publishes_nothing(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    _decline(ctl)
    with pytest.raises(ConvergeAborted):
        ctl.access('me', cat.resolve(['remote']))
    assert ledger.profile() is None
    assert ledger.publication_pending() is None
    assert not (ctl.backend.state_dir / 'litellm_config.yaml').exists()


def test_a_crash_before_the_first_commit_publishes_nothing(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)

    def crash(*args, **kwargs):
        raise RuntimeError('killed')

    ledger.publish_profile = crash
    with pytest.raises(RuntimeError, match='killed'):
        ctl.access('me', cat.resolve(['remote']))
    assert ledger.profile() is None and ledger.publication_pending() is None


def test_a_declined_first_acquire_leaves_no_lease_and_no_profile(tmp_path):
    cat = Catalog.from_dict(catalog())
    ledger, ctl = controller(tmp_path, catalog=cat)
    _decline(ctl)
    with pytest.raises(ConvergeAborted):
        ctl.acquire('me', cat.resolve_requests(['local']), wait=False)
    assert ledger.profile() is None
    assert ledger.status() == ([], [])


def test_a_crash_at_the_first_acquire_commit_publishes_nothing(tmp_path):
    cat = Catalog.from_dict(catalog())
    ledger, ctl = controller(tmp_path, catalog=cat)

    def crash(*args, **kwargs):
        raise RuntimeError('killed')

    ledger.acquire = crash
    with pytest.raises(RuntimeError, match='killed'):
        ctl.acquire('me', cat.resolve_requests(['local']), wait=False)
    assert _published(ledger) == []


def test_the_first_external_access_commits_profile_and_marker_together(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    seen = []
    real = ledger.publish_profile

    def watch(profile, **kwargs):
        seen.append((ledger.profile(), ledger.publication_pending()))
        real(profile, **kwargs)
        seen.append((ledger.profile(), ledger.publication_pending()))

    ledger.publish_profile = watch
    ctl.access('me', cat.resolve(['remote']))
    (before_profile, before_marker), (after_profile, after_marker) = seen
    assert before_profile is None and before_marker is None
    assert 'remote' in _published(ledger) and after_marker is not None


def test_an_acquire_commits_its_catalog_with_its_lease(tmp_path):
    """A candidate profile (a new catalog on an existing ledger) is written in
    the lease's transaction, not before it."""
    first = Catalog.from_dict(catalog())
    ledger, ctl = controller(tmp_path, catalog=first)
    ctl.acquire('me', first.resolve_requests(['local']), wait=False)
    more = catalog(remote=REMOTE)
    more['endpoints']['other'] = {'engine': 'vllm', 'model': 'm',
                                  'runtime': {'max_model_len': 1024}}
    wider = Catalog.from_dict(more)
    _, ctl2 = controller(tmp_path, catalog=wider)

    def crash(*args, **kwargs):
        raise RuntimeError('killed')

    ledger2 = ctl2.ledger
    ledger2.acquire = crash
    with pytest.raises(RuntimeError, match='killed'):
        ctl2.acquire('me', wider.resolve_requests(['other']), wait=False)
    assert 'other' not in _published(ledger)


def test_gc_on_a_fresh_ledger_publishes_no_endpoint(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    ctl.gc()
    assert _published(ledger) == []
    profile = ledger.profile()
    assert profile is None or profile.get('backend') == 'compose'


def test_a_first_apply_publishes_with_approval_or_not_at_all(tmp_path):
    """`stack up` / `apply` on a fresh ledger publishes the invocation's
    endpoints (so catalog routes exist before any model runs), through the
    same approved single commit as access."""
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    real = ctl.backend._approve_changes
    _decline(ctl)
    with pytest.raises(ConvergeAborted):
        ctl.apply_now()
    assert ledger.profile() is None and ledger.publication_pending() is None
    ctl.backend._approve_changes = real
    ctl.apply_now()
    assert _published(ledger) == ['local', 'remote']
    assert ledger.publication_pending() is None
