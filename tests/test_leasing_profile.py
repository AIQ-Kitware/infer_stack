"""The published profile: renders use frozen settings, not each caller's."""

from __future__ import annotations

import json

import pytest
import yaml

from infer_stack.leasing import Catalog, Controller, Ledger, SqliteStore
from infer_stack.leasing.profile import CatalogConflict, CatalogUnion, ProfileMismatch
from test_leasing_compose import IMAGES, PORTS, STATE, FakeDocker, FakeHttp

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.compose import ComposeBackend


def cat(endpoint, source=None, **extra):
    return {
        'models': {'m': {'source': f'hf://org/{source or endpoint}'}},
        'endpoints': {endpoint: {'engine': 'vllm', 'model': 'm', **extra}},
    }


def backend(state_dir, *, catalog=None, **kw):
    return ComposeBackend(
        state_dir=state_dir, inventory=simulate_inventory('4x80'), run=FakeDocker(),
        http=FakeHttp(state_dir), images=kw.pop('images', IMAGES), ports=PORTS,
        state=STATE, catalog=catalog, **kw,
    )


def controller(tmp_path, **kw):
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    return ledger, Controller(ledger, backend(tmp_path / 'state', **kw))


def compose(ctl):
    return yaml.safe_load(ctl.backend.compose_file.read_text())


def test_first_mutation_freezes_the_invocation_settings(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, ui=True)
    assert ledger.profile() is None                      # opening freezes nothing
    out = ctl.acquire('x', a.resolve_names(['alpha']), wait=False)
    profile = ledger.profile()
    assert profile['backend'] == 'compose' and profile['ui'] is True
    assert 'allowed_gpus' not in profile
    assert 'open-webui' in compose(ctl)['services']

    # A later caller with different settings and newer image pins renders the same.
    images = {**IMAGES, 'litellm': 'litellm:someday'}
    _, ctl2 = controller(tmp_path, catalog=a, ui=False, images=images)
    ctl2.release(out.lease.id)
    after = compose(ctl2)
    assert 'open-webui' in after['services']
    assert after['services']['litellm']['image'] == IMAGES['litellm']
    assert ledger.profile() == profile


def test_drift_is_warned_once(tmp_path):
    from infer_stack._log import logger

    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, ui=True)
    ctl.gc()
    seen = []
    logger.enable('infer_stack')
    handle = logger.add(lambda m: seen.append(m.record['message']), level='WARNING')
    try:
        _, ctl2 = controller(tmp_path, catalog=a, ui=False)
        ctl2.gc()
        ctl2.gc()
    finally:
        logger.remove(handle)
        logger.disable('infer_stack')
    warnings = [m for m in seen if 'published profile' in m]
    assert len(warnings) == 1 and 'ui' in warnings[0]


def test_acquire_outside_the_published_catalog_is_refused_before_writing(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    b = Catalog.from_dict(cat('beta'))
    ledger, ctl = controller(tmp_path, catalog=a)
    ctl.gc()                                              # freezes {alpha}
    _, ctl2 = controller(tmp_path, catalog=b)
    with pytest.raises(ProfileMismatch, match='config publish'):
        ctl2.acquire('x', b.resolve_names(['beta']), wait=False)
    assert ledger.status()[0] == [] and ledger.publication_pending() is None


def test_a_changed_definition_is_refused(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    changed = Catalog.from_dict(cat('alpha', runtime={'max_model_len': 1024}))
    _, ctl = controller(tmp_path, catalog=a)
    ctl.gc()
    with pytest.raises(ProfileMismatch, match='differs'):
        ctl.acquire('x', changed.resolve_names(['alpha']), wait=False)


def test_union_of_catalogs_serves_either_runbook(tmp_path):
    union = CatalogUnion.from_sources([cat('alpha'), cat('beta')])
    ledger, ctl = controller(tmp_path, catalog=union)
    ctl.gc()
    # A runbook that only knows `beta` passes its own catalog: no drift, accepted.
    b = Catalog.from_dict(cat('beta'))
    _, ctl2 = controller(tmp_path, catalog=b)
    ctl2.acquire('x', b.resolve_names(['beta']), wait=False)
    assert ctl2.backend.catalog.endpoints.keys() == {'alpha', 'beta'}


def test_union_rejects_conflicts_and_deduplicates_identical_definitions():
    assert sorted(CatalogUnion.from_sources([cat('alpha'), cat('alpha')]).endpoints) == ['alpha']
    with pytest.raises(CatalogConflict):
        CatalogUnion.from_sources([cat('alpha'), cat('alpha', source='other')])
    bundles = [{**cat('alpha'), 'bundles': {'b': ['alpha']}},
               {**cat('alpha'), 'bundles': {'b': []}}]
    with pytest.raises(CatalogConflict, match='bundle'):
        CatalogUnion.from_sources(bundles)


def test_recovery_places_a_crashed_acquire_within_its_own_gpus(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, allowed_gpus=[3])
    ctl.gc()
    # The acquire commits with its scope, then dies before rendering.
    ledger.mark_publication_pending(apply_requested=True,
                                    placement_context={'allowed_gpus': [3]})
    res = ledger.acquire('x', a.resolve_names(['alpha']))
    # A different caller, allowed only GPU 0, runs the next operation.
    _, ctl2 = controller(tmp_path, catalog=a, allowed_gpus=[0])
    ctl2.gc()
    sidecar = json.loads((tmp_path / 'state' / 'leasing-compose-state.json').read_text())
    assert sidecar['assignments'][res.deployments[0].id] == [3]
    assert ledger.publication_pending() is None


def test_an_acquire_clears_its_scope_once_rendered(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, allowed_gpus=[2])
    ctl.acquire('x', a.resolve_names(['alpha']), wait=False, apply=False)
    assert ledger.publication_pending()['placement_context'] is None


def test_reverse_proxy_config_is_snapshotted(tmp_path):
    conf = tmp_path / 'nginx.conf'
    conf.write_text('events {}\n# v1\n')
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, reverse_proxy=True,
                             reverse_proxy_config=str(conf))
    ctl.gc()
    conf.write_text('events {}\n# edited later\n')
    _, ctl2 = controller(tmp_path, catalog=a, reverse_proxy=True,
                         reverse_proxy_config=str(conf))
    ctl2.gc()
    snapshot = tmp_path / 'state' / 'reverse-proxy.conf'
    assert snapshot.read_text().endswith('# v1\n')
    assert str(snapshot) in json.dumps(compose(ctl2))


def test_a_different_backend_kind_is_refused(tmp_path):
    from infer_stack.backends.kubeai import KubeaiBackend

    ledger, ctl = controller(tmp_path)
    ctl.gc()
    other = Controller(ledger, KubeaiBackend(state_dir=tmp_path / 'k', assume_yes=True))
    with pytest.raises(ProfileMismatch, match='compose'):
        other.gc()                         # refused on the first mutation...


def test_config_publish_refuses_to_change_the_backend_kind(tmp_path):
    from infer_stack.backends.kubeai import KubeaiBackend

    ledger, ctl = controller(tmp_path)
    ctl.gc()
    kube = KubeaiBackend(state_dir=tmp_path / 'k', assume_yes=True, run=lambda args: '')
    other = Controller(ledger, kube)          # opening still works
    with pytest.raises(ProfileMismatch, match='not supported'):
        other.publish_profile(kube.render_profile())
    assert ledger.profile()['backend'] == 'compose'


def test_explicit_missing_or_broken_catalog_is_an_error_after_publishing(tmp_path, monkeypatch):
    from infer_stack.cli import commands_leasing as cl

    state = tmp_path / 'state'
    monkeypatch.setattr(cl, '_make_backend',
                        lambda config, *, interactive=False: backend(state))
    db = str(tmp_path / 'ledger.db')
    f = tmp_path / 'a.yaml'
    f.write_text(yaml.safe_dump(cat('alpha')))
    assert cl.ConfigPublishCLI.main(argv=['--ledger', db, str(f), '--yes']) == 0
    with pytest.raises(SystemExit, match='catalog not found'):
        cl.AcquireCLI.main(argv=['alpha', '--ledger', db, '--catalog',
                                 str(tmp_path / 'typo.yaml'), '--no-wait', '--yes'])
    broken = tmp_path / 'broken.yaml'
    broken.write_text('endpoints: {e: {engine: nope}}\n')
    with pytest.raises(SystemExit, match='invalid catalog'):
        cl.AcquireCLI.main(argv=['alpha', '--ledger', db, '--catalog', str(broken),
                                 '--no-wait', '--yes'])


@pytest.mark.parametrize('failure', ['declined', 'raises'])
def test_publish_on_a_fresh_ledger_that_does_not_complete_stores_nothing(tmp_path, failure):
    from infer_stack.leasing.backend import ConvergeAborted

    ledger, ctl = controller(tmp_path, catalog=Catalog.from_dict(cat('alpha')))

    def fail(planned):
        raise ConvergeAborted('no') if failure == 'declined' else RuntimeError('boom')

    ctl.backend._approve_changes = fail
    candidate = {**ctl.backend.render_profile(), 'catalogs': [cat('alpha'), cat('beta')]}
    with pytest.raises((ConvergeAborted, RuntimeError)):
        ctl.publish_profile(candidate)
    assert ledger.profile() is None
    assert ledger.publication_pending() is None


def test_a_declined_recovery_render_keeps_the_crashed_acquires_scope(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a)
    ctl.gc()
    ledger.mark_publication_pending(apply_requested=True,
                                    placement_context={'allowed_gpus': [3]})
    ledger.acquire('x', a.resolve_names(['alpha']))
    _, ctl2 = controller(tmp_path, catalog=a, allowed_gpus=[0])

    def decline(planned):
        raise ConvergeAborted('no')

    ctl2.backend._approve_changes = decline
    with pytest.raises(ConvergeAborted):
        ctl2.gc()
    assert ledger.publication_pending()['placement_context'] == {'allowed_gpus': [3]}


def test_cli_refuses_an_edited_catalog_instead_of_serving_old_definitions(tmp_path, monkeypatch):
    from infer_stack.cli import commands_leasing as cl

    state = tmp_path / 'state'
    monkeypatch.setattr(cl, '_make_backend',
                        lambda config, *, interactive=False: backend(state))
    db = str(tmp_path / 'ledger.db')
    f = tmp_path / 'a.yaml'
    f.write_text(yaml.safe_dump(cat('alpha')))
    assert cl.ConfigPublishCLI.main(argv=['--ledger', db, str(f), '--yes']) == 0
    f.write_text(yaml.safe_dump(cat('alpha', runtime={'max_model_len': 1024})))
    with pytest.raises(SystemExit, match='not part of the published profile'):
        cl.AcquireCLI.main(argv=['alpha', '--ledger', db, '--catalog', str(f),
                                 '--no-wait', '--yes'])


# -- config publish (quiescent only) ----------------------------------------------


def test_publish_replaces_the_profile_when_quiescent(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, ui=True)
    out = ctl.acquire('x', a.resolve_names(['alpha']), wait=False)
    ctl.release_leases([out.lease.id], evict=True)

    new = {**ledger.profile(), 'ui': False,
           'catalogs': [cat('alpha'), cat('beta')]}
    ctl.publish_profile(new)
    assert ledger.profile() == new
    assert 'open-webui' not in compose(ctl)['services']
    b = Catalog.from_dict(cat('beta'))
    ctl.acquire('y', b.resolve_names(['beta']), wait=False)      # now published


def test_publish_refuses_with_an_active_lease(tmp_path):
    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a)
    ctl.acquire('x', a.resolve_names(['alpha']), wait=False)
    before = ledger.profile()
    with pytest.raises(ProfileMismatch, match='active lease'):
        ctl.publish_profile({**before, 'ui': not before['ui']})
    assert ledger.profile() == before and ledger.publication_pending() is None


def test_publish_refuses_while_a_deployment_container_exists(tmp_path):
    from infer_stack.leasing.residency import Container, Residency

    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a)
    ctl.gc()
    ctl.backend.residency = lambda: Residency({'grp-warm': (Container('c1', 'grp-warm', 'running'),)})
    with pytest.raises(ProfileMismatch, match='grp-warm'):
        ctl.publish_profile({**ledger.profile(), 'ui': False})


def test_a_declined_publish_stores_nothing(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    a = Catalog.from_dict(cat('alpha'))
    ledger, ctl = controller(tmp_path, catalog=a, ui=True)
    ctl.gc()
    before = ledger.profile()

    def decline(planned):
        raise ConvergeAborted('declined')

    ctl.backend._approve_changes = decline
    with pytest.raises(ConvergeAborted):
        ctl.publish_profile({**before, 'ui': False})
    assert ledger.profile() == before
    assert ctl.backend.ui is True


def test_config_publish_cli_publishes_a_union_and_rejects_conflicts(tmp_path, monkeypatch, capsys):
    from infer_stack.cli import commands_leasing as cl

    state = tmp_path / 'state'
    monkeypatch.setattr(cl, '_make_backend', lambda config, *, interactive=False: backend(state))
    db = str(tmp_path / 'ledger.db')
    fa, fb, fc = (tmp_path / n for n in ('a.yaml', 'b.yaml', 'c.yaml'))
    fa.write_text(yaml.safe_dump(cat('alpha')))
    fb.write_text(yaml.safe_dump(cat('beta')))
    fc.write_text(yaml.safe_dump(cat('alpha', source='other')))

    assert cl.ConfigPublishCLI.main(argv=['--ledger', db, str(fa), str(fb), '--yes']) == 0
    assert len(Ledger(SqliteStore(db)).profile()['catalogs']) == 2
    with pytest.raises(SystemExit, match='defined differently'):
        cl.ConfigPublishCLI.main(argv=['--ledger', db, str(fa), str(fc), '--yes'])
