"""Explicit, archived backend transitions under the publication lock."""
from __future__ import annotations

from pathlib import Path

from .models import LeaseState
from .profile import CatalogUnion, ProfileMismatch, canonical_digest


def rotate_backend(controller, old_backend, *, apply=False) -> dict:
    """Verify the frozen backend is quiescent, archive history, start a new epoch.

    Preview and commit both recheck leases and strict runtime observations. No
    observe() fallback is allowed to interpret a probe failure as quiescence.
    Archive-before-commit and one SQLite reset transaction make interruptions
    safe: before commit the old epoch remains intact; afterward retries are no-op.
    """
    store = controller.ledger.store
    target = controller.invocation_profile()
    if target is None or target.get('backend') not in {'compose', 'kubeai'}:
        raise ProfileMismatch('ledger rotate requires a configured Compose or KubeAI backend')
    # Validate proposed catalogs without publishing them or changing config files.
    if target.get('catalogs'):
        CatalogUnion.from_sources(target['catalogs'])
    with controller._global_lock():
        current = store.profile()
        archives = store.meta_json('ledger_archives', [])
        result = {'from_backend': (current or {}).get('backend'),
                  'to_backend': target['backend'], 'changed': False,
                  'archive': None, 'archives': archives}
        if current is None:
            raise ProfileMismatch('No active recovery snapshot to rotate; acquire normally starts one')
        if current.get('backend') == target['backend']:
            result['detail'] = 'Recovery backend already matches the configured backend; no rotation needed'
            return result
        leases, deployments = controller.ledger.status(virtual_expiry=True)
        active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
        if active:
            raise ProfileMismatch('ledger rotate refused: active leases remain: ' + ', '.join(active)
                                  + f". Release them using --backend={current['backend']} first")
        if old_backend is None:
            raise ProfileMismatch('Cannot verify the old backend; rotation refused')
        old_backend.recovery_profile.use_profile(current)
        try:
            blockers = getattr(old_backend, 'recovery_blockers', None)
            running = blockers() if blockers else [i.name for i in old_backend.instances()]
        except Exception as ex:
            raise ProfileMismatch(f'Cannot verify old backend quiescence: {ex}; rotation refused') from ex
        if running:
            raise ProfileMismatch('ledger rotate refused: old backend still owns runtime objects: '
                                  + ', '.join(running)
                                  + f". Run infer-stack stack down --backend={current['backend']}")
        version = store.admission_state_version()
        signature = canonical_digest({'profile': current, 'state_version': version})[:16]
        if store.path == ':memory:':
            raise ProfileMismatch('ledger rotate requires a file-backed ledger for history archives')
        path = Path(store.path).resolve()
        archive_path = path.parent / 'archives' / f'{path.stem}-{current["backend"]}-{signature}.db'
        result.update(archive=str(archive_path), leases=len(leases), deployments=len(deployments),
                      detail='Old runtime is quiescent; archive history and start a new recovery epoch')
        if not apply:
            return result
        # Prepare the target backend before touching the old epoch. Catalogs are
        # imported by the next acquire; no workloads/routes are started here.
        fresh = {**target, 'catalogs': []}
        controller._use_profile_candidate(fresh)
        store.backup_to(archive_path)
        archive = {'path': str(archive_path), 'backend': current['backend'],
                   'next_backend': target['backend'], 'leases': len(leases),
                   'deployments': len(deployments), 'created_at': controller.clock()}
        store.start_backend_epoch(fresh, archive, expected_profile=current, expected_version=version)
        controller._profile_error = None
        controller._applied_profile = fresh
        controller._fresh_profile = None
        result.update(changed=True, archives=[*archives, archive])
        return result
