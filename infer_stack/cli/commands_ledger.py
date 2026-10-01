"""Recovery ledger lifecycle; configuration/catalogs remain user authority."""
from __future__ import annotations

import json

import kwconf as kw

from .commands_leasing import _ApprovalMixin, _make_backend, _open_controller


class LedgerRotateCLI(_ApprovalMixin):
    """Archive a quiescent old backend ledger and start the configured backend.

    Default: verify and preview. --yes authorizes rotation; it never releases
    leases or tears down workloads. Configuration and catalog files are retained.
    """

    __command__ = 'rotate'
    json = kw.Value(False, isflag=True, help='Emit the transition summary as JSON.')
    dry_run = kw.Value(False, isflag=True, alias=['plan'], help='Verify and preview even with --yes.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.profile import ProfileMismatch
        from ..leasing.transition import rotate_backend

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        current = controller.ledger.profile()
        target = controller.invocation_profile() or {}
        old_backend = None
        if current and current.get('backend') != target.get('backend'):
            old_config = cls.cli(argv=False, data={**config.asdict(), 'backend': current['backend']})
            old_backend = _make_backend(old_config)
        try:
            summary = rotate_backend(controller, old_backend, apply=bool(config.yes and not config.dry_run))
        except (ProfileMismatch, OSError, RuntimeError) as ex:
            raise SystemExit(f'ledger rotate: {ex}') from ex
        if config.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Recovery backend: {summary['from_backend']} -> {summary['to_backend']}")
            print(summary['detail'])
            if summary['archive']:
                print(f"History archive: {summary['archive']} ({summary['leases']} leases, {summary['deployments']} deployments)")
            if summary['changed']:
                print('New recovery epoch initialized; acquire now uses the configured backend.')
            elif summary['archive']:
                print('No changes made. Re-run infer-stack ledger rotate --yes to commit.')
        return 0


class LedgerArchivesCLI(_ApprovalMixin):
    """List archived recovery epochs and their inspectable SQLite ledger paths."""

    __command__ = 'archives'
    json = kw.Value(False, isflag=True, help='Emit archive metadata as JSON.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        archives = controller.ledger.store.meta_json('ledger_archives', [])
        if config.json:
            print(json.dumps(archives, indent=2))
        elif not archives:
            print('No archived recovery epochs.')
        else:
            for archive in archives:
                print(f"{archive['backend']}: {archive['path']} ({archive['leases']} leases, {archive['deployments']} deployments)")
                print(f"  inspect: infer-stack leases --backend={archive['backend']} --ledger={archive['path']}")
        return 0


class LedgerModalCLI(kw.ModalCLI):
    """Preview/rotate backend recovery epochs and inspect preserved history."""

    __command__ = 'ledger'
    rotate = LedgerRotateCLI
    archives = LedgerArchivesCLI
