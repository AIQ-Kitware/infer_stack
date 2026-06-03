from __future__ import annotations

from .backends import render_compose_artifacts, render_kubeai_artifacts


def render_from_lock(lock_data: dict, *, assume_yes: bool = True) -> None:
    """Render backend artifacts.

    ``assume_yes`` defaults to True so programmatic callers and tests
    are unaffected. CLI entry points pass ``assume_yes=False`` to surface
    the per-file diff confirmation prompt.
    """
    backend = str(
        lock_data.get('deployment', {}).get('backend', 'compose')
    ).lower()
    if backend == 'kubeai':
        render_kubeai_artifacts(lock_data, assume_yes=assume_yes)
        return
    render_compose_artifacts(lock_data, assume_yes=assume_yes)
