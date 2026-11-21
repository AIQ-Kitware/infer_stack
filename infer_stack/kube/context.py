"""One persisted administrative target for owned Kubernetes subprocesses."""
from __future__ import annotations

from pathlib import Path

from ..paths import get_setting, load_settings, save_settings


def command(args: list[str]) -> list[str]:
    """Explicit command flags win; otherwise use infer-stack's saved target.

    Leaving the setting absent preserves kubectl/Helm's environment defaults.
    Nothing changes the caller's shell or unrelated default Kubernetes context.
    """
    if not args or args[0] not in {'kubectl', 'helm'}:
        return args
    if any(a == '--kubeconfig' or a.startswith('--kubeconfig=') for a in args):
        return args
    path = get_setting('kubeconfig')
    if not path:
        return args
    if not isinstance(path, str):
        raise ValueError('infer-stack kubeconfig setting must be a path string')
    return [args[0], '--kubeconfig', str(Path(path).expanduser().resolve()), *args[1:]]


def remember(path: str | Path) -> None:
    """Persist only the path to the already provisioned private credentials."""
    settings = load_settings()
    settings['kubeconfig'] = str(Path(path).expanduser().resolve())
    save_settings(settings)
