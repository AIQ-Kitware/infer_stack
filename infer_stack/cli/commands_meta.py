"""Meta / introspection subcommands.

Hosts commands that report on infer-stack itself rather than render or run a
deployment:

* ``infer-stack version``      — print the installed package version.
* ``infer-stack config paths`` — show the config / data / bind-mount paths
  infer-stack reads and writes, with an exists/missing status for each
  (analogous to ``aivm config paths``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scriptconfig as scfg

from .. import __version__
from ..config import kubeai_local_values_path
from ..config import normalized_state
from .context import _apply_path_overrides
from .context import _safe_load_config
from .context import backend_name
from .context import config_path
from .context import generated_dir
from .context import kubeai_generated_dir
from .context import models_path
from .context import plan_path
from .context import runtime_dir_for_config
from .options import _PathOverridesMixin
from ..paths import config_root
from ..paths import data_root


class VersionCLI(scfg.DataConfig):
    """Print the installed infer-stack version."""

    __command__ = 'version'

    @classmethod
    def main(cls, argv=True, **kwargs):
        cls.cli(argv=argv, data=kwargs)
        print(f'infer-stack {__version__}')
        return 0


def _path_status(path: Path) -> str:
    try:
        if path.exists():
            return 'exists'
    except PermissionError:
        return 'permission-denied'
    except OSError as ex:
        return f'error:{ex.__class__.__name__}'
    return 'missing'


def _entry(label: str, path: Path, *, kind: str) -> dict[str, str]:
    return {
        'label': label,
        'kind': kind,
        'status': _path_status(path),
        'path': str(path),
    }


class ConfigPathsCLI(_PathOverridesMixin):
    """Show the config, data, and bind-mount paths infer-stack uses.

    Reports the editable config files under ``config_root()`` plus the
    rendered-artifact and bind-mount locations under ``data_root()``. Each
    entry is tagged with its kind (file/dir) and an exists/missing status, so
    the same command works as a layout reference and a quick health check.
    Honours ``--config-dir`` / ``--data-dir`` (and the matching env vars) so it
    reports the layout a given invocation would actually use.
    """

    __command__ = 'paths'

    target: Any = scfg.Value(
        'all',
        position=1,
        help='Path group to show: all, config, data, or state.',
    )
    json: Any = scfg.Value(
        False,
        isflag=True,
        help='Emit the path groups as JSON instead of human-readable text.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)

        target = str(config.target or 'all').strip().lower()
        valid = {'all', 'config', 'data', 'state'}
        if target not in valid:
            raise SystemExit(
                f'Unknown path group {target!r}. Expected one of: '
                + ', '.join(sorted(valid))
            )

        # Resolve against the persisted config when present, else defaults,
        # so paths reflect the user's actual state.* / generated_dir choices.
        cfg = _safe_load_config()
        backend = backend_name(cfg)

        groups: dict[str, list[dict[str, str]]] = {}
        if target in {'all', 'config'}:
            entries = [
                _entry('config_root', config_root(), kind='dir'),
                _entry('config.yaml', config_path(), kind='file'),
                _entry('models.yaml', models_path(), kind='file'),
                _entry(
                    'kubeai_values_local',
                    kubeai_local_values_path(),
                    kind='file',
                ),
            ]
            groups['config'] = entries
        if target in {'all', 'data'}:
            entries = [
                _entry('data_root', data_root(), kind='dir'),
                _entry('generated_dir', generated_dir(cfg), kind='dir'),
                _entry('plan.yaml', plan_path(cfg), kind='file'),
                _entry(
                    'runtime_dir', runtime_dir_for_config(cfg), kind='dir'
                ),
            ]
            if backend == 'kubeai':
                entries.append(
                    _entry(
                        'kubeai_generated_dir',
                        kubeai_generated_dir(cfg),
                        kind='dir',
                    )
                )
            groups['data'] = entries
        if target in {'all', 'state'}:
            state = normalized_state(cfg.get('state'))
            groups['state'] = [
                _entry(name, Path(value), kind='dir')
                for name, value in state.items()
            ]

        if config.json:
            print(json.dumps(groups, indent=2))
            return 0

        print('infer-stack paths')
        print(f'backend: {backend}')
        for group, entries in groups.items():
            print(f'{group}:')
            for e in entries:
                print(
                    f'  {e["label"]} ({e["kind"]}, {e["status"]}): {e["path"]}'
                )
        return 0


class ConfigModalCLI(scfg.ModalCLI):
    """Inspect infer-stack configuration."""

    __command__ = 'config'

    paths = ConfigPathsCLI
