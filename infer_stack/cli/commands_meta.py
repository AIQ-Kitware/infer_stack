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


def _status_style(status: str) -> str:
    if status == 'exists':
        return 'green'
    if status == 'missing':
        return 'yellow'
    if status == 'glob':
        return 'cyan'
    return 'red'  # error:* / permission-denied


def _render_plain(groups: dict, backend: str) -> None:
    print('infer-stack paths')
    print(f'backend: {backend}')
    for group, entries in groups.items():
        print(f'{group}:')
        for e in entries:
            print(
                f'  {e["label"]} ({e["kind"]}, {e["status"]}): {e["path"]}'
            )


def _render_rich(groups: dict, backend: str, console) -> None:
    from rich.table import Table
    from rich.text import Text

    console.print('[bold]infer-stack paths[/bold]')
    console.print(f'backend: [bold cyan]{backend}[/bold cyan]')
    for group, entries in groups.items():
        table = Table(
            title=group,
            title_justify='left',
            title_style='bold magenta',
            box=None,
            header_style='dim',
            pad_edge=False,
            padding=(0, 2, 0, 0),
        )
        table.add_column('name', style='cyan', no_wrap=True)
        table.add_column('kind', style='dim')
        table.add_column('status', no_wrap=True)
        table.add_column('path', overflow='fold')
        for e in entries:
            status = e['status']
            if status == 'exists':
                path_text = Text(e['path'])
            elif status == 'missing':
                path_text = Text(e['path'], style='dim')
            else:
                path_text = Text(e['path'], style='red')
            table.add_row(
                e['label'],
                e['kind'],
                Text(status, style=_status_style(status)),
                path_text,
            )
        console.print(table)


class ConfigPathsCLI(_PathOverridesMixin):
    """Show the config, data, and bind-mount paths infer-stack uses.

    Reports the editable config files under ``config_root()`` plus the
    resolved rendered-artifact and bind-mount locations (``generated_dir``,
    ``runtime_dir``, and the ``state.*`` caches). Each entry is tagged with its
    kind (file/dir) and an exists/missing status, so the same command works as
    a layout reference and a quick health check. Honours ``--config-dir`` /
    ``--data-dir`` (and the matching env vars) so it reports the layout a given
    invocation would actually use.

    Output is colorized when writing to a terminal; piped/redirected output
    stays plain so it remains greppable (use ``--json`` for machine parsing).
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

        from rich.console import Console

        console = Console()
        if console.is_terminal:
            _render_rich(groups, backend, console)
        else:
            _render_plain(groups, backend)
        return 0


class ConfigModalCLI(scfg.ModalCLI):
    """Inspect infer-stack configuration."""

    __command__ = 'config'

    paths = ConfigPathsCLI
