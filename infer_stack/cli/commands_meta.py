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
from ..config import kubeai_local_values_path, normalized_state
from ..paths import config_root
from .context import (
    _apply_path_overrides,
    _safe_load_config,
    backend_name,
    config_path,
    generated_dir,
    kubeai_generated_dir,
    models_path,
    plan_path,
    runtime_dir_for_config,
)
from .options import _PathOverridesMixin


class VersionCLI(scfg.DataConfig):
    """Print the installed infer-stack version."""

    __command__ = 'version'

    @classmethod
    def main(cls, argv=True, **kwargs):
        cls.cli(argv=argv, data=kwargs)
        print(f'infer-stack {__version__}')
        return 0


# ---------------------------------------------------------------------------
# help tree — the whole command surface at a glance (cf. `aivm help tree`)
# ---------------------------------------------------------------------------


def _iter_subcommands(modal: Any):
    """(name, cls) for each registered subcommand of a ModalCLI class."""
    out: dict[str, Any] = {}
    for attr, val in vars(modal).items():
        if attr.startswith('_'):
            continue
        if isinstance(val, type) and issubclass(
            val, (scfg.DataConfig, scfg.ModalCLI)
        ):
            name = getattr(val, '__command__', None) or attr.replace('_', '-')
            out[name] = val
    return sorted(out.items())


def _doc_one_line(cls: Any) -> str:
    doc = (getattr(cls, '__doc__', None) or '').strip()
    if not doc:
        doc = (getattr(cls, 'description', '') or '').strip()
    line = doc.splitlines()[0].strip() if doc else ''
    return line.replace('``', '').replace('`', '')  # drop rST backticks


def _is_group(sub: Any) -> bool:
    return isinstance(sub, type) and issubclass(sub, scfg.ModalCLI)


def _build_tree(modal: Any, node: Any) -> None:
    """Attach each subcommand of ``modal`` to a rich Tree ``node``."""
    from rich.text import Text

    for name, sub in _iter_subcommands(modal):
        label = Text()
        # Groups (submodals) in bold cyan, runnable leaves in green; the
        # one-line description trails in dim. Built from Text segments (not
        # markup) so brackets/backticks in docstrings can't break rendering.
        label.append(name, style='bold cyan' if _is_group(sub) else 'green')
        desc = _doc_one_line(sub)
        if desc:
            label.append('  ')
            label.append(desc, style='dim')
        child = node.add(label)
        if _is_group(sub):
            _build_tree(sub, child)


class HelpTreeCLI(scfg.DataConfig):
    """Print the full nested command tree with one-line descriptions."""

    __command__ = 'tree'

    @classmethod
    def main(cls, argv=True, **kwargs):
        cls.cli(argv=argv, data=kwargs)
        from rich.console import Console
        from rich.text import Text
        from rich.tree import Tree

        from infer_stack.cli import ManageCLI

        tree = Tree(Text('infer-stack', style='bold'))
        _build_tree(ManageCLI, tree)
        Console().print(tree)
        return 0


class HelpModalCLI(scfg.ModalCLI):
    """Help utilities (use ``infer-stack <command> --help`` for per-command help)."""

    __command__ = 'help'

    tree = HelpTreeCLI


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
        help='Path group to show: all, config, data, state, or leasing.',
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
        valid = {'all', 'config', 'data', 'state', 'leasing'}
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
        if target in {'all', 'leasing'}:
            from ..paths import data_root

            compose_dir = data_root() / 'leasing' / 'compose'
            groups['leasing'] = [
                _entry(
                    'ledger', data_root() / 'leasing' / 'ledger.db', kind='file'
                ),
                _entry('compose_dir', compose_dir, kind='dir'),
                _entry(
                    'docker-compose.yml',
                    compose_dir / 'docker-compose.yml',
                    kind='file',
                ),
                _entry(
                    'litellm_config.yaml',
                    compose_dir / 'litellm_config.yaml',
                    kind='file',
                ),
                _entry('env (secrets)', compose_dir / '.env', kind='file'),
                _entry(
                    'compose_state',
                    compose_dir / 'leasing-compose-state.json',
                    kind='file',
                ),
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


# ---------------------------------------------------------------------------
# config settings: durable defaults (backend, data_dir) in settings.yaml
# ---------------------------------------------------------------------------

# Keys the leasing world actually honors (others are allowed but warned about).
KNOWN_SETTINGS = {
    'backend': 'Default serving backend (compose | kubeai | null).',
    'data_dir': 'Where docker-mounted state lives (overrides the XDG default).',
}


class ConfigInitCLI(_PathOverridesMixin):
    """Set up the durable settings (data dir + default backend), interactively.

    Prompts for each setting and shows them for confirmation before writing
    (like ``aivm``). Use ``--yes`` for non-interactive scripting (accepts the
    provided flags / current values / defaults without prompting); the same
    non-interactive path is taken automatically when stdin is not a TTY.
    """

    __command__ = 'init'
    yes = scfg.Value(
        False, isflag=True, alias=['y'],
        help='Non-interactive: write without prompting/confirming.',
    )
    backend = scfg.Value(
        None, choices=['compose', 'kubeai', 'null'],
        help='Preset the default backend (skips that prompt).',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        import sys

        from ..paths import (
            data_root,
            load_settings,
            save_settings,
            settings_path,
        )

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        settings = load_settings()
        path = settings_path()

        # Proposed values: explicit flag > current setting > sensible default.
        # data_root() already resolves --data-dir override / $env / setting / XDG.
        data_dir = config.data_dir or settings.get('data_dir') or str(data_root())
        backend = config.backend or settings.get('backend') or 'compose'

        interactive = (
            not config.yes and sys.stdin.isatty() and sys.stdout.isatty()
        )
        if interactive:
            from rich.console import Console
            from rich.prompt import Confirm, Prompt
            from rich.table import Table

            console = Console()
            console.print('[bold]infer-stack config init[/]\n')
            data_dir = Prompt.ask(
                'Data dir (docker-mounted weight/state)', default=data_dir
            )
            backend = Prompt.ask(
                'Default backend',
                choices=['compose', 'kubeai', 'null'],
                default=backend,
            )
            table = Table(show_header=True, header_style='bold')
            table.add_column('setting')
            table.add_column('value', style='green')
            table.add_row('data_dir', data_dir)
            table.add_row('backend', backend)
            console.print(table)
            if not Confirm.ask(f'Write these to {path}?', default=True):
                console.print('[yellow]aborted — nothing written[/]')
                return 0

        settings['data_dir'] = data_dir
        settings['backend'] = backend
        save_settings(settings)
        print(f'wrote settings -> {path}')
        print('next: `infer-stack catalog init` to add models/endpoints')
        return 0


class ConfigSetCLI(_PathOverridesMixin):
    """Persist a durable default, e.g. ``config set backend compose``."""

    __command__ = 'set'
    key = scfg.Value(None, position=1, type=str)
    value = scfg.Value(None, position=2, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        import yaml

        from ..paths import load_settings, save_settings

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.key or config.value is None:
            raise SystemExit('config set: KEY and VALUE are required')
        if config.key not in KNOWN_SETTINGS:
            print(f"warning: '{config.key}' is not a recognized setting "
                  f'(known: {", ".join(sorted(KNOWN_SETTINGS))})')
        settings = load_settings()
        settings[config.key] = yaml.safe_load(config.value)
        path = save_settings(settings)
        print(f"set {config.key} = {settings[config.key]!r}  ({path})")
        return 0


class ConfigGetCLI(_PathOverridesMixin):
    """Print one setting's value (or all settings)."""

    __command__ = 'get'
    key = scfg.Value(None, position=1, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..paths import load_settings

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        settings = load_settings()
        if config.key:
            if config.key not in settings:
                raise SystemExit(f"'{config.key}' is not set")
            print(settings[config.key])
        else:
            for k, v in settings.items():
                print(f'{k}={v}')
        return 0


class ConfigShowCLI(_PathOverridesMixin):
    """Show the persisted settings and where they live."""

    __command__ = 'show'

    @classmethod
    def main(cls, argv=True, **kwargs):
        import yaml

        from ..paths import load_settings, settings_path
        from .commands_catalog import _print_yaml

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        settings = load_settings()
        print(f'# {settings_path()}')
        if settings:
            _print_yaml(yaml.safe_dump(settings, sort_keys=False))
        else:
            print('(no settings yet — `infer-stack config set …`)')
        return 0


class ConfigEditCLI(_PathOverridesMixin):
    """Open settings.yaml in $EDITOR."""

    __command__ = 'edit'

    @classmethod
    def main(cls, argv=True, **kwargs):
        import os
        import subprocess

        from ..paths import save_settings, settings_path

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        path = settings_path()
        if not path.exists():
            save_settings({})
        subprocess.run([*os.environ.get('EDITOR', 'vi').split(), str(path)],
                       check=False)
        return 0


class ConfigModalCLI(scfg.ModalCLI):
    """Inspect + manage infer-stack configuration (paths + durable settings)."""

    __command__ = 'config'

    init = ConfigInitCLI
    paths = ConfigPathsCLI
    show = ConfigShowCLI
    set = ConfigSetCLI
    get = ConfigGetCLI
    edit = ConfigEditCLI
