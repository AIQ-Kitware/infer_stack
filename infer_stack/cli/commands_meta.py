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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scriptconfig as scfg

from .. import __version__
from ..paths import config_root, data_root, settings_path
from .context import _apply_path_overrides
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


def _render_plain(groups: dict) -> None:
    print('infer-stack paths')
    for group, entries in groups.items():
        print(f'{group}:')
        for e in entries:
            print(
                f'  {e["label"]} ({e["kind"]}, {e["status"]}): {e["path"]}'
            )


def _render_rich(groups: dict, console) -> None:
    from rich.table import Table
    from rich.text import Text

    console.print('[bold]infer-stack paths[/bold]')
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
        help='Path group to show: all, config, data, or leasing.',
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
        valid = {'all', 'config', 'data', 'leasing'}
        if target not in valid:
            raise SystemExit(
                f'Unknown path group {target!r}. Expected one of: '
                + ', '.join(sorted(valid))
            )

        groups: dict[str, list[dict[str, str]]] = {}
        if target in {'all', 'config'}:
            groups['config'] = [
                _entry('config_root', config_root(), kind='dir'),
                _entry('settings.yaml', settings_path(), kind='file'),
                _entry('catalog.yaml', config_root() / 'catalog.yaml',
                       kind='file'),
            ]
        if target in {'all', 'data'}:
            groups['data'] = [
                _entry('data_root', data_root(), kind='dir'),
            ]
        if target in {'all', 'leasing'}:
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
            _render_rich(groups, console)
        else:
            _render_plain(groups)
        return 0


# ---------------------------------------------------------------------------
# config settings: durable defaults (backend, data_dir) in settings.yaml
# ---------------------------------------------------------------------------

# The settings the leasing world honors. One registry drives `config init`'s
# prompts, `config set`'s validation, and the help — so adding a setting here is
# enough for `config init` to ask about it (no separate prompt to wire up).
@dataclass(frozen=True)
class _Setting:
    key: str
    label: str          # short prompt label shown by `config init`
    help: str           # one-line description (config set / KNOWN_SETTINGS)
    kind: str           # 'path' | 'choice' | 'bool'
    default: object = None
    choices: tuple = ()


_SETTINGS: tuple[_Setting, ...] = (
    _Setting(
        'data_dir', 'Data dir (docker-mounted weight/state)',
        'Where docker-mounted state lives (overrides the XDG default).',
        'path',
    ),
    _Setting(
        'backend', 'Default backend',
        'Default serving backend (compose | kubeai | null).',
        'choice', 'compose', ('compose', 'kubeai', 'null'),
    ),
    _Setting(
        'ui', 'Manage an Open WebUI alongside the stack',
        'Render a managed Open WebUI with the compose stack (true | false).',
        'bool', True,
    ),
    _Setting(
        'skip_display_gpus', 'Skip display-attached GPUs (leave the monitor GPU free)',
        'Skip display-attached GPUs during placement (true | false). Off by '
        'default — every GPU is used; turn on to leave a monitor GPU free.',
        'bool', False,
    ),
    _Setting(
        'reverse_proxy', 'Front the stack with a single-port HTTP reverse proxy',
        'Front the gateway + UI with one HTTP reverse proxy — UI at /, API at '
        '/v1 (true | false, or a {enabled, port, config_path} block via '
        '`config edit`). No TLS/auth — localhost / trusted networks only.',
        'bool', False,
    ),
)

# Keys the leasing world actually honors (others are allowed but warned about).
KNOWN_SETTINGS = {s.key: s.help for s in _SETTINGS}


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


class ConfigInitCLI(_PathOverridesMixin):
    """Set up the durable settings (data dir + default backend).

    Prompts for each setting and shows them for confirmation before writing. Use
    ``--yes`` for non-interactive scripting (accepts the provided flags /
    current values / defaults without prompting); the same non-interactive path
    is taken automatically when stdin is not a TTY.

    Re-running edits the existing config in place — it keeps your other settings
    and just re-confirms the data dir and backend (or hand-edit the file with
    ``infer-stack config edit``). Pass ``--fresh`` to discard the existing config
    and start from defaults.
    """

    __command__ = 'init'
    yes = scfg.Value(
        False, isflag=True, alias=['y'],
        help='Non-interactive: write without prompting/confirming.',
    )
    fresh = scfg.Value(
        False, isflag=True,
        help='Start over: ignore any existing config and write a clean one from '
        'defaults (discards other persisted settings too).',
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
        path = settings_path()
        existing = load_settings()

        # Tell the user, every time, which mode this is (and how else to edit).
        if config.fresh and existing:
            print(f'config init: starting fresh — replacing the config at {path}')
        elif existing:
            print(f'config init: editing the existing config at {path} '
                  '(or hand-edit it with `infer-stack config edit`)')
        else:
            print(f'config init: initializing a new config from scratch ({path})')

        # On --fresh, ignore the old file entirely (clean slate); otherwise keep
        # other settings and seed the proposals from the current values.
        base: dict = {} if config.fresh else dict(existing)
        seed: dict = {} if config.fresh else existing
        # Flags `config init` exposes directly (the rest are prompt/default only).
        flags = {'data_dir': config.data_dir, 'backend': config.backend}

        def _proposed(s: _Setting):
            # explicit flag > current setting > sensible default.
            if flags.get(s.key) is not None:
                return flags[s.key]
            if s.key in seed:
                return seed[s.key]
            # data_root() resolves --data-dir override / $env / setting / XDG.
            return str(data_root()) if s.key == 'data_dir' else s.default

        values = {s.key: _proposed(s) for s in _SETTINGS}

        interactive = (
            not config.yes and sys.stdin.isatty() and sys.stdout.isatty()
        )
        if interactive:
            from rich.console import Console
            from rich.prompt import Confirm, Prompt
            from rich.table import Table

            console = Console()
            for s in _SETTINGS:
                cur = values[s.key]
                if s.kind == 'bool':
                    values[s.key] = Confirm.ask(s.label, default=_as_bool(cur))
                elif s.kind == 'choice':
                    values[s.key] = Prompt.ask(
                        s.label, choices=list(s.choices), default=str(cur)
                    )
                else:
                    values[s.key] = Prompt.ask(s.label, default=str(cur))
            table = Table(show_header=True, header_style='bold')
            table.add_column('setting')
            table.add_column('value', style='green')
            for s in _SETTINGS:
                table.add_row(s.key, str(values[s.key]))
            console.print(table)
            if not Confirm.ask(f'Write these to {path}?', default=True):
                console.print('[yellow]aborted — nothing written[/]')
                return 0

        # Normalize bools so they persist as YAML true/false (not strings).
        for s in _SETTINGS:
            if s.kind == 'bool':
                values[s.key] = _as_bool(values[s.key])
        base.update(values)
        save_settings(base)
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
