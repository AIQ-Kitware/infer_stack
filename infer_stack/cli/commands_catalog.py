"""`infer-stack catalog …` — edit the user serving catalog without raw YAML.

The catalog (``config_root()/catalog.yaml`` by default) is the durable list of
models / endpoints / runtime-hosts / bundles the leasing verbs (``acquire`` /
``serve`` / ``run``) read. Hand-editing YAML is the current workflow; this
submodal adds a flag-driven editor with a validating writer, so a bad edit is
rejected before it can land (the same :class:`~infer_stack.leasing.catalog.Catalog`
parser the leasing path uses).

Grammar is noun-verb: ``catalog model add``, ``catalog endpoint rm``, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import scriptconfig as scfg
import yaml

from ..leasing import Catalog, CatalogError
from ..paths import config_root
from .context import _apply_path_overrides
from .options import _PathOverridesMixin

SECTIONS = ('models', 'endpoints', 'runtime_hosts', 'bundles')


def _print_yaml(text: str) -> None:
    """Print YAML, syntax-highlighted when stdout is a terminal.

    Falls back to a plain ``print`` when piped/redirected so downstream
    parsers and tests get clean, color-free output.
    """
    if not sys.stdout.isatty():
        print(text, end='')
        return
    try:
        from rich.console import Console
        from rich.syntax import Syntax
    except ImportError:
        print(text, end='')
        return
    syntax = Syntax(
        text.rstrip('\n'), 'yaml', theme='ansi_dark', background_color='default'
    )
    Console().print(syntax)


# ---------------------------------------------------------------------------
# raw catalog load / mutate / save
# ---------------------------------------------------------------------------


def _catalog_path(config) -> Path:
    _apply_path_overrides(config)
    raw = getattr(config, 'catalog', None)
    if raw:
        return Path(raw).expanduser()
    return config_root() / 'catalog.yaml'


def _load_raw(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    for section in SECTIONS:
        data.setdefault(section, {})
    return data


def _validate(data: dict[str, Any]) -> None:
    """Refuse to persist a catalog the leasing path would reject."""
    try:
        Catalog.from_dict(data)
    except CatalogError as ex:
        raise SystemExit(f'refusing to write an invalid catalog: {ex}')


def _save_raw(path: Path, data: dict[str, Any], *, dry_run: bool = False) -> None:
    # Drop empty sections for a tidy file.
    out = {k: v for k, v in data.items() if v or k == 'models'}
    text = yaml.safe_dump(out, sort_keys=False, default_flow_style=False)
    if dry_run:
        _print_yaml(text)
        return
    _validate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _exists_guard(data, section, name, force, hint='') -> None:
    if name in data[section] and not force:
        raise SystemExit(
            f"{section[:-1]} '{name}' already exists; "
            f'pass --force to overwrite{hint}'
        )


def _next_indexed_name(existing, base: str) -> str:
    """First free ``{base}-{N}`` (N starting at 1) not already in ``existing``.

    Defaulted endpoint names get a numeric suffix so repeated ``endpoint add``
    for the same model don't collide — they accumulate as ``base-1``, ``base-2``,
    … and the first is deterministically ``base-1``.
    """
    n = 1
    while f'{base}-{n}' in existing:
        n += 1
    return f'{base}-{n}'


def _slug_alias(text: str) -> str:
    """Make ``text`` safe to use as an endpoint alias.

    An endpoint name doubles as the LiteLLM ``model_name`` (what clients ask for
    and what Open WebUI shows) and as a CLI-typed token, so keep it shell/URL
    friendly: collapse model/tag separators (``/``, ``:``) and any other
    non-``[A-Za-z0-9._-]`` runs to a single ``-``.
    """
    import re

    out = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-')
    return out or text


def _rm(config, section, names) -> int:
    names = [names] if isinstance(names, str) else list(names or [])
    if not names:
        raise SystemExit(f'{section[:-1]} rm: give at least one name')
    path = _catalog_path(config)
    data = _load_raw(path)
    missing = [n for n in names if n not in data[section]]
    if missing:
        raise SystemExit(
            f'{section[:-1]}(s) not found in {path}: {", ".join(missing)}'
        )
    for name in names:
        del data[section][name]
    _save_raw(path, data, dry_run=getattr(config, 'dry_run', False))
    for name in names:
        print(f"removed {section[:-1]} '{name}'")
    return 0


def _list(config, section) -> int:
    data = _load_raw(_catalog_path(config))
    names = sorted(data[section])
    print('\n'.join(names) if names else f'(no {section})')
    return 0


def _show(config, section, name) -> int:
    data = _load_raw(_catalog_path(config))
    entries = data.get(section) or {}
    # No name -> show every entry in the section (the whole `endpoints:` block),
    # rather than erroring on a `None` lookup.
    if name is None:
        if not entries:
            print(f'(no {section})')
            return 0
        _print_yaml(yaml.safe_dump({section: entries}, sort_keys=False))
        return 0
    if name not in entries:
        have = f" (have: {', '.join(sorted(entries))})" if entries else ''
        raise SystemExit(f"{section[:-1]} '{name}' not found{have}")
    _print_yaml(yaml.safe_dump({name: entries[name]}, sort_keys=False))
    return 0


def _parse_kv(items) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` pairs; values are YAML-typed (ints/floats/bools)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        key, _, val = item.partition('=')
        out[key.strip()] = yaml.safe_load(val)
    return out


# ---------------------------------------------------------------------------
# catalog-level commands: init / path / show / validate / edit
# ---------------------------------------------------------------------------


class _CatalogCommon(_PathOverridesMixin):
    catalog = scfg.Value(None, type=str, help='Catalog path (default: config dir).')
    dry_run = scfg.Value(
        False, isflag=True, help='Print the resulting YAML, do not write.'
    )


class CatalogInitCLI(_CatalogCommon):
    """Write a starter catalog.yaml (empty sections) if none exists."""

    __command__ = 'init'
    force = scfg.Value(False, isflag=True, help='Overwrite an existing catalog.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        path = _catalog_path(config)
        if path.exists() and not config.force:
            raise SystemExit(f'{path} already exists; pass --force to reset')
        _save_raw(path, {s: {} for s in SECTIONS}, dry_run=config.dry_run)
        if not config.dry_run:
            print(f'wrote starter catalog -> {path}')
        return 0


class CatalogPathCLI(_PathOverridesMixin):
    """Print the catalog path."""

    __command__ = 'path'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        print(_catalog_path(cls.cli(argv=argv, data=kwargs)))
        return 0


class CatalogShowCLI(_PathOverridesMixin):
    """Pretty-print the whole catalog (or one named entry across sections)."""

    __command__ = 'show'
    catalog = scfg.Value(None, type=str)
    name = scfg.Value(None, position=1, type=str, help='Optional entry name.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        data = _load_raw(_catalog_path(config))
        if config.name:
            hits = {
                s: data[s][config.name]
                for s in SECTIONS
                if config.name in data[s]
            }
            if not hits:
                raise SystemExit(f"'{config.name}' not found in any section")
            _print_yaml(yaml.safe_dump(hits, sort_keys=False))
        else:
            _print_yaml(yaml.safe_dump(data, sort_keys=False))
        return 0


class CatalogValidateCLI(_PathOverridesMixin):
    """Parse + cross-reference check the catalog."""

    __command__ = 'validate'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        path = _catalog_path(config)
        if not path.exists():
            raise SystemExit(f'no catalog at {path}')
        try:
            cat = Catalog.load(path)
        except CatalogError as ex:
            raise SystemExit(f'invalid: {ex}')
        print(
            f'ok: {len(cat.models)} model(s), {len(cat.endpoints)} endpoint(s), '
            f'{len(cat.hosts)} host(s), {len(cat.bundles)} bundle(s)'
        )
        return 0


class CatalogEditCLI(_PathOverridesMixin):
    """Open the catalog in $EDITOR (escape hatch), then validate it."""

    __command__ = 'edit'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        import os
        import subprocess

        config = cls.cli(argv=argv, data=kwargs)
        path = _catalog_path(config)
        if not path.exists():
            _save_raw(path, {s: {} for s in SECTIONS})
        editor = os.environ.get('EDITOR', 'vi')
        subprocess.run([*editor.split(), str(path)], check=False)
        try:
            Catalog.load(path)
        except CatalogError as ex:
            print(f'warning: catalog is invalid after edit: {ex}')
            return 1
        return 0


# ---------------------------------------------------------------------------
# model {add,list,show,rm}
# ---------------------------------------------------------------------------


class ModelAddCLI(_CatalogCommon):
    """Add (or --force overwrite) a model: a Hugging Face / local weight source."""

    __command__ = 'add'
    name = scfg.Value(None, position=1, type=str)
    source = scfg.Value(None, type=str, help='e.g. hf://org/Model or a path.')
    revision = scfg.Value(None, type=str)
    quantization = scfg.Value(None, type=str)
    dtype = scfg.Value(None, type=str)
    force = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        if not config.name or not config.source:
            raise SystemExit('model add: NAME and --source are required')
        path = _catalog_path(config)
        data = _load_raw(path)
        _exists_guard(data, 'models', config.name, config.force)
        entry: dict[str, Any] = {'source': config.source}
        for key in ('revision', 'quantization', 'dtype'):
            if getattr(config, key) is not None:
                entry[key] = getattr(config, key)
        data['models'][config.name] = entry
        _save_raw(path, data, dry_run=config.dry_run)
        if not config.dry_run:
            print(f"added model '{config.name}'")
        return 0


class ModelListCLI(_PathOverridesMixin):
    """List model names."""
    __command__ = 'list'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        return _list(cls.cli(argv=argv, data=kwargs), 'models')


class ModelShowCLI(_PathOverridesMixin):
    """Show a model entry, or all of them when no NAME is given."""
    __command__ = 'show'
    catalog = scfg.Value(None, type=str)
    name = scfg.Value(None, position=1, type=str,
                      help='Model name; omit to show every model.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _show(config, 'models', config.name)


class ModelRmCLI(_CatalogCommon):
    """Remove one or more models by name."""
    __command__ = 'rm'
    names = scfg.Value([], nargs='+', position=1, type=str,
                       help='Model name(s) to remove.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _rm(config, 'models', config.names)


class CatalogModelCLI(scfg.ModalCLI):
    """Manage catalog models."""
    __command__ = 'model'
    add = ModelAddCLI
    list = ModelListCLI
    show = ModelShowCLI
    rm = ModelRmCLI


# ---------------------------------------------------------------------------
# endpoint {add,list,show,rm}
# ---------------------------------------------------------------------------


class EndpointAddCLI(_CatalogCommon):
    """Add (or --force overwrite) an endpoint — the served API name.

    ``NAME`` is optional: when omitted it defaults to ``{model}-{N}`` (the vLLM
    model name, or the Ollama tag, slugified, with an auto-incrementing suffix —
    ``smol135-1``, ``smol135-2``, …). That keeps the served alias — what you ask
    for and what Open WebUI shows — tied to the model, and a repeated add for one
    model just gets the next index instead of colliding. Give an explicit
    ``NAME`` when you want a stable alias decoupled from the model (e.g. ``chat``
    you can re-point).
    """

    __command__ = 'add'
    name = scfg.Value(
        None, position=1, type=str,
        help='Endpoint alias (default: {model}-N, auto-incrementing).',
    )
    engine = scfg.Value('vllm', choices=['vllm', 'ollama'])
    model = scfg.Value(None, type=str, help='Model name (vllm) or tag (ollama).')
    host = scfg.Value(None, type=str, help='Runtime host (ollama).')
    public_name = scfg.Value(
        None, type=str, help='Served/public name (for coalescing aliases).'
    )
    reclaim = scfg.Value(
        None, choices=['keep-warm', 'stop', 'scale-to-zero'],
        help='Reclaim policy when idle.',
    )
    # vLLM runtime conveniences
    max_model_len = scfg.Value(None, type=int)
    gpu_mem = scfg.Value(
        None, type=float, help='gpu_memory_utilization (0-1).'
    )
    tensor_parallel = scfg.Value(None, type=int)
    extra_args = scfg.Value(
        None, type=str,
        help="Raw vLLM flags as one string (shell-split), "
        "e.g. --extra-args='--dtype=half --enforce-eager'.",
    )
    runtime = scfg.Value(
        [], nargs='*', type=str,
        help='Extra runtime KEY=VALUE pairs (YAML-typed).',
    )
    force = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        path = _catalog_path(config)
        data = _load_raw(path)
        if config.name:
            name = config.name
            _exists_guard(data, 'endpoints', name, config.force)
        else:
            if not config.model:
                raise SystemExit(
                    'endpoint add: give a NAME, or --model to derive the name '
                    'from it'
                )
            # Default name = {model}-N, auto-incrementing so repeated adds for
            # one model accumulate (smol135-1, smol135-2, …) instead of colliding.
            name = _next_indexed_name(
                data['endpoints'], _slug_alias(config.model)
            )
        entry: dict[str, Any] = {'engine': config.engine}
        if config.model:
            entry['model'] = config.model
        if config.host:
            entry['host'] = config.host
        if config.public_name:
            entry['public_name'] = config.public_name
        runtime: dict[str, Any] = _parse_kv(config.runtime)
        if config.max_model_len is not None:
            runtime['max_model_len'] = config.max_model_len
        if config.gpu_mem is not None:
            runtime['gpu_memory_utilization'] = config.gpu_mem
        if config.tensor_parallel is not None:
            runtime['tensor_parallel_size'] = config.tensor_parallel
        if config.extra_args:
            import shlex
            runtime['extra_args'] = shlex.split(config.extra_args)
        if runtime:
            entry['runtime'] = runtime
        if config.reclaim:
            entry['reclaim'] = {'policy': config.reclaim}
        data['endpoints'][name] = entry
        _save_raw(path, data, dry_run=config.dry_run)
        if not config.dry_run:
            model_note = f' -> {config.model}' if config.model else ''
            print(f"added endpoint '{name}'{model_note}")
        return 0


class EndpointListCLI(_PathOverridesMixin):
    """List endpoint names."""
    __command__ = 'list'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        return _list(cls.cli(argv=argv, data=kwargs), 'endpoints')


class EndpointShowCLI(_PathOverridesMixin):
    """Show an endpoint entry, or all of them when no NAME is given."""
    __command__ = 'show'
    catalog = scfg.Value(None, type=str)
    name = scfg.Value(None, position=1, type=str,
                      help='Endpoint name; omit to show every endpoint.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _show(config, 'endpoints', config.name)


class EndpointRmCLI(_CatalogCommon):
    """Remove one or more endpoints by name."""
    __command__ = 'rm'
    names = scfg.Value([], nargs='+', position=1, type=str,
                       help='Endpoint name(s) to remove.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _rm(config, 'endpoints', config.names)


class CatalogEndpointCLI(scfg.ModalCLI):
    """Manage catalog endpoints."""
    __command__ = 'endpoint'
    add = EndpointAddCLI
    list = EndpointListCLI
    show = EndpointShowCLI
    rm = EndpointRmCLI


# ---------------------------------------------------------------------------
# host {add,list,rm}  (runtime_hosts — ollama daemons)
# ---------------------------------------------------------------------------


class HostAddCLI(_CatalogCommon):
    """Add (or --force overwrite) a runtime host (e.g. an Ollama daemon)."""

    __command__ = 'add'
    name = scfg.Value(None, position=1, type=str)
    engine = scfg.Value('ollama', choices=['ollama'])
    gpu = scfg.Value([], nargs='*', type=int, help='GPU index/indices.')
    keep_alive = scfg.Value(None, type=str, help='Ollama keep_alive, e.g. 5m.')
    num_parallel = scfg.Value(None, type=int)
    max_loaded_models = scfg.Value(None, type=int)
    context_length = scfg.Value(None, type=int)
    image = scfg.Value(None, type=str)
    force = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        if not config.name:
            raise SystemExit('host add: NAME is required')
        path = _catalog_path(config)
        data = _load_raw(path)
        _exists_guard(data, 'runtime_hosts', config.name, config.force)
        entry: dict[str, Any] = {'engine': config.engine}
        if config.gpu:
            entry['placement'] = {'gpu_indices': list(config.gpu)}
        settings: dict[str, Any] = {}
        for flag, key in (
            ('keep_alive', 'keep_alive'),
            ('num_parallel', 'num_parallel'),
            ('max_loaded_models', 'max_loaded_models'),
            ('context_length', 'context_length'),
        ):
            if getattr(config, flag) is not None:
                settings[key] = getattr(config, flag)
        if settings:
            entry['settings'] = settings
        if config.image:
            entry['image'] = config.image
        data['runtime_hosts'][config.name] = entry
        _save_raw(path, data, dry_run=config.dry_run)
        if not config.dry_run:
            print(f"added host '{config.name}'")
        return 0


class HostListCLI(_PathOverridesMixin):
    """List runtime-host names."""
    __command__ = 'list'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        return _list(cls.cli(argv=argv, data=kwargs), 'runtime_hosts')


class HostRmCLI(_CatalogCommon):
    """Remove one or more runtime hosts by name."""
    __command__ = 'rm'
    names = scfg.Value([], nargs='+', position=1, type=str,
                       help='Runtime-host name(s) to remove.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _rm(config, 'runtime_hosts', config.names)


class CatalogHostCLI(scfg.ModalCLI):
    """Manage runtime hosts (Ollama daemons / placement)."""
    __command__ = 'host'
    add = HostAddCLI
    list = HostListCLI
    rm = HostRmCLI


# ---------------------------------------------------------------------------
# bundle {add,list,rm}
# ---------------------------------------------------------------------------


class BundleAddCLI(_CatalogCommon):
    """Add (or --force overwrite) a bundle: a named group of endpoints."""

    __command__ = 'add'
    name = scfg.Value(None, position=1, type=str)
    members = scfg.Value([], nargs='*', position=2, type=str)
    force = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        members = list(config.members or [])
        if not config.name or not members:
            raise SystemExit('bundle add: NAME and at least one endpoint required')
        path = _catalog_path(config)
        data = _load_raw(path)
        _exists_guard(data, 'bundles', config.name, config.force)
        data['bundles'][config.name] = members
        _save_raw(path, data, dry_run=config.dry_run)
        if not config.dry_run:
            print(f"added bundle '{config.name}' -> {', '.join(members)}")
        return 0


class BundleListCLI(_PathOverridesMixin):
    """List bundle names."""
    __command__ = 'list'
    catalog = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=True, **kwargs):
        return _list(cls.cli(argv=argv, data=kwargs), 'bundles')


class BundleRmCLI(_CatalogCommon):
    """Remove one or more bundles by name."""
    __command__ = 'rm'
    names = scfg.Value([], nargs='+', position=1, type=str,
                       help='Bundle name(s) to remove.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _rm(config, 'bundles', config.names)


class CatalogBundleCLI(scfg.ModalCLI):
    """Manage endpoint bundles."""
    __command__ = 'bundle'
    add = BundleAddCLI
    list = BundleListCLI
    rm = BundleRmCLI


# ---------------------------------------------------------------------------
# top-level catalog modal
# ---------------------------------------------------------------------------


class CatalogModalCLI(scfg.ModalCLI):
    """Edit the user serving catalog (models / endpoints / hosts / bundles)."""

    __command__ = 'catalog'

    init = CatalogInitCLI
    path = CatalogPathCLI
    show = CatalogShowCLI
    validate = CatalogValidateCLI
    edit = CatalogEditCLI
    model = CatalogModelCLI
    endpoint = CatalogEndpointCLI
    host = CatalogHostCLI
    bundle = CatalogBundleCLI
