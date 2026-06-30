"""CWD-independent locations for the infer-stack config and runtime data.

The CLI used to anchor every path on ``Path.cwd()``, which meant invoking
``infer-stack`` from a different directory silently changed where config
was read from, where rendered artifacts landed, and where bind-mount
state lived. This module replaces that with two stable roots:

* ``config_root()`` — where ``config.yaml`` / ``models.yaml`` /
  ``kubeai-values.local.yaml`` live. Defaults to
  ``ub.Path.appdir('infer_stack', type='config')`` (``~/.config/infer_stack``
  on Linux, respecting ``XDG_CONFIG_HOME``).
* ``data_root()`` — where ``generated/`` (rendered artifacts) and
  ``state/`` (hf-cache, postgres volumes, runtime bind-mounts) default
  to. ``ub.Path.appdir('infer_stack', type='data')``
  (``~/.local/share/infer_stack`` on Linux, respecting
  ``XDG_DATA_HOME``). Uses ``data`` and not ``cache`` because the stack
  hosts persistent state — postgres databases, Open WebUI chat history,
  and user accounts — that would be silently lost if treated as
  regenerable cache by a system cleanup tool.

Both can be overridden by env vars (``INFER_STACK_CONFIG_DIR`` /
``INFER_STACK_DATA_DIR``) or by the CLI flags ``--config-dir`` /
``--data-dir``. The CLI flags translate into process-wide overrides via
``set_config_root`` / ``set_data_root``.

``INFER_STACK_MODEL_PATH`` is a separate PATH-like catalog overlay for model and
profile YAML files. It augments the configured catalog; it does not change the
config root or data root.

``--data-dir`` is the single knob for "put everything I generate in one
place": at ``setup`` time it is baked into the absolute ``state.*`` and
``output.generated_dir`` paths written to ``config.yaml``. For a bespoke
split layout, edit those fields in ``config.yaml`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import ubelt as ub


CONFIG_DIR_ENV = 'INFER_STACK_CONFIG_DIR'
DATA_DIR_ENV = 'INFER_STACK_DATA_DIR'
MODEL_PATH_ENV = 'INFER_STACK_MODEL_PATH'

SETTINGS_FILENAME = 'settings.yaml'
# UI-only preferences for the TUI (poll cadences, pane sizes). Deliberately a
# SEPARATE file from settings.yaml: settings.yaml is the CLI-facing leasing
# config (backend, data dir, proxy) that changes how the stack runs; this one
# only tunes the dashboard and never affects a `config`/`acquire` from the CLI.
TUI_SETTINGS_FILENAME = 'tui_settings.yaml'

_config_root_override: Path | None = None
_data_root_override: Path | None = None


# ---------------------------------------------------------------------------
# durable user settings (config dir / leasing world): default backend, data dir
# ---------------------------------------------------------------------------


def settings_path() -> Path:
    """Where the leasing-world user settings live (``config set`` writes here)."""
    return config_root() / SETTINGS_FILENAME


def load_settings() -> dict:
    """Load ``settings.yaml`` (empty dict if absent)."""
    path = settings_path()
    if path.exists():
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    return {}


def save_settings(settings: dict) -> Path:
    import yaml

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def get_setting(key: str, default=None):
    return load_settings().get(key, default)


# ---------------------------------------------------------------------------
# UI-only preferences (TUI dashboard): poll cadences etc. Kept apart from the
# CLI's settings.yaml so tuning the dashboard never touches how the stack runs.
# ---------------------------------------------------------------------------


def tui_settings_path() -> Path:
    """Where the TUI's own UI preferences live (separate from settings.yaml)."""
    return config_root() / TUI_SETTINGS_FILENAME


def load_tui_settings() -> dict:
    """Load ``tui_settings.yaml`` (empty dict if absent)."""
    path = tui_settings_path()
    if path.exists():
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    return {}


def save_tui_settings(settings: dict) -> Path:
    import yaml

    path = tui_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def _default_config_root() -> Path:
    return Path(ub.Path.appdir('infer_stack', type='config'))


def _default_data_root() -> Path:
    return Path(ub.Path.appdir('infer_stack', type='data'))


def config_root() -> Path:
    if _config_root_override is not None:
        return _config_root_override
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return _default_config_root()


def data_root() -> Path:
    if _data_root_override is not None:
        return _data_root_override
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    # Honor a persisted `config set data_dir <path>` before the XDG default, so
    # "where my docker-mounted state lives" can live in the durable config
    # instead of being re-exported every shell.
    configured = load_settings().get('data_dir')
    if configured:
        return Path(configured).expanduser()
    return _default_data_root()


def set_config_root(path: Path | str | None) -> None:
    """Override ``config_root()`` for the lifetime of this process.

    Pass ``None`` to clear the override and fall back to env var / default.
    """
    global _config_root_override
    _config_root_override = (
        Path(path).expanduser() if path is not None else None
    )


def set_data_root(path: Path | str | None) -> None:
    """Override ``data_root()`` for the lifetime of this process."""
    global _data_root_override
    _data_root_override = Path(path).expanduser() if path is not None else None
