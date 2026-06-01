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

``--data-dir`` is the single knob for "put everything I generate in one
place": at ``setup`` time it is baked into the absolute ``state.*`` and
``output.generated_dir`` paths written to ``config.yaml``. For a bespoke
split layout, edit those fields in ``config.yaml`` directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import ubelt as ub


CONFIG_DIR_ENV = "INFER_STACK_CONFIG_DIR"
DATA_DIR_ENV = "INFER_STACK_DATA_DIR"

_config_root_override: Path | None = None
_data_root_override: Path | None = None


def _default_config_root() -> Path:
    return Path(ub.Path.appdir("infer_stack", type="config"))


def _default_data_root() -> Path:
    return Path(ub.Path.appdir("infer_stack", type="data"))


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
    return _default_data_root()


def set_config_root(path: Path | str | None) -> None:
    """Override ``config_root()`` for the lifetime of this process.

    Pass ``None`` to clear the override and fall back to env var / default.
    """
    global _config_root_override
    _config_root_override = Path(path).expanduser() if path is not None else None


def set_data_root(path: Path | str | None) -> None:
    """Override ``data_root()`` for the lifetime of this process."""
    global _data_root_override
    _data_root_override = Path(path).expanduser() if path is not None else None
