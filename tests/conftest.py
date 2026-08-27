"""
Keep the suite out of the developer's real infer-stack state.

``data_root()`` and ``config_root()`` default to ``~/.local/share/infer_stack``
and ``~/.config/infer_stack``. Tests that build a backend without overriding
them therefore read and WRITE the real ones -- ``docker-compose.yml``,
``.env``, the compose state sidecar and ``.converge.lock`` -- on the machine
running the tests. That is destructive on a developer box, and it is why
``tests/test_cli_leasing.py`` could hang: with real state present there are
real changes to apply, so ``acquire`` reached the diff-confirm prompt, and
``ConvergeScaffold._converge_lock`` took a blocking ``flock`` on the real lock
file that a live converge may already hold.

Isolation is autouse and per-test rather than opt-in: a test that forgets is
exactly the case that causes the damage.
"""

import pytest

from infer_stack.paths import CONFIG_DIR_ENV, DATA_DIR_ENV


@pytest.fixture(autouse=True)
def isolate_infer_stack_roots(tmp_path, monkeypatch):
    """Point the config and data roots at this test's tmp_path."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / 'config'))
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / 'data'))
