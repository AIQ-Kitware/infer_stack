from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from infer_stack import __version__

MANAGE_PY = Path(__file__).resolve().parents[1] / 'manage.py'


# Pre-leasing commands moved under `infer-stack legacy …`.
_LEGACY_CMDS = {
    'setup', 'init', 'resolve', 'validate', 'lock', 'render', 'switch',
    'list-models', 'list-profiles', 'explain', 'describe-profile',
    'verify-profile', 'kubeai-sync-resource-profiles', 'up', 'down', 'purge',
    'deploy', 'env', 'diagnose', 'wait-ready', 'smoke-test', 'benchmark',
    'ollama-pull', 'ollama-list', 'ollama-ps',
}


def run_cli(
    tmp_path: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    if args and args[0] in _LEGACY_CMDS:
        args = ('legacy', *args)
    env = os.environ.copy()
    # Force (not setdefault) so an ambient INFER_STACK_* in the caller's shell
    # can't leak in and point the test at the real config/data dir.
    env['INFER_STACK_CONFIG_DIR'] = str(tmp_path)
    env['INFER_STACK_DATA_DIR'] = str(tmp_path)
    return subprocess.run(
        [sys.executable, str(MANAGE_PY), *args],
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def test_version_prints_package_version(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'version').stdout
    assert __version__ in out
    assert 'infer-stack' in out


def test_config_paths_reports_groups(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'config', 'paths').stdout
    # Group headers are present.
    assert 'config:' in out
    assert 'data:' in out
    assert 'leasing:' in out
    # Anchored under tmp_path, and nothing exists yet on a fresh root.
    assert f'settings.yaml (file, missing): {tmp_path / "settings.yaml"}' in out
    assert f'catalog.yaml (file, missing): {tmp_path / "catalog.yaml"}' in out


def test_config_paths_json_emits_structured_groups(tmp_path: Path) -> None:
    payload = json.loads(run_cli(tmp_path, 'config', 'paths', '--json').stdout)
    assert set(payload) == {'config', 'data', 'leasing'}
    entry = payload['config'][0]
    assert set(entry) == {'label', 'kind', 'status', 'path'}


def test_config_paths_includes_leasing_group(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'config', 'paths', 'leasing').stdout
    assert 'leasing:' in out
    assert 'ledger' in out
    assert 'docker-compose.yml' in out
    assert 'env (secrets)' in out


def test_paths_top_level_alias_works(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'paths').stdout
    assert 'config:' in out
    assert 'leasing:' in out


def test_status_summarizes_leases_when_present(tmp_path: Path) -> None:
    from infer_stack.leasing import (
        EndpointRequest,
        Ledger,
        SqliteStore,
        vllm_structural,
    )

    # A lease in the ledger at the anchored data dir (default_ledger_path()).
    led = Ledger(SqliteStore(tmp_path / 'leasing' / 'ledger.db'))
    led.acquire('me', [EndpointRequest('m', 'vllm', vllm_structural(model_ref='m'))])
    out = run_cli(tmp_path, 'status').stdout
    assert 'leasing:' in out and '1 active' in out and 'lease(s)' in out


def test_config_paths_target_filters_to_single_group(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'config', 'paths', 'leasing').stdout
    assert 'leasing:' in out
    assert 'config:' not in out
    assert 'data:' not in out


def test_config_paths_rejects_unknown_target(tmp_path: Path) -> None:
    result = run_cli(tmp_path, 'config', 'paths', 'bogus', check=False)
    assert result.returncode != 0
    assert 'Unknown path group' in (result.stderr + result.stdout)


def test_render_rich_colorizes_status() -> None:
    import io

    from rich.console import Console

    from infer_stack.cli.commands_meta import _render_rich

    groups = {
        'config': [
            {
                'label': 'config.yaml',
                'kind': 'file',
                'status': 'exists',
                'path': '/tmp/x/config.yaml',
            },
            {
                'label': 'models.yaml',
                'kind': 'file',
                'status': 'missing',
                'path': '/tmp/x/models.yaml',
            },
        ]
    }
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100)
    _render_rich(groups, console)
    out = buf.getvalue()
    assert 'config.yaml' in out
    assert 'models.yaml' in out
    # ANSI escape sequences are emitted on a (forced) terminal.
    assert '\x1b[' in out


def test_day2_compose_base_prefers_leasing(tmp_path: Path, monkeypatch) -> None:
    from infer_stack.cli.commands_runtime import _day2_compose_base

    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    compose_file = tmp_path / 'leasing' / 'compose' / 'docker-compose.yml'
    compose_file.parent.mkdir(parents=True)
    compose_file.write_text('services: {}\n')
    # leasing compose present -> targets it without needing config.yaml
    base = _day2_compose_base(None, 'logs')
    assert base == [
        'docker', 'compose', '-p', 'infer-stack', '-f', str(compose_file)
    ]
