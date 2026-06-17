from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from infer_stack import __version__

MANAGE_PY = Path(__file__).resolve().parents[1] / 'manage.py'


def run_cli(
    tmp_path: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def test_config_paths_reports_config_and_state(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'config', 'paths').stdout
    # Group headers are present.
    assert 'config:' in out
    assert 'data:' in out
    assert 'state:' in out
    # Anchored under tmp_path, and nothing exists yet on a fresh root.
    assert f'config.yaml (file, missing): {tmp_path / "config.yaml"}' in out
    assert f'hf_cache (dir, missing): {tmp_path / "hf-cache"}' in out


def test_config_paths_status_tracks_existence(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        'setup',
        '--yes',
        '--backend',
        'compose',
        '--profile',
        'qwen2-5-7b-instruct-turbo-default',
    )
    out = run_cli(tmp_path, 'config', 'paths', 'config').stdout
    assert f'config.yaml (file, exists): {tmp_path / "config.yaml"}' in out


def test_config_paths_json_emits_structured_groups(tmp_path: Path) -> None:
    payload = json.loads(run_cli(tmp_path, 'config', 'paths', '--json').stdout)
    assert set(payload) == {'config', 'data', 'state'}
    entry = payload['config'][0]
    assert set(entry) == {'label', 'kind', 'status', 'path'}


def test_config_paths_target_filters_to_single_group(tmp_path: Path) -> None:
    out = run_cli(tmp_path, 'config', 'paths', 'state').stdout
    assert 'state:' in out
    assert 'config:' not in out
    assert 'data:' not in out


def test_config_paths_kubeai_includes_generated_dir(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        'setup',
        '--yes',
        '--backend',
        'kubeai',
        '--profile',
        'qwen2-72b-instruct-tp2-balanced',
    )
    out = run_cli(tmp_path, 'config', 'paths', 'data').stdout
    assert 'kubeai_generated_dir' in out


def test_config_paths_omits_data_root(tmp_path: Path) -> None:
    # data_root is only a default anchor for relative paths; with absolute
    # state/output paths it is unused, so it should not be reported.
    out = run_cli(tmp_path, 'config', 'paths').stdout
    assert 'data_root' not in out
    assert 'generated_dir' in out


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
    _render_rich(groups, 'compose', console)
    out = buf.getvalue()
    assert 'config.yaml' in out
    assert 'models.yaml' in out
    # ANSI escape sequences are emitted on a (forced) terminal.
    assert '\x1b[' in out
