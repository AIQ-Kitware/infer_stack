"""Built-in Qwen profile is path-independent and safe to import repeatedly."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from infer_stack.cli.commands_catalog import CatalogImportCLI
from infer_stack.leasing import Catalog


def test_qwen_catalog_preset_matches_documented_example():
    root = Path(__file__).resolve().parents[1]
    example = yaml.safe_load((root / 'examples/qwen38-flash-next-3090.yaml').read_text())
    packaged = yaml.safe_load(resources.files('infer_stack').joinpath(
        'templates').joinpath('catalog-qwen38-flash-next-3090.yaml').read_text())
    assert packaged == example
    assert Catalog.from_dict(packaged).resolve_endpoint('qwen38-flash-next-3090')


def test_import_idempotent_preserves_existing_catalog_and_noop_mtime(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path / 'data'))
    path = tmp_path / 'catalog.yaml'
    path.write_text(yaml.safe_dump({
        'models': {'existing-model': {'source': 'hf://test/example'}},
        'endpoints': {'existing': {'engine': 'vllm', 'model': 'existing-model'}},
    }))
    assert CatalogImportCLI.main(argv=['qwen38-flash-next-3090']) == 0
    installed = yaml.safe_load(path.read_text())
    assert 'existing' in installed['endpoints']
    assert 'qwen38-flash-next-3090' in installed['endpoints']
    Catalog.load(path)
    previous_text = path.read_text()
    old_time = path.stat().st_mtime_ns
    assert CatalogImportCLI.main(argv=['qwen38-flash-next-3090']) == 0
    assert path.stat().st_mtime_ns == old_time
    assert path.read_text() == previous_text


def test_import_conflict_is_non_destructive(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    path = tmp_path / 'catalog.yaml'
    original = yaml.safe_dump({'models': {
        'qwen38-flash-next-w4a16': {'source': 'hf://another/model'},
    }})
    path.write_text(original)
    with pytest.raises(SystemExit, match='already exists and differs'):
        CatalogImportCLI.main(argv=['qwen38-flash-next-3090'])
    assert path.read_text() == original


def test_import_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    path = tmp_path / 'catalog.yaml'
    assert CatalogImportCLI.main(argv=['qwen38-flash-next-3090', '--dry-run']) == 0
    assert not path.exists()
    assert 'qwen38-flash-next-3090' in capsys.readouterr().out


def test_import_rejects_unknown_preset(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    with pytest.raises(SystemExit, match='available: qwen38-flash-next-3090'):
        CatalogImportCLI.main(argv=['not-present'])
    assert not (tmp_path / 'catalog.yaml').exists()
