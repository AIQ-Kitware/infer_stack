"""Tests for the `infer-stack catalog …` editor and `help tree`."""

from __future__ import annotations

import pytest
import yaml

from infer_stack.cli.commands_catalog import (
    BundleAddCLI,
    CatalogInitCLI,
    EndpointAddCLI,
    EndpointRmCLI,
    HostAddCLI,
    ModelAddCLI,
)
from infer_stack.leasing import Catalog


def cat_path(tmp_path):
    return str(tmp_path / 'catalog.yaml')


def _opts(tmp_path):
    return ['--catalog', cat_path(tmp_path)]


def test_init_creates_sections(tmp_path):
    CatalogInitCLI.main(argv=_opts(tmp_path))
    data = yaml.safe_load((tmp_path / 'catalog.yaml').read_text())
    assert set(data) >= {'models'}
    # an empty catalog still parses
    Catalog.load(cat_path(tmp_path))


def test_add_model_endpoint_roundtrips_and_validates(tmp_path):
    ModelAddCLI.main(argv=['smol17b', '--source', 'hf://org/SmolLM2',
                           *_opts(tmp_path)])
    EndpointAddCLI.main(argv=[
        'chat', '--engine', 'vllm', '--model', 'smol17b',
        '--max-model-len', '8192', '--gpu-mem', '0.4',
        '--extra-args=--dtype=half --enforce-eager', '--reclaim', 'keep-warm',
        *_opts(tmp_path),
    ])
    cat = Catalog.load(cat_path(tmp_path))
    ep = cat.endpoints['chat']
    assert ep.model == 'smol17b'
    assert ep.runtime['max_model_len'] == 8192
    assert ep.runtime['extra_args'] == ['--dtype=half', '--enforce-eager']
    assert ep.reclaim == 'keep-warm'


def test_endpoint_referencing_missing_model_is_refused(tmp_path):
    CatalogInitCLI.main(argv=_opts(tmp_path))
    with pytest.raises(SystemExit):
        EndpointAddCLI.main(argv=['bad', '--engine', 'vllm', '--model', 'ghost',
                                  *_opts(tmp_path)])
    # nothing persisted
    cat = Catalog.load(cat_path(tmp_path))
    assert 'bad' not in cat.endpoints


def test_ollama_endpoint_needs_existing_host(tmp_path):
    CatalogInitCLI.main(argv=_opts(tmp_path))
    with pytest.raises(SystemExit):
        EndpointAddCLI.main(argv=['ot', '--engine', 'ollama', '--host', 'h',
                                  '--model', 'm:1b', *_opts(tmp_path)])
    HostAddCLI.main(argv=['h', '--engine', 'ollama', '--gpu', '0',
                          '--keep-alive', '5m', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['ot', '--engine', 'ollama', '--host', 'h',
                              '--model', 'm:1b', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert cat.hosts['h'].gpu_indices == [0]
    assert cat.endpoints['ot'].host == 'h'


def test_add_existing_without_force_errors(tmp_path):
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    with pytest.raises(SystemExit):
        ModelAddCLI.main(argv=['m', '--source', 'hf://b', *_opts(tmp_path)])
    ModelAddCLI.main(argv=['m', '--source', 'hf://b', '--force', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert cat.models['m'].source == 'hf://b'


def test_rm_missing_errors_and_rm_removes(tmp_path):
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['e', '--engine', 'vllm', '--model', 'm',
                              *_opts(tmp_path)])
    with pytest.raises(SystemExit):
        EndpointRmCLI.main(argv=['nope', *_opts(tmp_path)])
    EndpointRmCLI.main(argv=['e', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert 'e' not in cat.endpoints


def test_bundle_add_and_validate(tmp_path):
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['a', '--engine', 'vllm', '--model', 'm', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['b', '--engine', 'vllm', '--model', 'm', *_opts(tmp_path)])
    BundleAddCLI.main(argv=['pair', 'a', 'b', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert cat.bundles['pair'] == ['a', 'b']


def test_dry_run_does_not_write(tmp_path, capsys):
    CatalogInitCLI.main(argv=_opts(tmp_path))
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', '--dry-run', *_opts(tmp_path)])
    out = capsys.readouterr().out
    assert 'hf://a' in out                       # printed
    cat = Catalog.load(cat_path(tmp_path))
    assert 'm' not in cat.models                 # but not persisted


def test_show_piped_output_is_plain(tmp_path, capsys):
    # Under capsys stdout is not a tty -> no ANSI escapes leak into pipes.
    from infer_stack.cli.commands_catalog import CatalogShowCLI

    CatalogInitCLI.main(argv=_opts(tmp_path))
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    capsys.readouterr()
    CatalogShowCLI.main(argv=_opts(tmp_path))
    out = capsys.readouterr().out
    assert '\x1b[' not in out                      # color-free when piped
    assert yaml.safe_load(out)['models']['m']['source'] == 'hf://a'


def test_help_tree_lists_groups_and_leaves(capsys):
    from infer_stack.cli.commands_meta import HelpTreeCLI

    HelpTreeCLI.main(argv=[])
    out = capsys.readouterr().out
    assert 'catalog' in out
    assert 'model' in out and 'endpoint' in out
    assert 'acquire' in out                       # top-level leasing verb
    assert 'tree' in out                          # itself, under help
