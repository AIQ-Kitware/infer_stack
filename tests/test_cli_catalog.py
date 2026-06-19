"""Tests for the `infer-stack catalog …` editor and `help tree`."""

from __future__ import annotations

import pytest
import yaml

from infer_stack.cli.commands_catalog import (
    BundleAddCLI,
    CatalogInitCLI,
    CatalogSuggestCLI,
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


def test_endpoint_name_defaults_to_model(tmp_path):
    # No NAME given -> the endpoint alias defaults to `{model}-1`, so the served
    # alias / Open WebUI label is tied to the model.
    ModelAddCLI.main(argv=['smol135', '--source', 'hf://org/Smol',
                           *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['--model', 'smol135', '--max-model-len', '4096',
                              *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert cat.endpoints['smol135-1'].model == 'smol135'


def test_endpoint_default_name_slugs_ollama_tag(tmp_path):
    # An Ollama tag (`llama3:8b`) is slugified for the alias but kept as the tag.
    HostAddCLI.main(argv=['gpu0', '--engine', 'ollama', '--gpu', '0',
                          *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['--engine', 'ollama', '--model', 'llama3:8b',
                              '--host', 'gpu0', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert 'llama3-8b-1' in cat.endpoints
    assert cat.endpoints['llama3-8b-1'].model == 'llama3:8b'


def test_endpoint_no_name_no_model_errors(tmp_path):
    CatalogInitCLI.main(argv=_opts(tmp_path))
    with pytest.raises(SystemExit):
        EndpointAddCLI.main(argv=_opts(tmp_path))   # neither NAME nor --model


def test_endpoint_default_name_autoincrements(tmp_path):
    # Repeated default-named adds for one model accumulate as -1, -2 (no clobber).
    ModelAddCLI.main(argv=['m', '--source', 'hf://org/M', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['--model', 'm', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['--model', 'm', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert set(cat.endpoints) == {'m-1', 'm-2'}


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


def test_rm_multiple_endpoints(tmp_path):
    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    for nm in ('a', 'b', 'c'):
        EndpointAddCLI.main(argv=[nm, '--model', 'm', *_opts(tmp_path)])
    # rm accepts several names at once
    EndpointRmCLI.main(argv=['a', 'c', *_opts(tmp_path)])
    cat = Catalog.load(cat_path(tmp_path))
    assert set(cat.endpoints) == {'b'}

    # one missing name -> nothing removed (atomic)
    with pytest.raises(SystemExit):
        EndpointRmCLI.main(argv=['b', 'ghost', *_opts(tmp_path)])
    assert 'b' in Catalog.load(cat_path(tmp_path)).endpoints


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


def test_endpoint_show_no_name_lists_all(tmp_path, capsys):
    from infer_stack.cli.commands_catalog import EndpointShowCLI

    ModelAddCLI.main(argv=['m', '--source', 'hf://a', *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['e1', '--engine', 'vllm', '--model', 'm',
                              *_opts(tmp_path)])
    EndpointAddCLI.main(argv=['e2', '--engine', 'vllm', '--model', 'm',
                              *_opts(tmp_path)])
    capsys.readouterr()
    # no NAME -> show every endpoint (not "endpoint 'None' not found")
    assert EndpointShowCLI.main(argv=_opts(tmp_path)) == 0
    shown = yaml.safe_load(capsys.readouterr().out)['endpoints']
    assert set(shown) == {'e1', 'e2'}
    # a real name still shows just that one
    capsys.readouterr()
    EndpointShowCLI.main(argv=['e1', *_opts(tmp_path)])
    assert set(yaml.safe_load(capsys.readouterr().out)) == {'e1'}
    # an unknown name errors and lists what's available
    with pytest.raises(SystemExit) as exc:
        EndpointShowCLI.main(argv=['ghost', *_opts(tmp_path)])
    assert 'have: e1, e2' in str(exc.value)


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


# ---------------------------------------------------------------------------
# catalog suggest — seed from server introspection (simulated for determinism)
# ---------------------------------------------------------------------------


def test_suggest_render_only_writes_nothing(tmp_path, capsys):
    # Default is render: prints a catalog fragment, but creates no file.
    CatalogSuggestCLI.main(
        argv=['--simulate-hardware', '2x80', *_opts(tmp_path)]
    )
    out = capsys.readouterr().out
    rendered = yaml.safe_load(out)
    assert 'qwen2.5-72b' in rendered['models']        # 2 GPUs -> the 72B fits
    assert 'qwen2.5-72b' in rendered['endpoints']
    assert not (tmp_path / 'catalog.yaml').exists()    # nothing written


def test_suggest_apply_merges_and_validates(tmp_path):
    CatalogSuggestCLI.main(
        argv=['--simulate-hardware', '1x48', '--apply', *_opts(tmp_path)]
    )
    cat = Catalog.load(cat_path(tmp_path))            # parses + cross-refs ok
    assert 'qwen2.5-7b' in cat.endpoints
    # the largest fitting model is kept warm, the rest stop
    warm = [n for n, e in cat.endpoints.items() if e.reclaim == 'keep-warm']
    assert len(warm) == 1


def test_suggest_apply_is_additive_and_idempotent(tmp_path):
    # A hand-added entry survives; re-applying changes nothing without --force.
    ModelAddCLI.main(argv=['mine', '--source', 'hf://me/Model', *_opts(tmp_path)])
    CatalogSuggestCLI.main(
        argv=['--simulate-hardware', '1x24', '--apply', *_opts(tmp_path)]
    )
    after_first = (tmp_path / 'catalog.yaml').read_text()
    CatalogSuggestCLI.main(
        argv=['--simulate-hardware', '1x24', '--apply', *_opts(tmp_path)]
    )
    assert (tmp_path / 'catalog.yaml').read_text() == after_first
    cat = Catalog.load(cat_path(tmp_path))
    assert 'mine' in cat.models                       # hand-added entry preserved


def test_suggest_no_fit_writes_nothing(tmp_path, capsys):
    # A box too small for any pooled model: a friendly note, no file, no crash.
    CatalogSuggestCLI.main(
        argv=['--simulate-hardware', '1x0', '--apply', *_opts(tmp_path)]
    )
    err = capsys.readouterr().err
    assert 'no pooled model fits' in err
    assert not (tmp_path / 'catalog.yaml').exists()
