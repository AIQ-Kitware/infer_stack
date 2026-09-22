"""The CLI commands the TUI logs must be real commands.

A logged command that does not parse, or that writes a different catalog entry
than the TUI did, would teach the wrong thing -- worse than logging nothing.
"""

from __future__ import annotations

import shlex

import pytest
import scriptconfig as scfg

from infer_stack import cli_equivalent as cli
from infer_stack.cli import ManageCLI


def parse(text: str):
    """Resolve ``infer-stack a b ...`` to its CLI class and parse the rest."""
    words = shlex.split(text.split('   #', 1)[0])
    assert words[0] == 'infer-stack'
    node, rest = ManageCLI, words[1:]
    while isinstance(node, type) and issubclass(node, scfg.ModalCLI):
        node = getattr(node, rest[0].replace('-', '_'))
        rest = rest[1:]
    return node, node.cli(argv=rest, strict=True)


@pytest.mark.parametrize('text', [
    cli.command('acquire', 'qwen', '--owner', 'manual', '--no-wait', '--yes'),
    cli.command('release', 'lease-0123', '--yes'),
    cli.command('release', '--all', '--yes'),
    cli.command('evict', 'grp-a', 'grp-b', '--yes'),
    cli.command('evict', '--all', '--yes'),
    cli.command('apply', '--yes'),
    cli.command('stack', 'down'),
    cli.command('catalog', 'model', 'add', 'm', '--source', 'hf://org/m'),
    cli.command('catalog', 'endpoint', 'rm', 'e'),
    cli.command('catalog', 'model', 'rm', 'm'),
    cli.command('catalog', 'suggest', '--apply'),
])
def test_every_logged_command_parses(text):
    parse(text)


def test_acquire_matches_what_the_tui_does():
    _, config = parse(cli.command('acquire', 'qwen', '--owner', 'manual',
                                  '--no-wait', '--yes'))
    assert config.names == ['qwen'] and config.owner == 'manual'
    assert config.wait is False and config.ttl is None      # the TUI: no wait, no TTL


ENTRIES = [
    {'engine': 'vllm', 'model': 'm'},
    {'engine': 'vllm', 'model': 'm', 'reclaim': {'policy': 'stop'},
     'protocol': 'completions', 'public_name': 'shared',
     'placement': {'min_vram_gib': 24.0, 'gpu_indices': [0, 3]},
     'runtime': {'max_model_len': 65536, 'gpu_memory_utilization': 0.93,
                 'enable_prefix_caching': True, 'tensor_parallel_size': 2,
                 'image': 'ghcr.io/x/y:sha-1', 'dtype': 'half', 'quantization': 'true',
                 'extra_args': ['--enforce-eager', '--seed', '0']}},
]


@pytest.mark.parametrize('entry', ENTRIES)
def test_endpoint_add_round_trips_the_entry(tmp_path, entry):
    from infer_stack.cli.commands_catalog import _load_raw

    path = tmp_path / 'catalog.yaml'
    words = shlex.split(cli.command('catalog', 'model', 'add', 'm', '--source', 'hf://org/m'))
    ManageCLI.main(argv=[*words[1:], '--catalog', str(path)])
    words = shlex.split(cli.endpoint_add('e', entry))
    ManageCLI.main(argv=[*words[1:], '--catalog', str(path)])
    assert _load_raw(path)['endpoints']['e'] == entry


def test_fields_the_cli_cannot_set_are_named_not_dropped():
    text = cli.endpoint_add('e', {'engine': 'vllm', 'model': 'm', 'surprise': 1})
    assert text.endswith('# then `infer-stack catalog edit` for: surprise')
