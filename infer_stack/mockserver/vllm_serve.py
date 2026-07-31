"""
A ``vllm serve``-compatible entry point for the mock server.

The mock used to be started out-of-band, which meant a demo exercised the
card and the endpoint but skipped infer-stack's own acquire / converge /
release path -- exactly the machinery a dress rehearsal is meant to
validate. Presenting the same command line as vLLM lets an endpoint点 at a
mock image through ``runtime.image`` and be leased like any other:

.. code:: yaml

    endpoints:
      mock-smol:
        engine: vllm
        model: smol135
        runtime:
          image: aiq-mock-vllm:latest
          max_model_len: 2048

infer-stack renders ``<image-entrypoint> <hf_model_id> --host 0.0.0.0
--port 8000 --served-model-name=... --max-model-len=...`` and a long tail
of GPU knobs. Everything meaningful to a simulator is honoured; the rest
is accepted and ignored, because rejecting ``--gpu-memory-utilization``
would make the mock undeployable for a reason that has nothing to do with
what it simulates.

Behaviour comes from a config file when one is mounted
(``--mock-config``) and from defaults otherwise, so an endpoint can be
stood up with no fixture at all and still answer.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

__all__ = ['build_parser', 'config_from_args', 'main']

#: vLLM flags that take a value and mean nothing to a simulator. Accepted
#: so a real endpoint definition deploys unchanged.
_IGNORED_WITH_VALUE = [
    '--tensor-parallel-size', '--pipeline-parallel-size',
    '--data-parallel-size', '--gpu-memory-utilization',
    '--max-num-batched-tokens', '--max-num-seqs', '--revision',
    '--quantization', '--dtype', '--chat-template', '--tool-call-parser',
    '--kv-cache-dtype', '--block-size', '--swap-space', '--seed',
    '--download-dir', '--load-format', '--tokenizer',
]

#: Valueless vLLM flags, same rationale.
_IGNORED_FLAGS = [
    '--trust-remote-code', '--enable-prefix-caching',
    '--enable-auto-tool-choice', '--enforce-eager', '--disable-log-requests',
    '--enable-log-requests', '--disable-log-stats',
]


def build_parser() -> argparse.ArgumentParser:
    """The subset of ``vllm serve`` this simulator understands."""
    parser = argparse.ArgumentParser(
        prog='mock-vllm-serve',
        description='OpenAI-compatible mock that deploys like vLLM.',
    )
    parser.add_argument('model', nargs='?', default=None,
                        help='Model id, positional as in `vllm serve`.')
    parser.add_argument('--model', dest='model_flag', default=None,
                        help='Model id, flag form.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--served-model-name', default=None,
                        help='Alias clients request instead of the model id.')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Enforced: overflow returns '
                             'context_length_exceeded.')
    parser.add_argument('--api-key', default=None,
                        help='Require this bearer token.')

    # Simulator-specific, namespaced so they cannot collide with vLLM.
    parser.add_argument('--mock-config', default=None,
                        help='YAML/JSON cohort config to serve.')
    parser.add_argument('--mock-mode', default=None,
                        help='Override every model\'s response mode.')
    parser.add_argument('--mock-ability', type=float, default=0.6,
                        help='Ability for a model with no config entry.')
    parser.add_argument('--mock-seed', default=None)

    for flag in _IGNORED_WITH_VALUE:
        parser.add_argument(flag, dest=f'ignored_{flag.strip("-")}',
                            default=None, help=argparse.SUPPRESS)
    for flag in _IGNORED_FLAGS:
        parser.add_argument(flag, dest=f'ignored_{flag.strip("-")}',
                            action='store_true', help=argparse.SUPPRESS)
    return parser


def config_from_args(args, unknown=()) -> dict:
    """
    Build a mock-server config from parsed vLLM-style arguments.

    Args:
        args: the parsed namespace.
        unknown: leftover arguments, reported rather than silently dropped.

    Returns:
        dict: a config for :class:`infer_stack.mockserver.MockServer`.
    """
    import kwutil
    import ubelt as ub

    model = args.model_flag or args.model
    if not model:
        raise SystemExit(
            'no model given; pass it positionally as `vllm serve` does, or '
            'with --model'
        )

    if args.mock_config:
        config = dict(kwutil.Yaml.coerce(ub.Path(args.mock_config).read_text()))
    else:
        config = {}
    config.setdefault('models', {})

    # The served model must exist even when a mounted fixture does not
    # mention it, or the endpoint comes up and 404s every request.
    block: dict[str, Any] = dict(config['models'].get(model) or {})
    block.setdefault('ability', float(args.mock_ability))
    if args.served_model_name:
        block['served_model_name'] = args.served_model_name
    if args.max_model_len:
        block['max_model_len'] = int(args.max_model_len)
    config['models'][model] = block

    if args.mock_mode:
        for entry in config['models'].values():
            entry['mode'] = args.mock_mode
    if args.mock_seed:
        config['seed'] = args.mock_seed
    if args.api_key:
        config['api_keys'] = [args.api_key]
        config['require_auth'] = True

    if unknown:
        print(f'[mock-vllm] ignoring unrecognized arguments: {list(unknown)}',
              file=sys.stderr, flush=True)
    return config


def main(argv=None) -> int:
    """Serve until interrupted. Mirrors ``vllm serve``'s startup output."""
    from .server import MockServer

    parser = build_parser()
    args, unknown = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv)
    config = config_from_args(args, unknown)

    # Containers publish on a fixed port and are reached through the host
    # mapping, so bind wherever we were told.
    server = MockServer(config, host=args.host, port=int(args.port))
    served = sorted(
        profile.served_model_name or profile.model_id
        for profile in server.simulator.profiles.values()
    )
    print(f'[mock-vllm] serving {served} on {args.host}:{args.port}',
          flush=True)
    print(f'[mock-vllm] seed={server.simulator.seed} '
          f'modes={sorted({p.mode for p in server.simulator.profiles.values()})}',
          flush=True)
    print('[mock-vllm] THIS IS A SIMULATOR. Results describe it, not a model.',
          flush=True)
    # Mirror vLLM's readiness line so log-scraping health checks behave.
    print(f'INFO:     Uvicorn running on http://{args.host}:{args.port} '
          f'(Press CTRL+C to quit)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('[mock-vllm] shutting down', flush=True)
    finally:
        server.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
