"""
``infer-stack mock serve`` -- run the deterministic mock inference server.

This is a test fixture, not a deployment target.  It exists so cards,
HELM runs and deployment plumbing can be exercised end-to-end without a
GPU, an API key, or a network.
"""

from __future__ import annotations

from typing import Any

import scriptconfig as scfg
import ubelt as ub


class MockServeCLI(scfg.DataConfig):
    """
    Serve a deterministic OpenAI-compatible mock inference endpoint.
    """

    __command__ = 'serve'

    config_fpath = scfg.Value(
        None,
        position=1,
        help=ub.paragraph(
            """
            YAML/JSON file describing the simulated cohort.  Supports
            ``seed``, ``models`` (each with ``ability``, optional
            ``consistency``, ``failure_rate``, ``latency_s``),
            ``questions`` (id -> question text) and ``answer_key``
            (id -> gold answer).
            """
        ),
    )

    host = scfg.Value('127.0.0.1', help='Bind address.')

    port = scfg.Value(
        8100,
        help='Bind port.  Use 0 to pick a free port and print it.',
    )

    mode = scfg.Value(
        None,
        help=ub.paragraph(
            """
            Override every model's response mode. Useful for a demo or for
            exercising a client's parsing: 'sycophant' always agrees,
            'echo' returns the prompt it was sent, 'thinking' wraps the
            answer in <think> tags, 'truncated' stops early with
            finish_reason=length, 'magic_8ball' and 'pirate' are for
            morale. Default is 'simulate', which answers from the model's
            configured ability.
            """
        ),
    )

    require_auth = scfg.Value(
        False,
        isflag=True,
        help='Reject requests without a valid bearer token.',
    )

    api_key = scfg.Value(
        None,
        help=ub.paragraph(
            """
            Accept only this bearer token. Implies --require_auth. Without
            it but with --require_auth, any non-empty token is accepted,
            which is enough to catch a client that sends none.
            """
        ),
    )

    list_modes = scfg.Value(
        False, isflag=True, help='Print the available response modes and exit.',
    )

    seed = scfg.Value(
        None,
        help=ub.paragraph(
            """
            Override the config's server seed.  Changing it yields a
            different but equally reproducible world.
            """
        ),
    )

    print_url = scfg.Value(
        True,
        isflag=True,
        help='Print the bound base URL on startup.',
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        import yaml

        from ..mockserver.modes import available_modes
        from ..mockserver.server import MockServer

        config = cls.cli(argv=argv, data=kwargs, strict=True)

        if config.list_modes:
            print('available response modes:')
            for name in available_modes():
                print(f'  {name}')
            return

        # Annotated because the branches below add keys of other types
        # (api_keys is a list, require_auth a bool); without it the type is
        # inferred from the default literal alone and every later assignment
        # looks like a mistake. This is a free-form config mapping -- it
        # normally arrives from YAML.
        spec: dict[str, Any]
        if config.config_fpath:
            spec = yaml.safe_load(
                ub.Path(config.config_fpath).read_text()
            ) or {}
        else:
            # A cohort still has to *vary* to be useful, so the built-in
            # default is a spread of abilities rather than one model.
            spec = {
                'models': {
                    f'mock/model-{i}': {'ability': ability}
                    for i, ability in enumerate(
                        [0.15, 0.35, 0.55, 0.75, 0.95]
                    )
                }
            }

        if config.seed is not None:
            spec['seed'] = config.seed
        if config.mode:
            for block in spec.setdefault('models', {}).values():
                block['mode'] = config.mode
        if config.api_key:
            spec['api_keys'] = [config.api_key]
        if config.require_auth:
            spec['require_auth'] = True

        server = MockServer(spec, host=config.host, port=int(config.port))
        if config.print_url:
            print(f'infer-stack mock server listening on {server.url}')
            print(f'  models: {sorted(server.simulator.profiles)}')
            print(f'  seed:   {server.simulator.seed}')
            modes = sorted({p.mode for p in server.simulator.profiles.values()})
            print(f'  modes:  {modes}')
            if spec.get('require_auth') or spec.get('api_keys'):
                print('  auth:   bearer token required')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nshutting down mock server')
        finally:
            server.stop()


class MockModalCLI(scfg.ModalCLI):
    """
    Deterministic mock inference server for tests and dry runs.
    """

    __command__ = 'mock'

    serve = MockServeCLI


__cli__ = MockModalCLI
