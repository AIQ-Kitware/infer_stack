"""
``infer-stack mock serve`` -- run the deterministic mock inference server.

This is a test fixture, not a deployment target.  It exists so cards,
HELM runs and deployment plumbing can be exercised end-to-end without a
GPU, an API key, or a network.
"""

from __future__ import annotations

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
        import kwutil

        from ..mockserver.server import MockServer

        config = cls.cli(argv=argv, data=kwargs, strict=True)

        if config.config_fpath:
            spec = kwutil.Yaml.coerce(ub.Path(config.config_fpath).read_text())
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

        server = MockServer(spec, host=config.host, port=int(config.port))
        if config.print_url:
            print(f'infer-stack mock server listening on {server.url}')
            print(f'  models: {sorted(server.simulator.profiles)}')
            print(f'  seed:   {server.simulator.seed}')
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
