"""
OpenAI-compatible mock inference server.

Implements just enough of the chat-completions API for evaluation cards and
HELM to talk to it, backed by :mod:`infer_stack.mockserver.simulator` so
every response is deterministic.

Deliberately stdlib-only (``http.server``) -- this is a test fixture that
has to start in milliseconds inside CI and inside a card run, so it should
not drag a web framework into infer-stack's dependency set.

Endpoints
---------

``GET  /health``                 readiness probe
``GET  /v1/models``              configured models
``POST /v1/chat/completions``    chat completions
``POST /v1/completions``         raw text completions
``GET  /__mock__/requests``      every request received, verbatim
``POST /__mock__/reset``         clear the recorded requests

The ``__mock__`` endpoints are the reason a mock *server* is worth more
than an in-process fake: they let a test assert on the exact bytes a
client sent.  Comparing recorded payloads from two clients is how you
verify that a card's live calls and an upstream harness's calls really do
present the same prompt to the model, rather than merely documenting that
they should.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .simulator import ModelProfile, Simulator, flatten_messages

__all__ = ['MockServer', 'build_simulator']


def build_simulator(config: dict) -> Simulator:
    """
    Build a :class:`Simulator` from a config mapping.

    Args:
        config (dict): with optional keys ``seed``, ``models``,
            ``questions`` and ``answer_key``.

    Returns:
        Simulator

    Example:
        >>> from infer_stack.mockserver.server import build_simulator
        >>> sim = build_simulator({'models': {'m': {'ability': 0.9}}})
        >>> assert sim.profiles['m'].ability == 0.9
    """
    profiles = {
        model_id: ModelProfile.coerce(model_id, block)
        for model_id, block in (config.get('models') or {}).items()
    }
    return Simulator(
        profiles=profiles,
        answer_key=config.get('answer_key'),
        questions=config.get('questions'),
        composition=config.get('composition'),
        seed=str(config.get('seed', 'infer-stack-mock')),
    )


class _Handler(BaseHTTPRequestHandler):
    # Bound by MockServer.
    simulator: Simulator
    recorded: list
    record_lock: threading.Lock
    max_recorded: int
    draw_counts: dict
    draw_lock: threading.Lock
    api_keys: set
    require_auth: bool

    protocol_version = 'HTTP/1.1'

    def _next_draw(
        self, model_id: str, messages, temperature: float, n_samples: int = 1
    ) -> int:
        """
        Pick the sample index for this request.

        A real API at ``temperature > 0`` returns a *different* completion
        each time you send it the same request; the client does not tell it
        "this is sample 3". Clients that draw K samples therefore just send
        the same body K times.

        If the mock keyed only on request content, those K identical
        requests would collapse to one answer and any self-consistency
        score would be identically 1.0 -- the card would pass while
        measuring nothing. So repeated identical sampling requests advance a
        server-side counter.

        At ``temperature == 0`` decoding is greedy, so every request maps to
        index 0 and stays perfectly reproducible.
        """
        if temperature <= 0.0:
            return 0
        key = (model_id, flatten_messages(messages), round(temperature, 6))
        with self.draw_lock:
            index = self.draw_counts.get(key, 0)
            self.draw_counts[key] = index + max(1, int(n_samples))
        return index

    def log_message(self, fmt, *args):  # noqa: D102 - silence stderr spam
        pass

    # -- helpers ---------------------------------------------------------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """
        Enforce bearer-token auth, if the config asked for it.

        Real endpoints sit behind a key, and a client that forgets to send
        one should fail here the way it would in production rather than
        succeeding locally and 401-ing on first deployment.
        """
        if not self.require_auth:
            return True
        header = self.headers.get('Authorization', '')
        token = header[7:].strip() if header.startswith('Bearer ') else ''
        if token and (not self.api_keys or token in self.api_keys):
            return True
        self._send_json(
            {
                'error': {
                    'message': (
                        'Incorrect API key provided.' if token else
                        'You didn\'t provide an API key. You need to provide '
                        'your API key in an Authorization header using Bearer '
                        'auth (i.e. Authorization: Bearer YOUR_KEY).'
                    ),
                    'type': 'invalid_request_error',
                    'code': 'invalid_api_key',
                }
            },
            status=401,
        )
        return False

    def _read_json(self) -> dict:
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def _record(self, path: str, payload: dict) -> None:
        with self.record_lock:
            if len(self.recorded) >= self.max_recorded:
                return
            self.recorded.append({
                'path': path,
                'headers': dict(self.headers),
                'body': payload,
            })

    # -- routes ----------------------------------------------------------

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.rstrip('/') in ('/health', '/healthz'):
            self._send_json({'status': 'ok', 'mock': True})
        elif self.path.rstrip('/') == '/v1/models':
            if not self._check_auth():
                return
            self._send_json({
                'object': 'list',
                'data': [
                    {
                        'id': profile.served_model_name or profile.model_id,
                        'object': 'model',
                        'created': 0,
                        'owned_by': 'vllm',
                        'root': profile.model_id,
                        'parent': None,
                        'max_model_len': profile.max_model_len,
                        'permission': [],
                        # Non-standard, but the whole point of a mock is to
                        # be able to see what it was told to be.
                        'mock': {
                            'ability': profile.ability,
                            'consistency': profile.consistency,
                            'mode': profile.mode,
                        },
                        **profile.extra,
                    }
                    for profile in self.simulator.profiles.values()
                ],
            })
        elif self.path.rstrip('/') == '/__mock__/requests':
            with self.record_lock:
                self._send_json({'requests': list(self.recorded)})
        else:
            self._send_json({'error': f'unknown path {self.path}'}, status=404)

    def do_POST(self):  # noqa: N802 - http.server API
        path = self.path.rstrip('/')

        if path.startswith('/v1/') and not self._check_auth():
            return

        if path == '/__mock__/reset':
            with self.record_lock:
                self.recorded.clear()
            self._send_json({'status': 'reset'})
            return

        if path not in ('/v1/chat/completions', '/v1/completions'):
            self._send_json({'error': f'unknown path {self.path}'}, status=404)
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError as ex:
            self._send_json({'error': f'malformed JSON: {ex}'}, status=400)
            return

        self._record(path, payload)

        model_id = payload.get('model')
        is_chat = path == '/v1/chat/completions'
        if is_chat:
            messages = payload.get('messages') or []
        else:
            # A raw completion prompt is already fully rendered; treat it as
            # a single user turn so both endpoints key on the same text.
            messages = [{'role': 'user', 'content': payload.get('prompt', '')}]
        temperature = float(payload.get('temperature', 0.0) or 0.0)
        n_samples = int(payload.get('n', 1) or 1)

        profile = self.simulator.resolve_profile(model_id)
        if profile is None:
            self._send_json(
                {
                    'error': {
                        'message': (
                            f'model {model_id!r} is not configured on this '
                            f'mock server; configured: '
                            f'{sorted(self.simulator.profiles)}'
                        ),
                        'type': 'invalid_request_error',
                    }
                },
                status=404,
            )
            return

        if profile.max_model_len:
            # Rough proxy for tokens; enough to exercise the client's
            # context-overflow path, which is otherwise only reachable with
            # a real long prompt.
            approx_tokens = max(1, len(flatten_messages(messages)) // 4)
            requested = int(payload.get('max_tokens') or 0)
            if approx_tokens + requested > profile.max_model_len:
                self._send_json(
                    {
                        'error': {
                            'message': (
                                f"This model's maximum context length is "
                                f'{profile.max_model_len} tokens. However, '
                                f'you requested about '
                                f'{approx_tokens + requested} tokens '
                                f'({approx_tokens} in the messages, '
                                f'{requested} in the completion).'
                            ),
                            'type': 'invalid_request_error',
                            'code': 'context_length_exceeded',
                        }
                    },
                    status=400,
                )
                return

        if profile.latency_s:
            time.sleep(profile.latency_s)

        # Where this request's samples start. Clients that ask for n>1 get
        # a contiguous block; clients that re-send the same body K times
        # (the common case, since the OpenAI API has no "sample index")
        # advance one step per call.
        base_index = self._next_draw(
            model_id, messages, temperature, n_samples
        )

        choices = []
        for offset in range(n_samples):
            sample_index = base_index + offset
            completion = self.simulator.complete(
                model_id,
                messages,
                temperature=temperature,
                sample_index=sample_index,
            )
            if completion.should_fail:
                self._send_json(
                    {
                        'error': {
                            'message': 'injected mock failure',
                            'type': 'server_error',
                        }
                    },
                    status=profile.failure_status,
                )
                return
            choice = {
                'index': offset,
                'finish_reason': completion.finish_reason,
                # Non-standard, but invaluable when debugging why a card
                # scored the way it did.
                'mock': {
                    'is_correct': completion.is_correct,
                    'latent_key': completion.latent_key,
                },
            }
            if is_chat:
                choice['message'] = {
                    'role': 'assistant',
                    'content': completion.text,
                }
            else:
                choice['text'] = completion.text
            choices.append(choice)

        self._send_json({
            'id': 'chatcmpl-mock' if is_chat else 'cmpl-mock',
            'object': 'chat.completion' if is_chat else 'text_completion',
            'created': 0,
            'model': model_id,
            'choices': choices,
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            },
        })


class MockServer:
    """
    A running mock inference server.

    Args:
        config (dict): see :func:`build_simulator`.
        host (str): bind address.
        port (int): bind port; 0 picks a free one.
        max_recorded (int): cap on retained request records, so a long card
            run cannot exhaust memory.

    Example:
        >>> import json, urllib.request
        >>> from infer_stack.mockserver.server import MockServer
        >>> config = {'models': {'m': {'ability': 0.9}}, 'seed': 'demo'}
        >>> with MockServer(config, port=0) as server:
        ...     req = urllib.request.Request(
        ...         server.url + '/v1/chat/completions',
        ...         data=json.dumps({
        ...             'model': 'm',
        ...             'messages': [{'role': 'user', 'content': 'hi'}],
        ...         }).encode(),
        ...         headers={'Content-Type': 'application/json'},
        ...     )
        ...     body = json.loads(urllib.request.urlopen(req).read())
        >>> assert body['choices'][0]['message']['content']
    """

    def __init__(
        self,
        config: dict,
        host: str = '127.0.0.1',
        port: int = 0,
        max_recorded: int = 100000,
    ) -> None:
        self.simulator = build_simulator(config)
        self.recorded: list = []

        handler = type('_BoundHandler', (_Handler,), {
            'simulator': self.simulator,
            'recorded': self.recorded,
            'record_lock': threading.Lock(),
            'max_recorded': max_recorded,
            'draw_counts': {},
            'draw_lock': threading.Lock(),
            'api_keys': set(config.get('api_keys') or []),
            'require_auth': bool(
                config.get('require_auth',
                           bool(config.get('api_keys')))),
        })

        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """int: the bound port (resolved when ``port=0`` was requested)."""
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        """str: base URL of the running server."""
        host, port = self._httpd.server_address[:2]
        return f'http://{host}:{port}'

    def start(self) -> 'MockServer':
        """Start serving in a background thread."""
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name='infer-stack-mockserver',
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def serve_forever(self) -> None:
        """Serve in the foreground until interrupted."""
        self._httpd.serve_forever()

    def __enter__(self) -> 'MockServer':
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()
