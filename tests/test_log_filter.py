from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from infer_stack.log_filter import compact_litellm_tracebacks


def _known_segments():
    p = 'litellm-1 | '
    return [
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/aiohttp/connector.py", line 1298, in _wrap_create_connection\n',
            p + '    sock = await aiohappyeyeballs.start_connection(\n',
            p + '  File "/usr/lib/python3.13/site-packages/aiohappyeyeballs/impl.py", line 122, in start_connection\n',
            p + '  File "uvloop/loop.pyx", line 2633, in sock_connect\n',
            p + "ConnectionRefusedError: [Errno 111] Connect call failed ('172.18.0.4', 8000)\n",
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/custom_httpx/aiohttp_transport.py", line 61, in map_aiohttp_exceptions\n',
            p + '    yield\n',
            p + '  File "/usr/lib/python3.13/site-packages/aiohttp/client.py", line 779, in _request\n',
            p + 'INFO:     172.18.0.1:35494 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error\n',
            p + '  File "/usr/lib/python3.13/site-packages/aiohttp/connector.py", line 1321, in _wrap_create_connection\n',
            p + 'aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host vllm-qwen3-5-27b:8000 ssl:default [Connect call failed]\n',
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/openai/_base_client.py", line 1604, in request\n',
            p + '  File "/usr/lib/python3.13/site-packages/httpx/_client.py", line 1730, in _send_single_request\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/custom_httpx/aiohttp_transport.py", line 306, in handle_async_request\n',
            p + 'httpx.ConnectError: Cannot connect to host vllm-qwen3-5-27b:8000 ssl:default [Connect call failed]\n',
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/openai/openai.py", line 929, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/litellm_core_utils/logging_utils.py", line 297, in async_wrapper\n',
            p + '  File "/usr/lib/python3.13/site-packages/openai/resources/chat/completions/completions.py", line 2700, in create\n',
            p + '  File "/usr/lib/python3.13/site-packages/openai/_base_client.py", line 1636, in request\n',
            p + 'openai.APIConnectionError: Connection error.\n',
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/main.py", line 620, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/openai/openai.py", line 989, in acompletion\n',
            p + 'litellm.llms.openai.common_utils.OpenAIError: Connection error.\n',
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/router.py", line 5508, in async_function_with_retries\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/utils.py", line 2072, in wrapper_async\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/main.py", line 639, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 588, in exception_type\n',
            p + 'litellm.exceptions.InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.\n',
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/proxy/proxy_server.py", line 6768, in chat_completion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/proxy/common_request_processing.py", line 953, in base_process_llm_request\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/router.py", line 5665, in async_function_with_retries\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 588, in exception_type\n',
            p + 'litellm.exceptions.InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.. Received Model Group=qwen3.5-27b\n',
        ],
    ]


def _model_not_found_segments():
    p = 'litellm-1 | '
    return [
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/openai/openai.py", line 929, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/litellm_core_utils/logging_utils.py", line 297, in async_wrapper\n',
            p + '  File "/usr/lib/python3.13/site-packages/openai/resources/chat/completions/completions.py", line 2700, in create\n',
            p + '  File "/usr/lib/python3.13/site-packages/openai/_base_client.py", line 1669, in request\n',
            p + "openai.NotFoundError: Error code: 404 - {'error': {'message': 'The model `qwen3.5-27b` does not exist.', 'type': 'NotFoundError', 'param': 'model', 'code': 404}}\n",
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/main.py", line 620, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/llms/openai/openai.py", line 989, in acompletion\n',
            p + "litellm.llms.openai.common_utils.OpenAIError: Error code: 404 - {'error': {'message': 'The model `qwen3.5-27b` does not exist.', 'type': 'NotFoundError', 'param': 'model', 'code': 404}}\n",
        ],
        [
            p + 'Traceback (most recent call last):\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/proxy/proxy_server.py", line 6768, in chat_completion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/proxy/common_request_processing.py", line 953, in base_process_llm_request\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/router.py", line 1666, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/utils.py", line 2072, in wrapper_async\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/main.py", line 639, in acompletion\n',
            p + '  File "/usr/lib/python3.13/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 552, in exception_type\n',
            p + 'litellm.exceptions.NotFoundError: litellm.NotFoundError: NotFoundError: OpenAIException - The model `qwen3.5-27b` does not exist.. Received Model Group=qwen3.5-27b\n',
        ],
    ]


def test_captured_litellm_connection_refusal_is_compacted():
    fixture = Path(__file__).parent / 'data' / 'litellm_connection_refused.log'
    original = fixture.read_text().splitlines(keepends=True)
    output = list(compact_litellm_tracebacks(original))
    text = ''.join(output)

    assert len(original) == 257
    assert len(output) == 31
    assert 'Traceback (most recent call last):' not in text
    assert '  File "' not in text
    assert "ConnectionRefusedError: [Errno 111]" in text
    assert 'Cannot connect to host vllm-qwen3-5-27b:8000' in text
    assert 'openai.APIConnectionError: Connection error.' in text
    assert 'Received Model Group=qwen3.5-27b' in text
    assert 'LiteLLM Retried: 3 times, LiteLLM Max Retries: 3' in text
    assert 'POST /v1/chat/completions HTTP/1.1" 500' in text
    assert text.count('GET /v1/models HTTP/1.1" 200 OK') == 2


def test_captured_litellm_model_not_found_is_compacted():
    fixture = Path(__file__).parent / 'data' / 'litellm_model_not_found.log'
    original = fixture.read_text().splitlines(keepends=True)
    output = list(compact_litellm_tracebacks(original))
    text = ''.join(output)

    # The capture begins in the middle of an older traceback, so that clipped
    # prefix correctly remains raw. Every complete traceback in the capture is
    # one of the explicitly registered model-not-found shapes and is compacted.
    assert len(original) == 415
    assert len(output) == 91
    assert 'Traceback (most recent call last):' not in text
    assert 'openai.NotFoundError: Error code: 404' in text
    assert 'litellm.llms.openai.common_utils.OpenAIError: Error code: 404' in text
    assert 'litellm.exceptions.NotFoundError:' in text
    assert 'The model `qwen3.5-27b` does not exist.' in text
    assert 'POST /v1/chat/completions HTTP/1.1" 404 Not Found' in text
    assert 'Available Model Group Fallbacks=None' in text


def test_known_model_not_found_segments_drop_only_scaffolding():
    for segment in _model_not_found_segments():
        output = list(compact_litellm_tracebacks(segment))
        assert output == [segment[-1]]


@pytest.mark.parametrize(('segment_index', 'missing_fragment'), [
    (0, 'site-packages/openai/resources/chat/completions/completions.py'),
    (1, 'site-packages/litellm/main.py'),
    (2, 'site-packages/litellm/proxy/common_request_processing.py'),
])
def test_model_not_found_near_miss_fails_open(segment_index, missing_fragment):
    original = _model_not_found_segments()[segment_index]
    mutated = [line for line in original if missing_fragment not in line]
    assert list(compact_litellm_tracebacks(mutated)) == mutated


def test_known_connection_refusal_segments_drop_only_scaffolding():
    chain = []
    separators = [
        'The above exception was the direct cause of the following exception:',
        'The above exception was the direct cause of the following exception:',
        'During handling of the above exception, another exception occurred:',
        'During handling of the above exception, another exception occurred:',
        'During handling of the above exception, another exception occurred:',
        'During handling of the above exception, another exception occurred:',
    ]
    segments = _known_segments()
    for index, segment in enumerate(segments):
        chain.extend(segment)
        if index < len(separators):
            chain.extend([
                'litellm-1 | \n',
                f'litellm-1 | {separators[index]}\n',
                'litellm-1 | \n',
            ])

    output = list(compact_litellm_tracebacks(chain))
    text = ''.join(output)

    assert 'Traceback (most recent call last):' not in text
    assert '  File "' not in text
    assert 'sock = await' not in text
    assert 'ConnectionRefusedError: [Errno 111]' in text
    assert 'aiohttp.client_exceptions.ClientConnectorError:' in text
    assert 'httpx.ConnectError:' in text
    assert 'openai.common_utils.OpenAIError: Connection error.' in text
    assert 'Received Model Group=qwen3.5-27b' in text
    # A real access-log line interleaved with frames is never swallowed.
    assert 'POST /v1/chat/completions HTTP/1.1" 500' in text
    # Exception-chain prose is not traceback scaffolding and stays visible.
    assert 'direct cause of the following exception' in text


def test_unknown_litellm_traceback_is_byte_identical():
    original = [
        'litellm-1 | Traceback (most recent call last):\n',
        'litellm-1 |   File "/app/new_path.py", line 10, in run\n',
        'litellm-1 |     explode()\n',
        'litellm-1 | NewImportantError: something new happened\n',
    ]
    assert list(compact_litellm_tracebacks(original)) == original


@pytest.mark.parametrize(('segment_index', 'missing_fragment'), [
    (0, 'site-packages/aiohappyeyeballs/impl.py'),
    (1, 'site-packages/aiohttp/client.py'),
    (2, 'site-packages/httpx/_client.py'),
    (3, 'site-packages/openai/resources/chat/completions/completions.py'),
    (4, 'site-packages/litellm/main.py'),
    (5, 'site-packages/litellm/utils.py'),
    (6, 'site-packages/litellm/proxy/common_request_processing.py'),
])
def test_registered_shape_with_missing_fingerprint_fails_open(
    segment_index, missing_fragment
):
    original = _known_segments()[segment_index]
    mutated = [line for line in original if missing_fragment not in line]
    assert list(compact_litellm_tracebacks(mutated)) == mutated


def test_non_litellm_traceback_is_byte_identical():
    original = [
        'vllm-qwen | Traceback (most recent call last):\n',
        'vllm-qwen |   File "/app/server.py", line 1, in run\n',
        'vllm-qwen | RuntimeError: engine failed\n',
    ]
    assert list(compact_litellm_tracebacks(original)) == original


def test_interleaved_other_service_preserves_order():
    original = _known_segments()[1]
    original = original[:3] + [
        'vllm-qwen | ERROR engine-side detail must stay here\n'
    ] + original[3:]
    output = list(compact_litellm_tracebacks(original))
    text = ''.join(output)
    assert text.index('engine-side detail') < text.index('POST /v1/chat/completions')
    assert text.index('POST /v1/chat/completions') < text.index('ClientConnectorError')


def test_timestamped_compose_lines_can_match_without_changing_output():
    stamp = '2026-09-16T19:24:48.123456789Z '
    original = [line.replace(' | ', f' | {stamp}', 1) for line in _known_segments()[0]]
    output = list(compact_litellm_tracebacks(original))
    assert output == [original[-1]]
    assert stamp in output[0]


def test_ansi_colored_compose_prefix_can_match_without_changing_output():
    original = [
        line.replace('litellm-1', '\x1b[36mlitellm-1\x1b[0m', 1)
        for line in _known_segments()[0]
    ]
    output = list(compact_litellm_tracebacks(original))
    assert output == [original[-1]]
    assert '\x1b[36m' in output[0]


def test_eof_mid_traceback_fails_open():
    original = _known_segments()[0][:-1]
    assert list(compact_litellm_tracebacks(original)) == original


def _follow_logs(monkeypatch, lines, *, no_color, tty=True):
    """Run `infer-stack logs -f` over a fake instance whose log is ``lines``."""
    from infer_stack.cli import commands_runtime
    from infer_stack.leasing import instances as instances_mod
    from infer_stack.leasing.instances import Instance

    config = SimpleNamespace(follow=True, raw=False, no_color=no_color, tail=None,
                             timestamps=False, services=None)

    class Backend:
        def instances(self):
            return [Instance('litellm', 'c1', '', 'running')]

    class Follower:
        def __init__(self, listing, **kw):
            assert [i.name for i in listing()] == ['litellm']
            self.stdout = iter(lines)

        def terminate(self):
            pass

    class Out(io.StringIO):
        def isatty(self):
            return tty

    monkeypatch.setattr(commands_runtime.LogsCLI, 'cli', lambda *a, **kw: config)
    monkeypatch.setattr(commands_runtime, '_day2_backend', lambda config: Backend())
    monkeypatch.setattr(commands_runtime, '_served_by_deployment', lambda: {})
    monkeypatch.setattr(instances_mod, 'LogFollower', Follower)
    stream = Out()
    monkeypatch.setattr(commands_runtime.sys, 'stdout', stream)
    assert commands_runtime.LogsCLI.main(argv=False) == 0
    return stream.getvalue()


def test_cli_follow_compacts_a_known_trace(monkeypatch):
    lines = [line.replace('litellm-1 | ', 'litellm  | ') for line in _known_segments()[0]]
    out = _follow_logs(monkeypatch, lines, no_color=True)
    assert 'Traceback (most recent call last):' not in out
    assert 'ConnectionRefusedError: [Errno 111]' in out


@pytest.mark.parametrize('no_color', [False, True])
def test_cli_follow_colors_prefixes_unless_asked_not_to(monkeypatch, no_color):
    out = _follow_logs(monkeypatch, ['litellm  | hello\n'], no_color=no_color)
    assert ('\x1b[' in out) is (not no_color)
    assert 'hello' in out

