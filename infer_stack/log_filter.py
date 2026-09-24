"""Conservative streaming filters for noisy third-party service logs.

The filters in this module are intentionally allowlist-driven.  A traceback is
only compacted when its terminal exception and stack frames match a registered
LiteLLM traceback pattern.  Unknown and near-miss tracebacks are replayed
verbatim so adding this layer cannot silently hide a newly useful failure.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_DOCKER_TIMESTAMP_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?'
    r'(?:Z|[+-]\d{2}:\d{2})[ \t]'
)
_TRACEBACK_HEADER = 'Traceback (most recent call last):'
_MAX_TRACEBACK_LINES = 512
_MAX_TRACEBACK_BYTES = 256 * 1024

# Compose service names can carry a project prefix/suffix, so use a substring
# just as the TUI does for its gateway selector.
_LITELLM_SERVICE_HINT = 'litellm'

# An ordinary log record may be interleaved while Python is writing traceback
# frames.  It belongs to the surrounding log stream, not to traceback syntax.
_LOG_RECORD_RE = re.compile(
    r'^(?:(?:\d{2}:){2}\d{2}(?:\.\d+)?\s+-\s+)?'
    r'(?:(?:LiteLLM Proxy|LiteLLM):)?'
    r'(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b'
)


@dataclass(frozen=True)
class LiteLLMTracebackPattern:
    """One explicitly approved LiteLLM traceback shape.

    ``terminal`` identifies the final exception line for one Python traceback
    segment.  Every ``required_fragment`` must also occur somewhere in that
    segment before any stack scaffolding may be removed.
    """

    name: str
    terminal: re.Pattern[str]
    required_fragments: tuple[str, ...]

    def matches(self, traceback_text: str, terminal: str) -> bool:
        if self.terminal.search(terminal) is None:
            return False
        return all(
            fragment in traceback_text for fragment in self.required_fragments
        )


# Keep this registry deliberately small. Each entry corresponds to one stack
# segment from a captured LiteLLM failure family that maintainers have approved
# as redundant traceback scaffolding. Adding a new family should come with a
# captured regression fixture plus a near-miss test proving that missing
# fingerprints fail open.
LITELLM_TRACEBACK_PATTERNS = (
    LiteLLMTracebackPattern(
        name='backend_connection_refused.aiohttp_socket_connect',
        terminal=re.compile(
            r'^ConnectionRefusedError: \[Errno 111\] Connect call failed '
        ),
        required_fragments=(
            'site-packages/aiohttp/connector.py',
            'site-packages/aiohappyeyeballs/impl.py',
            'uvloop/loop.pyx',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.aiohttp_connector',
        terminal=re.compile(
            r'^aiohttp\.client_exceptions\.ClientConnectorError: '
            r'Cannot connect to host '
        ),
        required_fragments=(
            'site-packages/litellm/llms/custom_httpx/aiohttp_transport.py',
            'site-packages/aiohttp/client.py',
            'site-packages/aiohttp/connector.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.httpx',
        terminal=re.compile(r'^httpx\.ConnectError: Cannot connect to host '),
        required_fragments=(
            'site-packages/openai/_base_client.py',
            'site-packages/httpx/_client.py',
            'site-packages/litellm/llms/custom_httpx/aiohttp_transport.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.openai_sdk',
        terminal=re.compile(r'^openai\.APIConnectionError: Connection error\.$'),
        required_fragments=(
            'site-packages/litellm/llms/openai/openai.py',
            'site-packages/litellm/litellm_core_utils/logging_utils.py',
            'site-packages/openai/resources/chat/completions/completions.py',
            'site-packages/openai/_base_client.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.openai_adapter',
        terminal=re.compile(
            r'^litellm\.llms\.openai\.common_utils\.OpenAIError: '
            r'Connection error\.$'
        ),
        required_fragments=(
            'site-packages/litellm/main.py',
            'site-packages/litellm/llms/openai/openai.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.router',
        terminal=re.compile(
            r'^litellm\.exceptions\.InternalServerError: '
            r'litellm\.InternalServerError: InternalServerError: '
            r'OpenAIException - Connection error\.$'
        ),
        required_fragments=(
            'site-packages/litellm/router.py',
            'site-packages/litellm/utils.py',
            'site-packages/litellm/main.py',
            'site-packages/litellm/litellm_core_utils/exception_mapping_utils.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='backend_connection_refused.proxy',
        terminal=re.compile(
            r'^litellm\.exceptions\.InternalServerError: '
            r'litellm\.InternalServerError: InternalServerError: '
            r'OpenAIException - Connection error\.\.? Received Model Group='
        ),
        required_fragments=(
            'site-packages/litellm/proxy/proxy_server.py',
            'site-packages/litellm/proxy/common_request_processing.py',
            'site-packages/litellm/router.py',
            'site-packages/litellm/litellm_core_utils/exception_mapping_utils.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='model_not_found.openai_sdk',
        terminal=re.compile(
            r'^openai\.NotFoundError: Error code: 404 - .*'
            r"The model `[^`]+` does not exist\."
        ),
        required_fragments=(
            'site-packages/litellm/llms/openai/openai.py',
            'site-packages/litellm/litellm_core_utils/logging_utils.py',
            'site-packages/openai/resources/chat/completions/completions.py',
            'site-packages/openai/_base_client.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='model_not_found.openai_adapter',
        terminal=re.compile(
            r'^litellm\.llms\.openai\.common_utils\.OpenAIError: '
            r'Error code: 404 - .*The model `[^`]+` does not exist\.'
        ),
        required_fragments=(
            'site-packages/litellm/main.py',
            'site-packages/litellm/llms/openai/openai.py',
        ),
    ),
    LiteLLMTracebackPattern(
        name='model_not_found.proxy',
        terminal=re.compile(
            r'^litellm\.exceptions\.NotFoundError: '
            r'litellm\.NotFoundError: NotFoundError: OpenAIException - '
            r'The model `[^`]+` does not exist\.\.? Received Model Group='
        ),
        required_fragments=(
            'site-packages/litellm/proxy/proxy_server.py',
            'site-packages/litellm/proxy/common_request_processing.py',
            'site-packages/litellm/router.py',
            'site-packages/litellm/utils.py',
            'site-packages/litellm/main.py',
            'site-packages/litellm/litellm_core_utils/exception_mapping_utils.py',
        ),
    ),
)


@dataclass(frozen=True)
class _ComposeLine:
    raw: str
    service: str
    message: str


@dataclass(frozen=True)
class _BufferedLine:
    raw: str
    message: str | None
    role: str


class _TracebackBuffer:
    """Buffer one traceback segment until its terminal exception is known."""

    def __init__(self, service: str, first: _ComposeLine):
        self.service = service
        self.lines = [_BufferedLine(first.raw, first.message, 'header')]
        self.nbytes = len(first.raw.encode('utf8', errors='replace'))

    def add(self, line: _ComposeLine | None, raw: str, role: str) -> None:
        message = None if line is None else line.message
        self.lines.append(_BufferedLine(raw, message, role))
        self.nbytes += len(raw.encode('utf8', errors='replace'))

    @property
    def too_large(self) -> bool:
        return (
            len(self.lines) > _MAX_TRACEBACK_LINES
            or self.nbytes > _MAX_TRACEBACK_BYTES
        )

    def traceback_text(self) -> str:
        return '\n'.join(
            item.message for item in self.lines if item.message is not None
        )

    def raw_lines(self) -> Iterator[str]:
        for item in self.lines:
            yield item.raw

    def compacted_lines(self) -> Iterator[str]:
        # Once a segment is positively identified, only representation-level
        # traceback scaffolding is removed.  Terminal exception summaries and
        # unrelated/interleaved records remain exactly as Docker emitted them.
        for item in self.lines:
            if item.role in {'sideband', 'terminal'}:
                yield item.raw


def _parse_compose_line(line: str) -> _ComposeLine | None:
    clean = _ANSI_ESCAPE_RE.sub('', line.rstrip('\r\n'))
    prefix, sep, message = clean.partition('|')
    if not sep:
        return None
    service = prefix.strip()
    if not service:
        return None
    # docker compose formats records as ``<padded service> | <message>``.
    # Remove only the delimiter's one space so traceback indentation survives.
    if message.startswith(' '):
        message = message[1:]
    message = _DOCKER_TIMESTAMP_RE.sub('', message, count=1)
    return _ComposeLine(raw=line, service=service, message=message)


def _is_litellm(line: _ComposeLine | None) -> bool:
    return line is not None and _LITELLM_SERVICE_HINT in line.service.lower()


def _is_log_record(message: str) -> bool:
    return _LOG_RECORD_RE.match(message) is not None


def _match_pattern(buffer: _TracebackBuffer, terminal: str):
    traceback_text = buffer.traceback_text()
    for pattern in LITELLM_TRACEBACK_PATTERNS:
        if pattern.matches(traceback_text, terminal):
            return pattern
    return None


def compact_litellm_tracebacks(lines: Iterable[str]) -> Iterator[str]:
    """Compact only explicitly registered LiteLLM traceback segments.

    Unknown tracebacks, near misses, non-LiteLLM services, and malformed input
    pass through unchanged.  The function consumes the input incrementally and
    buffers at most one bounded traceback segment at a time.
    """

    pending: _TracebackBuffer | None = None
    passthrough_service: str | None = None

    for raw in lines:
        parsed = _parse_compose_line(raw)

        if passthrough_service is not None:
            yield raw
            if parsed is not None and parsed.service == passthrough_service:
                message = parsed.message
                if message and not message[:1].isspace() and not _is_log_record(message):
                    passthrough_service = None
            continue

        if pending is None:
            if parsed is not None and _is_litellm(parsed) and parsed.message == _TRACEBACK_HEADER:
                pending = _TracebackBuffer(parsed.service, parsed)
            else:
                yield raw
            continue

        # Preserve all interleaved records in-order while deciding whether the
        # LiteLLM traceback is one of the registered noisy shapes.
        if parsed is None or parsed.service != pending.service:
            pending.add(parsed, raw, 'sideband')
        else:
            message = parsed.message
            stripped = message.strip()
            if message == _TRACEBACK_HEADER:
                # A second header before a terminal exception is ambiguous.
                # Fail open for the first candidate, then let this header start
                # a fresh candidate on the next iteration path.
                yield from pending.raw_lines()
                pending = _TracebackBuffer(parsed.service, parsed)
                continue
            if not stripped:
                pending.add(parsed, raw, 'scaffold')
            elif message[:1].isspace():
                pending.add(parsed, raw, 'scaffold')
            elif _is_log_record(message):
                pending.add(parsed, raw, 'sideband')
            else:
                pending.add(parsed, raw, 'terminal')
                pattern = _match_pattern(pending, message)
                if pattern is None:
                    yield from pending.raw_lines()
                else:
                    yield from pending.compacted_lines()
                pending = None
                continue

        if pending.too_large:
            # A never-ending/malformed traceback must not hold a live log stream
            # hostage.  Flush exactly what we saw and stop filtering that segment
            # until its first terminal-looking line arrives.
            service = pending.service
            yield from pending.raw_lines()
            pending = None
            passthrough_service = service

    if pending is not None:
        # EOF in the middle of a traceback is incomplete evidence, therefore raw.
        yield from pending.raw_lines()


class LogLineSplitter:
    r"""Cut a container's raw output into display lines, ``\r`` included.

    A progress bar (a model download) redraws one line with ``\r`` and may not
    write a newline for minutes. Treating ``\r`` as a line end shows it while
    it moves; keeping at most one redraw per ``every`` seconds stops it
    flooding the pane.

    Example:
        >>> t = [0.0]
        >>> s = LogLineSplitter(every=2.0, clock=lambda: t[0])
        >>> s.feed('start\nfetch 1%\rfetch 2%\r')
        ['start', 'fetch 1%']
        >>> t[0] = 3.0
        >>> s.feed('fetch 9%\rdone\n')
        ['fetch 9%', 'done']
        >>> t[0] = 10.0                     # tqdm starts each redraw with \r
        >>> s.feed('\rpart 1\rpart 2\r')
        ['part 1']
    """

    def __init__(self, *, every: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.every = every
        self.clock = clock
        self._buf = ''
        self._last_redraw = float('-inf')

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            cut = min((i for i in (self._buf.find('\n'), self._buf.find('\r')) if i >= 0),
                      default=-1)
            if cut < 0:
                return out
            line, end = self._buf[:cut], self._buf[cut]
            self._buf = self._buf[cut + 1:]
            if not line.strip():
                if end == '\r' and self._buf.startswith('\n'):
                    self._buf = self._buf[1:]
                continue                                # nothing to show, no slot used
            if end == '\r':
                if self._buf.startswith('\n'):         # a CRLF line ending
                    self._buf = self._buf[1:]
                elif self.clock() - self._last_redraw < self.every:
                    continue                            # a redraw too soon after the last
                else:
                    self._last_redraw = self.clock()
            out.append(line)

    def flush(self) -> list[str]:
        line, self._buf = self._buf, ''
        return [line] if line.strip() else []
