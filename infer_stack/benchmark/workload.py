"""Synthetic, reproducible workload and OpenAI streaming observations."""
from __future__ import annotations

import json
import random
import time

import requests

WORKLOAD = 'synthetic-code-review-v1'
INSTRUCTION = ('Review this synthetic source listing. Write a detailed technical review, '
               'with concrete examples and a numbered list of improvements. Continue until the review is complete.\n')


def prompt(nlines, seed):
    rng = random.Random(seed)
    lines = [f'# Benchmark instance {rng.getrandbits(128):032x}\n', INSTRUCTION]
    for i in range(nlines):
        a, b, c = (rng.randrange(1, 99999) for _ in range(3))
        lines.append(f'def process_{i:06d}(value): return (value * {a} + {b}) % {c}\n')
    return ''.join(lines)


def request(base_url, key, model, text, *, protocol='chat', max_tokens=512,
            timeout=180, effort=None):
    payload = {'model': model, 'max_tokens': max_tokens, 'stream': True,
               'stream_options': {'include_usage': True}, 'temperature': 0.0}
    if protocol == 'completions':
        route = '/completions'
        payload['prompt'] = text
    else:
        route = '/chat/completions'
        payload['messages'] = [{'role': 'user', 'content': text}]
        if effort:
            payload['reasoning_effort'] = effort
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    started = time.monotonic()
    first = last = None
    usage = None
    chunks = 0
    finish = None
    with requests.post(base_url.rstrip('/') + route, json=payload, headers=headers,
                       stream=True, timeout=(15, timeout)) as response:
        if not response.ok:
            # Response bodies can contain credentials or echoed prompts. Keep
            # the status, not arbitrary upstream text, in the portable ledger.
            raise RuntimeError(f'HTTP {response.status_code}')
        for line in response.iter_lines(chunk_size=1, decode_unicode=False):
            now = time.monotonic()
            if now - started > timeout:
                raise TimeoutError('benchmark request exceeded wall-clock limit')
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError('stream reported an error')
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                if any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls')) or choice.get('text'):
                    first = now if first is None else first
                    last = now
                    chunks += 1
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
    elapsed = time.monotonic() - started
    if finish is None and usage is None:
        raise RuntimeError('incomplete SSE stream (no finish reason or usage)')
    usage = usage or {}
    p = usage.get('prompt_tokens')
    n = usage.get('completion_tokens')
    ttft = None if first is None else first - started
    duration = None if first is None or last is None else last - first
    return {'prompt_tokens': p, 'completion_tokens': n,
            'cached_prompt_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
            'elapsed_seconds': elapsed, 'ttft_seconds': ttft,
            'output_chunks': chunks, 'finish_reason': finish,
            'e2e_completion_tps': n / elapsed if n is not None and elapsed else None,
            'stream_tps_estimate': (n - 1) / duration if n is not None and n > 1 and duration and chunks > 1 else None,
            'prompt_tokens_per_ttft': p / ttft if p is not None and ttft else None}

