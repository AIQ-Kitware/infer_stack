from __future__ import annotations

import time
from typing import Any

import requests


def run_benchmark(
    base_url: str, api_key: str, model: str, prompts: list[str]
) -> dict[str, Any]:
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    results = []
    for prompt in prompts:
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 128,
        }
        t0 = time.time()
        resp = requests.post(
            f'{base_url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=300,
        )
        dt = time.time() - t0
        results.append(
            {
                'prompt': prompt,
                'status_code': resp.status_code,
                'elapsed_s': dt,
                'ok': resp.ok,
            }
        )
    return {'results': results}
